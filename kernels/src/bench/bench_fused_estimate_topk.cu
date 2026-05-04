#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <decode/decode_attn.cuh>
#include <nvbench/nvbench.cuh>
#include <topk/decode_select_k.cuh>

#include "cpu_utils.h"

using utils::vec_bytes;
using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

namespace adamas_bench {

template <typename T>
__global__ void BroadcastGroupTopKKernel(T* __restrict__ group_values,
										 int32_t* __restrict__ group_indices,
										 T* __restrict__ qo_values,
										 int32_t* __restrict__ qo_indices,
										 uint32_t num_qo_heads,
										 uint32_t num_kv_heads,
										 uint32_t k) {
	const uint32_t qo_head_idx = blockIdx.x;
	const uint32_t tx = threadIdx.x;
	const uint32_t group_size = num_qo_heads / num_kv_heads;
	const uint32_t kv_head_idx = qo_head_idx / group_size;

	for(uint32_t i = tx; i < k; i += blockDim.x) {
		qo_values[qo_head_idx * k + i] = group_values[kv_head_idx * k + i];
		qo_indices[qo_head_idx * k + i] = group_indices[kv_head_idx * k + i];
	}
}

template <typename T>
cudaError_t BroadcastGroupTopK(T* group_values,
							   int32_t* group_indices,
							   T* qo_values,
							   int32_t* qo_indices,
							   uint32_t num_qo_heads,
							   uint32_t num_kv_heads,
							   uint32_t k,
							   cudaStream_t stream = nullptr) {
	dim3 nblks(num_qo_heads);
	dim3 nthrs(256);
	auto kernel = BroadcastGroupTopKKernel<T>;
	void* args[] = {
		(void*)&group_values,
		(void*)&group_indices,
		(void*)&qo_values,
		(void*)&qo_indices,
		(void*)&num_qo_heads,
		(void*)&num_kv_heads,
		(void*)&k};
	return cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream);
}

} // namespace adamas_bench

template <typename T>
void bench_fused_estimate_topk(nvbench::state& state) {
	constexpr size_t batch_size = 1;
	constexpr size_t head_dim = 128;
	constexpr size_t hadamard_dim = head_dim / 8;
	constexpr auto rotary_mode = RotaryMode::kNone;

	size_t seqlen = state.get_int64("seqlen");
	size_t page_size = state.get_int64("page_size");
	size_t page_budget = state.get_int64("page_budget");
	size_t num_qo_heads = state.get_int64("num_qo_heads");
	size_t num_kv_heads = state.get_int64("num_kv_heads");

	if(num_qo_heads % num_kv_heads != 0) {
		state.skip("num_qo_heads must be divisible by num_kv_heads");
	}
	if(num_kv_heads != 8) {
		state.skip("decode_select_k instantiation in this bench supports num_kv_heads=8");
	}

	size_t num_pages = flashinfer::ceil_div(seqlen, page_size);
	size_t last_page_len = (seqlen - 1) % page_size + 1;
	int32_t last_page_idx = static_cast<int32_t>(num_pages - 1);
	size_t estimate_len = seqlen - 1;
	size_t k = std::min(page_budget, num_pages) - 1;

	if(k == 0 || k > estimate_len) {
		state.skip("invalid top-k size");
	}

	std::vector<int32_t> hadamard_indptr_host({0, static_cast<int32_t>(num_pages)});
	std::vector<int32_t> hadamard_indices_host(num_pages);
	std::iota(hadamard_indices_host.begin(), hadamard_indices_host.end(), 0);
	std::shuffle(hadamard_indices_host.begin(), hadamard_indices_host.end(), std::mt19937(0));
	last_page_idx = hadamard_indices_host.back();

	thrust::device_vector<T> q(batch_size * num_qo_heads * hadamard_dim);
	thrust::device_vector<T> hadamard_data(num_pages * page_size * num_kv_heads * hadamard_dim);
	thrust::device_vector<int32_t> hadamard_indptr(hadamard_indptr_host);
	thrust::device_vector<int32_t> hadamard_indices(hadamard_indices_host);

	thrust::device_vector<T> candidate_values(num_kv_heads * estimate_len);
	thrust::device_vector<T> group_topk_values(num_kv_heads * k);
	thrust::device_vector<int32_t> group_topk_indices(num_kv_heads * k);
	thrust::device_vector<T> qo_topk_values(num_qo_heads * k);
	thrust::device_vector<int32_t> qo_topk_indices(num_qo_heads * k);

	const size_t topk_buf_size_bytes =
		num_kv_heads * estimate_len * (sizeof(T) + sizeof(int32_t)) * 2 / 24;
	thrust::device_vector<char> topk_buf(std::max<size_t>(topk_buf_size_bytes, 1));

	paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_hadamard(
		num_kv_heads,
		page_size,
		hadamard_dim,
		batch_size,
		0,
		last_page_len,
		last_page_idx,
		thrust::raw_pointer_cast(hadamard_data.data()),
		thrust::raw_pointer_cast(hadamard_indices.data()),
		thrust::raw_pointer_cast(hadamard_indptr.data()));

	state.add_global_memory_reads<uint8_t>(
		vec_bytes(q) + vec_bytes(hadamard_data) + vec_bytes(hadamard_indptr) +
			vec_bytes(hadamard_indices) + vec_bytes(candidate_values),
		"Read");
	state.add_global_memory_writes<uint8_t>(
		vec_bytes(candidate_values) + vec_bytes(group_topk_values) +
			vec_bytes(group_topk_indices) + vec_bytes(qo_topk_values) + vec_bytes(qo_topk_indices),
		"Write");

	state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
		cudaError_t status =
			MaxPossibleSampleGroupMinWithPagedKVCache<PageStorage::kIndices,
													  kv_layout,
													  T,
													  T,
													  int32_t>(
				thrust::raw_pointer_cast(q.data()),
				paged_hadamard,
				thrust::raw_pointer_cast(candidate_values.data()),
				num_qo_heads,
				estimate_len,
				rotary_mode);
		if(status != cudaSuccess) {
			state.skip("group-min estimate CUDA error: " + std::string(cudaGetErrorString(status)));
			return;
		}

		decode_select_k<T, int32_t, 8>(
			thrust::raw_pointer_cast(candidate_values.data()),
			nullptr,
			thrust::raw_pointer_cast(topk_buf.data()),
			estimate_len,
			k,
			thrust::raw_pointer_cast(group_topk_values.data()),
			thrust::raw_pointer_cast(group_topk_indices.data()),
			false);

		status = adamas_bench::BroadcastGroupTopK<T>(
			thrust::raw_pointer_cast(group_topk_values.data()),
			thrust::raw_pointer_cast(group_topk_indices.data()),
			thrust::raw_pointer_cast(qo_topk_values.data()),
			thrust::raw_pointer_cast(qo_topk_indices.data()),
			num_qo_heads,
			num_kv_heads,
			k);
		if(status != cudaSuccess) {
			state.skip("broadcast CUDA error: " + std::string(cudaGetErrorString(status)));
		}
	});
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FUSED_ESTIMATE_TOPK(dtype)                                         \
	auto bench_fused_estimate_topk_##dtype##_ = bench_fused_estimate_topk<dtype>; \
	NVBENCH_BENCH(bench_fused_estimate_topk_##dtype##_)                           \
		.set_name("bench_fused_estimate_topk_" STR(dtype))                        \
		.add_int64_axis("seqlen", {16384, 32768, 65536, 131072})                  \
		.add_int64_axis("page_budget", {256, 512, 1024, 2048, 4096})                   \
		.add_int64_axis("page_size", {1})                                         \
		.add_int64_axis("num_qo_heads", {32})                                     \
		.add_int64_axis("num_kv_heads", {8})

BENCH_FUSED_ESTIMATE_TOPK(half);
