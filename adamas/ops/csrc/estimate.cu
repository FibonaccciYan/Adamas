#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

#include "decode/decode_attn.cuh"
#include "topk/decode_select_k.cuh"

using namespace flashinfer;

namespace adamas_estimate_topk {

template <typename DType>
__global__ void BroadcastGroupTopKKernel(DType* __restrict__ group_values,
										 int32_t* __restrict__ group_indices,
										 DType* __restrict__ qo_values,
										 int32_t* __restrict__ qo_indices,
										 uint32_t num_qo_heads,
										 uint32_t num_kv_heads,
										 uint32_t page_budget) {
	const uint32_t qo_head_idx = blockIdx.x;
	const uint32_t tx = threadIdx.x;
	const uint32_t group_size = num_qo_heads / num_kv_heads;
	const uint32_t kv_head_idx = qo_head_idx / group_size;
	for(uint32_t i = tx; i < page_budget; i += blockDim.x) {
		qo_values[qo_head_idx * page_budget + i] = group_values[kv_head_idx * page_budget + i];
		qo_indices[qo_head_idx * page_budget + i] = group_indices[kv_head_idx * page_budget + i];
	}
}

template <typename DType>
cudaError_t BroadcastGroupTopK(DType* group_values,
							   int32_t* group_indices,
							   DType* qo_values,
							   int32_t* qo_indices,
							   uint32_t num_qo_heads,
							   uint32_t num_kv_heads,
							   uint32_t page_budget,
							   cudaStream_t stream = nullptr) {
	dim3 nblks(num_qo_heads);
	dim3 nthrs(256);
	auto kernel = BroadcastGroupTopKKernel<DType>;
	void* args[] = {
		(void*)&group_values,
		(void*)&group_indices,
		(void*)&qo_values,
		(void*)&qo_indices,
		(void*)&num_qo_heads,
		(void*)&num_kv_heads,
		(void*)&page_budget};
	FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
	return cudaSuccess;
}

} // namespace adamas_estimate_topk

void estimate_attn_score(torch::Tensor q,
						torch::Tensor o,
						torch::Tensor hadamard_data,
						torch::Tensor hadamard_indices,
						torch::Tensor hadamard_indptr,
						unsigned int hadamard_last_page_len,
						unsigned int hadamard_last_page_idx,
						unsigned int layout) {
	constexpr size_t batch_size = 1;

	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(q); // [1, num_qo_heads, hadamard_dim]
	// (num_max_pages, 1, H_kv, page_size, hadamard_dim) for HND
	// (num_max_pages, 1, page_size, H_kv, hadamard_dim) for NHD
	CHECK_INPUT(hadamard_data);
	CHECK_INPUT(hadamard_indices);

	CHECK_DIM(3, q);
	CHECK_DIM(5, hadamard_data);
	CHECK_DIM(1, hadamard_indices);

	CHECK_EQ(q.size(0), 1);
	CHECK_EQ(hadamard_indices.scalar_type(), torch::kInt32);
	#endif

	size_t num_qo_heads = q.size(1);
	size_t hadamard_dim = q.size(2);
	size_t page_size, num_kv_heads;

	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		page_size = hadamard_data.size(3);
		num_kv_heads = hadamard_data.size(2);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(num_qo_heads % num_kv_heads, 0);
		CHECK_EQ(hadamard_data.size(4), hadamard_dim);
		#endif
	} else {
		page_size = hadamard_data.size(2);
		num_kv_heads = hadamard_data.size(3);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(num_qo_heads % num_kv_heads, 0);
		CHECK_EQ(hadamard_data.size(4), hadamard_dim);
		#endif
	}

	uint32_t output_len = o.size(1);
		
	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_hadamard(
				num_kv_heads,
				page_size,
				hadamard_dim,
				batch_size,
				0,
				hadamard_last_page_len,
				hadamard_last_page_idx,
				static_cast<c_type*>(hadamard_data.data_ptr()),
				static_cast<int32_t*>(hadamard_indices.data_ptr()),
				static_cast<int32_t*>(hadamard_indptr.data_ptr()));
			cudaError_t status =
				MaxPossibleSampleWithPagedKVCache<PageStorage::kIndices,
												KV_LAYOUT,
												c_type,
												c_type,
												int32_t>(static_cast<c_type*>(q.data_ptr()),
														paged_hadamard,
														static_cast<c_type*>(o.data_ptr()),
														num_qo_heads,
														output_len,
														/*rotary_mode*/ RotaryMode::kNone);
			TORCH_CHECK(status == cudaSuccess,
						"Estimate_attn_score failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});
	TORCH_CHECK(success, "Estimate_attn_score failed to dispatch with dtype ", q.scalar_type());
}

void estimate_topk_filtering(torch::Tensor q,
							 torch::Tensor group_topk_values,
							 torch::Tensor group_topk_indices,
							 torch::Tensor topk_values,
							 torch::Tensor topk_indices,
							 torch::Tensor candidate_values,
							 torch::Tensor candidate_indices,
							 torch::Tensor topk_buf,
							 torch::Tensor hadamard_data,
							 torch::Tensor hadamard_indices,
							 torch::Tensor hadamard_indptr,
							 unsigned int hadamard_last_page_len,
							 unsigned int hadamard_last_page_idx,
							 unsigned int layout,
							 unsigned int page_budget) {
	constexpr size_t batch_size = 1;

	#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(q);
	CHECK_INPUT(group_topk_values);
	CHECK_INPUT(group_topk_indices);
	CHECK_INPUT(topk_values);
	CHECK_INPUT(topk_indices);
	CHECK_INPUT(candidate_values);
	CHECK_INPUT(candidate_indices);
	CHECK_INPUT(topk_buf);
	CHECK_INPUT(hadamard_data);
	CHECK_INPUT(hadamard_indices);
	CHECK_INPUT(hadamard_indptr);
	CHECK_DIM(3, q);
	CHECK_DIM(2, group_topk_values);
	CHECK_DIM(2, group_topk_indices);
	CHECK_DIM(2, topk_values);
	CHECK_DIM(2, topk_indices);
	CHECK_DIM(2, candidate_values);
	CHECK_DIM(2, candidate_indices);
	CHECK_DIM(5, hadamard_data);
	CHECK_DIM(1, hadamard_indices);
	CHECK_DIM(1, hadamard_indptr);
	CHECK_EQ(q.size(0), 1);
	CHECK_EQ(group_topk_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(topk_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(candidate_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(hadamard_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(hadamard_indptr.scalar_type(), torch::kInt32);
	#endif

	size_t num_qo_heads = q.size(1);
	size_t hadamard_dim = q.size(2);
	size_t page_size, num_kv_heads;
	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		page_size = hadamard_data.size(3);
		num_kv_heads = hadamard_data.size(2);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(hadamard_data.size(4), hadamard_dim);
		#endif
	} else {
		page_size = hadamard_data.size(2);
		num_kv_heads = hadamard_data.size(3);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(hadamard_data.size(4), hadamard_dim);
		#endif
	}

	const uint32_t total_h_len =
		(static_cast<uint32_t>(hadamard_indices.size(0)) - 1U) * page_size + hadamard_last_page_len;
	const uint32_t estimate_len = total_h_len - 1U;

	#ifdef BSK_TORCH_CHECK
	CHECK_EQ(num_qo_heads % num_kv_heads, 0);
	CHECK_EQ(group_topk_values.size(0), num_kv_heads);
	CHECK_EQ(group_topk_indices.size(0), num_kv_heads);
	CHECK_EQ(group_topk_values.size(1), page_budget);
	CHECK_EQ(group_topk_indices.size(1), page_budget);
	CHECK_EQ(topk_values.size(0), num_qo_heads);
	CHECK_EQ(topk_indices.size(0), num_qo_heads);
	CHECK_EQ(topk_values.size(1), page_budget);
	CHECK_EQ(topk_indices.size(1), page_budget);
	CHECK_EQ(candidate_values.size(0), num_kv_heads);
	CHECK_EQ(candidate_indices.size(0), num_kv_heads);
	CHECK_EQ(candidate_values.size(1), estimate_len);
	CHECK_EQ(candidate_indices.size(1), estimate_len);
	CHECK_GE(estimate_len, page_budget);
	#endif

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_hadamard(
				num_kv_heads,
				page_size,
				hadamard_dim,
				batch_size,
				0,
				hadamard_last_page_len,
				hadamard_last_page_idx,
				static_cast<c_type*>(hadamard_data.data_ptr()),
				static_cast<int32_t*>(hadamard_indices.data_ptr()),
				static_cast<int32_t*>(hadamard_indptr.data_ptr()));

			cudaError_t status =
				MaxPossibleSampleGroupMinWithPagedKVCache<PageStorage::kIndices,
														  KV_LAYOUT,
														  c_type,
														  c_type,
														  int32_t>(
					static_cast<c_type*>(q.data_ptr()),
					paged_hadamard,
					static_cast<c_type*>(candidate_values.data_ptr()),
					num_qo_heads,
					estimate_len,
					/*rotary_mode*/ RotaryMode::kNone);
			TORCH_CHECK(status == cudaSuccess,
						"Estimate_topk group-min estimation failed with error code ",
						cudaGetErrorString(status));

			if(num_kv_heads == 8) {
				decode_select_k<c_type, int32_t, 8>(
					static_cast<c_type*>(candidate_values.data_ptr()),
					nullptr,
					static_cast<char*>(topk_buf.data_ptr()),
					estimate_len,
					page_budget,
					static_cast<c_type*>(group_topk_values.data_ptr()),
					static_cast<int32_t*>(group_topk_indices.data_ptr()),
					false);
			} else if(num_kv_heads == 32) {
				decode_select_k<c_type, int32_t, 32>(
					static_cast<c_type*>(candidate_values.data_ptr()),
					nullptr,
					static_cast<char*>(topk_buf.data_ptr()),
					estimate_len,
					page_budget,
					static_cast<c_type*>(group_topk_values.data_ptr()),
					static_cast<int32_t*>(group_topk_indices.data_ptr()),
					false);
			} else {
				TORCH_CHECK(false, "estimate_topk_filtering only supports num_kv_heads 8 or 32, got ", num_kv_heads);
			}

			status = adamas_estimate_topk::BroadcastGroupTopK<c_type>(
				static_cast<c_type*>(group_topk_values.data_ptr()),
				static_cast<int32_t*>(group_topk_indices.data_ptr()),
				static_cast<c_type*>(topk_values.data_ptr()),
				static_cast<int32_t*>(topk_indices.data_ptr()),
				num_qo_heads,
				num_kv_heads,
				page_budget);
			TORCH_CHECK(status == cudaSuccess,
						"Estimate_topk broadcast failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});
	TORCH_CHECK(success, "Estimate_topk_filtering failed to dispatch with dtype ", q.scalar_type());
}
