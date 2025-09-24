#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void append_kv_cache_decode(torch::Tensor k,
							torch::Tensor v,
							torch::Tensor h,
							torch::Tensor o,
							torch::Tensor kv_data,
							torch::Tensor kv_indices,
							torch::Tensor kv_indptr,
							unsigned int kv_last_page_len,
							unsigned int kv_last_page_idx,
							torch::Tensor hadamard_data,
							torch::Tensor hadamard_indices,
							torch::Tensor hadamard_indptr,
							unsigned int hadamard_last_page_len,
							unsigned int hadamard_last_page_idx,
							unsigned int layout) {
	constexpr size_t batch_size = 1;
	CHECK_INPUT(k); // [seq_len, num_heads, head_dim]
	CHECK_INPUT(v); // [seq_len, num_heads, head_dim]
	CHECK_INPUT(h); // [2*seq_len, num_heads, head_dim]
	CHECK_INPUT(o); // [seq_len, num_heads, head_dim/8]
	// (num_max_pages, 2, H_kv, page_size, head_dim) for HND
	// (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
	CHECK_INPUT(kv_data);
	CHECK_INPUT(kv_indices); // [num_pages]
	CHECK_INPUT(hadamard_data);
	CHECK_INPUT(hadamard_indices); // [num_pages]

	CHECK_DIM(1, kv_indices);
	CHECK_DIM(1, hadamard_indices);
	CHECK_DIM(3, k);
	CHECK_DIM(3, v);
	CHECK_DIM(3, h);
	CHECK_DIM(3, o);
	CHECK_DIM(5, kv_data);
	CHECK_DIM(5, hadamard_data);

	CHECK_EQ(k.size(0), 1); // decode
	CHECK_EQ(v.size(0), 1); // decode
	CHECK_EQ(h.size(0), 2*1); // decode
	CHECK_EQ(o.size(0), 1); // decode
	CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(hadamard_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);
	CHECK_EQ(hadamard_indptr.scalar_type(), torch::kInt32);

	size_t num_heads = k.size(1);
	size_t head_dim = k.size(2);
	size_t page_size;
	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		page_size = kv_data.size(3);
		CHECK_EQ(kv_data.size(2), num_heads);
		CHECK_EQ(kv_data.size(4), head_dim);
	} else {
		page_size = kv_data.size(2);
		CHECK_EQ(kv_data.size(3), num_heads);
		CHECK_EQ(kv_data.size(4), head_dim);
	}

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(k.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_heads,
				page_size,
				head_dim,
				batch_size,
				0,
				kv_last_page_len,
				kv_last_page_idx,
				static_cast<c_type*>(kv_data.data_ptr()),
				static_cast<int32_t*>(kv_indices.data_ptr()),
				static_cast<int32_t*>(kv_indptr.data_ptr()));

			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_hadamard(
				num_heads,
				page_size,
				head_dim/8,
				batch_size,
				0,
				hadamard_last_page_len,
				hadamard_last_page_idx,
				static_cast<c_type*>(hadamard_data.data_ptr()),
				static_cast<int32_t*>(hadamard_indices.data_ptr()),
				static_cast<int32_t*>(hadamard_indptr.data_ptr()));

			cudaError_t status =
				AppendPagedKVCacheDecode<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t>(
					paged_kv,
					paged_hadamard,
					static_cast<c_type*>(k.data_ptr()),
					static_cast<c_type*>(v.data_ptr()),
					static_cast<c_type*>(h.data_ptr()),
					static_cast<c_type*>(o.data_ptr()),
					nullptr);

			TORCH_CHECK(status == cudaSuccess,
						"Append_kv_cache_decode failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});

	TORCH_CHECK(success, "Append_kv_cache_decode failed to dispatch with dtype ", k.scalar_type());
}

void append_kv_cache_prefill(torch::Tensor k,
							 torch::Tensor v,
							 torch::Tensor h,
							 torch::Tensor kv_data,
							 torch::Tensor kv_indices,
							 torch::Tensor kv_indptr,
							 unsigned int kv_last_page_len,
							 unsigned int kv_last_page_idx,
							 torch::Tensor hadamard_data,
							 torch::Tensor hadamard_indices,
							 torch::Tensor hadamard_indptr,
							 unsigned int hadamard_last_page_len,
							 unsigned int hadamard_last_page_idx,
							 unsigned int layout) {
	constexpr size_t batch_size = 1;

#ifdef BSK_TORCH_CHECK
	CHECK_INPUT(k); // [seq_len, num_heads, head_dim]
	CHECK_INPUT(v); // [seq_len, num_heads, head_dim]
	CHECK_INPUT(h); // [seq_len, num_heads, head_dim]
	// (num_max_pages, 2, H_kv, page_size, head_dim) for HND
	// (num_max_pages, 2, page_size, H_kv, head_dim) for NHD
	CHECK_INPUT(kv_data);
	CHECK_INPUT(kv_indices); // [num_pages]
	CHECK_INPUT(hadamard_data);
	CHECK_INPUT(hadamard_indices); // [num_pages]

	CHECK_DIM(1, kv_indices);
	CHECK_DIM(1, hadamard_indices);
	CHECK_DIM(3, k);
	CHECK_DIM(3, v);
	CHECK_DIM(3, h);
	CHECK_DIM(5, kv_data);
	CHECK_DIM(5, hadamard_data);

	CHECK_GE(k.size(0), 2); // Prefill
	CHECK_GE(v.size(0), 2); // Prefill
	CHECK_GE(h.size(0), 2); // Prefill
	CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(hadamard_indices.scalar_type(), torch::kInt32);
	CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);
	CHECK_EQ(hadamard_indptr.scalar_type(), torch::kInt32);
#endif

	size_t seq_len = k.size(0);
	size_t num_heads = k.size(1);
	size_t head_dim = k.size(2);
	size_t page_size;
	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		page_size = kv_data.size(3);
#ifdef BSK_TORCH_CHECK
		CHECK_EQ(kv_data.size(2), num_heads);
		CHECK_EQ(kv_data.size(4), head_dim);
#endif
	} else {
		page_size = kv_data.size(2);
#ifdef BSK_TORCH_CHECK
		CHECK_EQ(kv_data.size(3), num_heads);
		CHECK_EQ(kv_data.size(4), head_dim);
#endif
	}

#ifdef BSK_TORCH_CHECK
	CHECK_EQ(seq_len, v.size(0));
	CHECK_EQ(seq_len, h.size(0));
#endif

	torch::Tensor append_indptr =
		torch::tensor({0, static_cast<int32_t>(seq_len)}, kv_indices.options());

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(k.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_kv(
				num_heads,
				page_size,
				head_dim,
				batch_size,
				0,
				kv_last_page_len,
				kv_last_page_idx,
				static_cast<c_type*>(kv_data.data_ptr()),
				static_cast<int32_t*>(kv_indices.data_ptr()),
				static_cast<int32_t*>(kv_indptr.data_ptr()));

			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_hadamard(
				num_heads,
				page_size,
				head_dim/8,
				batch_size,
				0,
				hadamard_last_page_len,
				hadamard_last_page_idx,
				static_cast<c_type*>(hadamard_data.data_ptr()),
				static_cast<int32_t*>(hadamard_indices.data_ptr()),
				static_cast<int32_t*>(hadamard_indptr.data_ptr()));

			cudaError_t status =
				AppendPagedKVCachePrefill<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t>(
					paged_kv,
					paged_hadamard,
					static_cast<c_type*>(k.data_ptr()),
					static_cast<c_type*>(v.data_ptr()),
					static_cast<c_type*>(h.data_ptr()),
					static_cast<int32_t*>(append_indptr.data_ptr()),
					nullptr);

			TORCH_CHECK(status == cudaSuccess,
						"Append_kv_cache_prefill failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});

	TORCH_CHECK(success, "Append_kv_cache_prefill failed to dispatch with dtype ", k.scalar_type());
}

void apply_rope_in_place(torch::Tensor q,
						 torch::Tensor k,
						 unsigned int past_kv_len,
						 float rope_scale,
						 float rope_theta) {
#ifdef BSK_TORCH_CHECK
	// Note: input layout is always NHD. Not Paged.
	CHECK_INPUT(q); // [seq_len, num_heads, head_dim]
	CHECK_INPUT(k); // [seq_len, num_heads, head_dim]

	CHECK_DIM(3, q);
	CHECK_DIM(3, k);

	CHECK_EQ(q.size(0), k.size(0));
	CHECK_EQ(q.size(1), k.size(1));
	CHECK_EQ(q.size(2), k.size(2));
#endif

	size_t seq_len = q.size(0);
	size_t num_heads = q.size(1);
	size_t head_dim = q.size(2);

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		cudaError_t status = QKApplyRotaryInPlace<c_type>(static_cast<c_type*>(q.data_ptr()),
														  static_cast<c_type*>(k.data_ptr()),
														  seq_len,
														  past_kv_len,
														  num_heads,
														  num_heads,
														  head_dim,
														  rope_scale,
														  rope_theta,
														  nullptr);

		TORCH_CHECK(status == cudaSuccess,
					"apply_rope_in_place failed with error code ",
					cudaGetErrorString(status));
		return true;
	});

	TORCH_CHECK(success, "apply_rope_in_place failed to dispatch with dtype ", k.scalar_type());
}

void pack_2bit(torch::Tensor h,
				torch::Tensor o,
				unsigned int past_kv_len) {
#ifdef BSK_TORCH_CHECK
	// Note: input layout is always NHD. Not Paged.
	CHECK_INPUT(h); // [2*seq_len, num_heads, head_dim]
	CHECK_INPUT(o); // [2*seq_len, num_heads, head_dim/8]

	CHECK_DIM(3, h);
	CHECK_DIM(3, o);
#endif

	size_t seq_len = h.size(0);
	size_t num_heads = h.size(1);
	size_t head_dim = h.size(2);

	cudaError_t status = HPack2bit<nv_half, nv_half>(static_cast<nv_half*>(h.data_ptr()),
													static_cast<nv_half*>(o.data_ptr()),
													seq_len,
													past_kv_len,
													num_heads,
													head_dim,
													nullptr);

	TORCH_CHECK(status == cudaSuccess,
				"pack_2bit failed with error code ",
				cudaGetErrorString(status));
}