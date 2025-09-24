#include "bsk_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

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
	CHECK_INPUT(q); // [1, num_heads, hadamard_dim]
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

	size_t num_heads = q.size(1);
	size_t hadamard_dim = q.size(2);
	size_t page_size;

	QKVLayout kv_layout = static_cast<QKVLayout>(layout);
	if(kv_layout == QKVLayout::kHND) {
		page_size = hadamard_data.size(3);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(hadamard_data.size(2), num_heads);
		CHECK_EQ(hadamard_data.size(4), hadamard_dim);
		#endif
	} else {
		page_size = hadamard_data.size(2);
		#ifdef BSK_TORCH_CHECK
		CHECK_EQ(hadamard_data.size(3), num_heads);
		CHECK_EQ(hadamard_data.size(4), hadamard_dim);
		#endif
	}

	// size_t output_len = (hadamard_indices.size(0) - 1) * page_size + hadamard_last_page_len - 1;
	// torch::Tensor o = torch::empty(
		// {static_cast<signed long>(num_heads), static_cast<signed long>(output_len)}, q.options());
		
	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
		SWITCH_LAYOUT(kv_layout, KV_LAYOUT, {
			paged_kv_t<PageStorage::kIndices, KV_LAYOUT, c_type, int32_t> paged_hadamard(
				num_heads,
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
														num_heads,
														/*rotary_mode*/ RotaryMode::kNone);
			TORCH_CHECK(status == cudaSuccess,
						"Estimate_attn_score failed with error code ",
						cudaGetErrorString(status));
		});
		return true;
	});
	TORCH_CHECK(success, "Estimate_attn_score failed to dispatch with dtype ", q.scalar_type());
}