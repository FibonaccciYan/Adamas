from adamas.utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper
from adamas.utils.kv_cache import KvCache
from adamas.utils.utils import TensorLayout

import torch

class InferenceController:
    def __init__(
        self,
        num_layers,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        page_budget, # Real page budget including the last page
        max_seq_len, # Real max for allocating kv / hadamard
        dtype,
        device,      
    ):
        max_kv_pages_num = (max_seq_len + page_size - 1) // page_size
        self.kv_cache = KvCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            page_size=page_size,
            items=2,
            dtype=dtype,
            device=device
        )
        self.hadamard_cache = KvCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim//8,
            max_seq_len=max_kv_pages_num,
            page_size=page_size,
            items=1,
            dtype=dtype,
            device=device
        )
        self.layout = TensorLayout.NHD # Arbitrarily choose NHD. 
        self.device = device
        self.dtype = dtype

        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_qo_heads // num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size

        self._page_budget = page_budget
        self._decode_handler = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")

        self.kv_indices_with_last = None
        self.kv_indices_without_last = None
        self.hadamard_indices = None
        self.kv_last_page_idx = None # For decoding self-attention
        self.hadamard_last_page_idx = None

        self.kv_indptr_for_append = None
        self.hadamard_indptr_for_append = None
        self.kv_indptr_for_approx_decode = None

        self.inference_page_budget = None

        self.topk_dout_buffer = None
        self.topk_dindices_buffer = None
        self.group_topk_dout_buffer = None
        self.group_topk_dindices_buffer = None
        self.topk_buf = None
        self.estimate_topk_candidate_values = None
        self.estimate_topk_candidate_indices = None
        self.estimate_topk_num_chunks = 0
        self.estimate_topk_local_k = 0
        self.estimate_topk_chunk_size = 1024

        self._kv_indices_all = torch.arange(
            self.kv_cache.pool.capacity, dtype=torch.int32, device=self.device
        )
        self._hadamard_indices_all = torch.arange(
            self.hadamard_cache.pool.capacity, dtype=torch.int32, device=self.device
        )

        self.kv_indptr_for_append = torch.empty(2, dtype=torch.int32, device=self.device)
        self.hadamard_indptr_for_append = torch.empty(2, dtype=torch.int32, device=self.device)
        self.kv_indptr_for_approx_decode = torch.empty(2, dtype=torch.int32, device=self.device)
        self.prefill_q_indptr = torch.empty(2, dtype=torch.int32, device=self.device)
        self.empty_indices = torch.empty(0, dtype=torch.int32, device=self.device)

        self.kv_indptr_for_append[0] = 0
        self.hadamard_indptr_for_append[0] = 0
        self.kv_indptr_for_approx_decode[0] = 0
        self.prefill_q_indptr[0] = 0
    
    # Used for controlling the number of pages
    # Here we skip first two layers by manipulating this.
    def set_page_budget(self, page_budget: int):
        self._page_budget = page_budget

    # Called once per forwarding in all layers
    # Adjust the hadamard data for paged_kv
    def prepare_hadamard(self, seq_len: int):
        # Allocate entry for tokens
        appended_new_pages = self.kv_cache.append_seq(seq_len)
        # Allocate entry for hadamard data
        _ = self.hadamard_cache.append_seq(appended_new_pages)
    
    # Prepare hadamard data used for inference under certain PAGE_BUDGET
    # Called multiple times for layer sensitivity
    def begin_forward(self, seq_len: int, updateTensor: bool = True):
        # Allocate tensor in advance
        # This is used for append kernels, which need original indices
        if updateTensor:
            self.kv_indptr_for_append[1] = len(self.kv_cache.indicies)
            self.hadamard_indptr_for_append[1] = len(self.hadamard_cache.indicies)
            self.kv_last_page_idx = self.kv_cache.indicies[-1]
            self.hadamard_last_page_idx = self.hadamard_cache.indicies[-1]

        if seq_len > 1:
            # prefill requests
            # append_kv_cache_prefill and prefill_with_paged_kv_cache
            if updateTensor:
                self.prefill_q_indptr[1] = seq_len
                self.kv_indices_with_last = self._kv_indices_all[:len(self.kv_cache.indicies)]
                self.hadamard_indices = self._hadamard_indices_all[:len(self.hadamard_cache.indicies)]
        else:
            # decode requests
            # append_kv_cache_decode, estimate_attn_score, topk_filtering
            cur_page_nums = len(self.kv_cache.indicies)
            assert cur_page_nums > 1 # at least two pages for excluding last page

            if updateTensor:
                # used for appending
                self.kv_indices_with_last = self._kv_indices_all[:len(self.kv_cache.indicies)]

                if cur_page_nums <= min(self._page_budget, cur_page_nums):
                    self.kv_indices_without_last = self._kv_indices_all[:cur_page_nums - 1].repeat(self.num_qo_heads, 1)
                else:
                    self.kv_indices_without_last = self.empty_indices

                # used for estimate
                self.hadamard_indices = self._hadamard_indices_all[:len(self.hadamard_cache.indicies)]

            # used as page_budget for topk and approx kernel
            self.inference_page_budget = min(self._page_budget, cur_page_nums)

            # Exclude the last page for decoding
            self.kv_indptr_for_approx_decode[1] = self.inference_page_budget - 1

            if cur_page_nums > self.inference_page_budget:
                # Allocate buffer for top-k filtering
                page_budget = self.inference_page_budget - 1
                estimate_len = self.hadamard_cache.seqlen - 1
                self.topk_dout_buffer = torch.empty((self.num_qo_heads, page_budget), dtype=self.dtype, device=self.device)
                self.topk_dindices_buffer = torch.empty((self.num_qo_heads, page_budget), dtype=torch.int32, device=self.device)
                self.group_topk_dout_buffer = torch.empty((self.num_kv_heads, page_budget), dtype=self.dtype, device=self.device)
                self.group_topk_dindices_buffer = torch.empty((self.num_kv_heads, page_budget), dtype=torch.int32, device=self.device)
                self.estimate_topk_candidate_values = torch.empty((self.num_kv_heads, estimate_len), dtype=self.dtype, device=self.device)
                self.estimate_topk_candidate_indices = torch.empty((self.num_kv_heads, estimate_len), dtype=torch.int32, device=self.device)
                self.topk_buf = torch.empty((self.num_qo_heads, 8192 * 2 * (2+4) // 2 // 48), dtype=self.dtype, device=self.device)

            self._decode_handler.begin_forward(
                self.kv_indptr_for_approx_decode,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                self.dtype
            )
    
    # Used for releasing resources
    # Free memory in CUDA side
    # called multiple times for layer sensitivity
    def end_forward(self):
        self._decode_handler.end_forward()
    
    def need_estimate(self) -> bool:
        if self.inference_page_budget is None:
            return False
        
        cur_page_nums = len(self.kv_cache.indicies)
        return cur_page_nums > self.inference_page_budget
    
    def clean_states(self):
        self.kv_cache.release()
        self.hadamard_cache.release()
        
