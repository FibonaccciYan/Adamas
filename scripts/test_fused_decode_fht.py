import argparse

import torch
import faster_hadamard_transform

import adamas.utils


def unpack_2bit(x):
    raw = x.view(torch.int16)
    shifts = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.int16, device=x.device)
    return torch.bitwise_and(torch.bitwise_right_shift(raw.unsqueeze(-1), shifts), 0x3)


def assert_packed_equal_or_boundary(name, got, ref, ref_hadamard, boundary_tol=2e-3):
    if torch.equal(got, ref):
        return

    got_code = unpack_2bit(got)
    ref_code = unpack_2bit(ref)
    diff = got_code != ref_code
    mismatch = diff.sum().item()

    ref_values = ref_hadamard.reshape_as(ref_code)[diff].float()
    thresholds = torch.tensor([-10.0, 0.0, 10.0], device=ref_values.device)
    distance = (ref_values[:, None] - thresholds[None, :]).abs().amin(dim=1)
    non_boundary = (distance > boundary_tol).sum().item()
    if non_boundary:
        max_distance = distance.max().item()
        raise AssertionError(
            f"{name} mismatch: {mismatch} code elements differ, "
            f"{non_boundary} are not within {boundary_tol} of a threshold; "
            f"max threshold distance among mismatches={max_distance:.6g}"
        )
    print(
        f"WARN {name}: {mismatch} code elements differ only at threshold boundary "
        f"(max distance={distance.max().item():.6g})"
    )


def make_controller(dtype, device, max_seq_len, page_budget):
    return adamas.utils.InferenceController(
        num_layers=1,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=1,
        page_budget=page_budget,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )


def active_kv(controller):
    idx = torch.tensor(controller.kv_cache.indicies, dtype=torch.long, device=controller.device)
    return controller.kv_cache.buf_layer(0).index_select(0, idx)


def active_hadamard(controller):
    idx = torch.tensor(controller.hadamard_cache.indicies, dtype=torch.long, device=controller.device)
    return controller.hadamard_cache.buf_layer(0).index_select(0, idx)


@torch.inference_mode()
def run_case(dtype, kv_len, seed):
    device = torch.device("cuda:0")
    torch.manual_seed(seed)

    page_budget = min(1024, kv_len)
    max_seq_len = max(4096, kv_len + 16)

    old = make_controller(dtype, device, max_seq_len, page_budget)
    fused = make_controller(dtype, device, max_seq_len, page_budget)

    k_prefill = torch.randn(kv_len - 1, 8, 128, dtype=dtype, device=device)
    v_prefill = torch.randn(kv_len - 1, 8, 128, dtype=dtype, device=device)
    q_decode = torch.randn(1, 32, 128, dtype=dtype, device=device)
    k_decode = torch.randn(1, 8, 128, dtype=dtype, device=device)
    v_decode = torch.randn(1, 8, 128, dtype=dtype, device=device)

    h_prefill = faster_hadamard_transform.hadamard_transform(k_prefill, inplace=False)

    for ctrl in (old, fused):
        ctrl.prepare_hadamard(kv_len - 1)
        ctrl.begin_forward(kv_len - 1)
        adamas.utils.append_kvh(k_prefill, v_prefill, h_prefill, ctrl, 0)
        ctrl.end_forward()

    h_q_decode = faster_hadamard_transform.hadamard_transform(q_decode, inplace=False)
    h_k_decode = faster_hadamard_transform.hadamard_transform(k_decode, inplace=False)

    old.prepare_hadamard(1)
    old.begin_forward(1)
    q_code_old = adamas.utils.append_kvh(
        k_decode,
        v_decode,
        h_k_decode,
        old,
        0,
        h_q_decode,
    )
    old.end_forward()

    fused.prepare_hadamard(1)
    fused.begin_forward(1)
    q_code_fused = adamas.utils.append_kvh_decode_fused(
        q_decode,
        k_decode,
        v_decode,
        fused,
        0,
    )
    fused.end_forward()

    torch.cuda.synchronize()

    checks = {
        "active_kv": (active_kv(fused), active_kv(old)),
    }
    assert_packed_equal_or_boundary("query_code", q_code_fused, q_code_old, h_q_decode)
    for name, (got, ref) in checks.items():
        if not torch.equal(got, ref):
            mismatch = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
            total = got.numel()
            raise AssertionError(
                f"{name} mismatch for dtype={dtype}, kv_len={kv_len}: "
                f"{mismatch}/{total} packed elements differ"
            )

    old_h = active_hadamard(old)
    fused_h = active_hadamard(fused)
    if not torch.equal(fused_h, old_h):
        assert_packed_equal_or_boundary(
            "active_hadamard",
            fused_h,
            old_h,
            faster_hadamard_transform.hadamard_transform(
                torch.cat([k_prefill, k_decode], dim=0),
                inplace=False,
            ).reshape_as(unpack_2bit(old_h)),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-lens", default="17,113,1025,4097")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-bf16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtypes = [torch.float16]
    if args.include_bf16 and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        for kv_len in [int(x) for x in args.kv_lens.split(",") if x]:
            run_case(dtype, kv_len, args.seed)
            print(f"PASS dtype={dtype} kv_len={kv_len}")


if __name__ == "__main__":
    main()
