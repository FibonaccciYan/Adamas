import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--decode-len", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--budgets", default="512,1024,2048,4096,102400")
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    result_dir = root / "test_results" / args.model
    archive_dir = root / "test_results" / "e2e_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.label
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    e2e_file = result_dir / "e2e_Adamas.txt"
    if e2e_file.exists():
        shutil.copy2(e2e_file, archive_dir / "e2e_Adamas.before.txt")
    e2e_file.write_text("")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--env expects KEY=VALUE, got {item}")
        env[key] = value

    budgets = [x for x in args.budgets.split(",") if x]
    contexts = [x for x in args.contexts.split(",") if x]
    for budget in budgets:
        for context in contexts:
            log_file = result_dir / f"log_{budget}_{context}.log"
            cmd = [
                args.python,
                str(root / "scripts" / "bench_textgen.py"),
                "--model",
                args.model,
                "--context_len",
                context,
                "--decode_len",
                str(args.decode_len),
                "--token_budget",
                budget,
                "--iteration",
                str(args.iteration),
                "--page_size",
                "1",
            ]
            with log_file.open("w") as f:
                subprocess.run(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    shutil.copy2(e2e_file, archive_dir / "e2e_Adamas.txt")
    for log_file in result_dir.glob("log_*.log"):
        shutil.copy2(log_file, archive_dir / log_file.name)
    subprocess.run(["git", "status", "--short"], cwd=root, text=True,
                   stdout=(archive_dir / "git_status.txt").open("w"), check=True)
    subprocess.run(["git", "diff"], cwd=root, text=True,
                   stdout=(archive_dir / "git_diff.patch").open("w"), check=True)
    print(archive_dir)


if __name__ == "__main__":
    main()
