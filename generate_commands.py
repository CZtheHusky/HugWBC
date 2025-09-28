#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import multiprocessing as mp
from datetime import datetime
import numpy as np


def build_argparser():
    p = argparse.ArgumentParser(
        description="Launch multiple data collectors in parallel and write logs per command."
    )
    # === 路径 & 并发参数 ===
    p.add_argument("--ckpt-dir", type=str, default="logs/h1_interrupt/Aug21_13-31-13_",
                   help="Directory containing *.pt checkpoints (e.g., logs/h1_interrupt/Aug21_13-31-13_)")
    p.add_argument("--num-slots", type=int, default=32,
                   help="Number of parallel slots (processes) to run concurrently.")
    p.add_argument("--num-gpus", type=int, default=8,
                   help="Number of available GPUs (for CUDA_VISIBLE_DEVICES assignment).")
    p.add_argument("--log-dir", type=str, default="/root/workspace/HugWBC/collector_log",
                   help="Directory to write per-command logs.")

    # === 采集器固定参数（按需改/加）===
    p.add_argument("--task", type=str, default="h1int")
    p.add_argument("--num-envs", type=int, default=100)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--num-total", type=int, default=2000)
    p.add_argument("--switch-prob", type=float, default=1.0)
    p.add_argument("--episode-length-s", type=int, default=4)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true", default=True)

    # === 可选：给“填充任务”的默认ckpt名 ===
    p.add_argument("--filler-ckpt", type=str, default="model_0.pt",
                   help="Checkpoint name used to balance per-slot loads when needed.")
    return p


def list_checkpoints(ckpt_dir: str):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    # 你的原逻辑：按 model_XXX.pt 中的数字排序
    def keyfn(x):
        try:
            return int(x.split("_")[1].split(".")[0])
        except Exception:
            # 兜底：不规则文件名放后面
            return 1 << 30
    ckpts = sorted(ckpts, key=keyfn)
    return ckpts


def make_slots(ckpts, num_slots: int, filler_ckpt: str):
    """
    将 ckpts 均匀分到 num_slots 个 slot；若某 slot 不足，填充 filler_ckpt。
    返回 slots: List[List[Tuple[ckpt_name, output_root_override_or_None]]]
    """
    slots = [[] for _ in range(num_slots)]
    for i, ckpt in enumerate(ckpts):
        slot_idx = i % num_slots
        slots[slot_idx].append((ckpt, None))

    # 目标每槽任务数
    target = int(np.ceil(len(ckpts) / max(1, num_slots)))
    app_idx = 0
    for s in range(num_slots):
        while len(slots[s]) < target:
            # 给填充任务一个独立 output_root，避免覆盖
            slots[s].append((filler_ckpt, f"{os.path.splitext(filler_ckpt)[0]}_{app_idx}"))
            app_idx += 1
    return slots


def build_job(slot_idx: int,
              num_gpus: int,
              ckpt_dir: str,
              ckpt_name: str,
              args: argparse.Namespace,
              output_root_override):
    """
    将单个任务打包为可在子进程执行的 dict.
    """
    gpu_idx = slot_idx % max(1, num_gpus)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    # output_root：默认用 ckpt 的去扩展名；也可外部覆盖（用于填充任务）
    if output_root_override is None:
        output_root = os.path.splitext(ckpt_name)[0]
    else:
        output_root = output_root_override

    argv = [
        "python", "./legged_gym/scripts/data_collector_runner.py",
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--load_checkpoint", ckpt_path,
        "--num_total", str(args.num_total),
        "--switch_prob", str(args.switch_prob),
        "--episode_length_s", str(args.episode_length_s),
        "--output_root", output_root,
        "--seed", str(args.seed),
    ]
    if args.headless:
        argv.append("--headless")
    if args.overwrite:
        argv.append("--overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    # 日志文件名：时间戳 + 槽位 + ckpt 名，避免重名覆盖
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ckpt = os.path.splitext(os.path.basename(ckpt_name))[0]
    log_name = f"{ts}_slot{slot_idx:02d}_gpu{gpu_idx}_{safe_ckpt}.log"

    job = {
        "argv": argv,
        "env": env,
        "log_path": os.path.join(args.log_dir, log_name),
        "desc": f"[slot {slot_idx:02d} | gpu {gpu_idx}] {ckpt_name} -> {output_root}",
    }
    return job


def run_job(job: dict):
    """
    子进程执行函数：启动命令并将输出写入日志文件。
    返回 (returncode, log_path, desc)
    """
    os.makedirs(os.path.dirname(job["log_path"]), exist_ok=True)
    with open(job["log_path"], "w", buffering=1) as logf:
        proc = subprocess.Popen(
            job["argv"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=job["env"],
        )
        rc = proc.wait()
    return rc, job["log_path"], job["desc"]


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"No .pt checkpoints found in: {args.ckpt_dir}")

    slots = make_slots(ckpts, args.num_slots, args.filler_ckpt)

    # 构造所有 job
    jobs: list[dict] = []
    for slot_idx, lst in enumerate(slots):
        for ckpt_name, out_override in lst:
            jobs.append(build_job(
                slot_idx=slot_idx,
                num_gpus=args.num_gpus,
                ckpt_dir=args.ckpt_dir,
                ckpt_name=ckpt_name,
                args=args,
                output_root_override=out_override
            ))

    print(f"[INFO] Prepared {len(jobs)} jobs across {args.num_slots} slots "
          f"on {args.num_gpus} GPUs. Logs -> {args.log_dir}")
    # for job in jobs:
    #     print(job['argv'])

    # 多进程并发执行（每个 job 在独立进程里启动外部脚本）
    with mp.get_context("spawn").Pool(processes=args.num_slots) as pool:
        try:
            for rc, log_path, desc in pool.imap_unordered(run_job, jobs):
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[{status}] {desc}\n  ↳ log: {log_path}")
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user. Some jobs may still be running.")
            # 进程池会在 with 退出时做清理


if __name__ == "__main__":
    main()
