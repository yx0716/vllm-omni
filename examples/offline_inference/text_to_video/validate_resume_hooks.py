# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the Wan2.2 hook pair: strength-resume + return_trajectory_latents.

Phase A (baseline): full denoise with ``return_trajectory_latents=True`` —
captures the post-step latent at every step plus the final video.
Phase B (resume): re-run the SAME prompt with ``latents = trajectory[k-1]``
and ``strength = (T - k) / T`` so denoising starts at step k.

With the single-step euler solver the resumed trajectory is mathematically
the tail of the baseline run, so Phase B's output should match Phase A's up
to numerical noise. With unipc (multistep) the solver restarts its history at
step k, so expect visually-identical-but-not-bitwise results.

Example (2x H100, CFG parallel):
  python validate_resume_hooks.py \
      --model /pvcplatform/model_zoo/Wan2.2-T2V-A14B-Diffusers \
      --height 480 --width 832 --num-frames 33 \
      --num-inference-steps 20 --skip-steps 10 \
      --cfg-parallel-size 2 --flow-shift 12.0 --sample-solver euler \
      --out-dir /tmp/wan22-resume-validation
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--skip-steps", type=int, default=10, help="k: steps to skip on resume")
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--flow-shift", type=float, default=12.0)
    p.add_argument("--sample-solver", default="euler", choices=["euler", "unipc"])
    p.add_argument("--cfg-parallel-size", type=int, default=2, choices=[1, 2])
    p.add_argument("--enable-cpu-offload", action="store_true")
    p.add_argument("--enable-layerwise-offload", action="store_true")
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--out-dir", default="/tmp/wan22-resume-validation")
    p.add_argument("--init-timeout", type=int, default=1800, help="Engine/stage init timeout (s)")
    return p.parse_args()


def _walk(result, name, depth=0):
    """DFS an OmniRequestOutput tree for the first non-None attribute `name`."""
    if result is None or depth > 4:
        return None
    if isinstance(result, list):
        for item in result:
            val = _walk(item, name, depth + 1)
            if val is not None:
                return val
        return None
    val = getattr(result, name, None)
    if val is not None:
        return val
    for child_attr in ("request_output", "outputs"):
        child = getattr(result, child_attr, None)
        if child is not None:
            val = _walk(child, name, depth + 1)
            if val is not None:
                return val
    return None


def _to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], np.ndarray):
        return torch.from_numpy(np.stack(x))
    return None


def _extract_frames(result):
    imgs = _walk(result, "images")
    t = _to_tensor(imgs[0] if isinstance(imgs, list) and imgs else imgs)
    return t


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T, k = args.num_inference_steps, args.skip_steps
    assert 0 < k < T, "--skip-steps must be in (0, num_inference_steps)"

    omni = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=args.cfg_parallel_size),
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        flow_shift=args.flow_shift,
        # Cold-loading 2x14B from shared storage can exceed the 600s/300s
        # defaults on the first run; be generous.
        init_timeout=args.init_timeout,
        stage_init_timeout=args.init_timeout,
    )

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    common = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=T,
        guidance_scale=args.guidance_scale,
    )

    # ---- Phase A: baseline + trajectory capture -------------------------
    sp_a = OmniDiffusionSamplingParams(
        seed=args.seed, return_trajectory_latents=True, **common
    )
    sp_a.extra_args["sample_solver"] = args.sample_solver
    t0 = time.perf_counter()
    res_a = omni.generate(prompt_dict, sp_a)
    t_a = time.perf_counter() - t0

    traj = _to_tensor(_walk(res_a, "trajectory_latents"))
    frames_a = _extract_frames(res_a)
    assert traj is not None, "PR-2 FAIL: trajectory_latents missing from output"
    print(f"[A] wall={t_a:.1f}s  trajectory shape={tuple(traj.shape)}  dtype={traj.dtype}")
    assert traj.shape[0] == T, f"PR-2 FAIL: expected {T} recorded steps, got {traj.shape[0]}"

    # Donor = post-step latent of step k-1 == the state entering step k.
    donor = traj[k - 1].clone().float()
    torch.save(donor, out_dir / f"donor_step{k}.pt")

    # ---- Phase B: resume from step k ------------------------------------
    sp_b = OmniDiffusionSamplingParams(
        seed=args.seed, latents=donor, strength=(T - k) / T, **common
    )
    sp_b.extra_args["sample_solver"] = args.sample_solver
    t0 = time.perf_counter()
    res_b = omni.generate(prompt_dict, sp_b)
    t_b = time.perf_counter() - t0
    frames_b = _extract_frames(res_b)
    print(f"[B] wall={t_b:.1f}s  (baseline {t_a:.1f}s, skipped {k}/{T} steps)")

    # ---- Compare ---------------------------------------------------------
    report = {
        "solver": args.sample_solver,
        "steps_total": T,
        "steps_skipped": k,
        "wall_baseline_s": round(t_a, 2),
        "wall_resume_s": round(t_b, 2),
        "trajectory_shape": list(traj.shape),
    }
    if frames_a is not None and frames_b is not None and frames_a.shape == frames_b.shape:
        fa = frames_a.float()
        fb = frames_b.float()
        mse = torch.mean((fa - fb) ** 2).item()
        peak = 255.0 if fa.max() > 2.0 else 1.0
        psnr = float("inf") if mse == 0 else 10 * np.log10(peak**2 / mse)
        report.update({"frames_shape": list(fa.shape), "mse": mse, "psnr_db": round(psnr, 2)})
        np.save(out_dir / "frames_baseline.npy", fa.cpu().numpy())
        np.save(out_dir / "frames_resume.npy", fb.cpu().numpy())
    else:
        report["frames_compare"] = (
            f"skipped (a={None if frames_a is None else tuple(frames_a.shape)}, "
            f"b={None if frames_b is None else tuple(frames_b.shape)})"
        )

    print(json.dumps(report, indent=2))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
