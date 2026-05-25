import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict

import torch
import torch.nn.functional as F


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_loaders.get_data import get_dataset_loader
from eval.eval_motion_quality import parse_checkpoint_args
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model


EPS = 1e-8


CHANNEL_GROUPS = {
    "root": (0, 4),
    "ric": (4, 67),
    "rot": (67, 193),
    "local_vel": (193, 259),
    "foot_contact": (259, 263),
}


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().square().mean().sqrt()


def mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.float().abs().mean()


class StatBag:
    def __init__(self) -> None:
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name: str, value) -> None:
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        self.sums[name] += float(value)
        self.counts[name] += 1

    def mean_dict(self) -> Dict[str, float]:
        return {
            name: self.sums[name] / max(self.counts[name], 1)
            for name in sorted(self.sums)
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose PlanOne local/global/fusion/head/refine signal scales without sampling."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[999, 500, 100])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--guidance_param", type=float, default=2.5)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--autoregressive_include_prefix", action="store_true")
    return parser


def prepare_model(model_path: str, args: argparse.Namespace):
    eval_args = parse_checkpoint_args(model_path, args)
    eval_args.batch_size = args.batch_size
    gen_loader = get_dataset_loader(
        name=eval_args.dataset,
        batch_size=eval_args.batch_size,
        num_frames=None,
        split=args.split,
        hml_mode="eval",
        fixed_len=eval_args.context_len + eval_args.pred_len,
        pred_len=eval_args.pred_len,
        device=dist_util.dev(),
        autoregressive=eval_args.autoregressive,
    )
    model, _ = create_model_and_diffusion(eval_args, gen_loader)
    load_saved_model(model, model_path, use_avg=eval_args.use_ema)
    model.to(dist_util.dev())
    model.eval()
    return model, gen_loader, eval_args


def to_device_y(model_kwargs: Dict) -> Dict:
    y = {}
    for key, value in model_kwargs["y"].items():
        y[key] = value.to(dist_util.dev()) if torch.is_tensor(value) else value
    return y


def fusion_zero_local_outputs(model, outputs: Dict):
    fusion = outputs["fusion"]
    if "local_pre_gated" in fusion:
        local_pre_raw = fusion["local_pre_raw"]
        local_pre_gated = fusion["local_pre_gated"]
        global_pre = fusion["global_pre"]
    else:
        l_t = outputs["local"]["l_t"]
        g_t = outputs["global"]["g_t"]
        d_model = l_t.shape[-1]
        weight = model.fusion.proj.weight
        bias = model.fusion.proj.bias
        local_pre_raw = F.linear(l_t, weight[:, :d_model], None)
        local_pre_gated = local_pre_raw
        global_pre = F.linear(g_t, weight[:, d_model:], bias)
    pre = local_pre_gated + global_pre
    f_t_zero_local = model.fusion.out_norm(global_pre)
    f_t_manual = model.fusion.out_norm(pre)
    return local_pre_raw, local_pre_gated, global_pre, f_t_zero_local, f_t_manual


def add_channel_stats(stats: StatBag, prefix: str, tensor: torch.Tensor) -> None:
    channels = tensor.shape[-1]
    for name, (start, end) in CHANNEL_GROUPS.items():
        if end <= channels:
            part = tensor[..., start:end]
            stats.add(f"{prefix}.{name}.rms", rms(part))
            stats.add(f"{prefix}.{name}.mean_abs", mean_abs(part))


def add_forward_stats(stats: StatBag, model, outputs: Dict, label: str) -> None:
    l_t = outputs["local"]["l_t"]
    g_t = outputs["global"]["g_t"]
    f_t = outputs["fusion"]["f_t"]
    y_base = outputs["heads"]["y_base_raw"]
    delta = outputs["residual_tcn"]["delta_raw"]
    y_raw = outputs["y_raw"]

    local_pre_raw, local_pre_gated, global_pre, f_t_zero_local, f_t_manual = fusion_zero_local_outputs(model, outputs)
    zero_head = model.refine_heads(f_t_zero_local)

    stats.add(f"{label}.local.l_t.rms", rms(l_t))
    stats.add(f"{label}.global.g_t.rms", rms(g_t))
    stats.add(f"{label}.fusion.local_pre_raw.rms", rms(local_pre_raw))
    stats.add(f"{label}.fusion.local_pre_gated.rms", rms(local_pre_gated))
    stats.add(f"{label}.fusion.global_pre.rms", rms(global_pre))
    stats.add(f"{label}.fusion.pre_local_raw_over_global", rms(local_pre_raw) / (rms(global_pre) + EPS))
    stats.add(f"{label}.fusion.pre_local_gated_over_global", rms(local_pre_gated) / (rms(global_pre) + EPS))
    if "local_gate" in outputs["fusion"]:
        stats.add(f"{label}.gate.fusion_local_gate", outputs["fusion"]["local_gate"])
    stats.add(f"{label}.fusion.f_t.rms", rms(f_t))
    stats.add(f"{label}.fusion.f_t_manual_error", mean_abs(f_t - f_t_manual))
    stats.add(f"{label}.fusion.zero_local_delta.rms", rms(f_t - f_t_zero_local))
    stats.add(f"{label}.fusion.zero_local_delta_over_f_t", rms(f_t - f_t_zero_local) / (rms(f_t) + EPS))

    stats.add(f"{label}.head.y_base.rms", rms(y_base))
    stats.add(f"{label}.head.zero_local_y_base.rms", rms(zero_head["y_base_raw"]))
    stats.add(
        f"{label}.head.zero_local_y_base_delta.rms",
        rms(y_base - zero_head["y_base_raw"]),
    )
    stats.add(
        f"{label}.head.zero_local_y_base_delta_over_y_base",
        rms(y_base - zero_head["y_base_raw"]) / (rms(y_base) + EPS),
    )

    stats.add(f"{label}.residual.delta_raw.rms", rms(delta))
    stats.add(f"{label}.residual.delta_over_y_base", rms(delta) / (rms(y_base) + EPS))
    stats.add(f"{label}.output.y_raw.rms", rms(y_raw))

    add_channel_stats(stats, f"{label}.head.y_base", y_base)
    add_channel_stats(stats, f"{label}.head.zero_local_delta", y_base - zero_head["y_base_raw"])
    add_channel_stats(stats, f"{label}.residual.delta_raw", delta)
    add_channel_stats(stats, f"{label}.output.y_raw", y_raw)

    local_branch = getattr(model, "local_branch", None)
    if local_branch is not None:
        if hasattr(local_branch, "local_gate_logit"):
            gate = torch.sigmoid(local_branch.local_gate_logit) * float(local_branch.local_gate_max)
            stats.add(f"{label}.gate.local_gate", gate)
        if hasattr(local_branch, "leg_gate_logit"):
            leg_gate = torch.sigmoid(local_branch.leg_gate_logit) * float(local_branch.leg_gate_max)
            torso_gate = torch.sigmoid(local_branch.torso_gate_logit) * float(local_branch.torso_gate_max)
            stats.add(f"{label}.gate.leg_gate", leg_gate)
            stats.add(f"{label}.gate.torso_gate", torso_gate)
    residual_tcn = getattr(model, "residual_tcn", None)
    if residual_tcn is not None and hasattr(residual_tcn, "get_gate"):
        stats.add(f"{label}.gate.residual_gate", residual_tcn.get_gate())


def diagnose_model(model_path: str, args: argparse.Namespace) -> Dict:
    model, gen_loader, eval_args = prepare_model(model_path, args)
    stats = StatBag()
    processed_batches = 0

    with torch.no_grad():
        for motion, model_kwargs in gen_loader:
            if processed_batches >= args.num_batches:
                break
            motion = motion.to(dist_util.dev())
            y = to_device_y(model_kwargs)

            for timestep in args.timesteps:
                timesteps = torch.full(
                    (motion.shape[0],),
                    int(timestep),
                    dtype=torch.long,
                    device=dist_util.dev(),
                )
                outputs = model(motion, timesteps, y=y, return_dict=True)
                add_forward_stats(stats, model, outputs, f"t{int(timestep):04d}")

            processed_batches += 1

    return {
        "model_path": os.path.abspath(model_path),
        "dataset": eval_args.dataset,
        "local_mode": getattr(model, "local_mode", None),
        "refine_mode": getattr(model, "refine_mode", None),
        "num_batches": processed_batches,
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "stats": stats.mean_dict(),
    }


def main() -> None:
    args = build_argparser().parse_args()
    if isinstance(args.device, str) and args.device.lstrip("-").isdigit():
        args.device = int(args.device)
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    results = {
        "target": diagnose_model(args.model_path, args),
    }

    output_path = args.output_path
    if output_path is None:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        output_path = os.path.join(model_dir, "plan_one_flow_diagnostics.json")

    wrote_file = False
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        wrote_file = True
    except PermissionError:
        fallback_dir = os.path.join(ROOT_DIR, "output")
        os.makedirs(fallback_dir, exist_ok=True)
        model_name = os.path.basename(os.path.dirname(os.path.abspath(args.model_path)))
        output_path = os.path.join(fallback_dir, f"{model_name}_plan_one_flow_diagnostics.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            wrote_file = True
        except PermissionError:
            output_path = None

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if wrote_file:
        print(f"Wrote diagnostics to {output_path}")
    else:
        print("Could not write diagnostics file due to permissions; printed diagnostics above.")


if __name__ == "__main__":
    main()
