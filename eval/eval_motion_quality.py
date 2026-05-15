import argparse
import csv
import json
import math
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_loaders.humanml.scripts.motion_process import recover_from_ric


EPS = 1e-8


JOINT_PROFILES = {
    "humanml": {
        "pelvis": 0,
        "left_leg": [1, 4, 7, 10],
        "right_leg": [2, 5, 8, 11],
        "left_feet": [7, 10],
        "right_feet": [8, 11],
    },
    "kit": {
        "pelvis": 0,
        "left_leg": [16, 17, 18, 19, 20],
        "right_leg": [11, 12, 13, 14, 15],
        "left_feet": [19, 20],
        "right_feet": [14, 15],
    },
}


LOWER_IS_BETTER = {
    "step_timing_cv",
    "leg_range_asymmetry",
    "local_accel_mean",
    "local_jerk_mean",
    "local_jerk_p95",
    "normalized_jerk",
    "foot_sliding",
    "contact_height_var",
    "floor_penetration",
    "double_support_ratio",
    "no_contact_ratio",
}


HIGHER_IS_BETTER = {
    "contact_alternation",
    "step_timing_score",
    "leg_motion_symmetry",
    "left_contact_ratio",
    "right_contact_ratio",
}


def _to_scalar_dict(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_motion_file(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """Return motions as [N, T, J, 3], optional lengths and texts."""
    raw = np.load(path, allow_pickle=True)
    if path.endswith(".npz"):
        raw = raw["arr_0"]
    raw = _to_scalar_dict(raw)

    lengths = None
    texts = None
    if isinstance(raw, dict):
        motions = raw["motion"]
        lengths = np.asarray(raw.get("lengths")) if "lengths" in raw else None
        texts = list(raw.get("text")) if "text" in raw else None
    else:
        motions = raw

    motions = np.asarray(motions)
    motions = normalize_motion_shape(motions)
    return motions, lengths, texts


def parse_checkpoint_args(model_path: str, quality_args: argparse.Namespace):
    from utils.parser_util import evaluation_parser

    argv = [
        sys.argv[0],
        "--model_path",
        model_path,
        "--eval_mode",
        "quick_debug",
        "--guidance_param",
        str(quality_args.guidance_param),
        "--seed",
        str(quality_args.seed),
        "--device",
        str(quality_args.device),
    ]
    if quality_args.use_ema:
        argv.append("--use_ema")
    if quality_args.autoregressive:
        argv.append("--autoregressive")
    if quality_args.autoregressive_include_prefix:
        argv.append("--autoregressive_include_prefix")

    old_argv = sys.argv
    try:
        sys.argv = argv
        return evaluation_parser()
    finally:
        sys.argv = old_argv


def default_sample_dir(model_path: str, tag: str, seed: int, num_samples: int) -> str:
    model_dir = os.path.dirname(os.path.abspath(model_path))
    niter = os.path.basename(model_path).replace("model", "").replace(".pt", "")
    return os.path.join(model_dir, f"motion_quality_{tag}_{niter}_seed{seed}_n{num_samples}")


def sample_checkpoint_to_results(
    model_path: str,
    quality_args: argparse.Namespace,
    tag: str,
    output_dir: str,
) -> str:
    from data_loaders.get_data import get_dataset_loader
    from utils import dist_util
    from utils.fixseed import fixseed
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.sampler_util import AutoRegressiveSampler, ClassifierFreeSampleModel

    eval_args = parse_checkpoint_args(model_path, quality_args)
    eval_args.batch_size = quality_args.sample_batch_size
    fixseed(eval_args.seed)
    dist_util.setup_dist(eval_args.device)

    gen_loader = get_dataset_loader(
        name=eval_args.dataset,
        batch_size=eval_args.batch_size,
        num_frames=None,
        split=quality_args.split,
        hml_mode="eval",
        fixed_len=eval_args.context_len + eval_args.pred_len,
        pred_len=eval_args.pred_len,
        device=dist_util.dev(),
        autoregressive=eval_args.autoregressive,
    )

    model, diffusion = create_model_and_diffusion(eval_args, gen_loader)
    load_saved_model(model, model_path, use_avg=eval_args.use_ema)
    if eval_args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    sample_fn = diffusion.p_sample_loop
    if eval_args.autoregressive:
        sample_fn = AutoRegressiveSampler(eval_args, sample_fn).sample

    all_motions = []
    all_lengths = []
    all_text = []
    max_samples = quality_args.eval_num_samples

    with torch.no_grad():
        for motion, model_kwargs in gen_loader:
            if len(all_text) >= max_samples:
                break

            motion = motion.to(dist_util.dev())
            model_kwargs["y"] = {
                key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
                for key, val in model_kwargs["y"].items()
            }

            if eval_args.guidance_param != 1:
                model_kwargs["y"]["scale"] = (
                    torch.ones(motion.shape[0], device=dist_util.dev()) * eval_args.guidance_param
                )

            sample = sample_fn(
                model,
                motion.shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            if "prefix" in model_kwargs["y"]:
                model_kwargs["y"]["lengths"] = model_kwargs["y"]["orig_lengths"]

            if model.data_rep == "hml_vec":
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = gen_loader.dataset.t2m_dataset.inv_transform(
                    sample.cpu().permute(0, 2, 3, 1)
                ).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            else:
                pose_rep = "xyz" if model.data_rep == "xyz" else model.data_rep
                mask = None if pose_rep == "xyz" else model_kwargs["y"]["mask"].reshape(motion.shape[0], -1).bool()
                sample = model.rot2xyz(
                    x=sample,
                    mask=mask,
                    pose_rep=pose_rep,
                    glob=True,
                    translation=True,
                    jointstype="smpl",
                    vertstrans=True,
                    betas=None,
                    beta=0,
                    glob_rot=None,
                    get_rotations_back=False,
                ).cpu()

            batch_np = sample.cpu().numpy()
            lengths_np = model_kwargs["y"]["lengths"].detach().cpu().numpy()
            text_key = "text" if "text" in model_kwargs["y"] else "action_text"
            texts = model_kwargs["y"].get(text_key, [""] * batch_np.shape[0])

            take = min(batch_np.shape[0], max_samples - len(all_text))
            all_motions.append(batch_np[:take])
            all_lengths.append(lengths_np[:take])
            all_text.extend(list(texts[:take]))

    if not all_motions:
        raise RuntimeError("No motions were sampled from the checkpoint.")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.npy")
    np.save(
        results_path,
        {
            "motion": np.concatenate(all_motions, axis=0),
            "text": all_text,
            "lengths": np.concatenate(all_lengths, axis=0),
            "num_samples": len(all_text),
            "num_repetitions": 1,
        },
    )
    return results_path


def normalize_motion_shape(motions: np.ndarray) -> np.ndarray:
    if motions.ndim == 4:
        if motions.shape[2] == 3:
            # MDM results.npy: [N, J, 3, T]
            return motions.transpose(0, 3, 1, 2).astype(np.float64)
        if motions.shape[-1] == 3:
            # Already [N, T, J, 3]
            return motions.astype(np.float64)
        raise ValueError(f"Unsupported 4D motion shape: {motions.shape}")

    if motions.ndim == 3:
        if motions.shape[-1] == 3:
            # [T, J, 3]
            return motions[None].astype(np.float64)
        if motions.shape[1] == 3:
            # [J, 3, T]
            return motions.transpose(2, 0, 1)[None].astype(np.float64)
        raise ValueError(f"Unsupported 3D motion shape: {motions.shape}")

    if motions.ndim == 2 and motions.shape[-1] in (251, 263):
        joints_num = 22 if motions.shape[-1] == 263 else 21
        tensor = torch.from_numpy(motions).float()
        xyz = recover_from_ric(tensor, joints_num).cpu().numpy()
        return xyz[None].astype(np.float64)

    raise ValueError(
        "Unsupported motion format. Expected results.npy dict, [N,J,3,T], "
        f"[T,J,3], [J,3,T], or HumanML/KIT vector [T,263/251], got {motions.shape}."
    )


def infer_profile(dataset: str, num_joints: int) -> Dict[str, List[int]]:
    if dataset != "auto":
        profile = JOINT_PROFILES[dataset]
    elif num_joints == 21:
        profile = JOINT_PROFILES["kit"]
    else:
        profile = JOINT_PROFILES["humanml"]

    all_indices = [profile["pelvis"]] + profile["left_leg"] + profile["right_leg"]
    if max(all_indices) >= num_joints:
        raise ValueError(
            f"Joint profile requires joint index {max(all_indices)}, "
            f"but motion only has {num_joints} joints."
        )
    return profile


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0 if len(arr) == 1 else float("nan")
    return float(arr.std(ddof=1))


def safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_var(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float("nan")
    return float(np.var(arr))


def safe_percentile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def contact_onsets(contact: np.ndarray) -> np.ndarray:
    if contact.size == 0:
        return np.zeros(0, dtype=np.int64)
    prev = np.concatenate([[False], contact[:-1]])
    return np.flatnonzero(contact & ~prev)


def evaluate_motion(
    joints: np.ndarray,
    profile: Dict[str, List[int]],
    length: Optional[int],
    sample_index: int,
    text: str,
    fps: float,
    floor_y: Optional[float],
    contact_height_threshold: float,
    contact_vel_threshold: float,
) -> Dict[str, float]:
    if length is not None and np.isfinite(length):
        joints = joints[: int(length)]

    row: Dict[str, float] = {
        "sample_index": sample_index,
        "text": text,
        "length": int(joints.shape[0]),
    }

    if joints.shape[0] < 4:
        for key in metric_keys():
            row[key] = float("nan")
        return row

    left_feet = profile["left_feet"]
    right_feet = profile["right_feet"]
    feet = left_feet + right_feet
    pelvis = profile["pelvis"]

    if floor_y is None:
        floor = float(np.min(joints[:, feet, 1]))
    else:
        floor = float(floor_y)
    row["floor_y"] = floor

    foot_pos = joints[:, feet]
    foot_height = foot_pos[:-1, :, 1] - floor
    foot_delta = foot_pos[1:] - foot_pos[:-1]
    vertical_delta = np.abs(foot_delta[:, :, 1])
    foot_contact = (foot_height <= contact_height_threshold) & (vertical_delta <= contact_vel_threshold)

    num_left = len(left_feet)
    left_contact_feet = foot_contact[:, :num_left]
    right_contact_feet = foot_contact[:, num_left:]
    left_contact = left_contact_feet.any(axis=1)
    right_contact = right_contact_feet.any(axis=1)
    any_contact = left_contact | right_contact

    row["left_contact_ratio"] = safe_mean(left_contact.astype(np.float64))
    row["right_contact_ratio"] = safe_mean(right_contact.astype(np.float64))
    row["double_support_ratio"] = safe_mean((left_contact & right_contact).astype(np.float64))
    row["no_contact_ratio"] = safe_mean((~any_contact).astype(np.float64))
    row["contact_alternation"] = (
        safe_mean((left_contact ^ right_contact)[any_contact].astype(np.float64))
        if any_contact.any()
        else float("nan")
    )

    left_onsets = contact_onsets(left_contact)
    right_onsets = contact_onsets(right_contact)
    step_events = np.sort(np.concatenate([left_onsets, right_onsets]))
    intervals = np.diff(step_events)
    intervals = intervals[intervals > 0]
    if len(intervals) >= 2:
        step_cv = float(np.std(intervals) / (np.mean(intervals) + EPS))
        row["step_timing_cv"] = step_cv
        row["step_timing_score"] = float(math.exp(-step_cv))
    else:
        row["step_timing_cv"] = float("nan")
        row["step_timing_score"] = float("nan")

    local_joints = joints - joints[:, pelvis : pelvis + 1]
    left_range = joint_motion_range(local_joints[:, profile["left_leg"]])
    right_range = joint_motion_range(local_joints[:, profile["right_leg"]])
    asymmetry = abs(left_range - right_range) / (left_range + right_range + EPS)
    row["left_leg_motion_range"] = left_range
    row["right_leg_motion_range"] = right_range
    row["leg_range_asymmetry"] = float(asymmetry)
    row["leg_motion_symmetry"] = float(max(0.0, 1.0 - asymmetry))

    lower_indices = sorted(set(profile["left_leg"] + profile["right_leg"]))
    lower_local = local_joints[:, lower_indices]
    vel = np.diff(lower_local, axis=0) * fps
    accel = np.diff(lower_local, n=2, axis=0) * (fps ** 2)
    jerk = np.diff(lower_local, n=3, axis=0) * (fps ** 3)
    vel_norm = np.linalg.norm(vel, axis=-1)
    accel_norm = np.linalg.norm(accel, axis=-1)
    jerk_norm = np.linalg.norm(jerk, axis=-1)

    row["local_accel_mean"] = safe_mean(accel_norm)
    row["local_jerk_mean"] = safe_mean(jerk_norm)
    row["local_jerk_p95"] = safe_percentile(jerk_norm, 95)
    row["normalized_jerk"] = row["local_jerk_mean"] / (safe_mean(vel_norm) + EPS)

    horizontal_delta = np.linalg.norm(foot_delta[:, :, [0, 2]], axis=-1)
    row["foot_sliding"] = safe_mean(horizontal_delta[foot_contact])
    row["contact_height_var"] = safe_var(foot_height[foot_contact])
    row["floor_penetration"] = safe_mean(np.maximum(0.0, floor - foot_pos[:, :, 1]))

    return row


def joint_motion_range(joints: np.ndarray) -> float:
    per_joint_range = np.linalg.norm(joints.max(axis=0) - joints.min(axis=0), axis=-1)
    return float(np.mean(per_joint_range))


def metric_keys() -> List[str]:
    return [
        "floor_y",
        "left_contact_ratio",
        "right_contact_ratio",
        "double_support_ratio",
        "no_contact_ratio",
        "contact_alternation",
        "step_timing_cv",
        "step_timing_score",
        "left_leg_motion_range",
        "right_leg_motion_range",
        "leg_range_asymmetry",
        "leg_motion_symmetry",
        "local_accel_mean",
        "local_jerk_mean",
        "local_jerk_p95",
        "normalized_jerk",
        "foot_sliding",
        "contact_height_var",
        "floor_penetration",
    ]


def evaluate_file(args, path: str, tag: str) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    motions, lengths, texts = load_motion_file(path)
    profile = infer_profile(args.dataset, motions.shape[2])

    rows = []
    for idx, joints in enumerate(motions):
        length = lengths[idx] if lengths is not None and idx < len(lengths) else None
        text = texts[idx] if texts is not None and idx < len(texts) else ""
        row = evaluate_motion(
            joints=joints,
            profile=profile,
            length=length,
            sample_index=idx,
            text=text,
            fps=args.fps,
            floor_y=args.floor_y,
            contact_height_threshold=args.contact_height_threshold,
            contact_vel_threshold=args.contact_vel_threshold,
        )
        row["tag"] = tag
        rows.append(row)

    return rows, summarize_rows(rows)


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in metric_keys():
        values = [row.get(key, float("nan")) for row in rows]
        summary[key] = {
            "mean": finite_mean(values),
            "std": finite_std(values),
        }
    return summary


def compare_summaries(
    baseline: Dict[str, Dict[str, float]], target: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    comparison = {}
    for key in metric_keys():
        base = baseline[key]["mean"]
        new = target[key]["mean"]
        delta = new - base if np.isfinite(base) and np.isfinite(new) else float("nan")
        rel = delta / (abs(base) + EPS) if np.isfinite(delta) else float("nan")

        if key in LOWER_IS_BETTER:
            improved = -delta
        elif key in HIGHER_IS_BETTER:
            improved = delta
        else:
            improved = float("nan")

        comparison[key] = {
            "baseline_mean": base,
            "target_mean": new,
            "delta": delta,
            "relative_delta": rel,
            "improvement_direction_delta": improved,
        }
    return comparison


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["tag", "sample_index", "text", "length"] + metric_keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)


def default_output_prefix(input_path: str) -> str:
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "results.npy")
    directory = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(directory, "motion_quality_eval")


def print_summary(title: str, summary: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    for key in metric_keys():
        mean = summary[key]["mean"]
        std = summary[key]["std"]
        print(f"{key:24s} mean={mean:.6g} std={std:.6g}")


def resolve_input_path(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "results.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def resolve_motion_source(
    path: str,
    quality_args: argparse.Namespace,
    tag: str,
    output_dir: str,
) -> str:
    path = resolve_input_path(path) if not path.endswith(".pt") else path
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".pt"):
        sample_dir = output_dir or default_sample_dir(path, tag, quality_args.seed, quality_args.eval_num_samples)
        print(f"Sampling checkpoint [{path}] to [{sample_dir}] before quality evaluation.")
        return sample_checkpoint_to_results(path, quality_args, tag=tag, output_dir=sample_dir)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate lower-body coordination, local smoothness and foot-ground "
            "stability from generated xyz joint motions."
        )
    )
    parser.add_argument("--input_path", default="", help="Path to results.npy, raw motion .npy, or model checkpoint .pt.")
    parser.add_argument("--model_path", default="", help="Path to model####.pt. Alias for passing a checkpoint as the target.")
    parser.add_argument("--baseline_path", default="", help="Optional baseline results.npy/raw .npy/checkpoint .pt.")
    parser.add_argument("--dataset", default="auto", choices=["auto", "humanml", "kit"])
    parser.add_argument("--output_prefix", default="", help="Prefix for CSV/JSON outputs.")
    parser.add_argument("--fps", default=1.0, type=float, help="Derivative scale. Keep 1.0 for frame-based metrics.")
    parser.add_argument("--floor_y", default=None, type=float, help="Fixed floor height. Default: min foot height per motion.")
    parser.add_argument("--contact_height_threshold", default=0.05, type=float)
    parser.add_argument("--contact_vel_threshold", default=0.02, type=float, help="Max vertical displacement per frame for contact.")
    parser.add_argument("--eval_num_samples", default=32, type=int, help="Number of test prompts to sample when input is a checkpoint.")
    parser.add_argument("--sample_batch_size", default=32, type=int, help="Sampling batch size when input is a checkpoint.")
    parser.add_argument("--sample_output_dir", default="", help="Where to save sampled target results.npy when input is a checkpoint.")
    parser.add_argument("--baseline_sample_output_dir", default="", help="Where to save sampled baseline results.npy.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--autoregressive_include_prefix", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.model_path and args.input_path:
        raise ValueError("Use either --model_path or --input_path, not both.")
    source_path = args.model_path or args.input_path
    if not source_path:
        raise ValueError("Please pass --model_path checkpoint.pt or --input_path results.npy.")

    input_path = resolve_motion_source(source_path, args, tag="target", output_dir=args.sample_output_dir)
    output_prefix = args.output_prefix or default_output_prefix(input_path)

    all_rows: List[Dict[str, float]] = []
    payload = {}

    target_rows, target_summary = evaluate_file(args, input_path, "target")
    all_rows.extend(target_rows)
    payload["target"] = {
        "path": input_path,
        "summary": target_summary,
    }
    print_summary("Target summary", target_summary)

    if args.baseline_path:
        baseline_path = resolve_motion_source(
            args.baseline_path,
            args,
            tag="baseline",
            output_dir=args.baseline_sample_output_dir,
        )
        baseline_rows, baseline_summary = evaluate_file(args, baseline_path, "baseline")
        all_rows = baseline_rows + all_rows
        payload["baseline"] = {
            "path": baseline_path,
            "summary": baseline_summary,
        }
        payload["comparison"] = compare_summaries(baseline_summary, target_summary)
        print_summary("Baseline summary", baseline_summary)

    csv_path = output_prefix + ".csv"
    json_path = output_prefix + "_summary.json"
    write_csv(csv_path, all_rows)
    write_json(json_path, payload)
    print(f"\nSaved per-sample metrics to: {csv_path}")
    print(f"Saved summary to: {json_path}")


if __name__ == "__main__":
    main()
