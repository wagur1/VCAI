"""Read-only V34 forensic audit and deterministic replay of saved measurements.

Run from any directory. Only the two report files under --out are written.
No models are trained, downloaded, or evaluated on new images. Missing evidence
is an explicit result, never a zero, an interpolated QP, or a fabricated metric.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from adaptive_vcm.legacy_bd import bd_rate, bd_metric
from adaptive_vcm.metrics import pchip_bd
from adaptive_vcm.v34.learning import domain_calibration, select_action
from adaptive_vcm.v34.protocol import ACTIONS

# HEAD when the bugs were audited. Pin it so replay survives committing the fixes.
HISTORICAL_REF = "307b77a05fc1084970ac2c3b1b1d74efb83fc806"
SEEDS = (303401, 303402, 303403)
GRIDS = ((30, 35, 40, 45), (30, 35, 40, 45, 50))
TASKS = ("ar", "od", "ot")
BASELINES = {
    "B0": "Identity, action 0",
    "B1": "Global Gaussian frequency attenuation, action 11",
    "B2_a6": "Fixed region-aware action 6 from the saved action bank",
    "B2_a5": "Fixed region-aware action 5 from the saved action bank",
    "B3": "forced_303401; direct argmax(logits)",
    "B4": "3 distinct random nonidentity candidates + identity; RNG seeds 0..19",
    "B5a": "3 model-ranked candidates + identity; minimum measured rate <= 1.01; no guard",
    "B5b": "calibrated_303401; calibrated risk/rate bounds, no source guard",
    "Full": "selected_303401; model candidates + codec trials + guard",
}
KAGGLE = Path.home() / "AppData/Local/Temp/opencode/v34_fix1/outputs/kaggle"
SCRATCH = Path.home() / ".gemini/antigravity-cli/brain/281602a8-5222-4a8a-9c2a-2617182a4026/scratch"
PAPER = Path.home() / "Downloads/prism_seed303401_qp30_45/Chapter"


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def stats(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                p50=float(np.quantile(a, .5)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def historical_source(relative, ref=HISTORICAL_REF):
    """Read audited source from Git, independently of the repaired workspace."""
    value = subprocess.run(["git", "show", f"{ref}:{relative}"], cwd=ROOT,
                           capture_output=True, check=True, timeout=30)
    return value.stdout.decode("utf-8-sig")


def source_lines(source, pattern):
    return [dict(line=i, text=line) for i, line in enumerate(source.splitlines(), 1)
            if re.search(pattern, line, re.I)]


class Evidence:
    def __init__(self):
        self.files = {}
        self.gaps = []

    def record(self, path):
        p = Path(path).resolve()
        key = str(p)
        if key not in self.files:
            self.files[key] = dict(exists=p.is_file())
            if p.is_file():
                self.files[key].update(bytes=p.stat().st_size, sha256=file_hash(p))
        return p.is_file()

    def gap(self, scope, reason):
        value = dict(scope=scope, status="unavailable", reason=reason)
        if value not in self.gaps:
            self.gaps.append(value)
        return value

    def read(self, path):
        if not self.record(path):
            self.gap(str(path), "Required input file is absent")
            return None
        raw = Path(path).read_bytes()
        if str(path).endswith(".gz"):
            raw = gzip.decompress(raw)
        try:
            import orjson
        except ImportError:
            return json.loads(raw)
        return orjson.loads(raw)

    def text(self, path):
        if not self.record(path):
            self.gap(str(path), "Required text file is absent")
            return ""
        return Path(path).read_text(encoding="utf-8-sig")

    def lines(self, path, pattern):
        return [dict(line=i, text=line) for i, line in enumerate(self.text(path).splitlines(), 1)
                if re.search(pattern, line, re.I)]


def protected_snapshot(root=ROOT):
    """Include every publication asset as well as the current runtime sources."""
    from scripts.reproduce_results import verify
    p = root / "results/v34_r1/publication_manifest.json"
    manifest = json.loads(p.read_text())
    if root == ROOT:
        verify(manifest)  # Also verifies compressed and uncompressed archive hashes.
    rel = set(manifest["runtime_sha256"]) | set(manifest["compressed_files"])
    rel.update(f"checkpoints/v34_r1/controller_{s}.pth" for s in manifest["checkpoints"])
    rel.update(f"results/v34_r1/shards/shard{i}/{name}"
               for i, files in manifest["shard_original_sha256"].items()
               for name in files if name != "test_rows.json")
    rel.update("results/v34_r1/" + f for f in (
        "publication_manifest.json", "independent_summary.json", "protocol/protocol.json",
        "protocol/annotations.csv", "protocol/exclusions.json"))
    return {name: file_hash(root / name) for name in sorted(rel)}


def validate_output_directory(out, protected):
    out = Path(out).resolve()
    # Reports cannot be redirected over any frozen asset, or into its directory.
    if out == ROOT or any(out == (ROOT / p).resolve().parent for p in protected):
        raise ValueError("Report directory overlaps a publication directory")
    for name in ("audit_summary.json", "AUDIT_REPORT.md"):
        target = (out / name).resolve()
        if any(target == (ROOT / p).resolve() for p in protected):
            raise ValueError("Refusing to overwrite a publication asset")
    return out


def pchip_quality(ra, qa, rb, qb):
    def fit(rate, quality):
        x = np.log(np.asarray(rate, float))
        y = np.asarray(quality, float)
        unique = np.unique(x)
        if len(unique) < 2:
            return None, None
        # Repeated bitrate uses mean quality; this is disclosed in the report.
        return unique, PchipInterpolator(unique, [y[x == u].mean() for u in unique])
    xa, a = fit(ra, qa)
    xb, b = fit(rb, qb)
    if a is None or b is None:
        return None
    low, high = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    return float((b.integrate(low, high) - a.integrate(low, high)) / (high - low)) if high > low else None


def compare_curves(anchor, adaptive, qps, grid, quality_scale=100.):
    """Report raw sensitivity even on nonmonotone curves, with reliability flags."""
    if len(set(qps)) != len(qps):
        raise ValueError("Duplicate QP labels")
    if not set(grid).issubset(qps):
        return dict(status="unavailable", reason="QP grid incomplete", missing_qps=sorted(set(grid) - set(qps)))
    arrays = [np.asarray(c[k], float) for c in (anchor, adaptive) for k in ("rate", "quality")]
    if any(len(a) != len(qps) for a in arrays):
        raise ValueError("Curve length differs from QP labels")
    ids = [qps.index(q) for q in grid]
    ra, qa, rb, qb = [a[ids] for a in arrays]
    if not all(np.isfinite(a).all() for a in (ra, qa, rb, qb)) or min(ra.min(), rb.min()) <= 0:
        raise ValueError("Nonfinite curve or nonpositive bitrate")
    monotone = dict(anchor=bool(np.all(np.diff(qa[np.argsort(ra)]) >= -1e-12)),
                    adaptive=bool(np.all(np.diff(qb[np.argsort(rb)]) >= -1e-12)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cubic = finite(bd_rate(ra, qa, rb, qb))
        pchip = finite(pchip_bd(ra, qa, rb, qb))
        cq = finite(bd_metric(ra, qa, rb, qb) * quality_scale)
        pq = pchip_quality(ra, qa, rb, qb)
    sensitive = cubic is not None and pchip is not None and abs(cubic - pchip) > 5
    return dict(status="computed", qps=list(grid), cubic_bd_rate_pct=cubic, pchip_bd_rate_pct=pchip,
                cubic_bd_quality=cq, pchip_bd_quality=None if pq is None else pq * quality_scale,
                quality_scale=quality_scale, quality_monotone_by_rate=monotone,
                reliable=all(monotone.values()) and cubic is not None and pchip is not None and not sensitive,
                interpolation_sensitive=sensitive, sensitivity_flag_threshold_pp=5.,
                quality_interval=[float(max(qa.min(), qb.min())), float(min(qa.max(), qb.max()))],
                cubic_minus_pchip_pp=None if cubic is None or pchip is None else cubic - pchip,
                active_qp_count=int(np.count_nonzero((ra != rb) | (qa != qb))),
                same_qp_rate_pct=((rb / ra - 1) * 100).tolist(),
                same_qp_quality_delta=((qb - qa) * quality_scale).tolist())


def mean_curves(curves):
    keys = [k for k in ("rate", "quality") if all(k in c for c in curves)]
    return {k: np.mean([c[k] for c in curves], axis=0).tolist() for k in keys}


def add_seed_means(curves):
    for prefix in ("forced", "calibrated", "selected", "unchecked"):
        names = [f"{prefix}_{seed}" for seed in SEEDS]
        if all(n in curves for n in names):
            curves[prefix + "_mean"] = mean_curves([curves[n] for n in names])
    random = [f"random_{s}" for s in range(20)]
    if all(n in curves for n in random):
        curves["B4"] = mean_curves([curves[n] for n in random])


def evaluate_curve_set(dataset, task, codec, metric, qps, curves, scale=100., **extra):
    add_seed_means(curves)
    comparisons = []
    for grid in GRIDS:
        for arm, curve in sorted(curves.items()):
            result = compare_curves(curves["anchor"], curve, list(qps), grid, scale)
            comparisons.append(dict(dataset=dataset, task=task, codec=codec, metric=metric,
                                    arm=arm, grid=len(grid), **result))
        for prefix in ("forced", "calibrated", "selected", "unchecked"):
            members = [r for r in comparisons if r["grid"] == len(grid) and r["arm"] in
                       [f"{prefix}_{s}" for s in SEEDS] and r["status"] == "computed"]
            if len(members) == 3:
                for r in comparisons:
                    if r["grid"] == len(grid) and r["arm"] == prefix + "_mean":
                        r["mean_of_seed_bd_rates"] = {k: stats([m[k] for m in members]) for k in
                                                       ("cubic_bd_rate_pct", "pchip_bd_rate_pct")}
    return dict(dataset=dataset, task=task, codec=codec, metric=metric, qps=list(qps),
                curves=curves, comparisons=comparisons, **extra)


def action_distribution(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["task"], r["arm"])].append(r)
    answer = {}
    for (task, arm), group in sorted(grouped.items()):
        actions = Counter(r["action"] for r in group)
        shortlists = Counter(tuple(r["tried_actions"]) for r in group if "tried_actions" in r)
        answer[f"{task}/{arm}"] = dict(total=len(group), source_count=len({r["id"] for r in group}),
            actions={str(k): dict(count=v, percent=100 * v / len(group)) for k, v in sorted(actions.items())},
            identity_percent=100 * actions[0] / len(group),
            shortlists=[dict(actions=list(k), count=v, percent=100 * v / len(group))
                        for k, v in sorted(shortlists.items(), key=lambda kv: (-kv[1], kv[0]))],
            source_guard_rows=sum("source_guard" in r for r in group))
    return answer


def shortlist(prediction, mask, calibrated=0, size=3, policy="published"):
    rank = prediction["rate"] + .25 * np.maximum(prediction["risk"][:, np.asarray(mask, bool)], 0).mean(1)
    forced = int(np.argmax(prediction["logits"]))
    order = [forced, *np.argsort(rank).tolist()] if policy == "published" else [calibrated, *np.argsort(rank).tolist()]
    candidates = []
    for action in order:
        if action and action not in candidates:
            candidates.append(int(action))
        if len(candidates) == size:
            break
    return [0, *candidates]


def choose_measured(row, candidates, threshold=None, budget=1.01):
    """Exact byte objective; cost-delta guard is a bank surrogate, not source-only."""
    if 0 not in candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Candidates require one identity and no duplicates")
    sizes = np.asarray([p["bytes"] for p in row["packets"]], float)
    if sizes[0] <= 0 or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError("Invalid measured bytes")
    d = TASKS.index(row["domain"])
    costs = np.array(row["cost"], float)[:, d]
    delta = costs - costs[0]
    eligible = [a for a in candidates if sizes[a] / sizes[0] <= budget and
                (threshold is None or (np.isfinite(delta[a]) and delta[a] <= threshold))]
    if not eligible:
        raise ValueError("Identity anchor is not feasible")
    return min(eligible, key=lambda a: (sizes[a] / sizes[0] + (1e-6 * delta[a] if threshold is not None else 0), a))


def validate_bank(rows):
    keys = [(r["id"], r["codec"], r["qp"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate measurement condition")
    for r in rows:
        if len(r["packets"]) != 13 or len(r["cost"]) != 13:
            raise ValueError("Incomplete 13-action bank")
        ratio = np.asarray([p["bytes"] / r["packets"][0]["bytes"] for p in r["packets"]])
        if not np.allclose(ratio, r["rate"], atol=1e-12, rtol=1e-10):
            raise ValueError("Bank rate disagrees with measured bytes")
    return dict(rows=len(rows), sources=len({r["id"] for r in rows}), actions=13,
                conditions_by_domain_split={"/".join(k): v for k, v in sorted(Counter(
                    (r["domain"], r["split"]) for r in rows).items())})


def load_controller(path, evidence, model_source=None):
    """Use the checkpoint's exact architecture; never silently drop head weights."""
    import torch
    from adaptive_vcm.v32vc.checkpoint import state_hash
    from adaptive_vcm.v34.protocol import digest
    evidence.record(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if digest(ckpt["metadata"]) != ckpt["metadata_sha256"] or state_hash(ckpt["state"]) != ckpt["state_sha256"]:
        raise ValueError("Checkpoint integrity mismatch: " + str(path))
    if model_source is not None:
        if not evidence.record(model_source):
            raise ValueError("Requested historical model source is absent")
        spec = importlib.util.spec_from_file_location("v34_audit_checkpoint_model", model_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kwargs = dict(task_logits=True) if ckpt.get("task_logits") else {}
        model = module.Controller(ckpt["actions"], conditions=ckpt.get("conditions", 6), **kwargs)
    else:
        from adaptive_vcm.v34.model import Controller
        model = Controller(ckpt["actions"], conditions=ckpt.get("conditions", 6), task_logits=ckpt.get("task_logits", False))
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()
    model.metadata = ckpt["metadata"]
    return model


def predict_bank(model, rows, inputs, batch_size=16):
    import torch
    result = [None] * len(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["id"] not in inputs:
            continue
        groups[tuple(inputs[row["id"]][0].shape)].append(i)
    with torch.inference_mode():
        for ids in groups.values():
            for offset in range(0, len(ids), batch_size):
                batch = ids[offset:offset + batch_size]
                x = torch.stack([inputs[rows[i]["id"]][0] for i in batch])
                m = torch.stack([inputs[rows[i]["id"]][1] for i in batch])
                c = torch.tensor([rows[i]["condition"] for i in batch], dtype=torch.float32)
                if c.shape[1] != model.conditions:
                    raise ValueError("Saved row condition incompatible with checkpoint")
                out = model(x, m, c)
                for j, i in enumerate(batch):
                    result[i] = {k: v[j].numpy() for k, v in out.items()}
    return result


def replay_bank(cache, predictions, models):
    rows = sorted(cache["rows"], key=lambda r: (r["domain"], r["id"], r["codec"], r["qp"]))
    # predictions are keyed by condition, so sorted replay never changes alignment.
    rngs = [np.random.default_rng(s) for s in range(20)]
    replay = []
    for row in rows:
        d = TASKS.index(row["domain"])
        threshold = .025 if d == 0 else .05
        choices = {"anchor": (0, [0]), "fixed": (11, [11]), "B2_a6": (6, [6]), "B2_a5": (5, [5])}
        choices["oracle"] = (choose_measured(row, list(range(13)), threshold), list(range(13)))
        for s, rng in enumerate(rngs):
            actions = [0, *map(int, rng.choice(np.arange(1, 13), 3, replace=False))]
            choices[f"random_{s}"] = (choose_measured(row, actions, threshold), actions)
        key = (row["id"], row["codec"], row["qp"])
        for seed, by_key in predictions.items():
            pred = by_key.get(key)
            if pred is None:
                continue
            meta = models[seed].metadata
            cal = domain_calibration(meta["calibration"], row["domain"])
            calibrated = select_action(pred["risk"], pred["rate"], cal, row["mask"],
                                       meta["settings"]["damage_tolerance"], 1.01)
            forced = int(np.argmax(pred["logits"]))
            policy = "rank" if meta["settings"].get("shortlist_seed") == "rank" else "published"
            actions = shortlist(pred, row["mask"], calibrated, size=3, policy=policy)
            choices[f"forced_{seed}"] = (forced, [forced])
            # B5b is the risk-bound decision, followed by the saved real-rate fallback.
            chosen = calibrated if row["rate"][calibrated] <= 1.01 else 0
            choices[f"calibrated_{seed}"] = (chosen, [0] + ([calibrated] if calibrated else []))
            choices[f"unchecked_{seed}"] = (choose_measured(row, actions), actions)
            choices[f"selected_{seed}"] = (choose_measured(row, actions, threshold), actions)
        for arm, (action, tried) in choices.items():
            packets = [row["packets"][a] for a in tried]
            guarded = arm.startswith(("selected_", "random_")) or arm == "oracle"
            replay.append(dict(id=row["id"], task=row["domain"], split=row["split"], codec=row["codec"],
                qp=row["qp"], arm=arm, action=action, rate=row["packets"][action]["bytes"],
                quality=-float(row["cost"][action][d]), tried_actions=tried,
                encode_count=len(tried), decode_count=len(tried),
                guard_candidate_count=sum(row["rate"][a] <= 1.01 for a in tried) if guarded else 0,
                preprocess_seconds=sum(p["preprocess_seconds"] for p in packets),
                encode_seconds=sum(p["encode_seconds"] for p in packets),
                decode_seconds=sum(p["decode_seconds"] for p in packets)))
    return replay


def curves_from_rows(rows, dataset, quality_key="quality", rate_key="rate", metric="negative_task_cost", scale=1.):
    keys = [(r["task"], r["id"], r["codec"], r["qp"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate paired evaluation row")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["codec"])].append(row)
    answer = []
    for (task, codec), group in sorted(groups.items()):
        qps = sorted({r["qp"] for r in group})
        cells = defaultdict(list)
        for r in group:
            cells[(r["arm"], r["qp"])].append(r)
        anchor_ids = {q: {r["id"] for r in cells[("anchor", q)]} for q in qps}
        if any(anchor_ids[q] != anchor_ids[qps[0]] for q in qps):
            raise ValueError("Anchor source population varies across QPs")
        curves = {}
        for arm in sorted({r["arm"] for r in group}):
            values = [cells[(arm, q)] for q in qps]
            if any(len(v) != len(anchor_ids[q]) or {r["id"] for r in v} != anchor_ids[q]
                   for q, v in zip(qps, values)):
                continue  # Missing controller inputs cannot create a biased partial curve.
            curves[arm] = dict(rate=[float(np.mean([r[rate_key] for r in v])) for v in values],
                               quality=[float(np.mean([r[quality_key] for r in v])) for v in values])
        if "anchor" in curves:
            answer.append(evaluate_curve_set(dataset, task, codec, metric, qps, curves, scale,
                                             source_count=len(anchor_ids[qps[0]])))
    return answer


def bank_costs(replay):
    groups = defaultdict(list)
    for r in replay:
        groups[(r["task"], r["codec"], r["arm"])].append(r)
        if r["arm"].startswith("random_"):
            groups[(r["task"], r["codec"], "B4")].append(r)
    output = []
    for (task, codec, arm), group in sorted(groups.items()):
        entry = dict(task=task, codec=codec, arm=arm, scope="sum of saved candidate timings; excludes support/CNN/guard/receiver")
        for k in ("encode_count", "decode_count", "guard_candidate_count", "preprocess_seconds", "encode_seconds", "decode_seconds"):
            entry[k] = stats([r[k] for r in group])
        subtotal = [r["preprocess_seconds"] + r["encode_seconds"] + r["decode_seconds"] for r in group]
        entry.update(candidate_work_seconds=stats(subtotal), sender_seconds=None, fps=None,
                     limitation="No per-action source-only guard timings in measurement bank; candidate count is not model forward count")
        output.append(entry)
    return output


def od_source_only_probe(source):
    """Execute the real costs method with tiny fake heads, never load a detector."""
    import torch
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "costs")
    method.decorator_list = []
    observed = []
    def forward(*args):
        observed.append("od_forward_with_targets" if len(args) > 3 else "od_forward_label_free")
        return [{}], torch.zeros(1)
    env = dict(torch=torch, tensor=lambda rgb, device: rgb, frame_indices=lambda n: [0],
               od_forward=forward, od_forward_label_free=forward, prediction=lambda x: x,
               detection_distance=lambda a, b: observed.append("detection_distance") or .125,
               distance=lambda a, b: torch.tensor(0.))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "<audited costs>", "exec"), env)
    record = SimpleNamespace(task="od", targets=None, mask=[False, True, False],
                             source_predictions={"od": [{}]}, source_features={"od": torch.zeros(1)})
    try:
        result = env["costs"](SimpleNamespace(device="cpu", detector=None), np.zeros((1, 8, 8, 3), np.uint8),
                              record, source_only=True)
        return dict(outcome="returned", value=result, calls=observed)
    except (AttributeError, TypeError) as error:
        return dict(outcome="error", exception=type(error).__name__, message=str(error), calls=observed)


def filter_probe():
    from adaptive_vcm.v34.preprocessing import filter_clip
    rng = np.random.default_rng(303401)
    rgb = rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    a = np.zeros(rgb.shape, np.float32)
    b = a.copy()
    a[..., 0] = 1
    b[..., 2] = 1
    differences = {}
    for action in (0, 5, 6, 11):
        difference = filter_clip(rgb, a, ACTIONS[action]).astype(float) - filter_clip(rgb, b, ACTIONS[action]).astype(float)
        differences[str(action)] = float(np.max(np.abs(difference)))
    return dict(actions={str(i): list(ACTIONS[i]) for i in (0, 5, 6, 11)},
                max_absolute_pixel_difference=differences,
                identity_unchanged=bool(np.array_equal(rgb, filter_clip(rgb, a, ACTIONS[0]))),
                interpretation="Action 11 is mask-independent global two-Gaussian frequency attenuation (B1).",
                requested_action5=[-.15, .05, -.3, -.05, -.5, -.15],
                action5_matches_request=list(ACTIONS[5]) == [-.15, .05, -.3, -.05, -.5, -.15])


def ffmpeg_probe():
    executable = shutil.which("ffmpeg")
    if executable is None:
        return dict(status="unavailable", reason="ffmpeg executable absent")
    version = subprocess.run([executable, "-version"], capture_output=True, text=True, timeout=15).stdout.splitlines()[0]
    result = dict(executable=executable, version=version, tests={})
    for name, flags in (("legacy_vsync", ["-vsync", "0"]), ("fps_mode", ["-fps_mode", "passthrough"])):
        command = [executable, "-nostdin", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                   "color=size=16x16:rate=1:duration=1", *flags, "-frames:v", "1", "-f", "null", "-"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result["tests"][name] = dict(command=command, returncode=completed.returncode, stderr=completed.stderr.strip())
    result["scope"] = "Tests this executable only; no blanket claim for every FFmpeg >=8 build"
    return result


def calibration_evidence(rows, checkpoint_dir, evidence):
    import torch
    result = {}
    for seed in SEEDS:
        path = checkpoint_dir / f"controller_{seed}.pth"
        if not evidence.record(path):
            continue
        metadata = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
        for task in sorted({r["task"] for r in rows}):
            group = [r for r in rows if r["task"] == task and r["arm"] == f"selected_{seed}"
                     and "predicted_risk" in r]
            if not group:
                continue
            cal = domain_calibration(metadata["calibration"], task)
            d = TASKS.index(task)
            risk = np.asarray([r["predicted_risk"] for r in group], float)
            rates = np.asarray([r["predicted_rate"] for r in group], float)
            bound = risk[:, 1:, d] + np.asarray(cal["risk"])[None, 1:, d]
            tol = metadata["settings"]["damage_tolerance"][d]
            rate_bound = rates[:, 1:] + np.asarray(cal["rate"])[None, 1:]
            result[f"{task}/{seed}"] = dict(rows=len(group), tolerance=tol, coverage=cal["coverage"],
                training_risk_loss=metadata["settings"].get("risk_loss", "Smooth L1 if risk_scale is set; otherwise MSE"),
                training_quantile=metadata["settings"].get("risk_quantile"),
                q_hat=cal["risk"], nonidentity_task_bound=stats(bound.ravel().tolist()),
                all_nonidentity_risk_rejected=sum(bool(v) for v in (bound > tol).all(1)),
                all_nonidentity_risk_or_rate_rejected=sum(bool(v) for v in ((bound > tol) | (rate_bound > 1.01)).all(1)),
                calibrated_identity=sum(r.get("calibrated_action", r["action"]) == 0 for r in group))
    return result


def task1_audit(evidence, scratch, paper, distributions, calibrations):
    taskfile = ROOT / "adaptive_vcm/v34/tasks.py"
    runner = ROOT / "scripts/run_v34_od.py"
    paths = ["adaptive_vcm/v34/tasks.py", "scripts/run_v34_od.py", "configs/v34.json",
             "adaptive_vcm/v34/learning.py", "adaptive_vcm/v32/codec.py", "adaptive_vcm/v34/model.py",
             "adaptive_vcm/v34/evaluation.py", "adaptive_vcm/metrics.py"]
    historical = {p: historical_source(p) for p in paths}
    def packet_calls(source):
        return [dict(line=n.lineno, keywords=[k.arg for k in n.keywords],
                     source_score_present=any(k.arg == "source_score" for k in n.keywords))
                for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id == "choose_packet"]
    old_calls = packet_calls(historical["scripts/run_v34_od.py"])
    calls = packet_calls(evidence.text(runner))
    old_cfg = json.loads(historical["configs/v34.json"])
    cfg = evidence.read(ROOT / "configs/v34.json")
    r1cfg = evidence.read(ROOT / "configs/v34_arot_r1.json")
    learning = ROOT / "adaptive_vcm/v34/learning.py"
    revised = scratch / "v34_fix1/adaptive_vcm/v34/learning.py"
    old_probe = od_source_only_probe(historical["adaptive_vcm/v34/tasks.py"])
    fixed_source = evidence.text(taskfile)
    fixed_probe = od_source_only_probe(fixed_source)
    numerical_filter = filter_probe()
    historical_identity = all(distributions.get("coco_historical", {}).get(f"od/selected_{s}", {}).get(
        "identity_percent") == 100 for s in SEEDS)
    fallback_confirmed = bool(old_calls) and not any(c["source_score_present"] for c in old_calls)
    fallback_confirmed &= "source_guard_limits" not in old_cfg and historical_identity
    od_fixed = fixed_probe["outcome"] == "returned" and fixed_probe["calls"] == ["od_forward_label_free", "detection_distance"]
    runner_fixed = bool(calls) and all(c["source_score_present"] for c in calls)
    runner_fixed &= cfg.get("source_guard_limits") == [.025, .05, .05] and cfg.get("source_shortlist") == 5
    ffmpeg = ffmpeg_probe()
    from adaptive_vcm.v32.codec import _fps_sync_flags
    flags = _fps_sync_flags(ffmpeg["executable"]) if "executable" in ffmpeg else None
    ffmpeg_fixed = flags is not None and ffmpeg["tests"]["fps_mode" if flags[0] == "-fps_mode" else "legacy_vsync"]["returncode"] == 0
    from adaptive_vcm.v34.learning import calibrate, select_direct_action
    p = np.zeros((20, 2, 3)); actual = p.copy(); actual[-3:, 1, 0] = 10.
    cal_args = (p, actual, np.ones((20, 2)), np.ones((20, 2)), np.ones((20, 3), bool), .9)
    quantile, mean_ucb = calibrate(*cal_args), calibrate(*cal_args, mode="mean_ucb")
    calibration_fixed = 0 < mean_ucb["risk"][1][0] < quantile["risk"][1][0]
    direct_input = dict(logits=np.array([0., 10., 1.]), logits_task=np.array([[0.] * 3, [10.] * 3, [1.] * 3]),
                        risk=np.array([[0.] * 3, [1.] * 3, [0.] * 3]), rate=np.array([1., .7, .8]))
    direct = select_direct_action(direct_input, [True, False, False], [.05] * 3, 1.01, domain_idx=0)
    findings = [
        dict(id="od_source_only", status="confirmed" if old_probe["outcome"] == "error" else "not_confirmed",
             evidence=source_lines(historical["adaptive_vcm/v34/tasks.py"], r"source_only|r\.targets|r\.task=='od'"),
             original_probe=old_probe, fixed_probe=fixed_probe, fixed_in_workspace=od_fixed,
             fixed_evidence=evidence.lines(taskfile, r"r\.task=='od'|od_forward_label_free|detection_distance")),
        dict(id="historical_od_selected", status="confirmed" if fallback_confirmed else "not_confirmed",
             choose_packet_calls=old_calls, workspace_choose_packet_calls=calls,
             source_guard_limits_present="source_guard_limits" in old_cfg,
             workspace_source_guard_limits=cfg.get("source_guard_limits"), fixed_in_workspace=runner_fixed,
             implementation=source_lines(historical["adaptive_vcm/v34/evaluation.py"], r"source_score|select_action|proposed=selected"),
             paper=evidence.lines(paper / "4_experiment_result.tex", r"Selected.*Source Guard|Source Guard.*nhận"),
             result="Historical selected used calibrated select_action fallback. The repaired runner passes a cached label-free source guard and records calibrated arms separately."),
        dict(id="conformal_collapse", status="dataset_specific",
             result="Historical calibrated counts are retained. Quantile calibration remains the default; the workspace adds clipped mean-risk UCB calibration. Task 5 measures the policy change on saved banks.",
             calibration=calibrations, fixed_in_workspace=bool(calibration_fixed),
             workspace_probe=dict(quantile=quantile, mean_ucb=mean_ucb),
             workspace_evidence=evidence.lines(learning, r"def calibrate|mean_ucb|residual_clip|sem=")),
        dict(id="fixed_is_global", status="confirmed", **numerical_filter,
             paper_evidence=evidence.lines(paper / "4_experiment_result.tex", r"a_\{11\}|không phân biệt vùng")),
        dict(id="direct_controller_collapse", status="measured", distributions={k: v for k, v in distributions.items()
             if k in ("k400_publication", "ot_test", "ot_dev")},
             fixed_in_workspace=direct == 2, workspace_probe=dict(raw_argmax=1, risk_gated_direct=direct),
             interpretation="Historical forced action 11 is B1 behavior. The workspace adds task-logit selection gated by predicted risk/rate. "
                            "Frozen risk heads can still reject all nonidentity actions; Task 5 reports this without retraining."),
        dict(id="paper_formula_latency", status="partly_contradicted",
             configured_source_guard_limits=r1cfg["source_guard_limits"],
             frozen_loss=source_lines(historical["adaptive_vcm/v34/learning.py"], r"smooth_l1|pinball|rank_loss|loss=|scale is None"),
             scratch_loss=evidence.lines(revised, r"pinball|rank_loss|loss=loss|loss=F"),
             paper=evidence.lines(paper / "3_proposed_method.tex", r"epsilon_k =|Smooth|sigma_"),
             latency_claim=evidence.lines(paper / "4_experiment_result.tex", r"0,8 ms|0.8 ms|8.661"),
             result="Paper Smooth L1 agrees with frozen scaled-risk training. Pinball belongs to later variants; 1.5*rank_loss must be verified in the actual source, not asserted. Measured latency is in task3."),
        dict(id="ffmpeg_vsync", source=source_lines(historical["adaptive_vcm/v32/codec.py"], r"vsync"),
             workspace_evidence=evidence.lines(ROOT / "adaptive_vcm/v32/codec.py", r"fps_sync_flags|fps_mode|vsync"),
             workspace_flags=flags, fixed_in_workspace=ffmpeg_fixed, **{"status": "probed", **ffmpeg}),
    ]
    return dict(findings=findings, distributions=distributions,
                fixed_in_workspace=bool(od_fixed and runner_fixed and ffmpeg_fixed and calibration_fixed and direct == 2),
                historical_ref=HISTORICAL_REF,
                historical_sources={p: dict(git_object=f"{HISTORICAL_REF}:{p}", sha256=hashlib.sha256(s.encode()).hexdigest())
                                    for p, s in historical.items()})


def summarize_saved(rows, summary, dataset, evidence):
    output = []
    # AR correctness is additive, so recompute from all paired rows (not shard means).
    ar = [r for r in rows if r["task"] == "ar"]
    for metric in ("swin3d_t", "r3d_18"):
        values = [dict(r, quality=r["correct"][metric]) for r in ar if metric in r.get("correct", {})]
        if values:
            output.extend(curves_from_rows(values, dataset, rate_key="bpp", metric=metric, scale=100.))
    if not summary:
        return output
    # mAP must be computed jointly over detections. Read its exact saved COCOeval
    # QP values; averaging per-image AP or inferring mAP from cost is invalid.
    for key, data in summary.get("results", {}).items():
        if key.startswith("ar/"):
            continue
        codec = key.split("/")[-1]
        metrics = data["curves"]
        if "anchor" in metrics:
            metrics = {"resnet50": metrics}
        for metric, arm_curves in metrics.items():
            if metric not in ("resnet50",):
                continue
            n = len(arm_curves["anchor"]["bpp"])
            qps = summary.get("qps", list(GRIDS[1] if n == 5 else GRIDS[0]))
            curves = {a: dict(rate=c["bpp"], quality=c["quality"]) for a, c in arm_curves.items()}
            # Verify that archived mean rates and all observed source counts agree.
            for arm, curve in curves.items():
                for i, q in enumerate(qps):
                    values = [r["bpp"] for r in rows if r["task"] == "od" and r["codec"] == codec
                              and r["arm"] == arm and r["qp"] == q]
                    if not values or not math.isclose(float(np.mean(values)), curve["rate"][i], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"Saved COCO curve rate mismatch: {dataset}/{arm}/{q}")
            output.append(evaluate_curve_set(dataset, "od", codec, metric, qps, curves,
                provenance="Exact saved COCOeval mAP; byte-rate cross-check against raw rows; no new COCOeval without GT"))
    output.extend(tracking_summary_curves(summary, dataset))
    return output


def tracking_summary_curves(summary, dataset):
    official = summary.get("ot_official_metrics", summary.get("official_tracking_metrics", {}))
    if not official:
        return []
    tables = summary.get("ot_curves", summary.get("curves", {}))
    output = []
    for codec in ("h264", "h265"):
        if codec not in tables:
            continue
        saved = tables[codec].get("curves", tables[codec])
        anchor = saved["anchor"]
        qps = list(GRIDS[1] if len(anchor.get("bpp", anchor.get("rate"))) == 5 else GRIDS[0])
        for metric in ("HOTA", "IDF1", "MOTA"):
            curves = {}
            for arm, curve in saved.items():
                points = [official.get(f"{codec}_{q}_{arm}", {}).get("combined", {}).get(metric) for q in qps]
                if any(p is None for p in points):
                    continue
                curves[arm] = dict(rate=curve.get("bpp", curve.get("rate")), quality=points)
            if "anchor" in curves:
                # These baseline/summary TrackEval exports are in percentage units.
                output.append(evaluate_curve_set(dataset, "ot", codec, metric, qps, curves, 1.,
                                                  provenance="Saved official sequence-level TrackEval; no rerun"))
    return output


def heldout_replay_coverage(rows):
    """Count which additional policies have an actual saved quality observation."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["id"], r["codec"], r["qp"])].append(r)
    counts = Counter()
    for group in groups.values():
        available = {r["action"] for r in group}
        counts["conditions"] += 1
        for action in (5, 6):
            counts[f"B2_a{action}_observed"] += action in available
        for r in group:
            if r["arm"] != "selected_303401":
                continue
            work = r.get("deployed_candidate_work", [])
            # Timings do not contain candidate bytes or task quality. Do not use
            # predicted_rate as if it were an actual encoded packet size.
            counts["candidate_timing_rows"] += bool(work)
            counts["full_13_action_conditions"] += len(available) == 13
    return dict(counts, limitation="Unmeasured candidates prevent exact held-out B2/B4/B5a replay. A timing or source-cost record is not an accuracy observation.")


COST_FIELDS = (
    "support_seconds", "preprocess_seconds", "deployed_preprocess_seconds", "controller_inference_seconds",
    "encode_seconds", "decode_seconds", "deployed_encode_seconds", "deployed_probe_decode_seconds",
    "deployed_encode_count", "deployed_decode_count", "diagnostic_teacher_seconds", "deployed_source_guard_seconds",
    "receiver_seconds", "deployed_sender_seconds", "standalone_benchmark_seconds",
)


def cost_analysis(rows, dataset):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["codec"], row["arm"])].append(row)
    output = []
    for (task, codec, arm), group in sorted(groups.items()):
        frames = 1 if task == "od" else 16
        fields = {k: stats([r.get(k) for r in group]) for k in COST_FIELDS}
        entry = dict(dataset=dataset, task=task, codec=codec, arm=arm, rows=len(group),
                     shape=[frames, {"ar": 128, "od": 320, "ot": 960}[task], {"ar": 128, "od": 320, "ot": 960}[task]],
                     fields=fields)
        anchor = groups.get((task, codec, "anchor"), [])
        # Frozen rows give deployed work only for selected. A standard anchor
        # encoder needs neither source maps nor a sender-side decoded probe.
        b0_sender = [r["preprocess_seconds"] + r["encode_seconds"] for r in anchor
                     if "preprocess_seconds" in r and "encode_seconds" in r]
        sender = [r["deployed_sender_seconds"] for r in group if "deployed_sender_seconds" in r]
        end = [r["deployed_sender_seconds"] + r["decode_seconds"] + r["receiver_seconds"] for r in group
               if all(k in r for k in ("deployed_sender_seconds", "decode_seconds", "receiver_seconds"))]
        entry.update(b0_encoder_seconds=stats(b0_sender), sender_fps=stats([frames / s for s in sender if s > 0]),
                     aggregate_sender_fps=frames / np.mean(sender) if sender else None,
                     end_to_end_seconds=stats(end),
                     sender_slowdown_vs_b0=float(np.mean(sender) / np.mean(b0_sender)) if sender and b0_sender else None,
                     b0_scope="preprocess(identity copy)+encode; excludes unnecessary support and probe decode",
                     end_to_end_scope="sender + final receiver decode + receiver task/evaluator; transfer evaluator included")
        entry["derived_probe_decode_count"] = stats([len(r["deployed_candidate_work"]) for r in group
                                                     if "deployed_candidate_work" in r])
        entry["guard_candidate_count"] = stats([sum(any(v is not None for v in c) for c in r["source_guard"]["costs"])
                                                  for r in group if "source_guard" in r])
        entry["sg_model_forward_count"] = None  # May involve multiple heads and batches, not one call per candidate.
        output.append(entry)
    return output


def model_costs(evidence, scratch):
    import torch
    from adaptive_vcm.v34.model import Controller
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.misc import FrozenBatchNorm2d
    from torchvision.models.video import r3d_18, swin3d_t
    # Meta tensors count real architecture parameters without allocating weights,
    # downloading assets, or confusing model bytes with full-system peak VRAM.
    factories = {"Controller_frozen_9_conditions": lambda: Controller(13, conditions=9),
                 "Controller_frozen_6_conditions": lambda: Controller(13),
                 "FasterRCNN_ResNet50_FPN": lambda: FasterRCNN(resnet_fpn_backbone(
                     backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d), num_classes=91),
                 "Swin3D_T": lambda: swin3d_t(weights=None), "R3D_18": lambda: r3d_18(weights=None)}
    models = {}
    for name, factory in factories.items():
        with torch.device("meta"):
            model = factory()
        count = sum(p.numel() for p in model.parameters())
        models[name] = dict(parameters=count, fp32_parameter_bytes=count * 4, fp32_MB=count * 4 / 1e6,
                            buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()))
    path = scratch / "munnn01_test/docs/V34R1-COSTS.csv"
    raw = evidence.text(path)
    rows = list(csv.DictReader(raw.splitlines())) if raw else []
    for row in rows:
        for k, v in list(row.items()):
            if k not in ("task", "codec"):
                row[k] = float(v)
        row["peak_gpu_MB"] = row["whole_task_peak_torch_cuda_allocated_bytes"] / 1e6
    return dict(models=models, full_system_csv=rows, experiment_cpu_rss_bytes=None,
                cpu_rss_status="Not recorded in supplied CSV or selected raw-row timing schema",
                csv_crosscheck="Different experiment/policy: CSV has 5 or 5.515625 trials; later saved selected arms use 6. Latencies must not be conflated.")


def timing_crosschecks(timings, memory):
    comparisons = []
    for row in memory["full_system_csv"]:
        name = "ot_test" if row["task"] == "ot" else "scaled_test"
        measured = next((v for v in timings if v["dataset"] == name and v["task"] == row["task"] and
                         v["codec"] == row["codec"] and v["arm"] == "selected_303401"), None)
        if measured:
            mean = measured["fields"]["deployed_sender_seconds"]["mean"]
            comparisons.append(dict(dataset=name, task=row["task"], codec=row["codec"],
                row_sender_seconds=mean, csv_sender_seconds=row["mean_sender_selection_seconds"],
                difference_seconds=mean - row["mean_sender_selection_seconds"],
                row_encode_count=measured["fields"]["deployed_encode_count"]["mean"],
                csv_encode_count=row["mean_distinct_trial_encodes"],
                scope="Cross-experiment discrepancy, not a failed equality check on one run"))
    claims = []
    for task, requested_ms in (("ar", 2.29), ("ot", 2.45)):
        for entry in timings:
            if entry["task"] != task or entry["arm"] != "selected_303401":
                continue
            actual = entry["fields"]["controller_inference_seconds"]["mean"]
            sender = entry["fields"]["deployed_sender_seconds"]["mean"]
            claims.append(dict(dataset=entry["dataset"], task=task, codec=entry["codec"],
                               requested_controller_ms=requested_ms, measured_controller_ms=actual * 1000,
                               exceeds_paper_0_8ms=actual * 1000 > .8, measured_sender_ms=sender * 1000))
    return dict(csv=comparisons, latency_claims=claims)


def paper_tracking(evidence, paper):
    raw = evidence.text(paper / "tables/ot_qp_results.tex")
    codec = None
    cells = defaultdict(list)
    for line in raw.splitlines():
        if "&" not in line:
            continue
        parts = [v.strip().replace(",", ".").replace("\\", "") for v in line.split("&")]
        if len(parts) != 9:
            continue
        if parts[0] in ("H.264", "H.265"):
            codec = "h" + parts[0][2:]
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if codec:
            cells[codec].append(values)
    curves = []
    for codec, values in sorted(cells.items()):
        qps = [int(v[0]) for v in values]
        table = dict(anchor=dict(rate=[v[1] for v in values], quality=[v[3] for v in values]),
                     selected_303401=dict(rate=[v[2] for v in values], quality=[v[4] for v in values]))
        curves.append(evaluate_curve_set("MOT20-05_paper_rounded", "ot", codec, "HOTA", qps, table, 1.,
                                        precision="bpp 4 decimals; HOTA 2 decimals; not full-precision measurements"))
    expected = {}
    for line in evidence.text(paper / "tables/ot_bd_results.tex").splitlines():
        if line.startswith(("H.264 &", "H.265 &")):
            parts = line.replace(",", ".").replace("\\", "").split("&")
            expected["h" + parts[0].strip()[2:]] = dict(zip(
                ("cubic_bd_rate_pct", "pchip_bd_rate_pct", "cubic_bd_quality", "vs_fixed_cubic_bd_rate_pct"),
                map(float, parts[1:])))
    checks = []
    for c in curves:
        actual = next(v for v in c["comparisons"] if v["arm"] == "selected_303401" and v["grid"] == 4)
        checks.append(dict(codec=c["codec"], paper=expected.get(c["codec"]),
                           recomputed_minus_paper={k: actual[k] - expected[c["codec"]][k] for k in
                                                  ("cubic_bd_rate_pct", "pchip_bd_rate_pct", "cubic_bd_quality")}))
    evidence.gap("MOT20-05 IDF1/MOTA and exact precision", "LaTeX QP table gives only IDF1/MOTA deltas, no anchor values, and rounded rate/HOTA. Cannot verify claimed -12.88/-6.91 MOTA or exact unrounded HOTA.")
    return curves, checks


def mot17_analysis(evidence, scratch):
    directory = scratch / "munnn01_test/results/v34r1_a6/ot-full21"
    report = evidence.read(directory / "replay-report.json")
    if report is None:
        return dict(curves=[], status="unavailable")
    raw = evidence.text(directory / "raw-curves.csv")
    csv_rows = list(csv.DictReader(raw.splitlines()))
    curve_sets = []
    csv_checks = []
    # All FIT candidates plus the frozen reused-CAL decision. These are not a
    # held-out source-only benchmark, as the source report explicitly records.
    splits = {"fit": {"anchor": report["fit_anchor"], **{
        "full_lambda_" + k: v for k, v in report["fit_candidates"].items()}},
        "validation_reused_cal": {"anchor": report["calibration"]["anchor"],
                                  "full_lambda_32": report["calibration"]["selected"]}}
    sequence_names = set()
    for split, policies in splits.items():
        anchor = policies["anchor"]
        sequences = sorted({s for m in anchor["metrics"].values() for s in m["sequences"]})
        sequence_names.update(sequences)
        for codec in ("h264", "h265"):
            for sequence in ("pooled_official", *sequences):
                for metric in ("HOTA", "IDF1", "MOTA"):
                    curves = {}
                    for policy, value in policies.items():
                        quality = []
                        rate = []
                        for i, qp in enumerate(GRIDS[0]):
                            metrics = value["metrics"][f"{codec}_{qp}"]
                            quality.append(metrics["combined"][metric] if sequence == "pooled_official" else
                                           metrics["sequences"][sequence][metric])
                            if sequence == "pooled_official":
                                rate.append(value["curves"][codec]["rate"][i])
                            else:
                                all_rows = [r for r in value["selected"] if r["codec"] == codec and r["qp"] == qp]
                                seq_rows = [r for r in all_rows if r["sequence"] == sequence]
                                # Same 16x960x960 canonical pixel denominator across windows.
                                # Confirm it from exact pooled bytes instead of inventing rates.
                                denominator = float(np.mean([r["bytes"] for r in all_rows])) / value["curves"][codec]["rate"][i]
                                rate.append(float(np.mean([r["bytes"] for r in seq_rows])) / denominator)
                        curves[policy] = dict(rate=rate, quality=quality)
                    curve_sets.append(evaluate_curve_set(f"MOT17/{split}/{sequence}", "ot", codec, metric,
                        GRIDS[0], curves, 100., provenance="Exact selected packet bytes and saved TrackEval sequence/combined metrics"))
            for policy, value in policies.items():
                csv_policy = policy.replace("_", "-") if split == "fit" else ("anchor" if policy == "anchor" else "frozen-full-lambda-32")
                matches = [r for r in csv_rows if r["codec"] == codec and r["split"] ==
                           ("fit" if split == "fit" else "validation") and r["policy"] == csv_policy]
                if matches:
                    matches.sort(key=lambda r: int(r["qp"]))
                    csv_checks.append(dict(split=split, codec=codec, policy=policy,
                        rates_match=bool(np.allclose([float(r["paid_bpp"]) for r in matches], value["curves"][codec]["rate"], rtol=1e-12)),
                        hota_match=bool(np.allclose([float(r["hota"]) for r in matches], value["curves"][codec]["quality"], rtol=1e-12))))
    selection = [r for v in splits.values() for p in v.values() for r in p["selected"]]
    windows = {(r["sequence"], r["start"]) for r in selection}
    result = dict(curves=curve_sets, csv_crosschecks=csv_checks, protocol=report["protocol"],
                  observed_sequences=sorted(sequence_names), observed_windows=len(windows),
                  full21_rows=report["full21_rows"], independent_validation=report["independent_validation"],
                  test_scored=report["test_scored"], expanded_evaluation_not_executed=report["expanded_evaluation_not_executed"],
                  requested_sequences=["MOT17-02", "MOT17-05", "MOT17-09", "MOT17-11"],
                  interpretation="The supplied A6 artifact uses MOT17-02/05 FIT and MOT17-04/11 reused CAL. It is not a 120-window held-out benchmark. Sequence-level policy tuning cannot establish strict window-local source-only generalization.",
                  boundary_state_hypothesis="ByteTrack state can propagate across windows, but this report alone does not isolate causality or verify the claimed held-out eps=0.05 rejection/forced-drift experiment.")
    result["full_lambda_32_claim_check"] = [dict(codec=c["codec"], requested_bd_rate_pct={"h264": -23.94, "h265": -16.92}[c["codec"]],
        **{k: r[k] for k in ("cubic_bd_rate_pct", "pchip_bd_rate_pct", "reliable")})
        for c in curve_sets if c["dataset"] == "MOT17/fit/pooled_official" and c["metric"] == "HOTA"
        for r in c["comparisons"] if r["arm"] == "full_lambda_32" and r["grid"] == 4]
    evidence.gap("MOT17 requested held-out 02/05/09/11, 120 windows", "Input report has no MOT17-09, uses MOT17-05 for FIT, flags independent_validation=false and expanded_evaluation_not_executed=true. Strict window-local ablation results are absent.")
    return result


def combine_mot20(curve_sets, evidence):
    output = []
    for codec in ("h264", "h265"):
        for metric in ("HOTA", "IDF1", "MOTA"):
            wanted = [("ot_test", 256), ("ot_dev", 256), ("MOT20-05_paper_rounded", 1024)]
            parts = []
            for name, weight in wanted:
                item = next((c for c in curve_sets if c["dataset"] == name and c["codec"] == codec
                             and c["metric"] == metric), None)
                if item:
                    parts.append((item, weight))
            if len(parts) != 3:
                evidence.gap(f"MOT20-01+03+05/{codec}/{metric}", "Complete comparable per-sequence QP curves absent")
                continue
            for arm in ("selected_303401", "selected_mean", "fixed"):
                if not all(arm in c["curves"] for c, _ in parts):
                    continue
                individual = [compare_curves(c["curves"]["anchor"], c["curves"][arm], c["qps"], GRIDS[0], 1.) for c, _ in parts]
                pooled = {}
                for a in ("anchor", arm):
                    pooled[a] = {k: np.average([c["curves"][a][k] for c, _ in parts], axis=0,
                                               weights=[w for _, w in parts]).tolist() for k in ("rate", "quality")}
                output.append(dict(codec=codec, metric=metric, arm=arm, sequences=[n for n, _ in wanted],
                    macro_bd={k: stats([v[k] for v in individual]) for k in ("cubic_bd_rate_pct", "pchip_bd_rate_pct")},
                    descriptive_pooled_curve=pooled,
                    descriptive_pooled_bd=compare_curves(pooled["anchor"], pooled[arm], list(GRIDS[0]), GRIDS[0], 1.),
                    limitation="Frame-weighted descriptive rate/metric curve, NOT official combined TrackEval HOTA/IDF1/MOTA. Includes rounded MOT20-05 and different policy versions; no combined generalization claim."))
    evidence.gap("Official pooled MOT20-01+03+05 and +02", "Sequence-specific TrackEval sufficient statistics/tracks are absent; MOT20-02 bank contains only GT-conditioned window costs, not official HOTA/IDF1/MOTA. Averaging HOTA is not official TrackEval pooling.")
    return output


def task2_ablation(evidence, kaggle, scratch):
    import torch
    output = []
    costs = []
    banks = []
    specs = [("scaled", "v34-fix2-ar-od-scaled", "v34-fix2-ar-od-scaled"),
             ("coco256_rank", "v34-fix1-od-coco", "v34-fix2-od-rank"),
             ("coco256_scaled_controller", "v34-fix1-od-coco", "v34-fix2-ar-od-scaled"),
             ("MOT20-02", "v34-ot-measure", "v34-ot-baseline")]
    for label, bank_name, model_name in specs:
        print(f"Replaying bank {label}...", flush=True)
        directory = kaggle / bank_name / "outputs" / bank_name
        model_dir = kaggle / model_name / "outputs" / model_name
        cache = evidence.read(directory / "measurements.json")
        if cache is None:
            continue
        bank_info = validate_bank(cache["rows"])
        bank_info.update(name=label, action5=cache["actions"][5], settings=cache["settings"])
        if cache["actions"][5] != [-.15, .05, -.3, -.05, -.5, -.15]:
            evidence.gap("B2_a5 requested coefficients", "The requested tuple is not action 5 in the saved bank. Tables report the actual measured action 5; the requested coefficient tuple has no saved measurement.")
        if any(list(ACTIONS[i]) != cache["actions"][i] for i in range(13)):
            bank_info["bank_differs_from_frozen_actions"] = True
        input_path = directory / "controller_inputs.pth"
        input_cache = cache
        if not evidence.record(input_path):
            # The scaled bank may cover an exact subset of the COCO IDs. Never
            # evaluate uncovered rows with fabricated or approximate inputs.
            input_path = kaggle / "v34-fix2-ar-od-scaled/outputs/v34-fix2-ar-od-scaled/controller_inputs.pth"
            input_cache = evidence.read(input_path.parent / "measurements.json")
        inputs = torch.load(input_path, weights_only=True, map_location="cpu") if evidence.record(input_path) else {}
        mismatched = [key for key in inputs if key in cache["sources"] and
                      cache["sources"][key]["pixel_sha256"] != input_cache["sources"][key]["pixel_sha256"]]
        if mismatched:
            raise ValueError("Controller input source pixel hashes differ from bank: " + repr(mismatched[:3]))
        bank_info["input_source_hashes_match"] = True
        missing = sorted({r["id"] for r in cache["rows"]} - inputs.keys())
        bank_info["missing_controller_input_ids"] = missing
        if missing:
            evidence.gap(label + " controller replay", f"{len(missing)} of {bank_info['sources']} bank source IDs lack saved inputs; partial controller curves are suppressed")
        models, predictions = {}, {}
        for seed in SEEDS:
            checkpoint = model_dir / f"controller_{seed}.pth"
            if not evidence.record(checkpoint):
                evidence.gap(label + f"/{seed}", "Controller checkpoint absent")
                continue
            model_source = kaggle / model_name / "v34_fix1/adaptive_vcm/v34/model.py"
            if not model_source.exists():
                model_source = scratch / "v34_fix1/adaptive_vcm/v34/model.py"
            model = load_controller(checkpoint, evidence, model_source)
            models[seed] = model
            pred = predict_bank(model, cache["rows"], inputs)
            predictions[seed] = {(r["id"], r["codec"], r["qp"]): p for r, p in zip(cache["rows"], pred) if p is not None}
            bank_info.setdefault("controllers", {})[str(seed)] = dict(parameters=sum(p.numel() for p in model.parameters()),
                conditions=model.conditions, task_logits=getattr(model, "task_logits", False), predictions=sum(p is not None for p in pred),
                input_call="model(x, maps, saved_condition); condition contains codec/QP/mask/budget and optional domain one-hot")
        replay = replay_bank(cache, predictions, models)
        costs.extend(dict(bank=label, **r) for r in bank_costs(replay))
        for split in ("cal", "all"):
            rows = replay if split == "all" else [r for r in replay if r["split"] == split]
            output.extend(curves_from_rows(rows, label + "/" + split))
        if missing and len(missing) < bank_info["sources"]:
            # A named matched subset provides valid controller comparisons while
            # retaining the full-population static/random baselines separately.
            covered = {r["id"] for r in cache["rows"]} & inputs.keys()
            for split in ("cal", "all"):
                rows = [r for r in replay if r["id"] in covered and (split == "all" or r["split"] == split)]
                output.extend(curves_from_rows(rows, label + "/input_covered_" + split))
        bank_info["distribution"] = action_distribution(replay)
        bank_info["damage_negative_values"] = sum(v is not None and v < 0 for r in cache["rows"] for a in r["damage"] for v in a)
        banks.append(bank_info)
        del inputs, replay, predictions, models, cache
    return dict(baselines=BASELINES, banks=banks, full_bank_curves=output, candidate_costs=costs,
                scope="Exact stored bytes and GT-conditioned task-cost replay. Quality=-cost; BD-quality in cost units. CAL held out from gradient training but used for calibration. ALL includes TRAIN.",
                guard_scope="User-requested cost[a][domain]-cost[0][domain] replay at AR .025, OD/OT .05; these measurement costs include labels/GT and are NOT the label-free source guard costs from held-out selected rows.",
                shortlist_scope="Three nonidentity candidates for requested ablations; use checkpoint-specific published forced-first or fix2 rank-first rule. Saved later selected runs actually use five candidates; held-out saved results keep that policy.")


def bugfix_bank_replay(cache, predictions, models):
    """Apply repaired policies to live predictions and exact saved candidate costs."""
    from adaptive_vcm.v34.evaluation import shortlist_actions, source_guard_choice
    from adaptive_vcm.v34.learning import calibrate, calibrate_domains, relative_damage, select_direct_action, targets
    rows = cache["rows"]
    replay, decisions, calibrations = [], [], {}
    normalization = []
    old_units = relative_damage(rows, cache["settings"].get("damage_floor", [1e-3] * 3))
    new_units = relative_damage(rows)
    for task in sorted({r["domain"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["domain"] == task]
        d = TASKS.index(task)
        cfg = cache["settings"]
        labels = {}
        for name, values in (("before", old_units), ("after", new_units)):
            chosen = [targets(values[i]["rate"], values[i]["damage"], values[i]["mask"], cfg["damage_tolerance"],
                              cfg["budget_ratio"], cfg.get("oracle_damage_weight")) for i in ids]
            labels[name] = dict(Counter(map(str, chosen)))
        normalization.append(dict(task=task, rows=len(ids), before_floor=cfg.get("damage_floor", [1e-3] * 3),
            after_floor=[1., .25, .25], before_absolute_relative_damage=stats([
                abs(a[d]) for i in ids for a in old_units[i]["damage"]]),
            after_absolute_relative_damage=stats([abs(a[d]) for i in ids for a in new_units[i]["damage"]]),
            oracle_action_counts=labels, scope="Supervision diagnostic only; existing checkpoints are not retrained or reinterpreted in new units"))
    def observation(row, arm, action):
        d = TASKS.index(row["domain"])
        size = row["packets"][action]["bytes"]
        return dict(id=row["id"], task=row["domain"], split=row["split"], codec=row["codec"], qp=row["qp"],
                    arm=arm, action=int(action), rate=size, anchor_bytes=row["packets"][0]["bytes"],
                    quality=-float(row["cost"][action][d]), cost_excess=float(row["cost"][action][d] - row["cost"][0][d]))
    replay.extend(observation(r, "anchor", 0) for r in rows)
    for seed, model in models.items():
        cfg = model.metadata["settings"]
        pred = [predictions[seed][(r["id"], r["codec"], r["qp"])] for r in rows]
        # Keep the checkpoint's training units. The new floor applies to future
        # training; silently changing frozen prediction units would invalidate CAL.
        actual = relative_damage(rows, cfg.get("damage_floor", [1e-3] * 3)) if cfg.get("damage_units") == "relative" else rows
        ids = [i for i, r in enumerate(rows) if r["split"] == "cal"]
        if not ids:
            raise ValueError("Mean UCB replay requires held-out CAL rows")
        cal_rows, cal_pred = [actual[i] for i in ids], [pred[i] for i in ids]
        before_cal = model.metadata["calibration"]
        if "domains" in before_cal:
            after_cal = calibrate_domains(cal_rows, cal_pred, cfg["cal_coverage"], mode="mean_ucb")
        else:
            after_cal = calibrate(np.array([p["risk"] for p in cal_pred]), np.array([r["damage"] for r in cal_rows], float),
                                  np.array([p["rate"] for p in cal_pred]), np.array([r["rate"] for r in cal_rows]),
                                  np.array([r["mask"] for r in cal_rows]), cfg["cal_coverage"], mode="mean_ucb")
        calibrations[str(seed)] = dict(before=before_cal, after=after_cal,
            cal_sources=sorted({r["id"] for r in cal_rows}), cal_rows=len(ids),
            units="Checkpoint training units, same saved CAL masks; no TEST residuals")
        for row, out in zip(rows, pred):
            mask, tolerance, budget = row["mask"], cfg["damage_tolerance"], cfg["budget_ratio"]
            before_direct = int(np.argmax(out["logits"]))
            after_direct = select_direct_action(out, mask, tolerance, budget, cfg.get("oracle_damage_weight"), TASKS.index(row["domain"]))
            before_bound = select_action(out["risk"], out["rate"], domain_calibration(before_cal, row["domain"]), mask, tolerance, budget)
            after_bound = select_action(out["risk"], out["rate"], domain_calibration(after_cal, row["domain"]), mask, tolerance, budget)
            size = cfg.get("source_shortlist", 5)
            policy = "rank" if cfg.get("shortlist_seed") == "rank" else "published"
            before_trials = shortlist(out, mask, before_bound, size, policy)
            after_trials = shortlist_actions(out, mask, after_bound, after_direct, size, cfg.get("shortlist_seed", "forced"))
            def guarded(actions, limits):
                rates = [row["packets"][a]["bytes"] / row["packets"][0]["bytes"] for a in actions]
                costs = [row["cost"][a] if rate <= budget else [None] * 3 for a, rate in zip(actions, rates)]
                choice, _ = source_guard_choice(rates, costs, mask, limits, budget)
                return actions[choice]
            def paid(action):
                return action if row["packets"][action]["bytes"] <= row["packets"][0]["bytes"] * budget else 0
            before = dict(B3=before_direct, B5b=paid(before_bound), Full=guarded(before_trials, cfg["source_guard_limits"]))
            after = dict(B3=after_direct, B5b=paid(after_bound), Full=guarded(after_trials, [.025, .05, .05]))
            decisions.append(dict(id=row["id"], task=row["domain"], split=row["split"], codec=row["codec"], qp=row["qp"], seed=seed,
                                  before=before, after=after, before_shortlist=before_trials, after_shortlist=after_trials))
            for stage, choices in (("before", before), ("after", after)):
                replay.extend(observation(row, f"{stage}_{policy}_{seed}", action) for policy, action in choices.items())
    metrics = []
    for task, codec in sorted({(r["task"], r["codec"]) for r in replay}):
        for seed in models:
            for policy in ("B3", "B5b", "Full"):
                value = dict(task=task, codec=codec, seed=seed, policy=policy)
                for stage in ("before", "after"):
                    selected = [r for r in replay if (r["task"], r["codec"], r["arm"]) == (task, codec, f"{stage}_{policy}_{seed}")]
                    value[stage] = dict(rows=len(selected), sources=len({r["id"] for r in selected}),
                        actions=dict(Counter(str(r["action"]) for r in selected)),
                        identity_percent=100 * sum(r["action"] == 0 for r in selected) / len(selected),
                        byte_rate_pct_vs_anchor=100 * (sum(r["rate"] for r in selected) / sum(r["anchor_bytes"] for r in selected) - 1),
                        mean_cost_excess=float(np.mean([r["cost_excess"] for r in selected])),
                        actual_budget_violations=sum(r["rate"] > r["anchor_bytes"] * models[seed].metadata["settings"]["budget_ratio"] for r in selected))
                value["changed_actions"] = sum(r["before"][policy] != r["after"][policy] for r in decisions
                    if (r["task"], r["codec"], r["seed"]) == (task, codec, seed))
                metrics.append(value)
    return dict(normalization=normalization, calibrations=calibrations, comparisons=metrics, decisions=decisions,
                curves=curves_from_rows(replay, "bugfix_replay"))


def replay_heldout_guard(rows, before_limits=(0., .05, 0.), after_limits=(.025, .05, .05)):
    """Replay recorded label-free costs; unobserved task predictions stay missing."""
    from adaptive_vcm.v34.evaluation import source_guard_choice
    lookup = defaultdict(dict)
    for r in rows:
        if r["task"] != "ar":
            continue
        key = (r["id"], r["codec"], r["qp"])
        previous = lookup[key].get(r["action"])
        if previous is not None and (previous["bytes"] != r["bytes"] or previous["correct"] != r["correct"]):
            raise ValueError("Conflicting saved predictions for the same action")
        lookup[key][r["action"]] = r
    decisions = []
    for r in rows:
        if r["task"] != "ar" or r["arm"] not in {f"selected_{s}" for s in SEEDS}:
            continue
        guard = r.get("source_guard")
        if guard is None or list(guard["limits"]) != list(before_limits):
            raise ValueError("Held-out replay requires the recorded pre-fix guard limits")
        actions = r["tried_actions"]
        if not actions or actions[0] != 0 or len(actions) != len(set(actions)) or len(actions) != len(guard["costs"]):
            raise ValueError("Invalid saved source shortlist")
        mask = [v is not None for v in guard["costs"][0]]
        before, _ = source_guard_choice(guard["rate"], guard["costs"], mask, before_limits, 1.01)
        after, _ = source_guard_choice(guard["rate"], guard["costs"], mask, after_limits, 1.01)
        if actions[before] != r["action"]:
            raise ValueError("Historical source guard decision does not replay exactly")
        known = lookup[(r["id"], r["codec"], r["qp"])]
        anchor = known[0]
        selected = known.get(actions[after])
        estimated_bytes = guard["rate"][after] * anchor["bytes"]
        if not math.isclose(estimated_bytes, round(estimated_bytes), abs_tol=1e-6, rel_tol=0):
            raise ValueError("Recorded ratio does not reconstruct integer packet bytes")
        after_bytes = round(estimated_bytes)
        if not math.isclose(guard["rate"][before] * anchor["bytes"], r["bytes"], abs_tol=1e-6, rel_tol=0):
            raise ValueError("Historical source guard bytes do not match the saved arm")
        if selected is not None and selected["bytes"] != after_bytes:
            raise ValueError("Saved candidate bytes disagree with the source guard")
        decisions.append(dict(id=r["id"], codec=r["codec"], qp=r["qp"], arm=r["arm"],
            before_action=r["action"], after_action=actions[after], anchor_bytes=anchor["bytes"],
            before_bytes=r["bytes"], after_bytes=after_bytes, before_correct=r["correct"],
            after_correct=selected["correct"] if selected is not None else None))
    comparisons = []
    for codec, arm in sorted({(r["codec"], r["arm"]) for r in decisions}):
        group = [r for r in decisions if (r["codec"], r["arm"]) == (codec, arm)]
        known = [r for r in group if r["after_correct"] is not None]
        value = dict(codec=codec, arm=arm, rows=len(group), changed_actions=sum(r["before_action"] != r["after_action"] for r in group),
                     after_prediction_rows=len(known), after_prediction_coverage=len(known) / len(group))
        for stage in ("before", "after"):
            value[stage] = dict(identity_percent=100 * sum(r[stage + "_action"] == 0 for r in group) / len(group),
                actions=dict(Counter(str(r[stage + "_action"]) for r in group)),
                byte_rate_pct_vs_anchor=100 * (sum(r[stage + "_bytes"] for r in group) / sum(r["anchor_bytes"] for r in group) - 1))
        names = sorted(group[0]["before_correct"])
        value["same_qp_accuracy_delta_pp"] = {name: float(np.mean([r["after_correct"][name] - r["before_correct"][name] for r in known])) * 100
                                              if len(known) == len(group) else None for name in names}
        value["known_subset_accuracy_delta_pp"] = {name: float(np.mean([r["after_correct"][name] - r["before_correct"][name] for r in known])) * 100
                                                  if known else None for name in names}
        comparisons.append(value)
    return dict(dataset="scaled_test", task="ar", before_limits=list(before_limits), after_limits=list(after_limits),
                historical_decisions_match=True, comparisons=comparisons, decisions=decisions,
                scope="Same recorded five-candidate shortlists and label-free costs; only limits change. Packet bytes replay exactly. "
                      "Accuracy is available only when the chosen action has a saved task prediction; matched-subset accuracy is not full-cohort accuracy.")


def prediction_digest(predictions):
    value = hashlib.sha256()
    for prediction in predictions:
        if prediction is None:
            raise ValueError("Incomplete live controller predictions")
        for key, array in sorted(prediction.items()):
            array = np.ascontiguousarray(array)
            value.update(key.encode())
            value.update(str((array.shape, array.dtype.str)).encode())
            value.update(array.tobytes())
    return value.hexdigest()


def frozen_controller_verification(evidence):
    import torch
    from adaptive_vcm.v34.checkpoint import load_checkpoint
    env = {}
    exec(compile(historical_source("adaptive_vcm/v34/model.py"), "<historical Controller>", "exec"), env)
    results = {}
    pair = torch.linspace(0, 1, 2 * 2 * 3 * 48 * 48).reshape(2, 2, 3, 48, 48)
    maps = torch.full_like(pair, 1 / 3)
    for seed in SEEDS:
        path = ROOT / f"checkpoints/v34_r1/controller_{seed}.pth"
        if not evidence.record(path):
            raise ValueError("Missing frozen V34-R1 checkpoint")
        model = load_checkpoint(path)
        reference = env["Controller"](model.actions, model.conditions).eval()
        reference.load_state_dict(model.state_dict(), strict=True)
        condition = torch.tensor([[0., .7, 1., 0., 1., 1.01, 1., 0., 0.],
                                  [1., .9, 1., 0., 1., 1.01, 0., 0., 1.]])[:, :model.conditions]
        with torch.inference_mode():
            before = {k: v.numpy() for k, v in reference(pair, maps, condition).items()}
            after = {k: v.numpy() for k, v in model(pair, maps, condition).items()}
        same = prediction_digest([before]) == prediction_digest([after])
        if not same:
            raise ValueError("Default Controller changed frozen checkpoint predictions")
        results[str(seed)] = dict(exact_outputs_equal=same, parameters=sum(p.numel() for p in model.parameters()),
                                 task_logits=model.task_logits, checkpoint_sha256=file_hash(path),
                                 output_sha256=prediction_digest([after]))
    return results


def publication_metric_verification(evidence):
    """Separate code compatibility from last-bit differences across numeric libraries."""
    from adaptive_vcm.metrics import curve_summary
    import scipy
    published = evidence.read(ROOT / "results/v34_r1/independent_summary.json")
    manifest = evidence.read(ROOT / "results/v34_r1/publication_manifest.json")
    env = {"__name__": "adaptive_vcm._historical_metrics", "__package__": "adaptive_vcm"}
    exec(compile(historical_source("adaptive_vcm/metrics.py"), "<historical metrics>", "exec"), env)
    checks, differences = [], []
    for codec, seeds in published["comparisons"].items():
        for seed, heads in seeds.items():
            for head, entry in heads.items():
                a, b = entry["anchor"], entry["adaptive"]
                args = (a["bpp"], a["quality"], b["bpp"], b["quality"])
                before, after = env["curve_summary"](*args), curve_summary(*args)
                if before != after:
                    raise ValueError("Repaired metrics changed a frozen monotone publication curve")
                point_differences = {key: after[key] - entry["comparison"][key]
                                     for key in ("bd_rate_pct", "pchip_bd_rate_pct", "bd_quality_pp")}
                differences.extend(abs(v) for v in point_differences.values())
                checks.append(dict(codec=codec, seed=seed, head=head, historical_repaired_exact=True,
                    published_pchip_exact=after["pchip_bd_rate_pct"] == entry["comparison"]["pchip_bd_rate_pct"],
                    published_guards_equal=after["guards"] == entry["comparison"]["guards"],
                    point_differences=point_differences))
    return dict(curve_pairs=len(checks), checks=checks, historical_repaired_exact=True,
        published_pchip_exact=all(c["published_pchip_exact"] for c in checks),
        max_published_point_difference=max(differences, default=0.),
        unchanged_primary_implementations={name: file_hash(ROOT / name) == manifest["v34_r1_historical_runtime_sha256"][name]
            for name in ("adaptive_vcm/legacy_bd.py", "adaptive_vcm/v34/holdout.py", "adaptive_vcm/v33ap/evaluate.py")},
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, platform=sys.platform),
        scope="Historical and repaired implementations run on the same published raw curves. Unchanged cubic fits can differ from the "
              "publication in their final floating-point bits on this runtime. Strict full-merge comparison retains exact equality; verify-only validates original file hashes.")


def task5_bugfix_verification(evidence=None, kaggle=KAGGLE, scratch=SCRATCH):
    """Live deterministic Before/After inference and saved-packet policy replay."""
    import torch
    evidence = Evidence() if evidence is None else evidence
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        frozen = frozen_controller_verification(evidence)
        publication = publication_metric_verification(evidence)
        banks = []
        specs = [("scaled/all", "v34-fix2-ar-od-scaled", "v34-fix2-ar-od-scaled"),
                 ("MOT20-02/all", "v34-ot-measure", "v34-ot-baseline")]
        for label, bank_name, model_name in specs:
            print(f"Verifying Before/After bugfix replay {label}...", flush=True)
            directory = Path(kaggle) / bank_name / "outputs" / bank_name
            model_dir = Path(kaggle) / model_name / "outputs" / model_name
            cache = evidence.read(directory / "measurements.json")
            input_path = directory / "controller_inputs.pth"
            if cache is None or not evidence.record(input_path):
                evidence.gap(label + " bugfix replay", "Saved measurement bank or controller inputs absent")
                continue
            info = validate_bank(cache["rows"])
            inputs = torch.load(input_path, weights_only=True, map_location="cpu")
            if any(r["id"] not in inputs for r in cache["rows"]):
                evidence.gap(label + " bugfix replay", "Incomplete source inputs; all-row replay unavailable")
                continue
            models, predictions, checks = {}, {}, {}
            for seed in SEEDS:
                path = model_dir / f"controller_{seed}.pth"
                if not evidence.record(path):
                    evidence.gap(label + f"/{seed} bugfix replay", "Saved controller absent")
                    continue
                original_source = Path(kaggle) / model_name / "v34_fix1/adaptive_vcm/v34/model.py"
                if not original_source.is_file():
                    original_source = Path(scratch) / "v34_fix1/adaptive_vcm/v34/model.py"
                if not evidence.record(original_source):
                    evidence.gap(label + f"/{seed} bugfix replay", "Historical task-logit model source absent")
                    continue
                before_model = load_controller(path, evidence, original_source)
                model = load_controller(path, evidence)
                before_pred = predict_bank(before_model, cache["rows"], inputs)
                pred = predict_bank(model, cache["rows"], inputs)
                before_hash, after_hash = prediction_digest(before_pred), prediction_digest(pred)
                if before_hash != after_hash:
                    raise ValueError("Native Controller differs from the saved checkpoint architecture: " + label)
                checks[str(seed)] = dict(rows=len(pred), before_prediction_sha256=before_hash, after_prediction_sha256=after_hash,
                                         exact_outputs_equal=True, checkpoint_sha256=file_hash(path), historical_model_source=str(original_source))
                models[seed] = model
                predictions[seed] = {(r["id"], r["codec"], r["qp"]): p for r, p in zip(cache["rows"], pred)}
            if set(models) != set(SEEDS):
                continue
            result = bugfix_bank_replay(cache, predictions, models)
            for curve in result["curves"]:
                curve["dataset"] = label
                for comparison in curve["comparisons"]:
                    comparison["dataset"] = label
            banks.append(dict(name=label, **info, controllers=checks, **result))
        path = Path(kaggle) / "v34-fix2-ar-od-scaled/outputs/v34-fix2-ar-od-scaled/test_rows.json"
        rows = evidence.read(path)
        heldout = replay_heldout_guard(rows) if rows is not None else evidence.gap("scaled_test bugfix replay", "Saved TEST rows absent")
        return dict(status="verified" if len(banks) == 2 and heldout.get("decisions") else "incomplete",
            frozen_checkpoint_compatibility=frozen, publication_metric_compatibility=publication, banks=banks, heldout_source_guard=heldout,
            identity_only_after=[dict(bank=b["name"], task=r["task"], codec=r["codec"], seed=r["seed"], policy=r["policy"])
                                 for b in banks for r in b["comparisons"] if r["after"]["identity_percent"] == 100],
            deterministic=dict(torch_threads=2, deterministic_algorithms=True, new_encodes=0, new_task_model_inferences=0,
                               parameter_updates=0, live_controller_inference=True),
            scope="Before uses saved logits, quantile bounds and original shortlists/limits. After gates task logits with predicted "
                  "risk/rate, recalibrates clipped mean UCB on CAL only, and uses repaired rank seeding and [.025,.05,.05] limits. "
                  "Bank quality is negative GT-conditioned cost; ALL includes TRAIN and is a policy diagnostic, not held-out accuracy. "
                  "Relative-floor diagnostics affect future supervision; frozen predictions and calibration keep their original units.")
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def render_report(audit):
    lines = ["# V34 verification and experimental audit", "",
             "Publication data, checkpoints, and current runtime hashes verified before and after the audit. Historical source is read from Git.", "",
             "This is a saved-artifact audit and deterministic replay. Missing measurements remain unavailable. "
             "The requested findings are hypotheses checked against evidence, not assumptions.", "",
             "## Scope corrections", "",
             "- Historical learning.py uses scaled Smooth L1 (MSE without scale); later checkpoints use pinball. The repaired workspace supports both losses and preserves the historical default. The supplied sources do not establish the claimed 1.5 × rank loss.",
             "- Action 11 is global filtering. Saved action 5 differs from the coefficients in the request; replay uses the actual saved action.",
             "- Full-bank quality is negative GT-conditioned task cost, not accuracy, mAP, or official tracking metrics. Its cost-delta guard is a supervised surrogate.",
             "- Later selected runs use five nonidentity candidates. Task 2 bank ablations use three; Task 5 replays five. The native Controller supports both the frozen 65-value head and the later 91-value task-logit head.",
             "- No fifth QP is invented. Each five-QP result is unavailable unless all five points exist.",
             "- MOT17 input lists 02/05 FIT and 04/11 reused CAL. It does not establish the requested held-out 02/05/09/11, 120-window benchmark.",
             "- MOT20-05 LaTeX has rounded HOTA/rates and only IDF1/MOTA deltas. Exact MOTA divergence cannot be reconstructed from it.", "",
             "## Task 1: seven findings", ""]
    for f in audit["task1_audit"]["findings"]:
        lines.extend([f"### {f['id']}: {f['status']}", "", f.get("result", f.get("interpretation", "Structured evidence is retained in audit_summary.json.")), ""])
        if "fixed_in_workspace" in f:
            lines.extend([f"Historical finding retained; `fixed_in_workspace: {str(f['fixed_in_workspace']).lower()}`.", ""])
        if "original_probe" in f:
            lines.extend([f"Original dummy OD call: `{f['original_probe']}`. Fixed call: `{f['fixed_probe']}`.", ""])
        if "max_absolute_pixel_difference" in f:
            lines.extend([f"Maximum pixel differences for different masks: `{f['max_absolute_pixel_difference']}`.", ""])
        if "tests" in f:
            lines.extend([f"{f['version']}. Legacy return code {f['tests']['legacy_vsync']['returncode']}; replacement return code {f['tests']['fps_mode']['returncode']}.", ""])
    lines.extend(["### Action counts and shortlist collapse", "", "| Dataset/task/arm | Rows | Identity % | Action counts | Most frequent shortlist (count) |",
                  "|---|---:|---:|---|---|"])
    for dataset, values in audit["task1_audit"]["distributions"].items():
        for key, d in values.items():
            if not any(t in key for t in ("forced_303", "selected_303", "calibrated_303")):
                continue
            top = d["shortlists"][0] if d["shortlists"] else None
            lines.append(f"| {dataset}/{key} | {d['total']} | {d['identity_percent']:.4f} | "
                         + ", ".join(f"{a}: {v['count']}" for a, v in d["actions"].items())
                         + f" | {top['actions'] if top else 'N/A'} ({top['count'] if top else 0}) |")
    lines.extend(["", "## Task 2: ablation definitions and measurement-bank results", "",
                  audit["task2_ablation"]["scope"], "", audit["task2_ablation"]["guard_scope"], "",
                  "| Baseline | Definition |", "|---|---|"])
    lines.extend(f"| {k} | {v} |" for k, v in BASELINES.items())
    lines.extend(["", "| Bank | Unique sources | Conditions by task/split | Missing controller inputs |", "|---|---:|---|---:|"])
    for bank in audit["task2_ablation"]["banks"]:
        lines.append(f"| {bank['name']} | {bank['sources']} | {bank['conditions_by_domain_split']} | {len(bank['missing_controller_input_ids'])} |")
    lines.extend(["", "Seed means average the QP curves first. The JSON also records the mean of independently computed seed BD-rates. "
                  "B4 averages 20 deterministic random-policy curves; individual RNG-seed results remain in JSON.", ""])
    def curve_table(sets):
        lines.extend(["| Dataset | Task/codec/metric | Arm | QPs | Cubic rate % | PCHIP rate % | Cubic quality | PCHIP quality | Reliable |",
                      "|---|---|---|---:|---:|---:|---:|---:|---|"])
        for c in sets:
            for r in c["comparisons"]:
                if r["arm"].startswith("random_"):
                    continue
                lines.append(f"| {r['dataset']} | {r['task']}/{r['codec']}/{r['metric']} | {r['arm']} | {r['grid']} | "
                    + " | ".join(fmt(r.get(k)) for k in ("cubic_bd_rate_pct", "pchip_bd_rate_pct", "cubic_bd_quality", "pchip_bd_quality"))
                    + f" | {r.get('reliable', False)} |")
    curve_table(audit["task2_ablation"]["full_bank_curves"])
    lines.extend(["", "## Held-out curves and Task 4: cubic/PCHIP sensitivity", "",
                  "AR curves are recomputed from paired correctness. COCO and official TrackEval quality uses exact saved evaluator curves; "
                  "COCO mean bytes are cross-checked against rows. These are not newly executed task-model/TrackEval runs.", "",
                  "Cubic fits a degree-3 polynomial (least squares for five points); PCHIP uses local shape-preserving interpolation. "
                  "Both integrate only on common support. PCHIP projects quality onto the nondecreasing Pareto envelope sorted by bitrate; plateaus use minimum rate. "
                  "PCHIP BD-quality uses mean quality at duplicate rates. Quality differences are percentage points for accuracy/TrackEval and cost units for bank proxies.", "",
                  "Original nonmonotone rate-quality curves remain flagged unreliable; their PCHIP envelope and raw same-QP results are retained. "
                  "Sharp four-point curvature and single active QPs can "
                  "make cubic overshoot; this alone does not prove classical Runge oscillation.", ""])
    lines.extend(["A cubic/PCHIP difference above 5 percentage points is additionally flagged as interpolation-sensitive. "
                  "This is a disclosed diagnostic threshold, not a confidence interval or statistical test.", ""])
    curve_table(audit["task4_bdrate_and_ot"]["curves"])
    lines.extend(["", "### MOT17 requested FIT numbers checked against this artifact", "",
                  "| Codec | Requested BD-rate % | Recomputed cubic % | Recomputed PCHIP % | Reliable |", "|---|---:|---:|---:|---|"])
    for row in audit["task4_bdrate_and_ot"]["mot17"].get("full_lambda_32_claim_check", []):
        lines.append(f"| {row['codec']} | {row['requested_bd_rate_pct']} | {fmt(row['cubic_bd_rate_pct'])} | {fmt(row['pchip_bd_rate_pct'])} | {row['reliable']} |")
    lines.extend(["", "These input curves do not reproduce the requested FIT numbers. Particularly large cubic values reflect "
                  "an unstable quality-to-log-rate polynomial; they are not evidence of a comparable bitrate gain.", ""])
    lines.extend(["", "### Largest interpolation discrepancies", "", "| Dataset/codec/metric/arm | QPs | Cubic % | PCHIP % | Difference pp | Active QPs |",
                  "|---|---:|---:|---:|---:|---:|"])
    for r in audit["task4_bdrate_and_ot"]["largest_divergences"][:25]:
        lines.append(f"| {r['dataset']}/{r['codec']}/{r['metric']}/{r['arm']} | {r['grid']} | {fmt(r['cubic_bd_rate_pct'])} | "
                     f"{fmt(r['pchip_bd_rate_pct'])} | {fmt(r['cubic_minus_pchip_pp'])} | {r['active_qp_count']} |")
    lines.extend(["", "### MOT20 macro and descriptive pooling", "",
                  "Official combined TrackEval cannot be obtained by averaging HOTA/IDF1. The following frame-weighted curves are descriptive only, "
                  "include rounded MOT20-05, and compare different checkpoint versions. Four-sequence official pooling is unavailable.", "",
                  "| Codec/metric/arm | Macro cubic % | Macro PCHIP % | Descriptive pooled cubic % | Descriptive pooled PCHIP % |", "|---|---:|---:|---:|---:|"])
    for r in audit["task4_bdrate_and_ot"]["mot20_combined"]:
        lines.append(f"| {r['codec']}/{r['metric']}/{r['arm']} | {fmt(r['macro_bd']['cubic_bd_rate_pct']['mean'])} | "
                     f"{fmt(r['macro_bd']['pchip_bd_rate_pct']['mean'])} | {fmt(r['descriptive_pooled_bd']['cubic_bd_rate_pct'])} | "
                     f"{fmt(r['descriptive_pooled_bd']['pchip_bd_rate_pct'])} |")
    lines.extend(["", "## Task 3: measured computational costs", "",
                  "All stage timings below are seconds (mean / p50 / p90), grouped by task, codec, and arm; JSON retains every field and arm. "
                  "End-to-end adds final receiver decoding and task evaluation to sender time. FPS is frames divided by sender time. "
                  "The baseline denominator is the standard identity encoder, not unnecessary source-map/guard work.", "",
                  "| Dataset/task/codec/arm | Field | Mean | p50 | p90 | N |", "|---|---|---:|---:|---:|---:|"])
    for entry in audit["task3_cost_analysis"]["timings"]:
        if entry["arm"] not in ("anchor", "fixed", "forced_303401", "calibrated_303401", "selected_303401"):
            continue
        for key, value in entry["fields"].items():
            lines.append(f"| {entry['dataset']}/{entry['task']}/{entry['codec']}/{entry['arm']} | {key} | "
                         f"{fmt(value['mean'])} | {fmt(value['p50'])} | {fmt(value['p90'])} | {value['n']} |")
        for key in ("aggregate_sender_fps", "sender_slowdown_vs_b0"):
            lines.append(f"| {entry['dataset']}/{entry['task']}/{entry['codec']}/{entry['arm']} | {key} | {fmt(entry[key])} | - | - | - |")
    lines.extend(["", "### Parameters, weights, and system memory", "", "| Model | Parameters | FP32 parameter MB |", "|---|---:|---:|"])
    for name, value in audit["task3_cost_analysis"]["memory"]["models"].items():
        lines.append(f"| {name} | {value['parameters']} | {value['fp32_MB']:.4f} |")
    lines.extend(["", "| CSV task/codec | Peak allocated GPU MB | Trial encodes | Sender seconds |", "|---|---:|---:|---:|"])
    for r in audit["task3_cost_analysis"]["memory"]["full_system_csv"]:
        lines.append(f"| {r['task']}/{r['codec']} | {r['peak_gpu_MB']:.4f} | {r['mean_distinct_trial_encodes']} | {r['mean_sender_selection_seconds']:.4f} |")
    lines.extend(["", audit["task3_cost_analysis"]["memory"]["csv_crosscheck"], "",
                  "CPU RSS is not recorded in these experiment artifacts. Model FP32 weight bytes are not full-system VRAM. "
                  "Bank timing subtotals omit unrecorded support/guard/CNN/receiver time; full ablation latency/FPS cannot be claimed from those subtotals.", "",
                  "### Ablation candidate-work comparison", "", "| Bank/task/codec/arm | Encodes | Probe decodes | Guard candidates | Candidate subtotal seconds |", "|---|---:|---:|---:|---:|"])
    for r in audit["task2_ablation"]["candidate_costs"]:
        if r["arm"].startswith("random_") and r["arm"] != "random_0":
            continue
        lines.append(f"| {r['bank']}/{r['task']}/{r['codec']}/{r['arm']} | {fmt(r['encode_count']['mean'])} | "
                     f"{fmt(r['decode_count']['mean'])} | {fmt(r['guard_candidate_count']['mean'])} | {fmt(r['candidate_work_seconds']['mean'])} |")
    verification = audit["task5_bugfix_verification"]
    lines.extend(["", "## 6. Bugfix verification: live Before vs After replay", "", verification["scope"], "",
                  f"Status: **{verification['status']}**. Torch uses two CPU threads and deterministic algorithms. "
                  "The three frozen V34-R1 checkpoints produce exactly the same outputs under the historical and repaired default Controller. "
                  "Saved task-logit checkpoints are also checked against their original architecture on every bank row.", "",
                  "Rates below are total selected packet bytes relative to total identity bytes. Cost excess is the mean raw primary-task cost difference. "
                  "B3 uses predicted feasibility only; actual budget violations are retained in JSON. B5b applies the measured-rate fallback; Full uses measured costs.", "",
                  "| Bank/task/codec | Seed/policy | Rows | Before rate % | After rate % | Before identity % | After identity % | Before cost excess | After cost excess | Changed actions |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for bank in verification["banks"]:
        for r in bank["comparisons"]:
            b, a = r["before"], r["after"]
            lines.append(f"| {bank['name']}/{r['task']}/{r['codec']} | {r['seed']}/{r['policy']} | {b['rows']} | "
                         f"{fmt(b['byte_rate_pct_vs_anchor'])} | {fmt(a['byte_rate_pct_vs_anchor'])} | "
                         f"{fmt(b['identity_percent'])} | {fmt(a['identity_percent'])} | "
                         f"{fmt(b['mean_cost_excess'])} | {fmt(a['mean_cost_excess'])} | {r['changed_actions']} |")
    heldout = verification["heldout_source_guard"]
    lines.extend(["", "### scaled_test AR: source-guard limit repair", "", heldout.get("scope", heldout.get("reason", "Unavailable")), "",
                  "Limits change from `[0, 0.05, 0]` to `[0.025, 0.05, 0.05]`. Every historical decision is checked against its saved action and paid bytes.", "",
                  "| Codec/arm | Rows | Before rate % | After rate % | Before identity % | After identity % | Changed actions | After accuracy coverage % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in heldout.get("comparisons", []):
        b, a = r["before"], r["after"]
        lines.append(f"| {r['codec']}/{r['arm']} | {r['rows']} | {fmt(b['byte_rate_pct_vs_anchor'])} | {fmt(a['byte_rate_pct_vs_anchor'])} | "
                     f"{fmt(b['identity_percent'])} | {fmt(a['identity_percent'])} | {r['changed_actions']} | {fmt(100*r['after_prediction_coverage'])} |")
    lines.extend(["", "Unobserved selected actions have no imputed accuracy. JSON retains matched-subset accuracy, coverage, all Before/After decisions, "
                  "CAL bounds, normalization diagnostics, and exact prediction hashes. Mean UCB is a bound on clipped average residuals, not an individual-risk coverage guarantee.", ""])
    if verification.get("identity_only_after"):
        lines.extend(["Some frozen risk heads still reject every nonidentity action under B3 or B5b. "
                      "The report records those identity-only results; applying the revised losses and normalization to new weights requires retraining.", ""])
    publication = verification["publication_metric_compatibility"]
    lines.extend([f"The historical and repaired metric implementations agree exactly on all {publication['curve_pairs']} published curve pairs. "
                  f"Published PCHIP values match exactly: `{str(publication['published_pchip_exact']).lower()}`. "
                  f"Maximum point-metric difference from the published snapshot on this runtime: `{publication['max_published_point_difference']:.3g}`. "
                  "The cubic and bootstrap source files are unchanged. The strict full-merge equality check is retained; `--verify-only` validates the original artifacts by hash.", ""])
    lines.extend(["", "## Unavailable evidence and limitations", ""])
    for gap in audit["limitations"]:
        lines.append(f"- **{gap['scope']}**: {gap['reason']}")
    lines.extend(["", "All input paths, SHA-256 hashes, complete curves, per-seed metrics, code excerpts with line numbers, "
                  "and missing-data evidence are in `audit_summary.json`. Re-run `python scripts/verify_and_ablate_v34.py` to regenerate both reports.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", type=Path, default=KAGGLE)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--paper", type=Path, default=PAPER)
    parser.add_argument("--out", type=Path, default=ROOT / "results/verification_audit")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(303401)
    torch.use_deterministic_algorithms(True)
    print("Verifying publication and reading saved experiments...", flush=True)
    before = protected_snapshot()
    out = validate_output_directory(args.out, before)
    evidence = Evidence()
    evidence.record(Path(__file__))
    distributions, calibrations, saved_curves, timing, coverage = {}, {}, [], [], {}
    specs = [
        ("k400_publication", [ROOT / f"results/v34_r1/shards/shard{i}/test_rows.json.gz" for i in range(4)], None, ROOT / "checkpoints/v34_r1"),
        ("coco_historical", [ROOT / "results/v34_od_coco/test_rows.json.gz"], ROOT / "results/v34_od_coco/test_summary.json", ROOT / "checkpoints/v34_od_coco"),
    ]
    for label, name, split in (("scaled_test", "v34-fix2-ar-od-scaled", "test"),
                               ("coco_rank", "v34-fix2-od-rank", "test"),
                               ("ot_test", "v34-ot-baseline", "test"),
                               ("ot_dev", "v34-ot-baseline", "dev")):
        p = args.kaggle / name / "outputs" / name
        specs.append((label, [p / f"{split}_rows.json"], p / f"{split}_summary.json", p))
    for label, paths, summary_path, checkpoint_dir in specs:
        print(f"Auditing saved rows {label}...", flush=True)
        rows = []
        for path in paths:
            value = evidence.read(path)
            if value is not None:
                # COCO predictions are large; saved COCOeval curves provide joint AP.
                for r in value:
                    r.pop("coco", None)
                rows.extend(value)
        if not rows:
            continue
        distributions[label] = action_distribution(rows)
        calibrations[label] = calibration_evidence(rows, checkpoint_dir, evidence)
        summary = evidence.read(summary_path) if summary_path else None
        saved_curves.extend(summarize_saved(rows, summary, label, evidence))
        if label in ("scaled_test", "ot_test", "ot_dev"):
            timing.extend(cost_analysis(rows, label))
        coverage[label] = heldout_replay_coverage(rows)
        evidence.gap(label + " held-out B2/B4/B5a", "Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed")
        if any(r["task"] == "ot" for r in rows):
            evidence.gap(label + " proxy tracking accuracy", "Rows contain AR source KL, not a GT tracking-accuracy observation; use official sequence TrackEval metrics")
        del rows
    first = task1_audit(evidence, args.scratch, args.paper, distributions, calibrations)
    second = task2_ablation(evidence, args.kaggle, args.scratch)
    fifth = task5_bugfix_verification(evidence, args.kaggle, args.scratch)
    memory = model_costs(evidence, args.scratch)
    for name in ("ot-test", "ot-dev-A1", "ot-dev-A2"):
        summary = evidence.read(args.scratch / f"munnn01_test/results/v34r1/{name}/summary.json")
        if summary:
            saved_curves.extend(tracking_summary_curves(summary, "ByteTrack_" + name))
            if summary.get("grid_audit", {}).get("source_count") != 16 and name != "ot-test":
                evidence.gap("ByteTrack_" + name + " 256-frame comparison", f"Artifact has {summary['grid_audit']['source_count']} windows, not 16; keep this run separate")
    paper_curves, paper_checks = paper_tracking(evidence, args.paper)
    saved_curves.extend(paper_curves)
    mot17 = mot17_analysis(evidence, args.scratch)
    saved_curves.extend(mot17.pop("curves"))
    pooled = combine_mot20(saved_curves, evidence)
    all_comparisons = [r for c in saved_curves for r in c["comparisons"]]
    divergences = sorted([r for r in all_comparisons if r.get("cubic_minus_pchip_pp") is not None],
                         key=lambda r: -abs(r["cubic_minus_pchip_pp"]))
    audit = dict(schema="v34-verification-audit-1", task1_audit=first, task2_ablation=second,
                 task3_cost_analysis=dict(timings=timing, memory=memory, crosschecks=timing_crosschecks(timing, memory)),
                 task4_bdrate_and_ot=dict(curves=saved_curves, largest_divergences=divergences,
                     mot17=mot17, mot20_combined=pooled, paper_tracking_crosscheck=paper_checks),
                 task5_bugfix_verification=fifth,
                 heldout_replay_coverage=coverage, limitations=evidence.gaps,
                 provenance=evidence.files,
                 reproducibility=dict(rng_seeds=list(range(20)), torch_threads=2, controller_batch_size=16,
                                      new_encodes=0, new_task_model_inferences=0, saved_controller_inference="CPU deterministic"))
    after = protected_snapshot()
    if before != after:
        raise RuntimeError("Publication asset changed during audit; refusing to publish report")
    audit["publication_integrity"] = dict(unchanged=True, protected_file_count=len(before), before=before, after=after)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "AUDIT_REPORT.md").write_text(render_report(audit), encoding="utf-8")
    print(f"Audit complete: {len(all_comparisons)} held-out/tracking BD comparisons; {len(evidence.gaps)} evidence gaps; publication unchanged. Reports: {out}", flush=True)
    return audit


if __name__ == "__main__":
    main()
