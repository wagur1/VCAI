"""CPU guard-distillation pilot; saved banks are read-only.

Example: python scripts/distill_guard_v34.py --quick --out output/distill_smoke
K counts extra nonidentity codec/guard trials, not the final output encoding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("v34_distill_audit", ROOT / "scripts/verify_and_ablate_v34.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
choose_measured, shortlist = audit.choose_measured, audit.shortlist
validate_bank, load_controller, predict_bank = audit.validate_bank, audit.load_controller, audit.predict_bank
replay_bank, curves_from_rows = audit.replay_bank, audit.curves_from_rows
evaluate_curve_set, compare_curves = audit.evaluate_curve_set, audit.compare_curves
TASKS, SEEDS = audit.TASKS, audit.SEEDS
from adaptive_vcm.v34.model import Controller

KAGGLE = audit.KAGGLE
BANKS = (("ar_od", "v34-fix2-ar-od-scaled", "v34-fix2-ar-od-scaled"),
         ("ot", "v34-ot-measure", "v34-ot-baseline"))
STUDENTS = ("distill_k0_argmax", "distill_k0_feasible", "distill_k1", "distill_k2",
            "distill_k3", "gbm_k0", "gbm_k1", "gbm_k3")
CI_PREFIXES = ("distill_k0_argmax", "distill_k0_feasible", "distill_k1", "selected")
GRID = tuple(audit.GRIDS[0])
TIMINGS = ("preprocess_seconds", "encode_seconds", "decode_seconds")
CAVEATS = [
    "Quality is negative task cost, a surrogate, NOT Top-1, mAP, or HOTA.",
    "The bank contains train/cal only, not the Kaggle test split.",
    "OT contains windows from a single sequence, MOT20-02; source-id folds do not establish sequence generalization or exclude temporal dependence.",
    "Teacher costs include labels/GT (AR includes label CE). Student fitting, early stopping and tau tuning use only outer-training sources. Evaluation costs are used only for post-hoc metrics and explicitly guarded replay arms.",
    "The replay guard is the requested measured task-cost surrogate, not a new execution of the label-free frozen task model.",
    "P1 CAL is held out from existing-controller gradient training, but was used for their calibration. P2 existing-controller numbers include in-sample TRAIN rows.",
    "K counts nonidentity probe trials. Guarded policies also encode identity; direct K=0 policies still encode their chosen output once. Calibrated replay includes its saved measured-rate fallback.",
    "Sender estimate is saved preprocess+encode+decode work summed over tried actions, as in replay_bank. Guard/task-model, support, controller inference and receiver timings are unavailable; guard cost scales with trials.",
    "Seed means average rate/quality curves (not a voting ensemble or the mean of BD rates). Individual seed BD rates and their arithmetic mean are also reported.",
    "Recovery columns are the literal signed BD_student/BD_reference ratios (undefined for a zero reference). A ratio only measures recovered gain when the reference BD-rate is negative; GO additionally requires a negative selected BD-rate.",
    "fixed is action 11; forced is B3; calibrated is B5; selected is Full with the imported replay's three-candidate shortlist. random3+guard averages RNG seeds 0 through 19.",
    "Bootstrap resamples source ids jointly across QPs, arms and seeds, without refitting students. P2 intervals are conditional on the fitted out-of-fold predictions.",
    "The primary BD grid is QP 30/35/40/45. Both audit grids and all interpolation reliability flags are retained in summary.json.",
    "S2 features are six channel means and six population standard deviations across time/space of concatenated x/maps, plus six conditions (18 total). The requested 12 means+12 stds is inconsistent with two three-channel tensors.",
    "Students use the first six saved conditions. Existing controllers receive the original conditions, including the OT bank's appended domain one-hot.",
]


def key(row):
    return row["id"], row["codec"], row["qp"]


def threshold(row):
    return .025 if row["domain"] == "ar" else .05


def teacher_labels(row):
    """Called for fitting only on training folds; evaluation labels are post-hoc."""
    d = TASKS.index(row["domain"])
    costs = np.asarray(row["cost"], dtype=float)[:, d]
    feasible = (np.asarray(row["rate"]) <= 1.01) & (costs - costs[0] <= threshold(row))
    return choose_measured(row, list(range(13)), threshold(row)), feasible


def trial_candidates(logits, k):
    if k not in (1, 2, 3):
        raise ValueError("K must be 1, 2 or 3")
    logits = np.asarray(logits)
    if logits.shape != (13,) or not np.isfinite(logits).all():
        raise ValueError("Expected 13 finite action logits")
    order = np.argsort(-logits, kind="stable")
    return [0, *[int(a) for a in order if a != 0][:k]]


def student_decision(prediction, k=0, row=None, tau=None, domain=None):
    """K=0 never accesses a measurement row; K>0 explicitly executes its guard."""
    if k:
        candidates = trial_candidates(prediction["logits"], k)
        return choose_measured(row, candidates, threshold(row)), candidates
    if tau is None:
        action = int(np.argmax(prediction["logits"]))
    else:
        risk = np.asarray(prediction["risk"], dtype=float)[:, TASKS.index(domain)]
        probability = 1. / (1. + np.exp(-np.clip(risk, -700, 700)))
        feasible = np.flatnonzero(probability >= tau)
        action = int(feasible[np.argmin(np.asarray(prediction["rate"])[feasible])]) if len(feasible) else 0
    return action, [action]


def source_domains(rows):
    domains = {}
    for r in rows:
        if r["id"] in domains and domains[r["id"]] != r["domain"]:
            raise ValueError("A source id occurs in multiple domains")
        domains[r["id"]] = r["domain"]
    return domains


def grouped_folds(rows, n_splits=5):
    """Domain-stratified GroupKFold when each domain has at least five sources."""
    domains = source_domains(rows)
    if len(domains) < n_splits:
        raise ValueError(f"Need at least {n_splits} source ids for GroupKFold")
    counts = Counter(domains.values())
    stratified = min(counts.values()) >= n_splits
    try:
        from sklearn.model_selection import GroupKFold
    except ImportError:
        GroupKFold = None
    assignments = {}
    strata = sorted(counts) if stratified else [None]
    for domain in strata:
        indices = [i for i, r in enumerate(rows) if domain is None or r["domain"] == domain]
        groups = np.asarray([rows[i]["id"] for i in indices])
        if GroupKFold is not None:
            for fold, (_, val) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(indices)), groups=groups)):
                for i in val:
                    assignments[str(groups[i])] = fold
        else:
            # Same greedy group-size balancing objective, usable without sklearn.
            sizes = Counter(groups.tolist())
            loads = [0] * n_splits
            for source in sorted(sizes, key=lambda s: (-sizes[s], s)):
                fold = int(np.argmin(loads))
                assignments[source] = fold
                loads[fold] += sizes[source]
    folds = [(np.asarray([i for i, r in enumerate(rows) if assignments[r["id"]] != f]),
              np.asarray([i for i, r in enumerate(rows) if assignments[r["id"]] == f])) for f in range(n_splits)]
    return folds, dict(method="GroupKFold" if GroupKFold else "greedy grouped fallback (sklearn missing)",
                       domain_stratified=stratified, source_to_fold=assignments)


def inner_split(rows, seed):
    domains = source_domains(rows)
    sources = sorted(domains)
    if len(sources) < 2:
        raise ValueError("Need two outer-training sources for inner validation")
    rng = np.random.default_rng(seed)
    target = min(len(sources) - 1, max(1, math.ceil(.2 * len(sources))))
    # Proportional allocation keeps the validation size exactly ceil(20% sources).
    by_domain = {d: [s for s in sources if domains[s] == d] for d in sorted(set(domains.values()))}
    allocation = {d: int(target * len(v) / len(sources)) for d, v in by_domain.items()}
    for d in sorted(by_domain, key=lambda d: (-(target * len(by_domain[d]) / len(sources) - allocation[d]), d)):
        if sum(allocation.values()) < target:
            allocation[d] += 1
    validation = set()
    for d, ids in by_domain.items():
        validation.update(rng.permutation(ids)[:allocation[d]].tolist())
    return [r for r in rows if r["id"] not in validation], [r for r in rows if r["id"] in validation]


def inference_rows(rows):
    # Deliberately remove costs, rates and split labels from the inference interface.
    return [dict(id=r["id"], condition=r["condition"][:6]) for r in rows]


def batches(rows, inputs, rng=None, size=32):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(inputs[row["id"]][0].shape)].append(i)
    result = []
    for ids in groups.values():
        ids = np.asarray(ids)
        if rng is not None:
            ids = rng.permutation(ids)
        result.extend(ids[i:i + size] for i in range(0, len(ids), size))
    if rng is not None:
        rng.shuffle(result)
    return result


def tensors(rows, inputs, ids):
    group = [rows[int(i)] for i in ids]
    return (torch.stack([inputs[r["id"]][0] for r in group]),
            torch.stack([inputs[r["id"]][1] for r in group]),
            torch.tensor([r["condition"][:6] for r in group], dtype=torch.float32))


def training_targets(rows):
    labels = [teacher_labels(r) for r in rows]
    return (torch.tensor([v[0] for v in labels], dtype=torch.long),
            torch.tensor(np.stack([v[1] for v in labels]), dtype=torch.float32),
            torch.tensor([r["rate"] for r in rows], dtype=torch.float32).log(),
            torch.tensor([TASKS.index(r["domain"]) for r in rows], dtype=torch.long))


def distillation_loss(output, targets, ids):
    teacher, feasible, log_rate, domain = [v[ids] for v in targets]
    risk = output["risk"][torch.arange(len(ids)), :, domain]
    return (F.cross_entropy(output["logits"], teacher) + F.binary_cross_entropy_with_logits(risk, feasible)
            + .5 * F.smooth_l1_loss(output["rate"].log(), log_rate))


def fit_distill(train, validation, inputs, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Controller(13, conditions=6, task_logits=True).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_targets, val_targets = training_targets(train), training_targets(validation)
    rng = np.random.default_rng(seed)
    best_loss, best_state, best_epoch = math.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for ids in batches(train, inputs, rng):
            optimizer.zero_grad(set_to_none=True)
            loss = distillation_loss(model(*tensors(train, inputs, ids)), train_targets, ids)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite student training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        model.eval()
        val_total = 0.
        with torch.inference_mode():
            for ids in batches(validation, inputs):
                val_total += float(distillation_loss(model(*tensors(validation, inputs, ids)), val_targets, ids)) * len(ids)
        val_loss = val_total / len(validation)
        history.append(dict(epoch=epoch, train_loss=total / len(train), validation_loss=val_loss))
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 10:
            break
    if best_state is None:
        raise ValueError("No finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_epoch=best_epoch, epochs_run=len(history), history=history,
                       parameters=sum(p.numel() for p in model.parameters()))


def tune_tau(rows, predictions):
    """Exact threshold sweep on inner-validation data, never outer evaluation."""
    risks = np.stack([p["risk"][:, TASKS.index(r["domain"])] for r, p in zip(rows, predictions)])
    probabilities = 1. / (1. + np.exp(-np.clip(risks.astype(float), -700, 700)))
    predicted_rate = np.stack([p["rate"] for p in predictions])
    actual_rate = np.asarray([r["rate"] for r in rows])
    feasible = np.stack([teacher_labels(r)[1] for r in rows])
    # A feasible set changes immediately above an observed probability (>= rule).
    thresholds = np.unique(np.r_[0., np.nextafter(probabilities.ravel(), np.inf), 1.])
    best = None
    for tau in thresholds:
        eligible = probabilities >= tau
        actions = np.where(eligible, predicted_rate, np.inf).argmin(axis=1)
        actions[~eligible.any(axis=1)] = 0
        chosen = np.arange(len(rows)), actions
        pass_rate = float(feasible[chosen].mean())
        gain = float((1. - actual_rate[chosen]).mean())
        if pass_rate >= .9:
            candidate = (gain, pass_rate, float(tau))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise ValueError("Identity did not supply a feasible validation fallback")
    return dict(tau=best[2], mean_rate_saving=best[0], feasibility_rate=best[1],
                candidate_thresholds=len(thresholds), validation_rows=len(rows))


def feature_matrix(rows, inputs):
    cached = {}
    for r in rows:
        if r["id"] not in cached:
            six = torch.cat(inputs[r["id"]], dim=1).double()
            cached[r["id"]] = torch.cat((six.mean((0, 2, 3)), six.std((0, 2, 3), correction=0))).numpy()
    return np.stack([np.r_[cached[r["id"]], r["condition"][:6]] for r in rows])


def fit_gbm(train, evaluation, inputs, seed):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        estimator = HistGradientBoostingClassifier(random_state=seed, early_stopping=False)
        backend = "HistGradientBoostingClassifier"
    except ImportError:
        try:
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(random_state=seed, max_iter=1000)
            backend = "LogisticRegression fallback"
        except ImportError:
            return None, dict(status="missing", reason="sklearn unavailable; S2 skipped")
    labels = np.asarray([teacher_labels(r)[0] for r in train])
    classes = np.unique(labels)
    x_train, x_eval = feature_matrix(train, inputs), feature_matrix(evaluation, inputs)
    if len(classes) == 1:
        # LogisticRegression rejects a single class; the empirical model is constant.
        probability = np.ones((len(evaluation), 1))
        backend += " (constant single-class training labels)"
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            estimator.fit(x_train, labels)
            probability = estimator.predict_proba(x_eval)
        classes = estimator.classes_
    # Missing classes have zero mass and rank after observed classes. Stable ties by action.
    logits = np.full((len(evaluation), 13), -1e9)
    logits[:, classes] = np.log(np.maximum(probability, 1e-300))
    return [dict(logits=p) for p in logits], dict(status="trained", backend=backend,
            features=x_train.shape[1], classes=classes.tolist(), fit_rows=len(train))


def decision_record(row, arm, action, tried, trials, guarded):
    teacher, feasible = teacher_labels(row)
    record = dict(id=row["id"], task=row["domain"], split=row["split"], codec=row["codec"],
                  qp=row["qp"], arm=arm, action=action, rate=row["packets"][action]["bytes"],
                  quality=-float(row["cost"][action][TASKS.index(row["domain"])]),
                  tried_actions=tried, trials=trials, guard_trials=trials if guarded else 0,
                  encode_count=len(tried), decode_count=len(tried), guard_pass=float(feasible[action]),
                  teacher=teacher, teacher_agreement=float(action == teacher))
    for field in TIMINGS:
        values = [row["packets"][a].get(field) for a in tried]
        record[field] = sum(values) if all(v is not None and math.isfinite(v) for v in values) else None
    record["estimated_sender_seconds"] = (sum(record[k] for k in TIMINGS)
                                             if all(record[k] is not None for k in TIMINGS) else None)
    return record


def baseline_records(rows, predictions, models):
    lookup = {key(r): r for r in rows}
    # replay_bank requires timing keys. Missing timing is explicitly unknown, never zero.
    replay_rows = []
    for row in rows:
        adjusted = dict(row, packets=[dict(p, **{k: p.get(k) if p.get(k) is not None else math.nan
                                                for k in TIMINGS}) for p in row["packets"]])
        replay_rows.append(adjusted)
    result = []
    for r in replay_bank(dict(rows=replay_rows), predictions, models):
        arm = r["arm"]
        if arm.startswith("unchecked_") or arm.startswith("B2_"):
            continue
        guarded = arm == "oracle" or arm.startswith(("selected_", "random_"))
        trials = len(r["tried_actions"]) - 1 if guarded or arm.startswith("calibrated_") else 0
        result.append(decision_record(lookup[key(r)], arm, r["action"], r["tried_actions"], trials, guarded))
    return result


MEAN_FIELDS = ("rate", "quality", "trials", "guard_trials", "encode_count", "decode_count",
               "guard_pass", "teacher_agreement", "estimated_sender_seconds", *TIMINGS)


def with_seed_means(records, seeds):
    """Mean at each paired source/QP before BD fitting, exactly audit semantics."""
    output = list(records)
    by_arm = defaultdict(dict)
    for r in records:
        by_arm[r["arm"]][key(r)] = r
    groups = {prefix + "_mean": [f"{prefix}_{s}" for s in seeds]
              for prefix in (*STUDENTS, "forced", "calibrated", "selected")}
    groups["random3+guard"] = [f"random_{s}" for s in range(20)]
    for name, members in groups.items():
        if not all(m in by_arm for m in members):
            continue
        common = set(by_arm[members[0]])
        for member in members[1:]:
            common &= by_arm[member].keys()
        for k in sorted(common):
            values = [by_arm[m][k] for m in members]
            r = {field: values[0][field] for field in ("id", "task", "split", "codec", "qp")}
            r.update(arm=name, members=members)
            for field in MEAN_FIELDS:
                r[field] = (float(np.mean([v[field] for v in values]))
                            if all(v[field] is not None for v in values) else None)
            output.append(r)
    return output


def paired_bootstrap(records, arms, draws=1000, seed=0, grid=GRID):
    """One domain/codec, cluster-resampling ids with identical draws for all arms."""
    lookup = {(r["id"], r["qp"], r["arm"]): r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate bootstrap observation")
    ids = sorted({r["id"] for r in records if r["arm"] == "anchor"})
    arrays, output = {}, {}
    for arm in ["anchor", *arms]:
        if not ids or any((i, q, arm) not in lookup for i in ids for q in grid):
            output[arm] = dict(status="missing", reason="Incomplete paired bootstrap population")
            continue
        arrays[arm] = np.asarray([[[lookup[(i, q, arm)][field] for field in ("rate", "quality")]
                                   for q in grid] for i in ids])
    if "anchor" not in arrays:
        return {arm: dict(status="missing", reason="Incomplete anchor QP grid") for arm in arms}
    values = {arm: [] for arm in arms if arm in arrays}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sample = rng.integers(0, len(ids), size=len(ids))
        anchor = arrays["anchor"][sample].mean(axis=0)
        for arm in values:
            adaptive = arrays[arm][sample].mean(axis=0)
            # Exactly the PCHIP implementation used by imported compare_curves.
            value = audit.pchip_bd(anchor[:, 0], anchor[:, 1], adaptive[:, 0], adaptive[:, 1])
            if math.isfinite(value):
                values[arm].append(float(value))
    for arm, samples in values.items():
        lo, hi = np.quantile(samples, [.025, .975]).tolist() if samples else (None, None)
        output[arm] = dict(status="computed" if samples else "unavailable", lo=lo, hi=hi,
                           finite_draws=len(samples), draws=draws, finite_fraction=len(samples) / draws,
                           seed=seed, source_count=len(ids), paired=True, refit=False)
    return {arm: output[arm] for arm in arms}


def recovery(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and abs(denominator) > 1e-12 else None


def summarize(records, seeds, dataset, bootstrap=True):
    records = with_seed_means(records, seeds)
    # Do not plot the 20 random draws separately; their mean is random3+guard.
    shown = [r for r in records if not r["arm"].startswith("random_")]
    curve_sets = curves_from_rows(shown, dataset)
    metrics = []
    for curves in curve_sets:
        task, codec = curves["task"], curves["codec"]
        group = [r for r in shown if r["task"] == task and r["codec"] == codec]
        by_arm = defaultdict(list)
        for r in group:
            by_arm[r["arm"]].append(r)
        primary = {c["arm"]: c for c in curves["comparisons"] if c["grid"] == len(GRID)}
        requested_ci = [a for a in primary if any(a.startswith(p + "_") for p in CI_PREFIXES)]
        intervals = paired_bootstrap(group, requested_ci) if bootstrap else {}
        for arm, comparison in primary.items():
            value = dict(task=task, codec=codec, arm=arm, **{k: v for k, v in comparison.items()
                         if k not in ("task", "codec", "arm", "dataset", "metric")})
            rows = by_arm[arm]
            value.update(rows=len(rows), sources=len({r["id"] for r in rows}), ci95=intervals.get(arm))
            for field in MEAN_FIELDS[2:]:
                value["mean_" + field] = (float(np.mean([r[field] for r in rows]))
                                           if rows and all(r[field] is not None for r in rows) else None)
            seed = next((s for s in seeds if arm.endswith("_" + str(s))), None)
            reference = f"selected_{seed}" if seed is not None else "selected_mean"
            bd = comparison.get("pchip_bd_rate_pct")
            selected = primary.get(reference, {}).get("pchip_bd_rate_pct")
            oracle = primary.get("oracle", {}).get("pchip_bd_rate_pct")
            value.update(selected_reference=reference, selected_pchip_bd_rate_pct=selected,
                         oracle_pchip_bd_rate_pct=oracle, gain_recovered_selected=recovery(bd, selected),
                         gain_recovered_oracle=recovery(bd, oracle),
                         selected_has_gain=selected is not None and selected < -1e-12,
                         oracle_has_gain=oracle is not None and oracle < -1e-12)
            if arm.endswith("_mean"):
                prefix = arm[:-5]
                members = [primary.get(f"{prefix}_{s}", {}) for s in seeds]
                value["mean_of_seed_bd_rates"] = {field: audit.stats([m.get(field) for m in members])
                                                  for field in ("cubic_bd_rate_pct", "pchip_bd_rate_pct")}
            metrics.append(value)
        suppressed = set(by_arm) - set(primary)
        for arm in sorted(suppressed):
            metrics.append(dict(task=task, codec=codec, arm=arm, status="missing",
                                reason="Arm does not cover the complete paired evaluation population"))
    return dict(curve_sets=curve_sets, metrics=metrics)


def verdicts(protocols, quick):
    p1 = [m for p in protocols if p["protocol"] == "P1" for m in p.get("metrics", [])]
    result = []
    for domain in TASKS:
        for codec in ("h264", "h265"):
            tests = {}
            for prefix in ("distill_k0_argmax", "distill_k0_feasible"):
                m = next((m for m in p1 if m["task"] == domain and m["codec"] == codec and m["arm"] == prefix + "_mean"), {})
                recovered = m.get("gain_recovered_selected")
                has_gain = m.get("selected_has_gain", False)
                tests[prefix] = dict(gain_recovered_selected=recovered,
                                     selected_has_gain=has_gain,
                                     passes=bool(has_gain and recovered >= .7) if recovered is not None else None)
            evaluable = any(v["passes"] is not None for v in tests.values())
            passed = any(v["passes"] is True for v in tests.values())
            result.append(dict(domain=domain, codec=codec, verdict="GO" if passed else "NO-GO",
                               evidence_status="evaluated" if evaluable else "missing_or_undefined",
                               reason=("At least one prespecified K=0 rule recovers 70% of selected gain" if passed else
                                       "Neither K=0 rule reaches 70%" if evaluable else "No evaluable P1 student/selected gain; GO not established"),
                               rules=tests, provisional=quick, required_for_overall=domain in ("ar", "od")))
    return result


def record_file(path, evidence):
    exists = evidence.record(path)
    evidence.files[str(Path(path).resolve())]["status"] = "available" if exists else "missing"
    return exists


def load_bank(kaggle, spec, seeds, evidence):
    name, bank_name, model_name = spec
    directory = kaggle / bank_name / "outputs" / bank_name
    measurement = directory / "measurements.json"
    info = dict(name=name, measurement_path=str(measurement), controllers={}, issues=[])
    if not record_file(measurement, evidence):
        info.update(status="missing", reason="measurements.json missing")
        return None, {}, {}, {}, info
    cache = json.loads(measurement.read_bytes())
    rows = sorted(cache["rows"], key=lambda r: (r["domain"], r["id"], r["codec"], r["qp"]))
    info.update(validate_bank(rows), status="available", actions=cache.get("actions"), settings=cache.get("settings"))
    source_domains(rows)
    for row in rows:
        if row["split"] not in ("train", "cal") or len(row["condition"]) not in (6, 9):
            raise ValueError("Unexpected split or condition size")
        if not np.isfinite(np.asarray(row["cost"], float)[:, TASKS.index(row["domain"])]).all():
            raise ValueError("Nonfinite active-domain task costs")
    info["condition_lengths"] = dict(Counter(str(len(r["condition"])) for r in rows))
    if any(len(r["condition"]) != 6 for r in rows):
        info["issues"].append("Student conditions use first six values; original nine-value conditions retained for existing controllers.")
    input_path = directory / "controller_inputs.pth"
    inputs = torch.load(input_path, weights_only=True, map_location="cpu") if record_file(input_path, evidence) else {}
    missing = sorted({r["id"] for r in rows} - inputs.keys())
    info["missing_input_ids"] = missing
    info["input_shapes"] = dict(Counter(str(tuple(v[0].shape)) for v in inputs.values()))
    for source, (x, maps) in inputs.items():
        if x.ndim != 4 or x.shape != maps.shape or x.shape[1] != 3 or not torch.isfinite(x).all() or not torch.isfinite(maps).all():
            raise ValueError("Invalid controller input: " + source)
    if missing:
        info["issues"].append("Missing controller inputs: student arms skipped; incomplete existing-controller curves suppressed.")
    models, predictions = {}, {}
    for seed in seeds:
        checkpoint = kaggle / model_name / "outputs" / model_name / f"controller_{seed}.pth"
        if not record_file(checkpoint, evidence):
            info["controllers"][str(seed)] = dict(status="missing", path=str(checkpoint), reason="Checkpoint missing")
            continue
        original = kaggle / model_name / "v34_fix1/adaptive_vcm/v34/model.py"
        source = original if record_file(original, evidence) else audit.SCRATCH / "v34_fix1/adaptive_vcm/v34/model.py"
        if not record_file(source, evidence):
            info["controllers"][str(seed)] = dict(status="missing", reason="Exact historical model_source and audit scratch fallback missing")
            continue
        model = load_controller(checkpoint, evidence, source)
        predicted = predict_bank(model, rows, inputs)
        models[seed] = model
        predictions[seed] = {key(r): p for r, p in zip(rows, predicted) if p is not None}
        info["controllers"][str(seed)] = dict(status="available" if not missing else "missing_inputs", predictions=len(predictions[seed]),
                model_source=str(source), used_model_source_fallback=source != original,
                conditions=model.conditions, parameters=sum(p.numel() for p in model.parameters()))
    return rows, inputs, models, predictions, info


def run_protocol(name, rows, inputs, models, predictions, info, seeds, epochs, protocol):
    result = dict(bank=name, protocol=protocol, folds=[], missing_arms=[],
                  existing_controller_scope=("CAL held out from gradient fitting; used for calibration" if protocol == "P1"
                                             else "IN-SAMPLE on original TRAIN rows; mixed train/cal population"))
    if protocol == "P1":
        train = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"], dtype=int)
        evaluation = np.asarray([i for i, r in enumerate(rows) if r["split"] == "cal"], dtype=int)
        folds = [(train, evaluation)]
        result["split_method"] = "original train/cal by source"
    else:
        try:
            folds, result["split_method"] = grouped_folds(rows)
        except ValueError as error:
            result.update(status="missing", reason=str(error))
            return result
    all_evaluation, student_records = [], []
    for fold, (train_ids, eval_ids) in enumerate(folds):
        outer_train, evaluation = [rows[i] for i in train_ids], [rows[i] for i in eval_ids]
        training_sources, evaluation_sources = {r["id"] for r in outer_train}, {r["id"] for r in evaluation}
        if training_sources & evaluation_sources:
            raise ValueError("Source leakage between outer train/evaluation folds")
        if not outer_train or not evaluation:
            result.update(status="missing", reason="Empty training or evaluation split")
            return result
        all_evaluation.extend(evaluation)
        fold_info = dict(fold=fold, train_ids=sorted(training_sources), eval_ids=sorted(evaluation_sources), fits=[])
        result["folds"].append(fold_info)
        if info["missing_input_ids"]:
            for seed in seeds:
                result["missing_arms"].extend(dict(arm=f"{a}_{seed}", fold=fold, status="missing", reason="Controller inputs missing") for a in STUDENTS)
            continue
        for seed in seeds:
            print(f"{name} {protocol} fold {fold + 1}/{len(folds)} seed {seed}: training", flush=True)
            try:
                train, validation = inner_split(outer_train, seed)
            except ValueError as error:
                result["missing_arms"].extend(dict(arm=f"{a}_{seed}", fold=fold, status="missing", reason=str(error)) for a in STUDENTS)
                continue
            fit = dict(seed=seed, inner_train_ids=sorted({r["id"] for r in train}),
                       inner_validation_ids=sorted({r["id"] for r in validation}))
            model, fit["distill"] = fit_distill(train, validation, inputs, seed, epochs)
            val_predictions = predict_bank(model, inference_rows(validation), inputs)
            fit["tau_selection"] = tune_tau(validation, val_predictions)
            student_predictions = predict_bank(model, inference_rows(evaluation), inputs)
            tau = fit["tau_selection"]["tau"]
            for row, pred in zip(evaluation, student_predictions):
                for arm, k, cutoff in (("distill_k0_argmax", 0, None), ("distill_k0_feasible", 0, tau),
                                       ("distill_k1", 1, None), ("distill_k2", 2, None), ("distill_k3", 3, None)):
                    action, tried = student_decision(pred, k=k, row=row if k else None, tau=cutoff, domain=row["domain"])
                    student_records.append(decision_record(row, f"{arm}_{seed}", action, tried, k, bool(k)))
            gbm_predictions, fit["gbm"] = fit_gbm(train, evaluation, inputs, seed)
            if gbm_predictions is None:
                result["missing_arms"].extend(dict(arm=f"gbm_k{k}_{seed}", fold=fold, **fit["gbm"]) for k in (0, 1, 3))
            else:
                for row, pred in zip(evaluation, gbm_predictions):
                    for k in (0, 1, 3):
                        action, tried = student_decision(pred, k=k, row=row if k else None)
                        student_records.append(decision_record(row, f"gbm_k{k}_{seed}", action, tried, k, bool(k)))
            fold_info["fits"].append(fit)
    if len({key(r) for r in all_evaluation}) != len(all_evaluation):
        raise ValueError("Evaluation rows appear in multiple outer folds")
    for seed in seeds:
        if seed not in models or any(key(r) not in predictions.get(seed, {}) for r in all_evaluation):
            for arm in ("forced", "calibrated", "selected"):
                result["missing_arms"].append(dict(arm=f"{arm}_{seed}", status="missing", reason=info["controllers"].get(str(seed), {}).get("reason", "Incomplete input coverage")))
    records = baseline_records(all_evaluation, predictions, models) + student_records
    print(f"{name} {protocol}: paired source bootstrap (1000 draws)", flush=True)
    result.update(summarize(records, seeds, f"{name}/{protocol}"), status="computed", evaluation_rows=len(all_evaluation),
                  evaluation_sources=len({r["id"] for r in all_evaluation}), decisions=records)
    for metric in result["metrics"]:
        if metric["arm"].startswith(("forced_", "calibrated_", "selected_")):
            metric["existing_controller_scope"] = result["existing_controller_scope"]
    if protocol == "P2":
        for fold in result["folds"]:
            ids = set(fold["eval_ids"])
            fold["evaluation"] = summarize([r for r in records if r["id"] in ids], seeds, f"{name}/P2/fold{fold['fold']}", bootstrap=False)
            fold["existing_controller_scope"] = result["existing_controller_scope"]
    return result


def fmt(value, places=3):
    return "missing/undefined" if value is None else f"{value:.{places}f}"


def render_report(summary):
    lines = ["# V34 guard-distillation pilot", "", "**QUICK SMOKE TEST: verdicts below are provisional; not the full experiment.**" if summary["settings"]["quick"] else "CPU experiment with three independent training seeds.", "",
             "GO requires a prespecified distill K=0 rule to recover at least 70% of the P1 selected seed-mean PCHIP gain in each of AR/OD × H.264/H.265. Both K=0 rules are shown; either may qualify, and the qualifying rule can differ by cell. OT is supplementary. A missing/undefined comparison cannot establish GO.", "",
             "| Domain | Codec | K0 argmax recovery | K0 feasible recovery | Verdict | Evidence |",
             "|---|---|---:|---:|---|---|"]
    for v in summary["verdicts"]:
        rules = v["rules"]
        lines.append(f"| {v['domain']} | {v['codec']} | {fmt(rules['distill_k0_argmax']['gain_recovered_selected'])} | {fmt(rules['distill_k0_feasible']['gain_recovered_selected'])} | {v['verdict']} | {v['evidence_status']} |")
    lines.extend(["", f"Overall AR+OD gate: **{summary['overall_verdict']}**" + (" (provisional quick run)." if summary["settings"]["quick"] else ".")])
    for protocol in summary["protocols"]:
        lines.extend(["", f"## {protocol['bank']} / {protocol['protocol']}", "", protocol["existing_controller_scope"], ""])
        if protocol.get("status") != "computed":
            lines.append("Missing: " + protocol.get("reason", "unavailable"))
            continue
        lines.extend(["Negative BD-rate is better. Recovery is the signed student/reference BD ratio (0.70 = 70% when the reference is negative); a positive reference has no gain to recover. Sender seconds exclude unmeasured task-model/guard work. Pass and agreement are fractions over all evaluation rows.", "",
                      "| Domain/codec | Arm | Cubic % | PCHIP % | PCHIP 95% CI | Trials | Pass | Teacher agreement | Sender s | Full recovery | Oracle recovery | Reliable |",
                      "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"])
        for m in protocol["metrics"]:
            ci = m.get("ci95")
            interval = f"[{fmt(ci.get('lo'))}, {fmt(ci.get('hi'))}] ({ci.get('finite_draws', 0)}/1000)" if ci else "—"
            lines.append(f"| {m['task']}/{m['codec']} | {m['arm']} | {fmt(m.get('cubic_bd_rate_pct'))} | {fmt(m.get('pchip_bd_rate_pct'))} | {interval} | {fmt(m.get('mean_trials'))} | {fmt(m.get('mean_guard_pass'))} | {fmt(m.get('mean_teacher_agreement'))} | {fmt(m.get('mean_estimated_sender_seconds'), 6)} | {fmt(m.get('gain_recovered_selected'))} | {fmt(m.get('gain_recovered_oracle'))} | {m.get('reliable', False)} |")
        if protocol["missing_arms"]:
            lines.extend(["", "Missing arms:", ""])
            lines.extend(f"- {m['arm']}: {m['reason']}" for m in protocol["missing_arms"])
        if protocol["protocol"] == "P2":
            lines.extend(["", "P2 tables pool the five disjoint out-of-fold predictions before fitting curves. Per-fold curves, metrics, source assignments and fitting histories are in summary.json."])
    lines.extend(["", "## Data and provenance", ""])
    for bank in summary["banks"]:
        lines.append(f"- {bank['name']}: {bank['status']}; {bank.get('rows', 'missing')} rows, {bank.get('sources', 'missing')} sources.")
        lines.extend(f"  - {issue}" for issue in bank.get("issues", []))
    missing = [p for p, data in summary["provenance"]["files"].items() if data.get("status") == "missing"]
    lines.extend(f"- Missing input/source file: `{p}`" for p in missing)
    lines.extend(["", "Input SHA256 hashes, software versions, settings, seeds, source splits, training histories, tau selections, replay decisions and all curve comparisons are in summary.json.", "", "## Caveats", ""])
    lines.extend("- " + c for c in summary["caveats"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", type=Path, default=KAGGLE)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v34_distill_pilot")
    parser.add_argument("--protocol", choices=("P1", "P2", "both"), default="both")
    parser.add_argument("--quick", action="store_true", help="2 epochs, seed 303401; retains all folds and 1000 bootstrap draws")
    args = parser.parse_args(argv)
    out, kaggle = args.out.resolve(), args.kaggle.resolve()
    for protected in (kaggle, ROOT / "adaptive_vcm", ROOT / "checkpoints"):
        if out == protected or protected in out.parents:
            raise ValueError("Output overlaps read-only input/source directory")
    # Never overwrite pre-existing reports under the user-protected results tree.
    if ROOT / "results" in out.parents and any((out / n).exists() for n in ("summary.json", "REPORT.md")):
        raise ValueError("Refusing to modify existing files under results/; choose a new --out")
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    seeds = SEEDS[:1] if args.quick else SEEDS
    epochs = 2 if args.quick else 60
    evidence = audit.Evidence()
    for path in (Path(__file__), ROOT / "scripts/verify_and_ablate_v34.py", ROOT / "adaptive_vcm/v34/model.py",
                 ROOT / "adaptive_vcm/metrics.py", ROOT / "adaptive_vcm/legacy_bd.py"):
        record_file(path, evidence)
    settings = dict(quick=args.quick, protocol=args.protocol, seeds=list(seeds), device="cpu", torch_threads=2,
                    deterministic_algorithms=True, epochs=epochs, batch_size=32, learning_rate=1e-3,
                    weight_decay=1e-4, optimizer="Adam", early_stopping_patience=10, early_stopping_min_delta=1e-8,
                    inner_validation_source_fraction=.2, inner_validation_rounding="ceil",
                    conditions=6, task_logits=True, actions=13, bootstrap_draws=1000, bootstrap_seed=0,
                    random_guard_seeds=list(range(20)), p2_folds=5, primary_qps=list(GRID),
                    loss="CE(logits,teacher)+BCEWithLogits(risk[:,:,domain],feasible)+0.5*SmoothL1(log(rate),log(measured_rate))",
                    tau_selection="inner validation: max mean(1-rate), feasibility >=0.9; ties prefer pass rate then larger tau",
                    gbm_features=18, gbm_fit_population="same inner-training sources as distill",
                    gbm_early_stopping=False, student_condition_projection="first six saved values")
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None
    summary = dict(schema="v34_guard_distillation_v1", settings=settings, banks=[], protocols=[], caveats=CAVEATS,
                   provenance=dict(kaggle=str(kaggle), argv=sys.argv[1:] if argv is None else list(map(str, argv)),
                                   python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                                   sklearn=sklearn_version))
    for spec in BANKS:
        print(f"Loading {spec[0]} bank", flush=True)
        rows, inputs, models, predictions, info = load_bank(kaggle, spec, seeds, evidence)
        summary["banks"].append(info)
        for protocol in (("P1", "P2") if args.protocol == "both" else (args.protocol,)):
            if rows is None:
                summary["protocols"].append(dict(bank=spec[0], protocol=protocol, status="missing", reason=info["reason"],
                                                 existing_controller_scope="No measurements", missing_arms=["all"]))
            else:
                summary["protocols"].append(run_protocol(spec[0], rows, inputs, models, predictions, info, seeds, epochs, protocol))
    summary["verdicts"] = verdicts(summary["protocols"], args.quick)
    required = [v for v in summary["verdicts"] if v["required_for_overall"]]
    summary["overall_verdict"] = "GO" if all(v["verdict"] == "GO" for v in required) else "NO-GO"
    summary["provenance"]["files"] = evidence.files
    summary["elapsed_seconds"] = time.perf_counter() - start
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote {out / 'summary.json'} and {out / 'REPORT.md'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
