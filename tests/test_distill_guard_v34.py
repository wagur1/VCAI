"""Synthetic-only tests for the guard-distillation pilot."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import distill_guard_v34 as pilot


def synthetic_rows(sources=10, domains=("ar", "od")):
    rows = []
    for domain in domains:
        d = pilot.TASKS.index(domain)
        for source in range(sources):
            for codec in ("h264", "h265"):
                for qp in pilot.GRID:
                    base = (46 - qp) * 1000 + source * 5
                    rate = np.array([1., .99, 1.02, .96, .94, .90, .85, 1.05, .88, .92, .98, .7, .95])
                    cost = (qp - 20) / 100 + source / 10000
                    delta = [0., .001, -.01, .2, .1, .003, .2, 0., .2, .1, .2, .4, .2]
                    costs = [[None] * 3 for _ in range(13)]
                    for action in range(13):
                        costs[action][d] = cost + delta[action]
                    mask = [int(i == d) for i in range(3)]
                    rows.append(dict(id=f"{domain}/{source:02d}", domain=domain,
                        split="train" if source < sources - 2 else "cal", codec=codec, qp=qp,
                        condition=[float(codec == "h265"), qp / 51, *mask, 1.01], mask=mask,
                        rate=rate.tolist(), cost=costs, damage=[[delta[a] if i == d else None for i in range(3)] for a in range(13)],
                        packets=[dict(bytes=float(base * r), preprocess_seconds=.001,
                                      encode_seconds=.02, decode_seconds=.003) for r in rate]))
    return rows


@pytest.fixture
def row():
    return synthetic_rows(2, ("ar",))[0]


@pytest.mark.parametrize("domain", pilot.TASKS)
def test_teacher_and_feasibility_equal_measured_guard(domain):
    row = synthetic_rows(2, (domain,))[0]
    teacher, feasible = pilot.teacher_labels(row)
    assert teacher == pilot.choose_measured(row, list(range(13)), pilot.threshold(row)) == 5
    for action in range(1, 13):
        assert bool(feasible[action]) == (pilot.choose_measured(row, [0, action], pilot.threshold(row)) == action)
    assert feasible[0]
    assert not feasible[2]  # favourable cost but over byte budget
    assert not feasible[11]  # favourable bytes but failed quality guard


@pytest.mark.parametrize("k", (1, 2, 3))
def test_trials_include_identity_without_duplicates_and_obey_guard(row, k):
    for logits in (np.zeros(13), np.arange(13.), -np.arange(13.)):
        candidates = pilot.trial_candidates(logits, k)
        assert candidates[0] == 0
        assert len(candidates) == len(set(candidates)) == k + 1
        action, tried = pilot.student_decision(dict(logits=logits), k=k, row=row)
        assert tried == candidates
        assert action == pilot.choose_measured(row, candidates, pilot.threshold(row))
        assert pilot.teacher_labels(row)[1][action]


def test_k0_needs_no_evaluation_labels_and_feasible_fallback():
    pred = dict(logits=np.arange(13.), risk=np.zeros((13, 3)), rate=np.arange(13., 0., -1.))
    assert pilot.student_decision(pred) == (12, [12])
    assert pilot.student_decision(pred, tau=.6, domain="ar") == (0, [0])
    assert pilot.student_decision(pred, tau=.4, domain="ar") == (12, [12])


def test_groupkfold_and_inner_validation_have_no_source_leakage():
    rows = synthetic_rows()
    folds, info = pilot.grouped_folds(rows)
    assert len(folds) == 5 and info["domain_stratified"]
    all_eval = []
    for train, evaluation in folds:
        training_ids = {rows[i]["id"] for i in train}
        eval_ids = {rows[i]["id"] for i in evaluation}
        assert training_ids.isdisjoint(eval_ids)
        assert training_ids | eval_ids == {r["id"] for r in rows}
        assert {rows[i]["domain"] for i in evaluation} == {"ar", "od"}
        inner_train, inner_validation = pilot.inner_split([rows[i] for i in train], pilot.SEEDS[0])
        it = {r["id"] for r in inner_train}
        iv = {r["id"] for r in inner_validation}
        assert it.isdisjoint(iv) and (it | iv).isdisjoint(eval_ids)
        assert len(iv) == int(np.ceil(.2 * len(training_ids)))
        all_eval.extend(evaluation.tolist())
    assert sorted(all_eval) == list(range(len(rows)))


def test_bootstrap_is_deterministic_paired_and_matches_known_gain():
    records = []
    for r in synthetic_rows(5, ("ar",)):
        if r["codec"] != "h264":
            continue
        anchor = pilot.decision_record(r, "anchor", 0, [0], 0, False)
        adaptive = dict(anchor, arm="adaptive", rate=anchor["rate"] * .8)
        records.extend([anchor, adaptive])
    first = pilot.paired_bootstrap(records, ["adaptive"], draws=30)
    second = pilot.paired_bootstrap(list(reversed(records)), ["adaptive"], draws=30)
    assert first == second
    assert first["adaptive"]["lo"] == pytest.approx(-20.)
    assert first["adaptive"]["hi"] == pytest.approx(-20.)
    assert first["adaptive"]["finite_draws"] == 30
    broken = pilot.paired_bootstrap(records[:-1], ["adaptive"], draws=3)
    assert broken["adaptive"]["status"] == "missing"
    # Heterogeneous effects ensure this also detects accidental unseeded sampling.
    varied = [dict(r, rate=r["rate"] * (1 + int(r["id"].split("/")[1]) / 20))
              if r["arm"] == "adaptive" else r for r in records]
    a = pilot.paired_bootstrap(varied, ["adaptive"], draws=30)
    b = pilot.paired_bootstrap(varied, ["adaptive"], draws=30)
    c = pilot.paired_bootstrap(varied, ["adaptive"], draws=30, seed=1)
    assert a == b and a["adaptive"]["lo"] != c["adaptive"]["lo"]


def test_tau_selection_uses_measured_feasibility_and_fallback(row):
    pred = dict(logits=np.zeros(13), risk=np.full((13, 3), -10.), rate=np.ones(13))
    pred["risk"][0, 0] = 10.
    pred["risk"][11, 0] = 0.
    pred["risk"][5, 0] = 2.
    pred["rate"][[5, 11]] = [.9, .7]
    selected = pilot.tune_tau([row], [pred])
    action, _ = pilot.student_decision(pred, tau=selected["tau"], domain="ar")
    assert action == 5
    assert selected["feasibility_rate"] == 1.
    assert selected["mean_rate_saving"] == pytest.approx(.1)


def test_seed_means_and_timing_follow_replay(row):
    records = []
    for seed, action in zip(pilot.SEEDS, (0, 5, 11)):
        records.append(pilot.decision_record(row, f"distill_k0_argmax_{seed}", action, [action], 0, False))
    means = pilot.with_seed_means(records, pilot.SEEDS)
    assert means[-1]["rate"] == pytest.approx(np.mean([r["rate"] for r in records]))
    assert means[-1]["trials"] == 0
    guarded = pilot.decision_record(row, "distill_k3", 5, [0, 11, 5, 6], 3, True)
    assert guarded["estimated_sender_seconds"] == pytest.approx(4 * .024)
    assert guarded["encode_count"] == 4 and guarded["trials"] == 3
    missing = deepcopy(row)
    del missing["packets"][5]["decode_seconds"]
    assert pilot.decision_record(missing, "arm", 5, [5], 0, False)["estimated_sender_seconds"] is None


def test_features_and_condition_projection_are_input_only(row):
    inputs = {row["id"]: (torch.ones(1, 3, 4, 4), torch.full((1, 3, 4, 4), 2.))}
    row = dict(row, condition=row["condition"] + [1, 0, 0])
    feature = pilot.feature_matrix([row], inputs)
    assert feature.shape == (1, 18)
    np.testing.assert_array_equal(feature[0, :6], [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(feature[0, 6:12], np.zeros(6))
    assert set(pilot.inference_rows([row])[0]) == {"id", "condition"}
    assert len(pilot.inference_rows([row])[0]["condition"]) == 6


def test_verdict_does_not_call_missing_or_nonbeneficial_full_a_go():
    metric = dict(task="ar", codec="h264", arm="distill_k0_argmax_mean", gain_recovered_selected=.71, selected_has_gain=True)
    result = pilot.verdicts([dict(protocol="P1", metrics=[metric])], quick=False)
    assert result[0]["verdict"] == "GO"
    assert result[1]["verdict"] == "NO-GO"
    assert result[1]["evidence_status"] == "missing_or_undefined"
    assert pilot.recovery(-10, 2) == -5
    assert pilot.recovery(-10, 0) is None
    metric.update(gain_recovered_selected=2., selected_has_gain=False)
    assert pilot.verdicts([dict(protocol="P1", metrics=[metric])], quick=False)[0]["verdict"] == "NO-GO"


def test_quick_cli_synthetic_end_to_end(tmp_path):
    rows = synthetic_rows(7, ("ar",))
    kaggle = tmp_path / "kaggle"
    name = pilot.BANKS[0][1]
    directory = kaggle / name / "outputs" / name
    directory.mkdir(parents=True)
    (directory / "measurements.json").write_text(json.dumps(dict(rows=rows)), encoding="utf-8")
    generator = torch.Generator().manual_seed(17)
    inputs = {r["id"]: (torch.rand(1, 3, 8, 8, generator=generator), torch.rand(1, 3, 8, 8, generator=generator)) for r in rows}
    torch.save(inputs, directory / "controller_inputs.pth")
    # Exercise the real strict checkpoint loader and importlib model-source path.
    from adaptive_vcm.v34.checkpoint import save_checkpoint
    model = pilot.Controller(13, conditions=6, task_logits=True)
    metadata = dict(calibration=dict(risk=[[0.] * 3] * 13, rate=[0.] * 13),
                    settings=dict(damage_tolerance=[.05] * 3))
    save_checkpoint(directory / f"controller_{pilot.SEEDS[0]}.pth", model, metadata)
    model_source = kaggle / name / "v34_fix1/adaptive_vcm/v34/model.py"
    model_source.parent.mkdir(parents=True)
    model_source.write_text("from adaptive_vcm.v34.model import Controller\n", encoding="utf-8")
    out = tmp_path / "output"
    completed = subprocess.run([sys.executable, str(Path(pilot.__file__)), "--quick", "--kaggle", str(kaggle),
                                "--out", str(out)], capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["epochs"] == 2
    assert summary["settings"]["seeds"] == [303401]
    assert summary["banks"][1]["status"] == "missing"
    p1 = summary["protocols"][0]
    p2 = next(p for p in summary["protocols"] if p["bank"] == "ar_od" and p["protocol"] == "P2")
    assert len(p2["folds"]) == 5
    assert "IN-SAMPLE" in p2["existing_controller_scope"]
    assert p1["evaluation_rows"] == 16
    expected = {"anchor", "fixed", "oracle", "random3+guard", "forced_303401", "calibrated_303401", "selected_303401", "selected_mean"}
    expected |= {f"{arm}_303401" for arm in pilot.STUDENTS}
    assert expected <= {m["arm"] for m in p1["metrics"]}
    assert all(m["ci95"]["draws"] == 1000 for m in p1["metrics"] if m.get("ci95"))
    assert all(r["split"] == "cal" for r in p1["decisions"])
    for protocol in (p1, p2):
        for fold in protocol["folds"]:
            for fit in fold["fits"]:
                assert set(fold["eval_ids"]).isdisjoint(fit["inner_train_ids"] + fit["inner_validation_ids"])
    measurements = summary["provenance"]["files"][str((directory / "measurements.json").resolve())]
    assert len(measurements["sha256"]) == 64
    assert "QUICK SMOKE TEST" in (out / "REPORT.md").read_text(encoding="utf-8")
    assert "single sequence" in (out / "REPORT.md").read_text(encoding="utf-8")


def test_p1_rejects_source_split_leakage():
    rows = synthetic_rows(2, ("ar",))
    rows[0]["split"] = "train"
    with pytest.raises(ValueError, match="Source leakage"):
        pilot.run_protocol("fixture", rows, {}, {}, {}, dict(missing_input_ids=[]), pilot.SEEDS[:1], 2, "P1")
