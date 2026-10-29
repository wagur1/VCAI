"""Numerical, policy, provenance, and saved-artifact regression tests for the audit."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import verify_and_ablate_v34 as audit


@pytest.fixture
def bank_row():
    sizes = [1000, 1010, 1011, 980, 950, 900, 800, 1050, 1200, 960, 1000, 700, 1100]
    costs = [.2, .1, .1, .3, .25, .21, .24, .2, .2, .3, .2, .8, .1]
    return dict(id="ar/example", domain="ar", split="cal", codec="h264", qp=30,
                condition=[0, 30 / 51, 1, 0, 0, 1.01], mask=[True, False, False],
                rate=[s / 1000 for s in sizes], cost=[[v, None, None] for v in costs],
                damage=[[v - costs[0], None, None] for v in costs],
                packets=[dict(bytes=s, preprocess_seconds=.01, encode_seconds=.1, decode_seconds=.02) for s in sizes])


@pytest.fixture
def prediction():
    logits = np.zeros(13)
    logits[11] = 2
    rate = np.ones(13)
    rate[[11, 6, 5]] = [.7, .8, .9]
    return dict(logits=logits, risk=np.zeros((13, 3)), rate=rate)


def curve(rate, quality):
    return dict(rate=list(rate), quality=list(quality))


@pytest.mark.parametrize("grid", audit.GRIDS)
def test_bd_constant_rate_reduction_and_identity(grid):
    n = len(grid)
    ra = np.exp(np.linspace(-3, 1, n))
    q = np.linspace(.2, .9, n)
    a = curve(ra, q)
    result = audit.compare_curves(a, curve(.8 * ra, q), list(grid), grid)
    assert result["cubic_bd_rate_pct"] == pytest.approx(-20, abs=1e-8)
    assert result["pchip_bd_rate_pct"] == pytest.approx(-20, abs=1e-8)
    assert result["reliable"]
    identity = audit.compare_curves(a, a, list(grid), grid)
    for key in ("cubic_bd_rate_pct", "pchip_bd_rate_pct", "cubic_bd_quality", "pchip_bd_quality"):
        assert identity[key] == pytest.approx(0, abs=1e-10)


def test_quality_units_and_missing_qp_are_not_zero():
    a = curve([1, 2, 4, 8], [.2, .4, .6, .8])
    b = curve([1, 2, 4, 8], [.21, .41, .61, .81])
    result = audit.compare_curves(a, b, list(audit.GRIDS[0]), audit.GRIDS[0])
    assert result["cubic_bd_quality"] == pytest.approx(1.)
    assert result["pchip_bd_quality"] == pytest.approx(1.)
    missing = audit.compare_curves(a, b, list(audit.GRIDS[0]), audit.GRIDS[1])
    assert missing == dict(status="unavailable", reason="QP grid incomplete", missing_qps=[50])


def test_flat_nonoverlap_nonmonotone_and_plateau_curves():
    qps = list(audit.GRIDS[0])
    a = curve([1, 2, 3, 4], [.1, .2, .3, .4])
    flat = audit.compare_curves(a, curve([1, 2, 3, 4], [.2] * 4), qps, qps)
    assert flat["cubic_bd_rate_pct"] is None
    assert flat["pchip_bd_rate_pct"] is None
    assert not flat["reliable"]
    no_overlap = audit.compare_curves(a, curve([1, 2, 3, 4], [.5, .6, .7, .8]), qps, qps)
    assert no_overlap["cubic_bd_rate_pct"] is None
    nonmono = audit.compare_curves(a, curve([1, 2, 3, 4], [.1, .3, .2, .4]), qps, qps)
    assert not nonmono["reliable"]
    assert nonmono["pchip_bd_rate_pct"] is not None
    plateau = curve([1, 2, 3, 4], [.1, .2, .2, .4])
    assert audit.compare_curves(plateau, plateau, qps, qps)["pchip_bd_rate_pct"] == pytest.approx(0)


def test_sharp_monotone_curve_is_flagged_for_cubic_instability():
    a = curve([.055, .0307, .0188, .0119], [.3809, .3622, .3498, .3230])
    b = curve([.045, .025, .016, .0105], [.3680, .3645, .3644, .3535])
    result = audit.compare_curves(a, b, list(audit.GRIDS[0]), audit.GRIDS[0])
    assert all(result["quality_monotone_by_rate"].values())
    assert result["interpolation_sensitive"] and not result["reliable"]


@pytest.mark.parametrize("bad", [curve([0, 2, 3, 4], [.1, .2, .3, .4]),
                                curve([1, 2, 3, 4], [.1, np.nan, .3, .4])])
def test_invalid_curves_fail_loudly(bad):
    with pytest.raises(ValueError):
        audit.compare_curves(bad, bad, list(audit.GRIDS[0]), audit.GRIDS[0])


def test_duplicate_qps_and_curve_length_rejected():
    a = curve([1, 2, 3, 4], [.1, .2, .3, .4])
    with pytest.raises(ValueError, match="Duplicate QP"):
        audit.compare_curves(a, a, [30, 35, 35, 45], [30, 35, 45])
    with pytest.raises(ValueError, match="length"):
        audit.compare_curves(a, a, [30, 35, 40, 45, 50], audit.GRIDS[0])


def test_guard_uses_actual_bytes_and_task_cost_delta(bank_row):
    # Lowest rate action 11 fails the guard; action 5 passes AR epsilon=.025.
    assert audit.choose_measured(bank_row, [0, 11, 6, 5]) == 11
    assert audit.choose_measured(bank_row, [0, 11, 6, 5], threshold=.025) == 5
    assert audit.choose_measured(bank_row, [0, 11, 6, 5], threshold=.05) == 6
    # Neither absolute cost nor a clipped damage field may replace the delta.
    bank_row["damage"] = [[999, None, None]] * 13
    bank_row["rate"] = [1.] * 13
    assert audit.choose_measured(bank_row, [0, 11, 6, 5], threshold=.025) == 5


def test_guard_budget_boundary_negative_cost_and_invalid_anchor(bank_row):
    assert audit.choose_measured(bank_row, [0, 1], threshold=0.) == 0  # identity is smaller
    bank_row["packets"][0]["bytes"] = 1000
    bank_row["packets"][1]["bytes"] = 900
    assert audit.choose_measured(bank_row, [0, 1], threshold=0.) == 1
    bank_row["packets"][1]["bytes"] = 1011
    assert audit.choose_measured(bank_row, [0, 1], threshold=0.) == 0
    with pytest.raises(ValueError, match="Candidates"):
        audit.choose_measured(bank_row, [11])
    with pytest.raises(ValueError, match="Candidates"):
        audit.choose_measured(bank_row, [0, 0])
    bank_row["packets"][0]["bytes"] = 0
    with pytest.raises(ValueError, match="Invalid measured"):
        audit.choose_measured(bank_row, [0, 1])


def test_bank_validation_catches_duplicate_missing_actions_and_bad_ratios(bank_row):
    assert audit.validate_bank([bank_row])["rows"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        audit.validate_bank([bank_row, bank_row])
    bad = deepcopy(bank_row)
    bad["packets"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        audit.validate_bank([bad])
    bad = deepcopy(bank_row)
    bad["rate"][6] = .001
    with pytest.raises(ValueError, match="rate disagrees"):
        audit.validate_bank([bad])


def test_shortlist_is_forced_then_predicted_risk_rate_rank(prediction):
    assert audit.shortlist(prediction, [True, False, False]) == [0, 11, 6, 5]
    prediction["logits"][3] = 3
    assert audit.shortlist(prediction, [True, False, False]) == [0, 3, 11, 6]
    assert audit.shortlist(prediction, [True, False, False], policy="rank") == [0, 11, 6, 5]
    prediction["risk"][11, 0] = 2
    assert audit.shortlist(prediction, [True, False, False], policy="rank")[:3] == [0, 6, 5]


def test_replay_all_policies_are_deterministic_and_use_seed_models(bank_row, prediction):
    calibration = dict(risk=[[0.] * 3] * 13, rate=[0.] * 13)
    model = SimpleNamespace(metadata=dict(calibration=calibration, settings=dict(damage_tolerance=[.05] * 3)))
    key = (bank_row["id"], bank_row["codec"], bank_row["qp"])
    predictions = {s: {key: prediction} for s in audit.SEEDS}
    models = {s: model for s in audit.SEEDS}
    a = audit.replay_bank(dict(rows=[bank_row]), predictions, models)
    b = audit.replay_bank(dict(rows=[bank_row]), predictions, models)
    assert a == b
    arms = {r["arm"]: r for r in a}
    assert arms["forced_303401"]["action"] == 11
    assert arms["unchecked_303401"]["action"] == 11
    assert arms["selected_303401"]["action"] == 5
    assert arms["selected_303401"]["quality"] == -.21
    assert arms["fixed"]["rate"] == 700
    for seed in range(20):
        row = arms[f"random_{seed}"]
        assert len(row["tried_actions"]) == len(set(row["tried_actions"])) == 4
        assert row["tried_actions"][0] == 0
        assert row["encode_count"] == row["decode_count"] == 4
    costs = audit.bank_costs(a)
    assert all(r["sender_seconds"] is None and r["fps"] is None for r in costs)


def test_seed_means_are_not_an_ensemble_and_missing_seeds_are_not_averaged():
    curves = {f"forced_{s}": curve([s - 303400] * 4, [.1, .2, .3, .4]) for s in audit.SEEDS}
    audit.add_seed_means(curves)
    assert curves["forced_mean"]["rate"] == [2.] * 4
    assert "forced_ensemble" not in curves
    del curves["forced_mean"]
    del curves["forced_303403"]
    audit.add_seed_means(curves)
    assert "forced_mean" not in curves


def test_paired_population_validation_and_partial_curve_suppression():
    rows = [dict(id="x", task="ar", codec="h264", qp=q, arm="anchor", rate=100 - q, quality=1 - q / 100)
            for q in audit.GRIDS[0]]
    rows.append(dict(rows[0], arm="forced_303401"))
    result = audit.curves_from_rows(rows, "fixture")
    assert "forced_303401" not in result[0]["curves"]
    rows[-1] = dict(rows[0], id="other", qp=50)
    with pytest.raises(ValueError, match="population varies"):
        audit.curves_from_rows(rows, "fixture")
    with pytest.raises(ValueError, match="Duplicate paired"):
        audit.curves_from_rows([rows[0], rows[0]], "fixture")


def test_distribution_denominators_and_shortlist_counts():
    rows = [dict(task="ar", id=str(i), arm="forced_303401", action=a, tried_actions=[0, 11, 6, 5])
            for i, a in enumerate([11, 11, 0, 6])]
    value = audit.action_distribution(rows)["ar/forced_303401"]
    assert value["actions"]["11"] == dict(count=2, percent=50.)
    assert value["identity_percent"] == 25
    assert value["shortlists"][0]["count"] == 4


def test_real_od_bug_and_global_filter_numerical_probe():
    historical = audit.historical_source("adaptive_vcm/v34/tasks.py")
    broken = audit.od_source_only_probe(historical)
    assert broken["outcome"] == "error" and "items" in broken["message"]
    source = (audit.ROOT / "adaptive_vcm/v34/tasks.py").read_text(encoding="utf-8")
    fixed = audit.od_source_only_probe(source)
    assert fixed["outcome"] == "returned"
    assert fixed["calls"] == ["od_forward_label_free", "detection_distance"]
    result = audit.filter_probe()
    assert result["identity_unchanged"]
    assert result["max_absolute_pixel_difference"]["11"] == 0
    assert result["max_absolute_pixel_difference"]["6"] > 0
    assert not result["action5_matches_request"]


def test_task1_separates_historical_faults_from_workspace_repairs(monkeypatch, tmp_path):
    from adaptive_vcm.v32 import codec
    monkeypatch.setattr(audit, "ffmpeg_probe", lambda: dict(executable="fixture", version="fixture FFmpeg 9", tests={
        "legacy_vsync": dict(returncode=1), "fps_mode": dict(returncode=0)}))
    monkeypatch.setattr(codec, "_fps_sync_flags", lambda _: ["-fps_mode", "passthrough"])
    distributions = {"coco_historical": {f"od/selected_{s}": dict(identity_percent=100) for s in audit.SEEDS}}
    result = audit.task1_audit(audit.Evidence(), tmp_path, tmp_path, distributions, {})
    assert result["fixed_in_workspace"]
    findings = {f["id"]: f for f in result["findings"]}
    assert findings["od_source_only"]["status"] == "confirmed"
    assert findings["od_source_only"]["original_probe"]["outcome"] == "error"
    assert findings["od_source_only"]["fixed_probe"]["outcome"] == "returned"
    runner = findings["historical_od_selected"]
    assert runner["status"] == "confirmed" and runner["fixed_in_workspace"]
    assert not runner["source_guard_limits_present"]
    assert all(c["source_score_present"] for c in runner["workspace_choose_packet_calls"])
    assert all(v["git_object"].startswith(audit.HISTORICAL_REF + ":") for v in result["historical_sources"].values())


def test_bugfix_bank_replay_uses_real_policy_functions_and_cal_only(bank_row, prediction):
    prediction["risk"][11, 0] = 2.
    cfg = dict(damage_tolerance=[.05] * 3, budget_ratio=1.01, cal_coverage=.9,
               source_shortlist=3, source_guard_limits=[0., .05, 0.], shortlist_seed="rank")
    model = SimpleNamespace(metadata=dict(settings=cfg, calibration=dict(risk=[[0.] * 3] * 13, rate=[0.] * 13)))
    models = {303401: model}
    key = (bank_row["id"], bank_row["codec"], bank_row["qp"])
    predictions = {303401: {key: prediction}}
    cache = dict(rows=[bank_row], settings=cfg)
    result = audit.bugfix_bank_replay(cache, predictions, models)
    assert result == audit.bugfix_bank_replay(cache, predictions, models)
    assert result["decisions"][0]["before"]["B3"] == 11
    assert result["decisions"][0]["after"]["B3"] == 6
    assert result["calibrations"]["303401"]["cal_rows"] == 1
    assert result["calibrations"]["303401"]["after"]["mode"] == "mean_ucb"
    assert result["normalization"][0]["after_floor"] == [1., .25, .25]
    bank_row["split"] = "train"
    with pytest.raises(ValueError, match="CAL"):
        audit.bugfix_bank_replay(cache, predictions, models)


def test_heldout_guard_replay_never_imputes_unobserved_accuracy():
    anchor = dict(id="clip", task="ar", codec="h264", qp=30, arm="anchor", action=0,
                  bytes=100, correct={"swin3d_t": 1})
    selected = dict(anchor, arm="selected_303401", tried_actions=[0, 1], source_guard=dict(
        limits=[0., .05, 0.], rate=[1., .8], costs=[[0., 0., None], [.01, 0., None]]))
    result = audit.replay_heldout_guard([anchor, selected])
    assert result["historical_decisions_match"]
    assert result["decisions"][0]["after_action"] == 1
    assert result["decisions"][0]["after_bytes"] == 80
    comparison = result["comparisons"][0]
    assert comparison["after"]["byte_rate_pct_vs_anchor"] == pytest.approx(-20.)
    assert comparison["after_prediction_coverage"] == 0.
    assert comparison["same_qp_accuracy_delta_pp"]["swin3d_t"] is None
    forced = dict(anchor, arm="forced_303401", action=1, bytes=80, correct={"swin3d_t": 0})
    known = audit.replay_heldout_guard([anchor, selected, forced])["comparisons"][0]
    assert known["after_prediction_coverage"] == 1.
    assert known["same_qp_accuracy_delta_pp"]["swin3d_t"] == -100.
    with pytest.raises(ValueError, match="decision"):
        audit.replay_heldout_guard([anchor, dict(selected, action=1)])


def test_inference_matches_single_row_calls_and_detects_missing_inputs(tmp_path):
    from adaptive_vcm.v34.model import Controller
    from adaptive_vcm.v34.checkpoint import save_checkpoint
    torch.manual_seed(11)
    torch.set_num_threads(2)
    original = Controller(13).eval()
    path = tmp_path / "controller.pth"
    save_checkpoint(path, original, {"seed": 303401})
    model = audit.load_controller(path, audit.Evidence())
    inputs = {"x": (torch.rand(1, 3, 48, 48), torch.rand(1, 3, 48, 48))}
    rows = [dict(id="x", condition=[0, q / 51, 0, 1, 0, 1.01]) for q in (30, 35)]
    result = audit.predict_bank(model, rows + [dict(rows[0], id="missing")], inputs)
    assert result[-1] is None
    with torch.inference_mode():
        for row, pred in zip(rows, result):
            expected = original(inputs["x"][0][None], inputs["x"][1][None], torch.tensor([row["condition"]]))
            for key in pred:
                np.testing.assert_allclose(pred[key], expected[key][0].numpy(), atol=1e-6)
    value = torch.load(path, weights_only=True)
    value["metadata"]["seed"] = 1
    torch.save(value, path)
    with pytest.raises(ValueError, match="integrity"):
        audit.load_controller(path, audit.Evidence())


def test_evidence_reading_json_gzip_hashes_and_missing(tmp_path):
    value = [{"id": "test"}]
    raw = json.dumps(value).encode()
    a, b = tmp_path / "a.json", tmp_path / "b.json.gz"
    a.write_bytes(raw)
    b.write_bytes(gzip.compress(raw, mtime=0))
    evidence = audit.Evidence()
    assert evidence.read(a) == evidence.read(b) == value
    assert evidence.files[str(a.resolve())]["sha256"] == audit.file_hash(a)
    assert evidence.read(tmp_path / "missing") is None
    assert evidence.gaps[0]["status"] == "unavailable"


def test_ffmpeg_probe_reports_observation_without_assuming_all_versions(monkeypatch):
    monkeypatch.setattr(audit.shutil, "which", lambda _: None)
    assert audit.ffmpeg_probe()["status"] == "unavailable"
    monkeypatch.setattr(audit.shutil, "which", lambda _: "ffmpeg")
    def fake_run(command, **kwargs):
        if "-version" in command:
            return SimpleNamespace(stdout="ffmpeg version 9.0.2\n", stderr="", returncode=0)
        legacy = "-vsync" in command
        return SimpleNamespace(stdout="", stderr="Unrecognized option vsync" if legacy else "", returncode=int(legacy))
    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    result = audit.ffmpeg_probe()
    assert result["tests"]["legacy_vsync"]["returncode"] == 1
    assert result["tests"]["fps_mode"]["returncode"] == 0
    assert "only" in result["scope"]


def test_output_cannot_overlap_frozen_directory(tmp_path):
    protected = {"results/v34_r1/independent_summary.json": "hash"}
    with pytest.raises(ValueError, match="publication"):
        audit.validate_output_directory(audit.ROOT / "results/v34_r1", protected)
    assert audit.validate_output_directory(tmp_path / "out", protected) == tmp_path / "out"


def test_timing_stage_quantiles_fps_and_anchor_slowdown():
    rows = []
    for i, sender in enumerate([2., 4.]):
        base = dict(task="ar", codec="h264", id=str(i), preprocess_seconds=.01, encode_seconds=.09,
                    decode_seconds=.2, receiver_seconds=.3)
        rows.extend([dict(base, arm="anchor"), dict(base, arm="selected_303401", deployed_sender_seconds=sender,
                    deployed_candidate_work=[{}, {}, {}, {}])])
    selected = next(r for r in audit.cost_analysis(rows, "fixture") if r["arm"] == "selected_303401")
    assert selected["fields"]["deployed_sender_seconds"] == dict(n=2, mean=3., p50=3., p90=3.8)
    assert selected["sender_slowdown_vs_b0"] == pytest.approx(30)
    assert selected["aggregate_sender_fps"] == pytest.approx(16 / 3)
    assert selected["end_to_end_seconds"]["mean"] == pytest.approx(3.5)
    assert selected["derived_probe_decode_count"]["mean"] == 4
    assert selected["fields"]["controller_inference_seconds"]["mean"] is None


@pytest.fixture(scope="module")
def generated_audit():
    path = audit.ROOT / "results/verification_audit/audit_summary.json"
    if not path.exists():
        pytest.skip("Run scripts/verify_and_ablate_v34.py for saved-artifact regression checks")
    return json.loads(path.read_bytes())


def test_generated_publication_and_exact_k400_counts(generated_audit):
    value = generated_audit
    integrity = value["publication_integrity"]
    assert integrity["before"] == integrity["after"]
    assert integrity["unchanged"]
    counts = value["task1_audit"]["distributions"]["k400_publication"]
    assert counts["ar/forced_303401"]["total"] == 10240
    assert counts["ar/forced_303401"]["actions"]["11"]["count"] == 9967
    assert counts["ar/forced_303401"]["shortlists"][0]["count"] == 9642
    assert counts["ar/selected_303401"]["actions"]["0"]["count"] == 3743
    for seed in audit.SEEDS:
        od = value["task1_audit"]["distributions"]["coco_historical"][f"od/selected_{seed}"]
        assert od["total"] == od["actions"]["0"]["count"] == 640


def test_generated_banks_and_tracking_limitations_are_explicit(generated_audit):
    data = generated_audit
    bank = next(b for b in data["task2_ablation"]["banks"] if b["name"] == "scaled")
    assert bank["rows"] == 1408 and bank["sources"] == 176
    assert bank["controllers"]["303401"]["predictions"] == 1408
    assert bank["controllers"]["303401"]["parameters"] == 9423
    mot17 = data["task4_bdrate_and_ot"]["mot17"]
    assert mot17["expanded_evaluation_not_executed"] and not mot17["independent_validation"]
    assert "MOT17-09-FRCNN" not in mot17["observed_sequences"]
    assert all(r["rates_match"] and r["hota_match"] for r in mot17["csv_crosschecks"])
    assert data["limitations"]
    curves = data["task2_ablation"]["full_bank_curves"]
    assert all(r["status"] == "unavailable" for c in curves for r in c["comparisons"] if r["grid"] == 5)


def test_generated_model_counts_and_memory_are_distinct(generated_audit):
    memory = generated_audit["task3_cost_analysis"]["memory"]
    expected = {"Controller_frozen_9_conditions": 8661, "FasterRCNN_ResNet50_FPN": 41755286,
                "Swin3D_T": 28158070, "R3D_18": 33371472}
    for name, count in expected.items():
        assert memory["models"][name]["parameters"] == count
        assert memory["models"][name]["fp32_parameter_bytes"] == count * 4
    assert memory["experiment_cpu_rss_bytes"] is None
    assert {r["task"] for r in memory["full_system_csv"]} == {"ar", "od", "ot"}


def test_generated_report_matches_json_and_cannot_hide_nan(generated_audit):
    raw = json.dumps(generated_audit, allow_nan=False)
    assert len(raw) > 100
    report = (audit.ROOT / "results/verification_audit/AUDIT_REPORT.md").read_text(encoding="utf-8")
    assert report == audit.render_report(generated_audit)
    assert "not official TrackEval" in report or "NOT official" in report or "not official combined" in report
    assert "## 6. Bugfix verification: live Before vs After replay" in report


def test_generated_bugfix_replay_is_complete_and_keeps_frozen_predictions(generated_audit):
    assert generated_audit["task1_audit"]["fixed_in_workspace"]
    verification = generated_audit["task5_bugfix_verification"]
    assert verification["status"] == "verified"
    assert verification["deterministic"]["torch_threads"] == 2
    assert verification["deterministic"]["parameter_updates"] == 0
    assert all(c["exact_outputs_equal"] and c["parameters"] == 8661 and not c["task_logits"]
               for c in verification["frozen_checkpoint_compatibility"].values())
    expected = {"scaled/all": 1408, "MOT20-02/all": 128}
    assert {b["name"]: b["rows"] for b in verification["banks"]} == expected
    for bank in verification["banks"]:
        assert len(bank["decisions"]) == bank["rows"] * 3
        assert all(c["before_prediction_sha256"] == c["after_prediction_sha256"] for c in bank["controllers"].values())
        assert all(c["after"]["actual_budget_violations"] == 0 for c in bank["comparisons"] if c["policy"] != "B3")
    heldout = verification["heldout_source_guard"]
    assert heldout["historical_decisions_match"] and heldout["after_limits"] == [.025, .05, .05]
    assert all(c["changed_actions"] > 0 and c["after"]["byte_rate_pct_vs_anchor"] <= c["before"]["byte_rate_pct_vs_anchor"]
               for c in heldout["comparisons"])


def test_manifest_keeps_exact_historical_runtime_hashes():
    historical = json.loads(audit.historical_source("results/v34_r1/publication_manifest.json"))
    current = json.loads((audit.ROOT / "results/v34_r1/publication_manifest.json").read_text())
    assert current["v34_r1_historical_runtime_sha256"] == historical["runtime_sha256"]
    assert current["independent_summary_sha256"] == historical["independent_summary_sha256"]
    assert current["checkpoints"] == historical["checkpoints"]
    assert current["runtime_sha256"] != current["v34_r1_historical_runtime_sha256"]


def test_pchip_repair_is_bit_exact_on_all_frozen_publication_curves():
    result = audit.publication_metric_verification(audit.Evidence())
    assert result["curve_pairs"] == 12
    assert result["historical_repaired_exact"] and result["published_pchip_exact"]
    assert all(result["unchanged_primary_implementations"].values())
    assert all(c["published_guards_equal"] for c in result["checks"])
