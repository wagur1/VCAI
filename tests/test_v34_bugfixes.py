"""Regression coverage for receiver, label-free guards, and controller repairs."""
from copy import deepcopy
from statistics import NormalDist
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_vcm.metrics import pchip_bd
from adaptive_vcm.v34.learning import (
    calibrate, calibrate_mean_ucb, oracle_weights, pinball_loss, relative_damage,
    select_direct_action, targets, targets_per_task, train_controller,
)
from adaptive_vcm.v34.model import Controller, EnsembleController


@pytest.mark.parametrize('help_text,expected', [
    ('-fps_mode mode  set framerate mode', ['-fps_mode', 'passthrough']),
    ('-fps_mode[:<stream_spec>]  set framerate mode', ['-fps_mode', 'passthrough']),
    ('-vsync parameter  video sync method', ['-vsync', '0']),
])
def test_fps_capability_probe_is_cached_by_executable(monkeypatch, help_text, expected):
    from adaptive_vcm.v32 import codec
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=help_text, stderr='')
    monkeypatch.setattr(codec.subprocess, 'run', run)
    codec._fps_sync_flags.cache_clear()
    try:
        assert codec._fps_sync_flags('ffmpeg-a') == expected
        assert codec._fps_sync_flags('ffmpeg-a') == expected
        assert codec._fps_sync_flags('ffmpeg-b') == expected
        assert len(calls) == 2
    finally:
        codec._fps_sync_flags.cache_clear()


def test_fps_probe_failure_is_not_misreported_as_legacy_support(monkeypatch):
    from adaptive_vcm.v32 import codec
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(codec.subprocess, 'run', fail)
    codec._fps_sync_flags.cache_clear()
    with pytest.raises(subprocess.CalledProcessError):
        codec._fps_sync_flags('broken-ffmpeg')


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_label_free_forward_restores_model_and_child_modes(training, fail):
    from adaptive_vcm.v33ap.tasks import od_forward_label_free
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.Dropout())
        def forward(self, images):
            assert not self.training and not self.backbone.training
            self.backbone(torch.stack(images))
            if fail:
                raise RuntimeError('inference failed')
            return [dict(boxes=torch.zeros(0, 4), labels=torch.zeros(0), scores=torch.zeros(0))]
    model = Detector().train(training)
    model.backbone[1].eval()  # A mixed mode must also survive the call.
    states = [m.training for m in model.modules()]
    call = lambda: od_forward_label_free(SimpleNamespace(model=model), torch.zeros(1, 3, 1, 8, 8), [0])
    if fail:
        with pytest.raises(RuntimeError, match='inference failed'):
            call()
    else:
        predictions, features = call()
        assert len(predictions) == 1 and features.shape == (1, 32)
    assert [m.training for m in model.modules()] == states
    assert not model.backbone._forward_hooks


@pytest.mark.parametrize('task,frames,experiment', [('ar', 2, 'ar_od'), ('od', 1, 'ar_od'), ('ot', 2, 'ar_ot')])
def test_source_primary_mask_and_label_free_od_cost(monkeypatch, task, frames, experiment):
    from adaptive_vcm.v34 import tasks, tracking
    heads = tasks.Heads.__new__(tasks.Heads)
    heads.device = 'cpu'; heads.experiment = experiment; heads.cfg = {'domain_primary_mask': True}
    heads.action = heads.detector = None
    prediction = dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0, int))
    heads.detections = lambda rgb: [prediction] * len(rgb)
    monkeypatch.setattr(tasks, 'ar_forward', lambda *args: (torch.zeros(1, 2), torch.ones(1, 2)))
    calls = []
    def forward(detector, x, indices):
        calls.append(tuple(indices))
        return [{k: torch.as_tensor(v) for k, v in prediction.items()} for _ in indices], torch.ones(1, 2)
    monkeypatch.setattr(tasks, 'od_forward_label_free', forward)
    monkeypatch.setattr(tracking, 'track_frames', lambda values: values)
    record = SimpleNamespace(task=task, rgb=np.zeros((frames, 8, 8, 3), np.uint8), targets=None)
    heads.source(record)
    assert record.mask == [task == 'ar', task == 'od', task == 'ot']
    if task == 'od':
        costs = heads.costs(record.rgb, record, source_only=True)
        assert costs[0] is costs[2] is None and costs[1] == pytest.approx(0., abs=1e-8)
        assert len(calls) == 2
    else:
        assert not calls
    heads.cfg = {}
    heads.source(record)
    assert record.mask == [frames > 1, experiment == 'ar_od', experiment == 'ar_ot' and frames > 1]


def test_pchip_uses_rate_ordered_envelope_and_dominated_points_do_not_change_gain():
    rate = np.array([1., 2., 3., 4.])
    quality = np.array([.1, .3, .2, .4])
    original = quality.copy()
    assert pchip_bd(rate, [.1, .3, .3, .4], .8 * rate, quality) == pytest.approx(-20.)
    order = [3, 0, 2, 1]
    assert pchip_bd(rate, [.1, .3, .3, .4], .8 * rate[order], quality[order]) == pytest.approx(-20.)
    np.testing.assert_array_equal(quality, original)


def test_nonmonotone_tracking_keeps_pchip_quality_and_raw_gaps():
    from adaptive_vcm.v33ap.evaluate import safe_curve
    from adaptive_vcm.v34.evaluation import tracking_curve
    a = dict(bpp=[1, 2, 3, 4], quality=[10, 30, 20, 40])
    b = dict(bpp=[.8, 1.6, 2.4, 3.2], quality=[10, 30, 30, 40])
    expected = safe_curve(dict(a, quality=np.array(a['quality']) / 100), dict(b, quality=np.array(b['quality']) / 100))
    actual = tracking_curve(a, b)
    assert actual['bd_rate_pct'] is None and not actual['bd_reliable']
    assert actual['pchip_bd_rate_pct'] == expected['pchip_bd_rate_pct'] == pytest.approx(-20.)
    assert actual['bd_quality_pp'] == expected['bd_quality_pp']
    assert actual['same_qp_gap_pp'] == expected['same_qp_gap_pp']


def test_weighted_oracle_respects_each_tasks_weight_and_inactive_tasks():
    damage = [[0, 0, None], [.04, -.02, None], [-.02, .04, None]]
    args = ([1., .9, .9], damage, [True, True, False], [.05] * 3, 1.01)
    assert targets(*args, weights=[10, 0, 999]) == 2
    assert targets(*args, weights=[0, 10, 999]) == 1
    assert targets(*args) == targets(*args, weights=[.25] * 3)
    assert targets_per_task(*args).tolist() == [2, 1, -1]
    for invalid in ([1, 2], [1, -1, 1], [1, np.inf, 1], [1, np.nan, 1]):
        with pytest.raises(ValueError):
            oracle_weights(invalid)


def test_relative_damage_floors_easy_tasks_without_mutating_or_double_scaling():
    rows = [dict(mask=[True, True, False], cost=[[.0005, .01, None], [.0105, .035, None]],
                 damage=[[0., 0., None], [.01, .025, None]])]
    original = deepcopy(rows)
    repaired = relative_damage(rows)
    assert repaired[0]['damage_reference'] == [1., .25, None]
    assert repaired[0]['damage'][1] == [.01, .1, None]
    assert relative_damage(rows, [1e-3] * 3)[0]['damage'][1][0] == 1000 * repaired[0]['damage'][1][0]
    assert relative_damage(repaired) == repaired and rows == original
    with pytest.raises(ValueError):
        relative_damage(rows, [0, .25, .25])


def test_pinball_loss_has_asymmetric_gradients_and_ignores_inactive_nan():
    prediction = torch.tensor([[[2., 9., -2.]]], requires_grad=True)
    target = torch.tensor([[[0., float('nan'), 0.]]])
    loss = pinball_loss(prediction, target, torch.tensor([[True, False, True]]), .9)
    loss.backward()
    assert loss.item() == pytest.approx(1.)
    assert prediction.grad.flatten().tolist() == pytest.approx([.05, 0., -.45])
    with pytest.raises(ValueError):
        pinball_loss(prediction, target, torch.ones(1, 3, dtype=torch.bool))


def test_mean_ucb_clips_residuals_uses_sem_and_preserves_quantile_default():
    predicted = np.zeros((4, 2, 3)); actual = predicted.copy()
    actual[:, 1, 0] = [-10, 0, 2, 10]
    rates = np.ones((4, 2)); mask = np.tile([True, False, False], (4, 1))
    args = (predicted, actual, rates, rates + .05, mask, .9)
    expected = .25 + NormalDist().inv_cdf(.9) * np.sqrt(11 / 48)
    calibration = calibrate_mean_ucb(*args)
    assert calibration == calibrate(*args, mode='mean_ucb')
    assert calibration['risk'][1][0] == pytest.approx(expected)
    assert calibration['risk'][1][1:] == [1e6, 1e6]
    assert calibration['rate'] == pytest.approx([0., .05])
    assert calibration['risk'][0] == [0.] * 3
    assert calibrate(*args)['risk'][1][0] == 10.
    singleton = calibrate_mean_ucb(predicted[:1], actual[:1], rates[:1], rates[:1], mask[:1], .9)
    assert singleton['risk'][1][0] == 0.
    with pytest.raises(ValueError):
        calibrate(*args, mode='unknown')


def test_direct_selection_gates_logits_by_risk_rate_and_primary_domain():
    out = dict(logits=np.array([0., 100., 2., 3.]),
               logits_task=np.array([[0., 0., 0.], [100., 100., 0.], [2., 3., 0.], [3., 2., 0.]]),
               risk=np.array([[9., 9., 9.], [.1, 0., 0.], [.01, .01, 0.], [.02, .02, 0.]]),
               rate=np.array([9., .7, .8, .9]))
    original = deepcopy(out)
    args = ([True, True, False], [.05] * 3, 1.01)
    assert select_direct_action(out, *args, domain_idx=0) == 3
    assert select_direct_action(out, *args, domain_idx=1) == 2
    assert select_direct_action(out, *args, weights=[0., 1., 0.]) == 2
    out['rate'][2:] = [1.02, np.nan]
    assert select_direct_action(out, *args) == 0
    for key in ('logits', 'logits_task', 'risk'):
        np.testing.assert_array_equal(out[key], original[key])


def test_controller_task_heads_checkpoint_and_ensemble(tmp_path, monkeypatch):
    from adaptive_vcm.v34.checkpoint import load_checkpoint, save_checkpoint
    from adaptive_vcm.v34 import model as module
    pair = torch.rand(2, 2, 3, 48, 48)
    condition = torch.tensor([[0., .7, 1., 0., 0., 1.01], [0., .7, 0., 1., 0., 1.01]])
    def redundant(*args, **kwargs):
        raise AssertionError('48x48 input was interpolated again')
    monkeypatch.setattr(module.F, 'interpolate', redundant)
    members = [Controller(3, task_logits=True).eval() for _ in range(2)]
    outputs = [m(pair, pair, condition) for m in members]
    for value in outputs:
        torch.testing.assert_close(value['logits'][0], value['logits_task'][0, :, 0])
        torch.testing.assert_close(value['logits'][1], value['logits_task'][1, :, 1])
    ensemble = EnsembleController(members)
    for key, value in ensemble(pair, pair, condition).items():
        torch.testing.assert_close(value, torch.stack([out[key] for out in outputs]).mean(0))
    path = tmp_path / 'controller.pth'
    save_checkpoint(path, members[0], {'seed': 1})
    loaded = load_checkpoint(path)
    assert loaded.task_logits
    for key, value in loaded(pair, pair, condition).items():
        torch.testing.assert_close(value, outputs[0][key], rtol=0, atol=0)
    legacy = Controller(3).eval()
    assert legacy.head[-1].out_features == 15 and not legacy.task_logits
    assert set(legacy(pair, pair, condition)) == {'logits', 'risk', 'rate'}
    with pytest.raises(ValueError):
        EnsembleController([])
    with pytest.raises(ValueError):
        EnsembleController([members[0], legacy])


@pytest.mark.parametrize('risk_loss', ['huber', 'pinball'])
def test_task_logits_train_with_each_risk_loss_and_finite_gradients(risk_loss):
    cfg = dict(epochs=1, lr=.001, task_logits=True, risk_loss=risk_loss,
               oracle_damage_weight=[2., .5, .25], damage_tolerance=[.05] * 3, budget_ratio=1.01)
    rows = [dict(id='clip', condition=[0., .7, 1., 0., 0., 1.01], mask=[True, False, False],
                 rate=[1., .9, .95], damage=[[0., None, None], [.01, None, None], [-.05, None, None]])]
    inputs = {'clip': (torch.rand(2, 3, 48, 48), torch.rand(2, 3, 48, 48))}
    model = Controller(3, task_logits=True)
    before = model.head[-1].weight.detach().clone()
    with torch.no_grad():
        initial = model(inputs['clip'][0][None], inputs['clip'][1][None], torch.tensor([rows[0]['condition']]))
        truth = torch.tensor([0., .01, -.05])
        residual = truth - initial['risk'][0, :, 0]
        risk_term = (torch.nn.functional.smooth_l1_loss(initial['risk'][0, :, 0], truth) if risk_loss == 'huber'
                     else torch.maximum(.9 * residual, -.1 * residual).mean())
        expected_loss = (2 * torch.nn.functional.cross_entropy(initial['logits'], torch.tensor([2])) + 4 * risk_term
                         + 4 * torch.nn.functional.mse_loss(initial['rate'], torch.tensor([[1., .9, .95]])))
    history = train_controller(model, rows, inputs, cfg, 'cpu')
    assert np.isfinite(history[0]['loss']) and not model.training
    assert history[0]['loss'] == pytest.approx(expected_loss.item())
    assert not torch.equal(before, model.head[-1].weight)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('seed,expected', [('forced', [0, 2, 3]), ('rank', [0, 1, 2])])
def test_choose_packet_shortlist_seed_order(monkeypatch, seed, expected):
    from adaptive_vcm.v34 import evaluation
    prediction = dict(logits=np.array([0., 0., 5., 0.]), risk=np.zeros((4, 3)), rate=np.array([1., .95, .8, .7]))
    monkeypatch.setattr(evaluation, 'infer', lambda *args: prediction)
    model = SimpleNamespace(metadata={'calibration': {'risk': [[0.] * 3, [0.] * 3, [1.] * 3, [1.] * 3], 'rate': [0.] * 4}})
    packets = [SimpleNamespace(coded_bytes=s, decoded=np.full((1, 1, 1, 3), i)) for i, s in enumerate([100, 95, 80, 70])]
    cfg = dict(damage_tolerance=[.05] * 3, budget_ratio=1.01, source_guard_limits=[.05] * 3,
               source_shortlist=2, shortlist_seed=seed)
    result = evaluation.choose_packet(model, None, 'h264', 35, [True, False, False], cfg, packets[0],
                                      lambda i: packets[i], source_score=lambda rgb: [0., None, None])
    assert result[3]['tried_actions'] == expected
    assert result[3]['calibrated_action'] == 1


def test_od_runner_caches_label_free_guard_and_records_all_three_arms(monkeypatch, tmp_path):
    import json
    from scripts import run_v34_od as runner
    from adaptive_vcm.v32.codec import Packet
    from adaptive_vcm.v33ap.sources import SourceRecord
    cfg = dict(schema='adaptive-vcm-v34-1', seeds=[303401, 303402], codecs=['h264'], qps=[30], preset='medium',
               fixed_action=11, budget_ratio=1.01, damage_tolerance=[.05] * 3, source_guard_limits=[.025, .05, .05],
               source_shortlist=2)
    rgb = np.zeros((1, 8, 8, 3), np.uint8)
    groups = {split: [] for split in ('train', 'cal', 'dev', 'test')}
    for split in ('dev', 'test'):
        record = SourceRecord(split, 'od', dict(image_id=1, label='held-out'), rgb, (1, 1, 0, 0), {})
        record.targets = {'held-out': 'must not read'}
        groups[split] = [record]
    monkeypatch.setattr(runner, 'settings', lambda path: cfg.copy())
    monkeypatch.setattr(runner, 'load_cohorts', lambda *args: (groups, {'annotations': []}))
    calls = []
    class Heads:
        def __init__(self, *args):
            self.detector = SimpleNamespace(predict=lambda rgb: {})
        def hashes(self):
            return {}
        def source(self, record):
            assert record.targets is None and 'label' not in record.item
            record.mask = [False, True, False]
            record.source_predictions = {'od': []}
        def costs(self, pixels, record, source_only=False):
            assert source_only and record.targets is None and 'label' not in record.item
            calls.append((record.id, int(pixels[0, 0, 0, 0])))
            return [None, 0., None]
    class Model(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))
            self.metadata = {'seed': seed, 'calibration': dict(risk=[[0.] * 3] * 13, rate=[0.] * 13)}
        def forward(self, *args):
            logits = torch.zeros(1, 13); logits[0, 1] = 10.
            rate = torch.ones(1, 13) * 2; rate[0, :3] = torch.tensor([1., .9, .8])
            return dict(logits=logits, risk=torch.zeros(1, 13, 3), rate=rate)
    def roundtrip(pixels, **kwargs):
        action = int(pixels[0, 0, 0, 0])
        size = {0: 100, 1: 90, 2: 80}.get(action, 110)
        return Packet(bytes([action]) * size, pixels, pixels.shape, size - 10, 10, 0.)
    monkeypatch.setattr(runner, 'Heads', Heads)
    monkeypatch.setattr(runner, 'load_checkpoint', lambda path, device: Model(int(path.stem.split('_')[-1])))
    monkeypatch.setattr(runner, 'region_maps', lambda rgb, pred: np.zeros_like(rgb, dtype=np.float32))
    monkeypatch.setattr(runner, 'filter_clip', lambda rgb, maps, action: np.full_like(rgb, runner.ACTIONS.index(action)))
    monkeypatch.setattr(runner, 'codec_factory', lambda *args: SimpleNamespace(roundtrip=roundtrip))
    monkeypatch.setattr(runner, 'coco_predictions', lambda *args: [])
    monkeypatch.setattr(runner, 'compute_od_summary', lambda *args: {})
    monkeypatch.setattr(runner, 'generate_report', lambda *args: 'fixture')
    (tmp_path / 'measurements.json').write_text('{}')
    for seed in cfg['seeds']:
        (tmp_path / f'controller_{seed}.pth').write_bytes(b'fixture')
    args = SimpleNamespace(config='fixture', train_count=0, cal_count=0, dev_count=1, test_count=1,
                           out=str(tmp_path), device='cpu', od_root='fixture', annotations='fixture',
                           frozen_checkpoint=str(tmp_path / 'absent.pth'))
    runner.run_benchmark(args)
    assert sorted(calls) == [(split, action) for split in ('dev', 'test') for action in (0, 1, 2)]
    for split in ('dev', 'test'):
        rows = json.loads((tmp_path / f'{split}_rows.json').read_text())
        arms = {r['arm']: r for r in rows}
        for seed in cfg['seeds']:
            assert arms[f'forced_{seed}']['action'] == 1
            assert arms[f'calibrated_{seed}']['action'] == 2
            assert arms[f'selected_{seed}']['action'] == 2
            assert arms[f'selected_{seed}']['source_guard']['limits'] == [.025, .05, .05]
