# V34 verification and experimental audit

Publication data, checkpoints, and current runtime hashes verified before and after the audit. Historical source is read from Git.

This is a saved-artifact audit and deterministic replay. Missing measurements remain unavailable. The requested findings are hypotheses checked against evidence, not assumptions.

## Scope corrections

- Historical learning.py uses scaled Smooth L1 (MSE without scale); later checkpoints use pinball. The repaired workspace supports both losses and preserves the historical default. The supplied sources do not establish the claimed 1.5 × rank loss.
- Action 11 is global filtering. Saved action 5 differs from the coefficients in the request; replay uses the actual saved action.
- Full-bank quality is negative GT-conditioned task cost, not accuracy, mAP, or official tracking metrics. Its cost-delta guard is a supervised surrogate.
- Later selected runs use five nonidentity candidates. Task 2 bank ablations use three; Task 5 replays five. The native Controller supports both the frozen 65-value head and the later 91-value task-logit head.
- No fifth QP is invented. Each five-QP result is unavailable unless all five points exist.
- MOT17 input lists 02/05 FIT and 04/11 reused CAL. It does not establish the requested held-out 02/05/09/11, 120-window benchmark.
- MOT20-05 LaTeX has rounded HOTA/rates and only IDF1/MOTA deltas. Exact MOTA divergence cannot be reconstructed from it.

## Task 1: seven findings

### od_source_only: confirmed

Structured evidence is retained in audit_summary.json.

Historical finding retained; `fixed_in_workspace: true`.

Original dummy OD call: `{'outcome': 'error', 'exception': 'AttributeError', 'message': "'NoneType' object has no attribute 'items'", 'calls': []}`. Fixed call: `{'outcome': 'returned', 'value': [None, 0.125, None], 'calls': ['od_forward_label_free', 'detection_distance']}`.

### historical_od_selected: confirmed

Historical selected used calibrated select_action fallback. The repaired runner passes a cached label-free source guard and records calibrated arms separately.

Historical finding retained; `fixed_in_workspace: true`.

### conformal_collapse: dataset_specific

Historical calibrated counts are retained. Quantile calibration remains the default; the workspace adds clipped mean-risk UCB calibration. Task 5 measures the policy change on saved banks.

Historical finding retained; `fixed_in_workspace: true`.

### fixed_is_global: confirmed

Action 11 is mask-independent global two-Gaussian frequency attenuation (B1).

Maximum pixel differences for different masks: `{'0': 0.0, '5': 50.0, '6': 58.0, '11': 0.0}`.

### direct_controller_collapse: measured

Historical forced action 11 is B1 behavior. The workspace adds task-logit selection gated by predicted risk/rate. Frozen risk heads can still reject all nonidentity actions; Task 5 reports this without retraining.

Historical finding retained; `fixed_in_workspace: true`.

### paper_formula_latency: partly_contradicted

Paper Smooth L1 agrees with frozen scaled-risk training. Pinball belongs to later variants; 1.5*rank_loss must be verified in the actual source, not asserted. Measured latency is in task3.

### ffmpeg_vsync: probed

Structured evidence is retained in audit_summary.json.

Historical finding retained; `fixed_in_workspace: true`.

ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers. Legacy return code 2880417800; replacement return code 0.

### Action counts and shortlist collapse

| Dataset/task/arm | Rows | Identity % | Action counts | Most frequent shortlist (count) |
|---|---:|---:|---|---|
| k400_publication/ar/calibrated_303401 | 10240 | 100.0000 | 0: 10240 | N/A (0) |
| k400_publication/ar/calibrated_303402 | 10240 | 100.0000 | 0: 10240 | N/A (0) |
| k400_publication/ar/calibrated_303403 | 10240 | 100.0000 | 0: 10240 | N/A (0) |
| k400_publication/ar/forced_303401 | 10240 | 2.6270 | 0: 269, 6: 4, 11: 9967 | [0, 11, 6, 5] (9642) |
| k400_publication/ar/forced_303402 | 10240 | 8.2324 | 0: 843, 1: 40, 6: 618, 11: 8739 | [0, 11, 6, 5] (7458) |
| k400_publication/ar/forced_303403 | 10240 | 0.0000 | 6: 1643, 11: 8597 | [0, 11, 6, 5] (8106) |
| k400_publication/ar/selected_303401 | 10240 | 36.5527 | 0: 3743, 1: 10, 2: 60, 3: 21, 4: 36, 5: 1073, 6: 2315, 8: 34, 10: 10, 11: 2938 | [0, 11, 6, 5] (9642) |
| k400_publication/ar/selected_303402 | 10240 | 36.7285 | 0: 3761, 1: 12, 2: 94, 3: 9, 4: 180, 5: 967, 6: 2200, 7: 13, 8: 162, 10: 29, 11: 2813 | [0, 11, 6, 5] (7458) |
| k400_publication/ar/selected_303403 | 10240 | 36.4844 | 0: 3736, 2: 1, 3: 10, 4: 139, 5: 1069, 6: 2347, 11: 2938 | [0, 11, 6, 5] (8106) |
| coco_historical/od/forced_303401 | 640 | 0.0000 | 4: 110, 6: 219, 11: 311 | [0] (640) |
| coco_historical/od/forced_303402 | 640 | 0.0000 | 4: 161, 6: 219, 11: 260 | [0] (640) |
| coco_historical/od/forced_303403 | 640 | 0.0000 | 4: 197, 6: 148, 11: 295 | [0] (640) |
| coco_historical/od/selected_303401 | 640 | 100.0000 | 0: 640 | [0] (640) |
| coco_historical/od/selected_303402 | 640 | 100.0000 | 0: 640 | [0] (640) |
| coco_historical/od/selected_303403 | 640 | 100.0000 | 0: 640 | [0] (640) |
| scaled_test/ar/calibrated_303401 | 1024 | 100.0000 | 0: 1024 | N/A (0) |
| scaled_test/ar/calibrated_303402 | 1024 | 100.0000 | 0: 1024 | N/A (0) |
| scaled_test/ar/calibrated_303403 | 1024 | 100.0000 | 0: 1024 | N/A (0) |
| scaled_test/ar/forced_303401 | 1024 | 36.0352 | 0: 369, 3: 8, 6: 375, 11: 272 | [0, 1, 12, 7, 10, 9] (48) |
| scaled_test/ar/forced_303402 | 1024 | 54.9805 | 0: 563, 3: 1, 4: 8, 6: 74, 11: 378 | [0, 12, 11, 1, 10, 9] (32) |
| scaled_test/ar/forced_303403 | 1024 | 61.4258 | 0: 629, 3: 17, 4: 5, 6: 159, 8: 4, 11: 210 | [0, 12, 11, 10, 1, 9] (63) |
| scaled_test/ar/selected_303401 | 1024 | 50.4883 | 0: 517, 1: 124, 2: 49, 3: 88, 5: 34, 6: 45, 7: 22, 9: 78, 10: 32, 11: 29, 12: 6 | [0, 1, 12, 7, 10, 9] (48) |
| scaled_test/ar/selected_303402 | 1024 | 49.3164 | 0: 505, 1: 130, 2: 56, 3: 56, 5: 9, 6: 60, 7: 41, 9: 86, 10: 29, 11: 46, 12: 6 | [0, 12, 11, 1, 10, 9] (32) |
| scaled_test/ar/selected_303403 | 1024 | 51.0742 | 0: 523, 1: 113, 2: 76, 3: 74, 5: 32, 6: 17, 7: 19, 9: 87, 10: 32, 11: 46, 12: 5 | [0, 12, 11, 10, 1, 9] (63) |
| scaled_test/od/calibrated_303401 | 1024 | 99.1211 | 0: 1015, 8: 1, 12: 8 | N/A (0) |
| scaled_test/od/calibrated_303402 | 1024 | 100.0000 | 0: 1024 | N/A (0) |
| scaled_test/od/calibrated_303403 | 1024 | 99.6094 | 0: 1020, 9: 4 | N/A (0) |
| scaled_test/od/forced_303401 | 1024 | 0.0000 | 2: 3, 4: 182, 6: 348, 8: 34, 11: 457 | [0, 11, 6, 5, 3, 9] (89) |
| scaled_test/od/forced_303402 | 1024 | 0.0000 | 2: 5, 4: 211, 6: 367, 8: 21, 11: 420 | [0, 11, 6, 5, 4, 1] (114) |
| scaled_test/od/forced_303403 | 1024 | 0.0000 | 4: 225, 6: 344, 8: 20, 11: 435 | [0, 11, 6, 5, 3, 4] (130) |
| scaled_test/od/selected_303401 | 1024 | 4.1016 | 0: 42, 2: 22, 3: 37, 4: 137, 5: 28, 6: 347, 7: 2, 8: 8, 9: 4, 11: 397 | [0, 11, 6, 5, 3, 9] (89) |
| scaled_test/od/selected_303402 | 1024 | 3.6133 | 0: 37, 1: 2, 2: 15, 3: 14, 4: 199, 5: 27, 6: 320, 7: 2, 8: 10, 9: 2, 11: 396 | [0, 11, 6, 5, 4, 1] (114) |
| scaled_test/od/selected_303403 | 1024 | 3.9062 | 0: 40, 2: 10, 3: 28, 4: 205, 5: 28, 6: 316, 7: 1, 8: 4, 9: 2, 11: 390 | [0, 11, 6, 5, 3, 4] (130) |
| coco_rank/od/calibrated_303401 | 2048 | 98.9746 | 0: 2027, 4: 1, 8: 13, 9: 1, 12: 6 | N/A (0) |
| coco_rank/od/calibrated_303402 | 2048 | 99.7559 | 0: 2043, 9: 5 | N/A (0) |
| coco_rank/od/calibrated_303403 | 2048 | 99.3652 | 0: 2035, 12: 13 | N/A (0) |
| coco_rank/od/forced_303401 | 2048 | 0.5859 | 0: 12, 2: 1, 4: 352, 6: 1046, 8: 1, 11: 636 | [0, 6, 11, 4, 3, 5] (92) |
| coco_rank/od/forced_303402 | 2048 | 1.9531 | 0: 40, 2: 6, 4: 350, 6: 794, 11: 858 | [0, 11, 6, 5, 3, 4] (217) |
| coco_rank/od/forced_303403 | 2048 | 0.5859 | 0: 12, 2: 5, 4: 331, 6: 977, 11: 723 | [0, 11, 6, 5, 4, 2] (147) |
| coco_rank/od/selected_303401 | 2048 | 4.2480 | 0: 87, 1: 10, 2: 21, 3: 65, 4: 387, 5: 67, 6: 634, 7: 5, 8: 12, 9: 17, 11: 743 | [0, 6, 11, 4, 3, 5] (92) |
| coco_rank/od/selected_303402 | 2048 | 4.3457 | 0: 89, 1: 1, 2: 25, 3: 60, 4: 421, 5: 83, 6: 642, 8: 13, 9: 2, 11: 712 | [0, 11, 6, 5, 3, 4] (217) |
| coco_rank/od/selected_303403 | 2048 | 4.4434 | 0: 91, 1: 3, 2: 27, 3: 56, 4: 425, 5: 67, 6: 630, 7: 1, 8: 7, 9: 4, 11: 737 | [0, 11, 6, 5, 4, 2] (147) |
| ot_test/ot/calibrated_303401 | 128 | 100.0000 | 0: 128 | N/A (0) |
| ot_test/ot/calibrated_303402 | 128 | 100.0000 | 0: 128 | N/A (0) |
| ot_test/ot/calibrated_303403 | 128 | 100.0000 | 0: 128 | N/A (0) |
| ot_test/ot/forced_303401 | 128 | 0.0000 | 11: 128 | [0, 11, 5, 9, 10, 7] (31) |
| ot_test/ot/forced_303402 | 128 | 0.0000 | 11: 128 | [0, 11, 9, 10, 1, 5] (32) |
| ot_test/ot/forced_303403 | 128 | 0.0000 | 11: 128 | [0, 11, 10, 5, 6, 1] (64) |
| ot_test/ot/selected_303401 | 128 | 6.2500 | 0: 8, 1: 7, 3: 4, 5: 17, 6: 36, 9: 1, 11: 55 | [0, 11, 5, 9, 10, 7] (31) |
| ot_test/ot/selected_303402 | 128 | 5.4688 | 0: 7, 1: 2, 3: 6, 5: 22, 6: 3, 7: 1, 9: 10, 10: 3, 11: 74 | [0, 11, 9, 10, 1, 5] (32) |
| ot_test/ot/selected_303403 | 128 | 7.0312 | 0: 9, 1: 6, 5: 13, 6: 36, 7: 1, 9: 5, 10: 3, 11: 55 | [0, 11, 10, 5, 6, 1] (64) |
| ot_dev/ot/calibrated_303401 | 128 | 100.0000 | 0: 128 | N/A (0) |
| ot_dev/ot/calibrated_303402 | 128 | 100.0000 | 0: 128 | N/A (0) |
| ot_dev/ot/calibrated_303403 | 128 | 100.0000 | 0: 128 | N/A (0) |
| ot_dev/ot/forced_303401 | 128 | 0.0000 | 11: 128 | [0, 11, 5, 9, 10, 7] (27) |
| ot_dev/ot/forced_303402 | 128 | 0.0000 | 11: 128 | [0, 11, 9, 10, 1, 5] (31) |
| ot_dev/ot/forced_303403 | 128 | 0.0000 | 11: 128 | [0, 11, 10, 5, 6, 1] (34) |
| ot_dev/ot/selected_303401 | 128 | 9.3750 | 0: 12, 1: 2, 5: 14, 6: 16, 9: 3, 11: 81 | [0, 11, 5, 9, 10, 7] (27) |
| ot_dev/ot/selected_303402 | 128 | 11.7188 | 0: 15, 1: 2, 3: 1, 5: 22, 9: 4, 11: 84 | [0, 11, 9, 10, 1, 5] (31) |
| ot_dev/ot/selected_303403 | 128 | 10.9375 | 0: 14, 1: 4, 5: 10, 6: 16, 9: 3, 11: 81 | [0, 11, 10, 5, 6, 1] (34) |

## Task 2: ablation definitions and measurement-bank results

Exact stored bytes and GT-conditioned task-cost replay. Quality=-cost; BD-quality in cost units. CAL held out from gradient training but used for calibration. ALL includes TRAIN.

User-requested cost[a][domain]-cost[0][domain] replay at AR .025, OD/OT .05; these measurement costs include labels/GT and are NOT the label-free source guard costs from held-out selected rows.

| Baseline | Definition |
|---|---|
| B0 | Identity, action 0 |
| B1 | Global Gaussian frequency attenuation, action 11 |
| B2_a6 | Fixed region-aware action 6 from the saved action bank |
| B2_a5 | Fixed region-aware action 5 from the saved action bank |
| B3 | forced_303401; direct argmax(logits) |
| B4 | 3 distinct random nonidentity candidates + identity; RNG seeds 0..19 |
| B5a | 3 model-ranked candidates + identity; minimum measured rate <= 1.01; no guard |
| B5b | calibrated_303401; calibrated risk/rate bounds, no source guard |
| Full | selected_303401; model candidates + codec trials + guard |

| Bank | Unique sources | Conditions by task/split | Missing controller inputs |
|---|---:|---|---:|
| scaled | 176 | {'ar/cal': 192, 'ar/train': 512, 'od/cal': 192, 'od/train': 512} | 0 |
| coco256_rank | 256 | {'od/cal': 512, 'od/train': 1536} | 168 |
| coco256_scaled_controller | 256 | {'od/cal': 512, 'od/train': 1536} | 168 |
| MOT20-02 | 16 | {'ot/cal': 32, 'ot/train': 96} | 0 |

Seed means average the QP curves first. The JSON also records the mean of independently computed seed BD-rates. B4 averages 20 deterministic random-policy curves; individual RNG-seed results remain in JSON.

| Dataset | Task/codec/metric | Arm | QPs | Cubic rate % | PCHIP rate % | Cubic quality | PCHIP quality | Reliable |
|---|---|---|---:|---:|---:|---:|---:|---|
| scaled/cal | ar/h264/negative_task_cost | B2_a5 | 4 | -33.0374 | -16.6282 | 0.2036 | 0.2483 | False |
| scaled/cal | ar/h264/negative_task_cost | B2_a6 | 4 | -2.8131 | -0.6079 | 0.0288 | 0.0156 | True |
| scaled/cal | ar/h264/negative_task_cost | B4 | 4 | -18.6503 | -31.6643 | 0.4647 | 0.4837 | False |
| scaled/cal | ar/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h264/negative_task_cost | fixed | 4 | -11.0224 | -17.0739 | 0.2623 | 0.2537 | False |
| scaled/cal | ar/h264/negative_task_cost | forced_303401 | 4 | 3.5984 | -11.3256 | 0.1573 | 0.1693 | False |
| scaled/cal | ar/h264/negative_task_cost | forced_303402 | 4 | 9.2706 | -13.8292 | 0.2035 | 0.2143 | False |
| scaled/cal | ar/h264/negative_task_cost | forced_303403 | 4 | -4.8862 | -7.7636 | 0.1229 | 0.1177 | True |
| scaled/cal | ar/h264/negative_task_cost | forced_mean | 4 | 2.1823 | -10.8391 | 0.1607 | 0.1670 | False |
| scaled/cal | ar/h264/negative_task_cost | oracle | 4 | -23.4481 | -38.9092 | 0.6308 | 0.6503 | False |
| scaled/cal | ar/h264/negative_task_cost | selected_303401 | 4 | 3.0853 | -32.5394 | 0.4721 | 0.4958 | False |
| scaled/cal | ar/h264/negative_task_cost | selected_303402 | 4 | 7.7487 | -32.4992 | 0.4553 | 0.4725 | False |
| scaled/cal | ar/h264/negative_task_cost | selected_303403 | 4 | -6.1100 | -32.4366 | 0.4655 | 0.4924 | False |
| scaled/cal | ar/h264/negative_task_cost | selected_mean | 4 | 2.0736 | -32.5124 | 0.4645 | 0.4871 | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_303401 | 4 | -77.3991 | -14.4057 | 0.1539 | 0.1958 | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_303402 | 4 | 90.7915 | -16.0711 | 0.1784 | 0.2143 | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_303403 | 4 | -27.3201 | -15.8000 | 0.1964 | 0.2326 | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_mean | 4 | -32.3419 | -15.2083 | 0.1760 | 0.2145 | False |
| scaled/cal | ar/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | B2_a5 | 4 | 85.0516 | 1.6033 | -0.0517 | -0.0375 | False |
| scaled/cal | ar/h265/negative_task_cost | B2_a6 | 4 | 24.2894 | 0.2817 | 0.0127 | 0.0128 | False |
| scaled/cal | ar/h265/negative_task_cost | B4 | 4 | 11.9155 | -15.9186 | 0.3385 | 0.3327 | False |
| scaled/cal | ar/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h265/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h265/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | ar/h265/negative_task_cost | fixed | 4 | -22.3402 | -2.7344 | 0.0081 | 0.0694 | False |
| scaled/cal | ar/h265/negative_task_cost | forced_303401 | 4 | 9.1298 | -3.0261 | 0.0899 | 0.0811 | False |
| scaled/cal | ar/h265/negative_task_cost | forced_303402 | 4 | 16.0611 | -0.0627 | -0.0011 | 0.0031 | False |
| scaled/cal | ar/h265/negative_task_cost | forced_303403 | 4 | 27.9140 | 0.5019 | -0.0049 | -0.0076 | False |
| scaled/cal | ar/h265/negative_task_cost | forced_mean | 4 | 16.4512 | -0.8393 | 0.0288 | 0.0258 | False |
| scaled/cal | ar/h265/negative_task_cost | oracle | 4 | 56.5487 | -21.4278 | 0.4784 | 0.4737 | False |
| scaled/cal | ar/h265/negative_task_cost | selected_303401 | 4 | -30.0488 | -12.9712 | 0.2593 | 0.2565 | False |
| scaled/cal | ar/h265/negative_task_cost | selected_303402 | 4 | 50.6404 | -14.0544 | 0.3214 | 0.3046 | False |
| scaled/cal | ar/h265/negative_task_cost | selected_303403 | 4 | 18.8078 | -13.1839 | 0.2791 | 0.2740 | False |
| scaled/cal | ar/h265/negative_task_cost | selected_mean | 4 | 11.5949 | -13.4069 | 0.2864 | 0.2796 | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_303401 | 4 | 15.7333 | 8.6293 | -0.1914 | -0.1880 | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_303402 | 4 | -18.3988 | 2.8905 | -0.1056 | -0.0751 | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_303403 | 4 | -39.7764 | 4.2590 | -0.1317 | -0.1077 | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_mean | 4 | -4.5670 | 5.2030 | -0.1432 | -0.1233 | False |
| scaled/cal | ar/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | ar/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | B2_a5 | 4 | 25.5910 | 16.8596 | -0.0353 | -0.0353 | False |
| scaled/cal | od/h264/negative_task_cost | B2_a6 | 4 | 25.8901 | 19.5419 | -0.0521 | -0.0529 | False |
| scaled/cal | od/h264/negative_task_cost | B4 | 4 | -21.2743 | -22.1047 | 0.0588 | 0.0594 | True |
| scaled/cal | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h264/negative_task_cost | fixed | 4 | 19.2611 | 11.7633 | -0.0237 | -0.0241 | False |
| scaled/cal | od/h264/negative_task_cost | forced_303401 | 4 | 21.6283 | 22.9752 | -0.0568 | -0.0534 | True |
| scaled/cal | od/h264/negative_task_cost | forced_303402 | 4 | -13.0143 | 5.3685 | -0.0154 | -0.0122 | False |
| scaled/cal | od/h264/negative_task_cost | forced_303403 | 4 | 12.5024 | 19.6683 | -0.0471 | -0.0440 | False |
| scaled/cal | od/h264/negative_task_cost | forced_mean | 4 | 7.2071 | 15.8272 | -0.0398 | -0.0366 | False |
| scaled/cal | od/h264/negative_task_cost | oracle | 4 | -93.8408 | -28.4138 | 0.0740 | 0.0757 | False |
| scaled/cal | od/h264/negative_task_cost | selected_303401 | 4 | -35.4558 | -22.7263 | 0.0631 | 0.0645 | False |
| scaled/cal | od/h264/negative_task_cost | selected_303402 | 4 | -95.3786 | -23.1273 | 0.0648 | 0.0665 | False |
| scaled/cal | od/h264/negative_task_cost | selected_303403 | 4 | -91.3967 | -26.5721 | 0.0722 | 0.0730 | False |
| scaled/cal | od/h264/negative_task_cost | selected_mean | 4 | -70.0762 | -24.0513 | 0.0667 | 0.0681 | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_303401 | 4 | 37.3563 | 28.9485 | -0.0710 | -0.0688 | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_303402 | 4 | 4.8346 | 17.5029 | -0.0419 | -0.0386 | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_303403 | 4 | 23.9457 | 22.0793 | -0.0529 | -0.0500 | True |
| scaled/cal | od/h264/negative_task_cost | unchecked_mean | 4 | 26.2663 | 22.8496 | -0.0552 | -0.0524 | True |
| scaled/cal | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | B2_a5 | 4 | -27.4913 | 6.5691 | -0.0382 | -0.0334 | False |
| scaled/cal | od/h265/negative_task_cost | B2_a6 | 4 | 1.2400 | 11.2559 | -0.0638 | -0.0607 | False |
| scaled/cal | od/h265/negative_task_cost | B4 | 4 | -5.1117 | -14.4708 | 0.0690 | 0.0684 | False |
| scaled/cal | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h265/negative_task_cost | calibrated_303401 | 4 | 0.0925 | -1.2717 | 0.0071 | 0.0070 | True |
| scaled/cal | od/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/cal | od/h265/negative_task_cost | calibrated_mean | 4 | 0.0638 | -0.4212 | 0.0023 | 0.0023 | True |
| scaled/cal | od/h265/negative_task_cost | fixed | 4 | -3.1229 | 1.7107 | -0.0060 | -0.0082 | True |
| scaled/cal | od/h265/negative_task_cost | forced_303401 | 4 | 2.5775 | 12.8652 | -0.0752 | -0.0731 | False |
| scaled/cal | od/h265/negative_task_cost | forced_303402 | 4 | 0.2007 | 10.4842 | -0.0676 | -0.0645 | False |
| scaled/cal | od/h265/negative_task_cost | forced_303403 | 4 | 1.1373 | 10.4742 | -0.0662 | -0.0631 | False |
| scaled/cal | od/h265/negative_task_cost | forced_mean | 4 | 1.2803 | 11.2946 | -0.0696 | -0.0669 | False |
| scaled/cal | od/h265/negative_task_cost | oracle | 4 | -15.6882 | -16.9235 | 0.0818 | 0.0824 | True |
| scaled/cal | od/h265/negative_task_cost | selected_303401 | 4 | -12.0946 | -12.8504 | 0.0639 | 0.0647 | True |
| scaled/cal | od/h265/negative_task_cost | selected_303402 | 4 | -13.9354 | -14.2830 | 0.0722 | 0.0713 | True |
| scaled/cal | od/h265/negative_task_cost | selected_303403 | 4 | -5.2333 | -13.4001 | 0.0725 | 0.0717 | False |
| scaled/cal | od/h265/negative_task_cost | selected_mean | 4 | -11.3398 | -13.5377 | 0.0695 | 0.0693 | True |
| scaled/cal | od/h265/negative_task_cost | unchecked_303401 | 4 | 3.4337 | 13.7285 | -0.0757 | -0.0757 | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_303402 | 4 | 0.0757 | 8.1567 | -0.0526 | -0.0535 | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_303403 | 4 | -3.8173 | 3.7743 | -0.0245 | -0.0262 | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_mean | 4 | -0.3456 | 8.4100 | -0.0510 | -0.0515 | False |
| scaled/cal | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/cal | od/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | B2_a5 | 4 | -4.0945 | -6.7290 | 0.0687 | 0.0854 | True |
| scaled/all | ar/h264/negative_task_cost | B2_a6 | 4 | -0.0221 | 0.2694 | -0.0389 | -0.0276 | True |
| scaled/all | ar/h264/negative_task_cost | B4 | 4 | -20.8216 | -24.7373 | 0.3836 | 0.3903 | True |
| scaled/all | ar/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h264/negative_task_cost | fixed | 4 | -7.0620 | -7.1943 | 0.1263 | 0.1281 | True |
| scaled/all | ar/h264/negative_task_cost | forced_303401 | 4 | -4.1355 | -4.4364 | 0.0662 | 0.0705 | True |
| scaled/all | ar/h264/negative_task_cost | forced_303402 | 4 | -4.2959 | -5.6346 | 0.0921 | 0.0930 | True |
| scaled/all | ar/h264/negative_task_cost | forced_303403 | 4 | -3.6192 | -4.2307 | 0.0768 | 0.0703 | True |
| scaled/all | ar/h264/negative_task_cost | forced_mean | 4 | -3.8864 | -4.7466 | 0.0787 | 0.0794 | True |
| scaled/all | ar/h264/negative_task_cost | oracle | 4 | -27.0511 | -31.5746 | 0.5214 | 0.5279 | True |
| scaled/all | ar/h264/negative_task_cost | selected_303401 | 4 | -17.3970 | -22.4797 | 0.3432 | 0.3476 | False |
| scaled/all | ar/h264/negative_task_cost | selected_303402 | 4 | -19.8598 | -22.2356 | 0.3270 | 0.3355 | True |
| scaled/all | ar/h264/negative_task_cost | selected_303403 | 4 | -17.7754 | -23.6244 | 0.3489 | 0.3613 | False |
| scaled/all | ar/h264/negative_task_cost | selected_mean | 4 | -18.4359 | -22.9857 | 0.3399 | 0.3485 | True |
| scaled/all | ar/h264/negative_task_cost | unchecked_303401 | 4 | -6.5155 | -6.8701 | 0.0689 | 0.0828 | True |
| scaled/all | ar/h264/negative_task_cost | unchecked_303402 | 4 | -5.8468 | -6.5993 | 0.0709 | 0.0826 | True |
| scaled/all | ar/h264/negative_task_cost | unchecked_303403 | 4 | -8.2009 | -6.2362 | 0.0569 | 0.0727 | True |
| scaled/all | ar/h264/negative_task_cost | unchecked_mean | 4 | -6.7147 | -6.5593 | 0.0656 | 0.0794 | True |
| scaled/all | ar/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | B2_a5 | 4 | 6.4485 | 3.0333 | -0.0806 | -0.0666 | True |
| scaled/all | ar/h265/negative_task_cost | B2_a6 | 4 | 10.9576 | 1.8073 | -0.0169 | -0.0272 | False |
| scaled/all | ar/h265/negative_task_cost | B4 | 4 | -15.2030 | -17.1185 | 0.3476 | 0.3446 | True |
| scaled/all | ar/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h265/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h265/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | ar/h265/negative_task_cost | fixed | 4 | 5.4113 | 0.6295 | -0.0241 | -0.0026 | True |
| scaled/all | ar/h265/negative_task_cost | forced_303401 | 4 | 1.6797 | -2.9380 | 0.0759 | 0.0720 | True |
| scaled/all | ar/h265/negative_task_cost | forced_303402 | 4 | 2.5676 | 0.0705 | 0.0008 | 0.0002 | True |
| scaled/all | ar/h265/negative_task_cost | forced_303403 | 4 | 2.4268 | -0.6923 | 0.0245 | 0.0189 | True |
| scaled/all | ar/h265/negative_task_cost | forced_mean | 4 | 2.2042 | -1.1898 | 0.0335 | 0.0300 | True |
| scaled/all | ar/h265/negative_task_cost | oracle | 4 | -22.6344 | -23.9854 | 0.4815 | 0.4907 | True |
| scaled/all | ar/h265/negative_task_cost | selected_303401 | 4 | -16.2835 | -14.1609 | 0.2762 | 0.2744 | True |
| scaled/all | ar/h265/negative_task_cost | selected_303402 | 4 | -16.5521 | -16.0939 | 0.3197 | 0.3181 | True |
| scaled/all | ar/h265/negative_task_cost | selected_303403 | 4 | -12.9677 | -13.7991 | 0.2735 | 0.2662 | True |
| scaled/all | ar/h265/negative_task_cost | selected_mean | 4 | -15.1596 | -14.7073 | 0.2898 | 0.2863 | True |
| scaled/all | ar/h265/negative_task_cost | unchecked_303401 | 4 | 4.2531 | 2.9963 | -0.0670 | -0.0643 | True |
| scaled/all | ar/h265/negative_task_cost | unchecked_303402 | 4 | -1.2314 | 1.5945 | -0.0470 | -0.0371 | True |
| scaled/all | ar/h265/negative_task_cost | unchecked_303403 | 4 | 6.5825 | 3.0870 | -0.0546 | -0.0611 | True |
| scaled/all | ar/h265/negative_task_cost | unchecked_mean | 4 | 3.7038 | 2.5530 | -0.0561 | -0.0541 | True |
| scaled/all | ar/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | ar/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | B2_a5 | 4 | 2.7816 | 2.2832 | -0.0036 | -0.0039 | True |
| scaled/all | od/h264/negative_task_cost | B2_a6 | 4 | 3.8801 | 3.5109 | -0.0072 | -0.0071 | True |
| scaled/all | od/h264/negative_task_cost | B4 | 4 | -25.5219 | -26.1453 | 0.0674 | 0.0676 | True |
| scaled/all | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h264/negative_task_cost | fixed | 4 | 3.4456 | 3.3028 | -0.0062 | -0.0064 | True |
| scaled/all | od/h264/negative_task_cost | forced_303401 | 4 | -2.0985 | 1.3128 | -0.0060 | -0.0047 | True |
| scaled/all | od/h264/negative_task_cost | forced_303402 | 4 | -8.7401 | -3.8461 | 0.0068 | 0.0082 | True |
| scaled/all | od/h264/negative_task_cost | forced_303403 | 4 | -3.7747 | -0.8711 | -0.0005 | 0.0007 | True |
| scaled/all | od/h264/negative_task_cost | forced_mean | 4 | -4.8611 | -1.1501 | 0.0001 | 0.0014 | True |
| scaled/all | od/h264/negative_task_cost | oracle | 4 | -33.9085 | -32.2525 | 0.0835 | 0.0843 | True |
| scaled/all | od/h264/negative_task_cost | selected_303401 | 4 | -28.0354 | -27.7276 | 0.0737 | 0.0743 | True |
| scaled/all | od/h264/negative_task_cost | selected_303402 | 4 | -31.4823 | -29.0753 | 0.0754 | 0.0762 | True |
| scaled/all | od/h264/negative_task_cost | selected_303403 | 4 | -31.7350 | -30.1936 | 0.0778 | 0.0785 | True |
| scaled/all | od/h264/negative_task_cost | selected_mean | 4 | -30.3064 | -29.0918 | 0.0756 | 0.0763 | True |
| scaled/all | od/h264/negative_task_cost | unchecked_303401 | 4 | 4.3962 | 4.6502 | -0.0119 | -0.0112 | True |
| scaled/all | od/h264/negative_task_cost | unchecked_303402 | 4 | -0.4022 | 0.9321 | -0.0048 | -0.0037 | True |
| scaled/all | od/h264/negative_task_cost | unchecked_303403 | 4 | 1.7275 | 2.4918 | -0.0077 | -0.0067 | True |
| scaled/all | od/h264/negative_task_cost | unchecked_mean | 4 | 2.0914 | 2.7504 | -0.0081 | -0.0072 | True |
| scaled/all | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | B2_a5 | 4 | -3.1349 | 1.3941 | -0.0085 | -0.0070 | True |
| scaled/all | od/h265/negative_task_cost | B2_a6 | 4 | -0.9520 | 2.2634 | -0.0166 | -0.0149 | True |
| scaled/all | od/h265/negative_task_cost | B4 | 4 | -16.1368 | -17.3474 | 0.0716 | 0.0723 | True |
| scaled/all | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h265/negative_task_cost | calibrated_303401 | 4 | -0.3096 | -0.2975 | 0.0014 | 0.0012 | True |
| scaled/all | od/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled/all | od/h265/negative_task_cost | calibrated_mean | 4 | -0.1016 | -0.0988 | 0.0005 | 0.0004 | True |
| scaled/all | od/h265/negative_task_cost | fixed | 4 | -0.9064 | 3.2132 | -0.0159 | -0.0148 | True |
| scaled/all | od/h265/negative_task_cost | forced_303401 | 4 | -2.6773 | 0.8124 | -0.0110 | -0.0091 | True |
| scaled/all | od/h265/negative_task_cost | forced_303402 | 4 | -2.5932 | 1.5848 | -0.0120 | -0.0104 | True |
| scaled/all | od/h265/negative_task_cost | forced_303403 | 4 | -2.4057 | 1.5832 | -0.0116 | -0.0100 | True |
| scaled/all | od/h265/negative_task_cost | forced_mean | 4 | -2.5788 | 1.3393 | -0.0115 | -0.0098 | True |
| scaled/all | od/h265/negative_task_cost | oracle | 4 | -19.4724 | -19.7359 | 0.0863 | 0.0871 | True |
| scaled/all | od/h265/negative_task_cost | selected_303401 | 4 | -17.0325 | -17.7918 | 0.0756 | 0.0767 | True |
| scaled/all | od/h265/negative_task_cost | selected_303402 | 4 | -17.6397 | -18.2377 | 0.0786 | 0.0797 | True |
| scaled/all | od/h265/negative_task_cost | selected_303403 | 4 | -16.6062 | -17.2390 | 0.0756 | 0.0760 | True |
| scaled/all | od/h265/negative_task_cost | selected_mean | 4 | -17.0787 | -17.7543 | 0.0766 | 0.0775 | True |
| scaled/all | od/h265/negative_task_cost | unchecked_303401 | 4 | -0.5793 | 2.7769 | -0.0189 | -0.0174 | True |
| scaled/all | od/h265/negative_task_cost | unchecked_303402 | 4 | -0.9955 | 0.4781 | -0.0116 | -0.0099 | True |
| scaled/all | od/h265/negative_task_cost | unchecked_303403 | 4 | -1.6530 | 1.7572 | -0.0103 | -0.0101 | True |
| scaled/all | od/h265/negative_task_cost | unchecked_mean | 4 | -1.1057 | 1.7084 | -0.0136 | -0.0125 | True |
| scaled/all | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled/all | od/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h264/negative_task_cost | B2_a5 | 4 | 13.2682 | 12.8722 | -0.0379 | -0.0375 | True |
| coco256_rank/cal | od/h264/negative_task_cost | B2_a6 | 4 | 15.5781 | 15.8704 | -0.0462 | -0.0459 | True |
| coco256_rank/cal | od/h264/negative_task_cost | B4 | 4 | -19.0595 | -21.3917 | 0.0730 | 0.0735 | True |
| coco256_rank/cal | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/cal | od/h264/negative_task_cost | fixed | 4 | 10.1137 | 10.0162 | -0.0288 | -0.0282 | True |
| coco256_rank/cal | od/h264/negative_task_cost | oracle | 4 | -30.7352 | -25.3674 | 0.0865 | 0.0880 | False |
| coco256_rank/cal | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h265/negative_task_cost | B2_a5 | 4 | -14.5867 | 9.9158 | -0.0655 | -0.0603 | False |
| coco256_rank/cal | od/h265/negative_task_cost | B2_a6 | 4 | -10.2308 | 10.4821 | -0.0676 | -0.0650 | False |
| coco256_rank/cal | od/h265/negative_task_cost | B4 | 4 | -8.4442 | -12.5251 | 0.0760 | 0.0757 | True |
| coco256_rank/cal | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/cal | od/h265/negative_task_cost | fixed | 4 | -12.5867 | 6.5796 | -0.0440 | -0.0402 | False |
| coco256_rank/cal | od/h265/negative_task_cost | oracle | 4 | -12.0393 | -16.6903 | 0.0996 | 0.1023 | True |
| coco256_rank/cal | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/cal | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h264/negative_task_cost | B2_a5 | 4 | 5.1909 | 4.5276 | -0.0147 | -0.0141 | True |
| coco256_rank/all | od/h264/negative_task_cost | B2_a6 | 4 | 7.6109 | 8.6677 | -0.0254 | -0.0247 | True |
| coco256_rank/all | od/h264/negative_task_cost | B4 | 4 | -21.8770 | -24.8265 | 0.0745 | 0.0748 | True |
| coco256_rank/all | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/all | od/h264/negative_task_cost | fixed | 4 | 7.5059 | 9.1331 | -0.0243 | -0.0239 | True |
| coco256_rank/all | od/h264/negative_task_cost | oracle | 4 | -25.7132 | -29.9313 | 0.0920 | 0.0927 | True |
| coco256_rank/all | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h265/negative_task_cost | B2_a5 | 4 | -1.5900 | 4.8617 | -0.0268 | -0.0246 | False |
| coco256_rank/all | od/h265/negative_task_cost | B2_a6 | 4 | -1.2548 | 5.9363 | -0.0354 | -0.0316 | False |
| coco256_rank/all | od/h265/negative_task_cost | B4 | 4 | -14.3983 | -15.0505 | 0.0760 | 0.0771 | True |
| coco256_rank/all | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/all | od/h265/negative_task_cost | fixed | 4 | -0.3887 | 8.3994 | -0.0450 | -0.0417 | False |
| coco256_rank/all | od/h265/negative_task_cost | oracle | 4 | -18.8399 | -18.1956 | 0.0946 | 0.0973 | True |
| coco256_rank/all | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/all | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | B2_a5 | 4 | 37.3298 | 17.0331 | -0.0265 | -0.0267 | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | B2_a6 | 4 | 313.1559 | 15.3609 | -0.0394 | -0.0407 | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | B4 | 4 | -26.6475 | -26.7107 | 0.0588 | 0.0587 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_303401 | 4 | -0.3219 | -0.3656 | 0.0008 | 0.0008 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_303402 | 4 | 0.3827 | 0.2127 | -0.0004 | -0.0004 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_mean | 4 | 0.0214 | -0.0427 | 0.0001 | 0.0001 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | fixed | 4 | 1.1451 | 1.1295 | -0.0012 | -0.0020 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_303401 | 4 | 20.1976 | 19.6745 | -0.0436 | -0.0427 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_303402 | 4 | 1.6689 | 1.9270 | -0.0075 | -0.0066 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_303403 | 4 | 20.5308 | 19.9097 | -0.0441 | -0.0429 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_mean | 4 | 13.6673 | 13.5497 | -0.0318 | -0.0307 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | oracle | 4 | -39.7548 | -32.8939 | 0.0715 | 0.0723 | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_303401 | 4 | -41.1037 | -28.9537 | 0.0625 | 0.0634 | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_303402 | 4 | -35.3566 | -27.1293 | 0.0588 | 0.0596 | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_303403 | 4 | -29.0256 | -29.2880 | 0.0713 | 0.0713 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_mean | 4 | -31.7002 | -29.1995 | 0.0642 | 0.0648 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_303401 | 4 | 22.3746 | 20.7158 | -0.0425 | -0.0407 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_303402 | 4 | 20.7304 | 20.0255 | -0.0398 | -0.0378 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_303403 | 4 | 13.4391 | 13.8844 | -0.0416 | -0.0424 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_mean | 4 | 18.6157 | 17.6712 | -0.0413 | -0.0401 | True |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | B2_a5 | 4 | -8.8551 | 6.3960 | -0.0323 | -0.0283 | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | B2_a6 | 4 | 2.4423 | 7.4404 | -0.0401 | -0.0380 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | B4 | 4 | -13.2442 | -17.0630 | 0.0759 | 0.0721 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_303401 | 4 | 2.8602 | -1.0808 | 0.0059 | 0.0047 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_mean | 4 | 0.7200 | -0.3443 | 0.0020 | 0.0018 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | fixed | 4 | -6.7377 | -1.4991 | 0.0141 | 0.0085 | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_303401 | 4 | 2.4274 | 6.4586 | -0.0396 | -0.0404 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_303402 | 4 | 0.0885 | 4.1999 | -0.0261 | -0.0260 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_303403 | 4 | 2.5056 | 6.6862 | -0.0401 | -0.0404 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_mean | 4 | 1.7947 | 5.8425 | -0.0353 | -0.0354 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | oracle | 4 | -17.3349 | -17.6188 | 0.0805 | 0.0796 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_303401 | 4 | -14.8042 | -15.0666 | 0.0696 | 0.0678 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_303402 | 4 | -14.7019 | -14.3549 | 0.0700 | 0.0676 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_303403 | 4 | -14.7048 | -15.5116 | 0.0707 | 0.0698 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_mean | 4 | -14.7617 | -14.9814 | 0.0700 | 0.0684 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_303401 | 4 | 3.9735 | 8.6971 | -0.0476 | -0.0473 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_303402 | 4 | 4.5165 | 9.4781 | -0.0507 | -0.0505 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_303403 | 4 | 4.2092 | 9.0969 | -0.0479 | -0.0479 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_mean | 4 | 4.2751 | 9.1138 | -0.0488 | -0.0486 | True |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_cal | od/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | B2_a5 | 4 | 2.7816 | 2.2832 | -0.0036 | -0.0039 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | B2_a6 | 4 | 3.8801 | 3.5109 | -0.0072 | -0.0071 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | B4 | 4 | -25.7709 | -26.3748 | 0.0677 | 0.0679 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_303401 | 4 | -0.0316 | -0.0531 | 0.0001 | 0.0001 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_303402 | 4 | 0.1899 | 0.0554 | -0.0001 | -0.0001 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_mean | 4 | 0.0522 | 0.0013 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | fixed | 4 | 3.4456 | 3.3028 | -0.0062 | -0.0064 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_303401 | 4 | -4.9749 | -2.4291 | 0.0024 | 0.0034 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_303402 | 4 | -5.6164 | -5.5378 | 0.0115 | 0.0124 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_303403 | 4 | -5.6672 | -2.2283 | 0.0019 | 0.0030 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_mean | 4 | -5.5820 | -3.4731 | 0.0053 | 0.0063 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | oracle | 4 | -33.9085 | -32.2525 | 0.0835 | 0.0843 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_303401 | 4 | -31.5989 | -29.7307 | 0.0766 | 0.0775 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_303402 | 4 | -32.2156 | -29.6353 | 0.0760 | 0.0768 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_303403 | 4 | -29.1768 | -29.3379 | 0.0784 | 0.0789 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_mean | 4 | -30.8328 | -29.6527 | 0.0770 | 0.0777 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_303401 | 4 | -0.3146 | 1.7614 | -0.0069 | -0.0058 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_303402 | 4 | -2.3744 | 0.6951 | -0.0047 | -0.0035 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_303403 | 4 | 3.5130 | 3.4403 | -0.0085 | -0.0079 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_mean | 4 | 0.9400 | 2.0787 | -0.0067 | -0.0057 | True |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | B2_a5 | 4 | -3.1349 | 1.3941 | -0.0085 | -0.0070 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | B2_a6 | 4 | -0.9520 | 2.2634 | -0.0166 | -0.0149 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | B4 | 4 | -16.4274 | -17.4816 | 0.0725 | 0.0732 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_303401 | 4 | 0.4470 | -0.2654 | 0.0012 | 0.0009 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_303403 | 4 | -0.0282 | -0.0047 | 0.0000 | 0.0000 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_mean | 4 | 0.1299 | -0.0887 | 0.0004 | 0.0003 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | fixed | 4 | -0.9064 | 3.2132 | -0.0159 | -0.0148 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_303401 | 4 | -1.6693 | 2.0808 | -0.0177 | -0.0157 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_303402 | 4 | -1.2872 | 0.5753 | -0.0100 | -0.0089 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_303403 | 4 | -2.8682 | 0.1010 | -0.0099 | -0.0076 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_mean | 4 | -1.8980 | 0.8738 | -0.0126 | -0.0107 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | oracle | 4 | -19.4724 | -19.7359 | 0.0863 | 0.0871 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_303401 | 4 | -16.9403 | -17.0230 | 0.0757 | 0.0758 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_303402 | 4 | -16.6640 | -16.8487 | 0.0762 | 0.0765 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_303403 | 4 | -17.5235 | -17.3525 | 0.0764 | 0.0773 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_mean | 4 | -17.0229 | -17.0733 | 0.0761 | 0.0766 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_303401 | 4 | -0.8305 | 2.1231 | -0.0176 | -0.0158 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_303402 | 4 | -0.3153 | 1.9724 | -0.0169 | -0.0153 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_303403 | 4 | -1.3939 | 2.1557 | -0.0157 | -0.0141 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_mean | 4 | -0.8867 | 2.0923 | -0.0167 | -0.0151 | True |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_rank/input_covered_all | od/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | B2_a5 | 4 | 13.2682 | 12.8722 | -0.0379 | -0.0375 | True |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | B2_a6 | 4 | 15.5781 | 15.8704 | -0.0462 | -0.0459 | True |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | B4 | 4 | -19.0595 | -21.3917 | 0.0730 | 0.0735 | True |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | fixed | 4 | 10.1137 | 10.0162 | -0.0288 | -0.0282 | True |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | oracle | 4 | -30.7352 | -25.3674 | 0.0865 | 0.0880 | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | B2_a5 | 4 | -14.5867 | 9.9158 | -0.0655 | -0.0603 | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | B2_a6 | 4 | -10.2308 | 10.4821 | -0.0676 | -0.0650 | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | B4 | 4 | -8.4442 | -12.5251 | 0.0760 | 0.0757 | True |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | fixed | 4 | -12.5867 | 6.5796 | -0.0440 | -0.0402 | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | oracle | 4 | -12.0393 | -16.6903 | 0.0996 | 0.1023 | True |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/cal | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h264/negative_task_cost | B2_a5 | 4 | 5.1909 | 4.5276 | -0.0147 | -0.0141 | True |
| coco256_scaled_controller/all | od/h264/negative_task_cost | B2_a6 | 4 | 7.6109 | 8.6677 | -0.0254 | -0.0247 | True |
| coco256_scaled_controller/all | od/h264/negative_task_cost | B4 | 4 | -21.8770 | -24.8265 | 0.0745 | 0.0748 | True |
| coco256_scaled_controller/all | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/all | od/h264/negative_task_cost | fixed | 4 | 7.5059 | 9.1331 | -0.0243 | -0.0239 | True |
| coco256_scaled_controller/all | od/h264/negative_task_cost | oracle | 4 | -25.7132 | -29.9313 | 0.0920 | 0.0927 | True |
| coco256_scaled_controller/all | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | B2_a5 | 4 | -1.5900 | 4.8617 | -0.0268 | -0.0246 | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | B2_a6 | 4 | -1.2548 | 5.9363 | -0.0354 | -0.0316 | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | B4 | 4 | -14.3983 | -15.0505 | 0.0760 | 0.0771 | True |
| coco256_scaled_controller/all | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/all | od/h265/negative_task_cost | fixed | 4 | -0.3887 | 8.3994 | -0.0450 | -0.0417 | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | oracle | 4 | -18.8399 | -18.1956 | 0.0946 | 0.0973 | True |
| coco256_scaled_controller/all | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/all | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | B2_a5 | 4 | 37.3298 | 17.0331 | -0.0265 | -0.0267 | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | B2_a6 | 4 | 313.1559 | 15.3609 | -0.0394 | -0.0407 | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | B4 | 4 | -26.6475 | -26.7107 | 0.0588 | 0.0587 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | fixed | 4 | 1.1451 | 1.1295 | -0.0012 | -0.0020 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_303401 | 4 | 22.5750 | 21.0774 | -0.0453 | -0.0436 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_303402 | 4 | -0.2479 | 2.3268 | -0.0092 | -0.0076 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_303403 | 4 | 22.7324 | 21.3561 | -0.0454 | -0.0440 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_mean | 4 | 14.5242 | 14.5073 | -0.0333 | -0.0317 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | oracle | 4 | -39.7548 | -32.8939 | 0.0715 | 0.0723 | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_303401 | 4 | -27.0277 | -26.7466 | 0.0613 | 0.0616 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_303402 | 4 | -45.3451 | -27.3641 | 0.0624 | 0.0635 | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_303403 | 4 | -42.2206 | -32.0316 | 0.0710 | 0.0714 | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_mean | 4 | -34.4958 | -29.0988 | 0.0649 | 0.0655 | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_303401 | 4 | 26.7522 | 26.6104 | -0.0614 | -0.0613 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_303402 | 4 | 18.8479 | 17.9640 | -0.0383 | -0.0367 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_303403 | 4 | 25.3097 | 23.8706 | -0.0521 | -0.0511 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_mean | 4 | 24.2859 | 23.0717 | -0.0506 | -0.0497 | True |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | B2_a5 | 4 | -8.8551 | 6.3960 | -0.0323 | -0.0283 | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | B2_a6 | 4 | 2.4423 | 7.4404 | -0.0401 | -0.0380 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | B4 | 4 | -13.2442 | -17.0630 | 0.0759 | 0.0721 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_303401 | 4 | -1.1676 | -1.4042 | 0.0075 | 0.0072 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_mean | 4 | -0.3592 | -0.4601 | 0.0024 | 0.0024 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | fixed | 4 | -6.7377 | -1.4991 | 0.0141 | 0.0085 | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_303401 | 4 | 5.2969 | 10.4269 | -0.0546 | -0.0535 | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_303402 | 4 | 3.5474 | 8.5464 | -0.0472 | -0.0468 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_303403 | 4 | 3.9888 | 8.6018 | -0.0457 | -0.0455 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_mean | 4 | 4.3345 | 9.2350 | -0.0492 | -0.0486 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | oracle | 4 | -17.3349 | -17.6188 | 0.0805 | 0.0796 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_303401 | 4 | -14.1366 | -14.2791 | 0.0652 | 0.0648 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_303402 | 4 | -15.6094 | -15.4084 | 0.0728 | 0.0705 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_303403 | 4 | -14.0965 | -14.7904 | 0.0737 | 0.0703 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_mean | 4 | -14.6708 | -14.8442 | 0.0704 | 0.0685 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_303401 | 4 | 6.1057 | 11.5546 | -0.0562 | -0.0572 | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_303402 | 4 | 1.8278 | 8.1379 | -0.0425 | -0.0461 | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_303403 | 4 | -2.6904 | 2.2011 | -0.0116 | -0.0146 | True |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_mean | 4 | 2.0415 | 7.2120 | -0.0368 | -0.0392 | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_cal | od/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | B2_a5 | 4 | 2.7816 | 2.2832 | -0.0036 | -0.0039 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | B2_a6 | 4 | 3.8801 | 3.5109 | -0.0072 | -0.0071 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | B4 | 4 | -25.7709 | -26.3748 | 0.0677 | 0.0679 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | fixed | 4 | 3.4456 | 3.3028 | -0.0062 | -0.0064 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_303401 | 4 | -2.0985 | 1.3128 | -0.0060 | -0.0047 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_303402 | 4 | -8.7401 | -3.8461 | 0.0068 | 0.0082 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_303403 | 4 | -3.7747 | -0.8711 | -0.0005 | 0.0007 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_mean | 4 | -4.8611 | -1.1501 | 0.0001 | 0.0014 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | oracle | 4 | -33.9085 | -32.2525 | 0.0835 | 0.0843 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_303401 | 4 | -28.0354 | -27.7276 | 0.0737 | 0.0743 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_303402 | 4 | -31.4823 | -29.0753 | 0.0754 | 0.0762 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_303403 | 4 | -31.7350 | -30.1936 | 0.0778 | 0.0785 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_mean | 4 | -30.3064 | -29.0918 | 0.0756 | 0.0763 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_303401 | 4 | 4.3962 | 4.6502 | -0.0119 | -0.0112 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_303402 | 4 | -0.4022 | 0.9321 | -0.0048 | -0.0037 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_303403 | 4 | 1.7275 | 2.4918 | -0.0077 | -0.0067 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_mean | 4 | 2.0914 | 2.7504 | -0.0081 | -0.0072 | True |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | B2_a5 | 4 | -3.1349 | 1.3941 | -0.0085 | -0.0070 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | B2_a6 | 4 | -0.9520 | 2.2634 | -0.0166 | -0.0149 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | B4 | 4 | -16.4274 | -17.4816 | 0.0725 | 0.0732 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_303401 | 4 | -0.3096 | -0.2975 | 0.0014 | 0.0012 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_mean | 4 | -0.1016 | -0.0988 | 0.0005 | 0.0004 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | fixed | 4 | -0.9064 | 3.2132 | -0.0159 | -0.0148 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_303401 | 4 | -2.6773 | 0.8124 | -0.0110 | -0.0091 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_303402 | 4 | -2.5932 | 1.5848 | -0.0120 | -0.0104 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_303403 | 4 | -2.4057 | 1.5832 | -0.0116 | -0.0100 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_mean | 4 | -2.5788 | 1.3393 | -0.0115 | -0.0098 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | oracle | 4 | -19.4724 | -19.7359 | 0.0863 | 0.0871 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_303401 | 4 | -17.0325 | -17.7918 | 0.0756 | 0.0767 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_303402 | 4 | -17.6397 | -18.2377 | 0.0786 | 0.0797 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_303403 | 4 | -16.6062 | -17.2390 | 0.0756 | 0.0760 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_mean | 4 | -17.0787 | -17.7543 | 0.0766 | 0.0775 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_303401 | 4 | -0.5793 | 2.7769 | -0.0189 | -0.0174 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_303402 | 4 | -0.9955 | 0.4781 | -0.0116 | -0.0099 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_303403 | 4 | -1.6530 | 1.7572 | -0.0103 | -0.0101 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_mean | 4 | -1.1057 | 1.7084 | -0.0136 | -0.0125 | True |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco256_scaled_controller/input_covered_all | od/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | B2_a5 | 4 | 10.7990 | -7.5614 | -0.0085 | -0.0082 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | B2_a6 | 4 | -7.5959 | -0.2508 | -0.0007 | -0.0005 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | B4 | 4 | -0.9706 | -0.6257 | 0.0004 | 0.0004 | True |
| MOT20-02/cal | ot/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_mean | 4 | -0.0000 | 0.0000 | -0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h264/negative_task_cost | fixed | 4 | 323977.2664 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_303401 | 4 | 323977.2664 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_303402 | 4 | 323977.2664 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_303403 | 4 | 323977.2664 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_mean | 4 | 323977.2665 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | oracle | 4 | 1.1043 | 9.8178 | -0.0067 | -0.0064 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_303401 | 4 | 0.1451 | 9.8956 | -0.0068 | -0.0064 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_303402 | 4 | 93405.5813 | -7.7758 | -0.0070 | -0.0065 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_303403 | 4 | 13922.7961 | -8.6628 | -0.0061 | -0.0056 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_mean | 4 | -79.9612 | 9.3353 | -0.0066 | -0.0062 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_303401 | 4 | -27.8197 | 11.6530 | -0.0083 | -0.0078 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_303402 | 4 | 323977.2664 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_303403 | 4 | 323977.2664 | -7.3973 | -0.0073 | -0.0068 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_mean | 4 | -92.9055 | 10.5394 | -0.0076 | -0.0072 | False |
| MOT20-02/cal | ot/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | B2_a5 | 4 | -12.9432 | -13.2871 | 0.0117 | 0.0117 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | B2_a6 | 4 | 247.6415 | -31.5305 | 0.0168 | 0.0171 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | B4 | 4 | -18.0978 | -15.9509 | 0.0128 | 0.0129 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | fixed | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_303401 | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_303402 | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_303403 | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_mean | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | oracle | 4 | -24.4889 | -18.2519 | 0.0137 | 0.0138 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_303401 | 4 | -22.1413 | -18.2859 | 0.0142 | 0.0143 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_303402 | 4 | -22.1413 | -18.2859 | 0.0142 | 0.0143 | True |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_303403 | 4 | -32.9085 | -13.8923 | 0.0103 | 0.0104 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_mean | 4 | -25.4260 | -16.9556 | 0.0129 | 0.0130 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_303401 | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_303402 | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_303403 | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_mean | 4 | -60.1035 | -8.6478 | 0.0052 | 0.0055 | False |
| MOT20-02/cal | ot/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/cal | ot/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | B2_a5 | 4 | -8.0972 | -6.0013 | 0.0011 | 0.0016 | True |
| MOT20-02/all | ot/h264/negative_task_cost | B2_a6 | 4 | 6.3796 | 2.0591 | -0.0031 | -0.0028 | True |
| MOT20-02/all | ot/h264/negative_task_cost | B4 | 4 | -6.6353 | -7.1919 | 0.0052 | 0.0055 | True |
| MOT20-02/all | ot/h264/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h264/negative_task_cost | fixed | 4 | 4.9956 | 6.9433 | -0.0060 | -0.0057 | True |
| MOT20-02/all | ot/h264/negative_task_cost | forced_303401 | 4 | 4.9956 | 6.9433 | -0.0060 | -0.0057 | True |
| MOT20-02/all | ot/h264/negative_task_cost | forced_303402 | 4 | 4.9956 | 6.9433 | -0.0060 | -0.0057 | True |
| MOT20-02/all | ot/h264/negative_task_cost | forced_303403 | 4 | 4.9956 | 6.9433 | -0.0060 | -0.0057 | True |
| MOT20-02/all | ot/h264/negative_task_cost | forced_mean | 4 | 4.9956 | 6.9433 | -0.0060 | -0.0057 | True |
| MOT20-02/all | ot/h264/negative_task_cost | oracle | 4 | -3.9936 | -3.9353 | 0.0032 | 0.0034 | True |
| MOT20-02/all | ot/h264/negative_task_cost | selected_303401 | 4 | -3.3941 | -2.8690 | 0.0021 | 0.0023 | True |
| MOT20-02/all | ot/h264/negative_task_cost | selected_303402 | 4 | -2.3995 | -1.6347 | 0.0011 | 0.0014 | True |
| MOT20-02/all | ot/h264/negative_task_cost | selected_303403 | 4 | -2.4516 | -2.4158 | 0.0021 | 0.0024 | True |
| MOT20-02/all | ot/h264/negative_task_cost | selected_mean | 4 | -2.8192 | -2.3434 | 0.0018 | 0.0020 | True |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_303401 | 4 | 2.6627 | 4.7120 | -0.0046 | -0.0042 | True |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_303402 | 4 | 4.9956 | 6.9433 | -0.0060 | -0.0057 | True |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_303403 | 4 | 5.3914 | 7.2065 | -0.0062 | -0.0059 | True |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_mean | 4 | 4.3575 | 6.2949 | -0.0056 | -0.0053 | True |
| MOT20-02/all | ot/h264/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h264/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | B2_a5 | 4 | 17.1461 | -10.2973 | 0.0102 | 0.0098 | False |
| MOT20-02/all | ot/h265/negative_task_cost | B2_a6 | 4 | -8.7876 | -12.0784 | 0.0114 | 0.0115 | True |
| MOT20-02/all | ot/h265/negative_task_cost | B4 | 4 | -9.1239 | -13.2690 | 0.0134 | 0.0133 | True |
| MOT20-02/all | ot/h265/negative_task_cost | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_mean | 4 | -0.0000 | -0.0000 | -0.0000 | -0.0000 | True |
| MOT20-02/all | ot/h265/negative_task_cost | fixed | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | forced_303401 | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | forced_303402 | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | forced_303403 | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | forced_mean | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | oracle | 4 | -6.7640 | -11.5574 | 0.0118 | 0.0118 | True |
| MOT20-02/all | ot/h265/negative_task_cost | selected_303401 | 4 | -3.3244 | -10.8144 | 0.0110 | 0.0110 | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_303402 | 4 | -1.8106 | -10.4202 | 0.0106 | 0.0106 | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_303403 | 4 | -3.0116 | -8.3402 | 0.0084 | 0.0084 | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_mean | 4 | -2.8208 | -9.8325 | 0.0100 | 0.0100 | False |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_303401 | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_303402 | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_303403 | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_mean | 4 | 1.2921 | -2.9621 | 0.0021 | 0.0022 | True |
| MOT20-02/all | ot/h265/negative_task_cost | B2_a5 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | B2_a6 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | B4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | fixed | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | oracle | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_303402 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_303403 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-02/all | ot/h265/negative_task_cost | unchecked_mean | 5 | N/A | N/A | N/A | N/A | False |

## Held-out curves and Task 4: cubic/PCHIP sensitivity

AR curves are recomputed from paired correctness. COCO and official TrackEval quality uses exact saved evaluator curves; COCO mean bytes are cross-checked against rows. These are not newly executed task-model/TrackEval runs.

Cubic fits a degree-3 polynomial (least squares for five points); PCHIP uses local shape-preserving interpolation. Both integrate only on common support. PCHIP projects quality onto the nondecreasing Pareto envelope sorted by bitrate; plateaus use minimum rate. PCHIP BD-quality uses mean quality at duplicate rates. Quality differences are percentage points for accuracy/TrackEval and cost units for bank proxies.

Original nonmonotone rate-quality curves remain flagged unreliable; their PCHIP envelope and raw same-QP results are retained. Sharp four-point curvature and single active QPs can make cubic overshoot; this alone does not prove classical Runge oscillation.

A cubic/PCHIP difference above 5 percentage points is additionally flagged as interpolation-sensitive. This is a disclosed diagnostic threshold, not a confidence interval or statistical test.

| Dataset | Task/codec/metric | Arm | QPs | Cubic rate % | PCHIP rate % | Cubic quality | PCHIP quality | Reliable |
|---|---|---|---:|---:|---:|---:|---:|---|
| k400_publication | ar/h264/swin3d_t | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | fixed | 4 | -2.3619 | -0.3081 | 0.2008 | 0.2012 | True |
| k400_publication | ar/h264/swin3d_t | forced_303401 | 4 | -2.2696 | -0.2385 | 0.2086 | 0.1989 | True |
| k400_publication | ar/h264/swin3d_t | forced_303402 | 4 | -1.5917 | -0.6087 | 0.2444 | 0.2420 | True |
| k400_publication | ar/h264/swin3d_t | forced_303403 | 4 | -2.3619 | -0.3081 | 0.2008 | 0.2012 | True |
| k400_publication | ar/h264/swin3d_t | forced_mean | 4 | -2.1173 | -0.3665 | 0.2177 | 0.2139 | True |
| k400_publication | ar/h264/swin3d_t | selected_303401 | 4 | -16.9663 | -18.6580 | 3.6182 | 3.6203 | True |
| k400_publication | ar/h264/swin3d_t | selected_303402 | 4 | -15.6109 | -18.7333 | 3.6616 | 3.6265 | True |
| k400_publication | ar/h264/swin3d_t | selected_303403 | 4 | -17.3418 | -19.3828 | 3.7453 | 3.7485 | True |
| k400_publication | ar/h264/swin3d_t | selected_mean | 4 | -16.6092 | -18.9186 | 3.6749 | 3.6696 | True |
| k400_publication | ar/h264/swin3d_t | anchor | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_303401 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_303402 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_303403 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | calibrated_mean | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/swin3d_t | fixed | 5 | -1.8630 | -1.7869 | 0.3655 | 0.4846 | True |
| k400_publication | ar/h264/swin3d_t | forced_303401 | 5 | -1.8233 | -1.8486 | 0.3664 | 0.4983 | True |
| k400_publication | ar/h264/swin3d_t | forced_303402 | 5 | -1.4842 | -1.6790 | 0.3707 | 0.4564 | True |
| k400_publication | ar/h264/swin3d_t | forced_303403 | 5 | -1.8630 | -1.7869 | 0.3655 | 0.4846 | True |
| k400_publication | ar/h264/swin3d_t | forced_mean | 5 | -1.7110 | -1.7615 | 0.3673 | 0.4796 | True |
| k400_publication | ar/h264/swin3d_t | selected_303401 | 5 | -14.4576 | -16.1267 | 4.1335 | 4.2470 | True |
| k400_publication | ar/h264/swin3d_t | selected_303402 | 5 | -14.4207 | -16.6085 | 4.2022 | 4.3276 | True |
| k400_publication | ar/h264/swin3d_t | selected_303403 | 5 | -14.6688 | -16.5527 | 4.2538 | 4.3707 | True |
| k400_publication | ar/h264/swin3d_t | selected_mean | 5 | -14.5145 | -16.4292 | 4.1964 | 4.3186 | True |
| k400_publication | ar/h265/swin3d_t | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | fixed | 4 | -1.5243 | -1.2917 | 0.4143 | 0.5016 | True |
| k400_publication | ar/h265/swin3d_t | forced_303401 | 4 | -2.1186 | -2.2568 | 0.7148 | 0.7953 | True |
| k400_publication | ar/h265/swin3d_t | forced_303402 | 4 | -1.7587 | -1.6640 | 0.6119 | 0.6492 | True |
| k400_publication | ar/h265/swin3d_t | forced_303403 | 4 | -0.9412 | -1.5220 | 0.6413 | 0.6546 | True |
| k400_publication | ar/h265/swin3d_t | forced_mean | 4 | -1.5760 | -1.7990 | 0.6550 | 0.6991 | True |
| k400_publication | ar/h265/swin3d_t | selected_303401 | 4 | -14.3841 | -15.8424 | 4.4932 | 4.4613 | True |
| k400_publication | ar/h265/swin3d_t | selected_303402 | 4 | -14.6730 | -16.3996 | 4.6236 | 4.5991 | True |
| k400_publication | ar/h265/swin3d_t | selected_303403 | 4 | -14.2581 | -15.8421 | 4.4913 | 4.4578 | True |
| k400_publication | ar/h265/swin3d_t | selected_mean | 4 | -14.4395 | -16.0278 | 4.5361 | 4.5061 | True |
| k400_publication | ar/h265/swin3d_t | anchor | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_303401 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_303402 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_303403 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | calibrated_mean | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/swin3d_t | fixed | 5 | -1.5132 | -0.7741 | 0.5147 | 0.4318 | True |
| k400_publication | ar/h265/swin3d_t | forced_303401 | 5 | -2.0418 | -1.2979 | 0.7693 | 0.6528 | True |
| k400_publication | ar/h265/swin3d_t | forced_303402 | 5 | -1.9142 | -1.4300 | 0.7060 | 0.6904 | True |
| k400_publication | ar/h265/swin3d_t | forced_303403 | 5 | -1.5417 | -1.0560 | 0.6595 | 0.5812 | True |
| k400_publication | ar/h265/swin3d_t | forced_mean | 5 | -1.8250 | -1.2525 | 0.7109 | 0.6414 | True |
| k400_publication | ar/h265/swin3d_t | selected_303401 | 5 | -11.4084 | -12.7175 | 4.8541 | 4.9792 | True |
| k400_publication | ar/h265/swin3d_t | selected_303402 | 5 | -11.6167 | -13.0056 | 4.9782 | 5.1007 | True |
| k400_publication | ar/h265/swin3d_t | selected_303403 | 5 | -11.3640 | -12.7261 | 4.8550 | 4.9803 | True |
| k400_publication | ar/h265/swin3d_t | selected_mean | 5 | -11.4635 | -12.8166 | 4.8958 | 5.0201 | True |
| k400_publication | ar/h264/r3d_18 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | fixed | 4 | 1.3961 | 6.7282 | -1.4474 | -1.3400 | False |
| k400_publication | ar/h264/r3d_18 | forced_303401 | 4 | 1.3812 | 6.5752 | -1.4028 | -1.2981 | False |
| k400_publication | ar/h264/r3d_18 | forced_303402 | 4 | 0.8258 | 4.4264 | -0.9827 | -0.9056 | True |
| k400_publication | ar/h264/r3d_18 | forced_303403 | 4 | 1.3961 | 6.7282 | -1.4474 | -1.3400 | False |
| k400_publication | ar/h264/r3d_18 | forced_mean | 4 | 1.1670 | 5.9068 | -1.2775 | -1.1811 | True |
| k400_publication | ar/h264/r3d_18 | selected_303401 | 4 | -0.5852 | 2.3304 | -0.5054 | -0.4584 | True |
| k400_publication | ar/h264/r3d_18 | selected_303402 | 4 | 0.5159 | 3.8316 | -0.7573 | -0.7085 | True |
| k400_publication | ar/h264/r3d_18 | selected_303403 | 4 | 0.0493 | 3.4249 | -0.6973 | -0.6471 | True |
| k400_publication | ar/h264/r3d_18 | selected_mean | 4 | -0.0169 | 3.1938 | -0.6534 | -0.6047 | True |
| k400_publication | ar/h264/r3d_18 | anchor | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_303401 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_303402 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_303403 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | calibrated_mean | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h264/r3d_18 | fixed | 5 | 4.6807 | 6.1177 | -1.2148 | -1.3504 | True |
| k400_publication | ar/h264/r3d_18 | forced_303401 | 5 | 4.6027 | 6.0375 | -1.1950 | -1.3260 | True |
| k400_publication | ar/h264/r3d_18 | forced_303402 | 5 | 3.1612 | 4.3137 | -0.8503 | -0.9631 | True |
| k400_publication | ar/h264/r3d_18 | forced_303403 | 5 | 4.6807 | 6.1177 | -1.2148 | -1.3504 | True |
| k400_publication | ar/h264/r3d_18 | forced_mean | 5 | 4.1452 | 5.4865 | -1.0868 | -1.2131 | True |
| k400_publication | ar/h264/r3d_18 | selected_303401 | 5 | 1.2497 | 2.3567 | -0.4106 | -0.4968 | True |
| k400_publication | ar/h264/r3d_18 | selected_303402 | 5 | 2.3916 | 3.5082 | -0.6682 | -0.7378 | True |
| k400_publication | ar/h264/r3d_18 | selected_303403 | 5 | 2.0186 | 3.1451 | -0.5939 | -0.6651 | True |
| k400_publication | ar/h264/r3d_18 | selected_mean | 5 | 1.8893 | 3.0037 | -0.5576 | -0.6332 | True |
| k400_publication | ar/h265/r3d_18 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | fixed | 4 | -3.3702 | 6.5034 | -1.2898 | -1.2261 | False |
| k400_publication | ar/h265/r3d_18 | forced_303401 | 4 | -3.4252 | 6.1500 | -1.1595 | -1.1320 | False |
| k400_publication | ar/h265/r3d_18 | forced_303402 | 4 | -4.5395 | 4.4934 | -0.7649 | -0.7597 | False |
| k400_publication | ar/h265/r3d_18 | forced_303403 | 4 | -1.4211 | 8.7637 | -1.6604 | -1.6136 | False |
| k400_publication | ar/h265/r3d_18 | forced_mean | 4 | -3.0697 | 6.5265 | -1.1948 | -1.1684 | False |
| k400_publication | ar/h265/r3d_18 | selected_303401 | 4 | -2.6019 | 4.2106 | -0.6753 | -0.6905 | False |
| k400_publication | ar/h265/r3d_18 | selected_303402 | 4 | -1.0334 | 4.3314 | -0.6060 | -0.7090 | False |
| k400_publication | ar/h265/r3d_18 | selected_303403 | 4 | -1.9454 | 4.1227 | -0.6205 | -0.6656 | False |
| k400_publication | ar/h265/r3d_18 | selected_mean | 4 | -1.8449 | 4.2356 | -0.6340 | -0.6859 | False |
| k400_publication | ar/h265/r3d_18 | anchor | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_303401 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_303402 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_303403 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | calibrated_mean | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| k400_publication | ar/h265/r3d_18 | fixed | 5 | 3.6400 | 4.3338 | -1.3634 | -1.2805 | True |
| k400_publication | ar/h265/r3d_18 | forced_303401 | 5 | 3.5629 | 4.1412 | -1.2957 | -1.2187 | True |
| k400_publication | ar/h265/r3d_18 | forced_303402 | 5 | 2.3815 | 2.6908 | -0.9115 | -0.7964 | True |
| k400_publication | ar/h265/r3d_18 | forced_303403 | 5 | 4.9914 | 5.5541 | -1.6852 | -1.5884 | True |
| k400_publication | ar/h265/r3d_18 | forced_mean | 5 | 3.6638 | 4.1726 | -1.2976 | -1.2012 | True |
| k400_publication | ar/h265/r3d_18 | selected_303401 | 5 | 2.1126 | 2.2254 | -0.6535 | -0.6014 | True |
| k400_publication | ar/h265/r3d_18 | selected_303402 | 5 | 2.4318 | 2.1403 | -0.6343 | -0.6098 | True |
| k400_publication | ar/h265/r3d_18 | selected_303403 | 5 | 2.1117 | 2.1458 | -0.6147 | -0.5832 | True |
| k400_publication | ar/h265/r3d_18 | selected_mean | 5 | 2.2153 | 2.1770 | -0.6342 | -0.5961 | True |
| coco_historical | od/h264/resnet50 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | fixed | 4 | 10.3274 | 7.4108 | -0.6973 | -0.6511 | True |
| coco_historical | od/h264/resnet50 | forced_303401 | 4 | 14478.8438 | -9.9931 | 0.6686 | 0.6718 | False |
| coco_historical | od/h264/resnet50 | forced_303402 | 4 | -99.8103 | -10.4039 | 0.9148 | 0.8837 | False |
| coco_historical | od/h264/resnet50 | forced_303403 | 4 | 325.6424 | -8.5698 | 0.6031 | 0.5835 | False |
| coco_historical | od/h264/resnet50 | forced_frozen_303401 | 4 | 10.3274 | 7.4108 | -0.6973 | -0.6511 | True |
| coco_historical | od/h264/resnet50 | forced_mean | 4 | 30247.3367 | -10.5734 | 0.7290 | 0.7099 | False |
| coco_historical | od/h264/resnet50 | selected_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | selected_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | selected_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | selected_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | anchor | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | fixed | 5 | -14.5499 | 4.9114 | -0.5413 | -0.6588 | False |
| coco_historical | od/h264/resnet50 | forced_303401 | 5 | -22.1537 | -2.5444 | 0.5766 | 0.4212 | False |
| coco_historical | od/h264/resnet50 | forced_303402 | 5 | -21.6874 | -3.1206 | 0.7626 | 0.6216 | False |
| coco_historical | od/h264/resnet50 | forced_303403 | 5 | -18.6515 | -2.0622 | 0.4538 | 0.3666 | False |
| coco_historical | od/h264/resnet50 | forced_frozen_303401 | 5 | -14.5499 | 4.9114 | -0.5413 | -0.6588 | False |
| coco_historical | od/h264/resnet50 | forced_mean | 5 | -20.8463 | -2.9790 | 0.5977 | 0.4676 | False |
| coco_historical | od/h264/resnet50 | selected_303401 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | selected_303402 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | selected_303403 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h264/resnet50 | selected_mean | 5 | 0.0000 | -0.0000 | -0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | fixed | 4 | 132.2712 | 12.8586 | -1.3264 | -1.2849 | False |
| coco_historical | od/h265/resnet50 | forced_303401 | 4 | -93.4645 | -3.6338 | 0.1349 | 0.3808 | False |
| coco_historical | od/h265/resnet50 | forced_303402 | 4 | 33.5131 | -0.3792 | -0.1469 | 0.0474 | False |
| coco_historical | od/h265/resnet50 | forced_303403 | 4 | 56.8133 | -3.1130 | 0.3211 | 0.4353 | False |
| coco_historical | od/h265/resnet50 | forced_frozen_303401 | 4 | 132.2712 | 12.8586 | -1.3264 | -1.2849 | False |
| coco_historical | od/h265/resnet50 | forced_mean | 4 | 32.4688 | -2.7082 | 0.1044 | 0.2965 | False |
| coco_historical | od/h265/resnet50 | selected_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | selected_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | selected_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | selected_mean | 4 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | anchor | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | fixed | 5 | 0.2803 | 5.2389 | -1.1147 | -1.0337 | True |
| coco_historical | od/h265/resnet50 | forced_303401 | 5 | -8.8938 | -2.6444 | 0.5413 | 0.4900 | False |
| coco_historical | od/h265/resnet50 | forced_303402 | 5 | -7.5879 | -1.2475 | 0.2265 | 0.2470 | False |
| coco_historical | od/h265/resnet50 | forced_303403 | 5 | -7.5598 | -2.6873 | 0.4396 | 0.5585 | True |
| coco_historical | od/h265/resnet50 | forced_frozen_303401 | 5 | 0.2803 | 5.2389 | -1.1147 | -1.0337 | True |
| coco_historical | od/h265/resnet50 | forced_mean | 5 | -8.1310 | -2.3433 | 0.4030 | 0.4389 | False |
| coco_historical | od/h265/resnet50 | selected_303401 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | selected_303402 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | selected_303403 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_historical | od/h265/resnet50 | selected_mean | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/swin3d_t | fixed | 4 | -2.1588 | -2.3891 | 0.6563 | 0.5713 | True |
| scaled_test | ar/h264/swin3d_t | forced_303401 | 4 | 4.0942 | 4.0989 | -1.2601 | -1.2221 | True |
| scaled_test | ar/h264/swin3d_t | forced_303402 | 4 | 5.3338 | 5.0762 | -1.6109 | -1.4997 | True |
| scaled_test | ar/h264/swin3d_t | forced_303403 | 4 | 4.1200 | 4.0872 | -1.2912 | -1.2184 | True |
| scaled_test | ar/h264/swin3d_t | forced_ensemble | 4 | 5.0861 | 4.8323 | -1.5480 | -1.4359 | True |
| scaled_test | ar/h264/swin3d_t | forced_mean | 4 | 4.4767 | 4.4344 | -1.3906 | -1.3156 | True |
| scaled_test | ar/h264/swin3d_t | selected_303401 | 4 | -30.3829 | -24.1315 | 6.9430 | 7.1777 | False |
| scaled_test | ar/h264/swin3d_t | selected_303402 | 4 | -20.7754 | -21.4168 | 6.0145 | 6.1243 | True |
| scaled_test | ar/h264/swin3d_t | selected_303403 | 4 | -24.5820 | -23.2209 | 6.6838 | 6.8296 | True |
| scaled_test | ar/h264/swin3d_t | selected_ensemble | 4 | -23.8310 | -21.8570 | 6.0783 | 6.2284 | True |
| scaled_test | ar/h264/swin3d_t | selected_mean | 4 | -25.0080 | -22.7729 | 6.5469 | 6.7124 | True |
| scaled_test | ar/h264/swin3d_t | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/swin3d_t | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/swin3d_t | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/swin3d_t | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/swin3d_t | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/swin3d_t | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/swin3d_t | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/swin3d_t | fixed | 4 | -6.0159 | -5.5076 | 2.5354 | 2.6164 | True |
| scaled_test | ar/h265/swin3d_t | forced_303401 | 4 | -2.6709 | -2.1673 | 1.1383 | 1.0375 | True |
| scaled_test | ar/h265/swin3d_t | forced_303402 | 4 | -0.8707 | -1.3091 | 0.2376 | 0.4400 | True |
| scaled_test | ar/h265/swin3d_t | forced_303403 | 4 | -1.2199 | -1.4024 | 0.4448 | 0.5518 | True |
| scaled_test | ar/h265/swin3d_t | forced_ensemble | 4 | -1.6119 | -1.4847 | 0.5519 | 0.6198 | True |
| scaled_test | ar/h265/swin3d_t | forced_mean | 4 | -1.4975 | -1.5699 | 0.6020 | 0.6735 | True |
| scaled_test | ar/h265/swin3d_t | selected_303401 | 4 | -10.1719 | -13.2915 | 5.5900 | 5.6687 | True |
| scaled_test | ar/h265/swin3d_t | selected_303402 | 4 | -11.8369 | -12.7556 | 5.5038 | 5.6232 | True |
| scaled_test | ar/h265/swin3d_t | selected_303403 | 4 | -12.1571 | -14.5117 | 6.3109 | 6.2884 | True |
| scaled_test | ar/h265/swin3d_t | selected_ensemble | 4 | -11.6537 | -12.5588 | 5.3977 | 5.5169 | True |
| scaled_test | ar/h265/swin3d_t | selected_mean | 4 | -11.6075 | -13.5266 | 5.8008 | 5.8836 | True |
| scaled_test | ar/h265/swin3d_t | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/swin3d_t | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/r3d_18 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/r3d_18 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/r3d_18 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/r3d_18 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/r3d_18 | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h264/r3d_18 | fixed | 4 | 11.6579 | 14.5241 | -4.4739 | -4.3175 | True |
| scaled_test | ar/h264/r3d_18 | forced_303401 | 4 | -6.4934 | 4.0336 | -1.5785 | -1.9150 | False |
| scaled_test | ar/h264/r3d_18 | forced_303402 | 4 | 5.6698 | 6.5988 | -2.3214 | -2.3495 | True |
| scaled_test | ar/h264/r3d_18 | forced_303403 | 4 | 2.0776 | 6.6671 | -1.8748 | -2.0131 | True |
| scaled_test | ar/h264/r3d_18 | forced_ensemble | 4 | 3.8553 | 6.8290 | -1.9686 | -2.0322 | True |
| scaled_test | ar/h264/r3d_18 | forced_mean | 4 | 1.7119 | 6.0599 | -1.9298 | -2.0950 | True |
| scaled_test | ar/h264/r3d_18 | selected_303401 | 4 | 0.2403 | 1.2602 | -0.3775 | -0.2489 | True |
| scaled_test | ar/h264/r3d_18 | selected_303402 | 4 | 4.2837 | 2.5103 | -0.8433 | -0.7010 | True |
| scaled_test | ar/h264/r3d_18 | selected_303403 | 4 | 0.3968 | -1.4437 | 0.3587 | 0.3799 | True |
| scaled_test | ar/h264/r3d_18 | selected_ensemble | 4 | 1.6630 | -0.0532 | -0.2003 | 0.0507 | True |
| scaled_test | ar/h264/r3d_18 | selected_mean | 4 | 1.5169 | 0.8277 | -0.2882 | -0.1900 | True |
| scaled_test | ar/h264/r3d_18 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h264/r3d_18 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/r3d_18 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/r3d_18 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/r3d_18 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/r3d_18 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/r3d_18 | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | ar/h265/r3d_18 | fixed | 4 | 15.3074 | 14.4045 | -4.4052 | -4.5294 | True |
| scaled_test | ar/h265/r3d_18 | forced_303401 | 4 | 11.2264 | 10.5063 | -3.3901 | -3.4369 | True |
| scaled_test | ar/h265/r3d_18 | forced_303402 | 4 | 0.4483 | -1.3059 | 0.3856 | 0.4078 | True |
| scaled_test | ar/h265/r3d_18 | forced_303403 | 4 | 3.2394 | 1.0237 | -0.5473 | -0.5564 | True |
| scaled_test | ar/h265/r3d_18 | forced_ensemble | 4 | 0.4088 | 0.8341 | -0.4860 | -0.3918 | True |
| scaled_test | ar/h265/r3d_18 | forced_mean | 4 | 4.5926 | 3.5266 | -1.1880 | -1.1992 | True |
| scaled_test | ar/h265/r3d_18 | selected_303401 | 4 | 4.1908 | 3.5753 | -1.1205 | -1.1595 | True |
| scaled_test | ar/h265/r3d_18 | selected_303402 | 4 | 6.3354 | 3.6401 | -1.2685 | -1.3354 | True |
| scaled_test | ar/h265/r3d_18 | selected_303403 | 4 | 3.7769 | 5.3971 | -2.1494 | -1.9728 | True |
| scaled_test | ar/h265/r3d_18 | selected_ensemble | 4 | 5.4818 | 2.8345 | -1.0948 | -1.1301 | True |
| scaled_test | ar/h265/r3d_18 | selected_mean | 4 | 4.7988 | 4.1398 | -1.5127 | -1.4842 | True |
| scaled_test | ar/h265/r3d_18 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | ar/h265/r3d_18 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h264/resnet50 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h264/resnet50 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h264/resnet50 | calibrated_303403 | 4 | -0.0787 | -0.0549 | 0.0063 | 0.0065 | True |
| scaled_test | od/h264/resnet50 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h264/resnet50 | calibrated_mean | 4 | -0.0269 | -0.0187 | 0.0021 | 0.0022 | True |
| scaled_test | od/h264/resnet50 | fixed | 4 | -1.6521 | -1.2574 | 0.1141 | 0.1223 | True |
| scaled_test | od/h264/resnet50 | forced_303401 | 4 | -7.7863 | -7.8953 | 1.2025 | 1.1807 | True |
| scaled_test | od/h264/resnet50 | forced_303402 | 4 | -3.6535 | -3.5284 | 0.5570 | 0.5226 | True |
| scaled_test | od/h264/resnet50 | forced_303403 | 4 | -4.6692 | -4.5760 | 0.7169 | 0.6907 | True |
| scaled_test | od/h264/resnet50 | forced_ensemble | 4 | -4.5085 | -4.4355 | 0.6855 | 0.6549 | True |
| scaled_test | od/h264/resnet50 | forced_mean | 4 | -5.3818 | -5.3324 | 0.8256 | 0.7981 | True |
| scaled_test | od/h264/resnet50 | selected_303401 | 4 | -6.6838 | -6.6441 | 0.9867 | 0.9749 | True |
| scaled_test | od/h264/resnet50 | selected_303402 | 4 | -8.0293 | -7.9871 | 1.1976 | 1.1889 | True |
| scaled_test | od/h264/resnet50 | selected_303403 | 4 | -6.7601 | -6.7357 | 1.0182 | 1.0036 | True |
| scaled_test | od/h264/resnet50 | selected_ensemble | 4 | -6.6376 | -6.5075 | 0.9632 | 0.9552 | True |
| scaled_test | od/h264/resnet50 | selected_mean | 4 | -7.1555 | -7.1225 | 1.0674 | 1.0557 | True |
| scaled_test | od/h264/resnet50 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h264/resnet50 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h265/resnet50 | calibrated_303401 | 4 | 0.0128 | 0.0081 | -0.0020 | -0.0018 | True |
| scaled_test | od/h265/resnet50 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h265/resnet50 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h265/resnet50 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| scaled_test | od/h265/resnet50 | calibrated_mean | 4 | 0.0043 | 0.0027 | -0.0007 | -0.0006 | True |
| scaled_test | od/h265/resnet50 | fixed | 4 | -7.0417 | -6.6309 | 1.4831 | 1.3944 | True |
| scaled_test | od/h265/resnet50 | forced_303401 | 4 | -6.4148 | -5.9473 | 1.1249 | 1.1963 | True |
| scaled_test | od/h265/resnet50 | forced_303402 | 4 | -9.9027 | -9.1164 | 1.7040 | 1.8123 | True |
| scaled_test | od/h265/resnet50 | forced_303403 | 4 | -9.0359 | -8.8253 | 1.7109 | 1.8012 | True |
| scaled_test | od/h265/resnet50 | forced_ensemble | 4 | -6.7361 | -5.8845 | 1.0286 | 1.1269 | True |
| scaled_test | od/h265/resnet50 | forced_mean | 4 | -8.4460 | -7.9571 | 1.5132 | 1.6034 | True |
| scaled_test | od/h265/resnet50 | selected_303401 | 4 | -8.8956 | -8.4865 | 1.6495 | 1.7108 | True |
| scaled_test | od/h265/resnet50 | selected_303402 | 4 | -8.2921 | -8.1840 | 1.6708 | 1.6921 | True |
| scaled_test | od/h265/resnet50 | selected_303403 | 4 | -8.1405 | -7.6305 | 1.4535 | 1.5204 | True |
| scaled_test | od/h265/resnet50 | selected_ensemble | 4 | -9.3629 | -9.1162 | 1.8025 | 1.8593 | True |
| scaled_test | od/h265/resnet50 | selected_mean | 4 | -8.4069 | -8.1042 | 1.5925 | 1.6420 | True |
| scaled_test | od/h265/resnet50 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| scaled_test | od/h265/resnet50 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_rank | od/h264/resnet50 | calibrated_303401 | 4 | -0.0266 | -0.0335 | 0.0022 | 0.0022 | True |
| coco_rank | od/h264/resnet50 | calibrated_303402 | 4 | 0.1644 | -0.0712 | 0.0060 | 0.0061 | True |
| coco_rank | od/h264/resnet50 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_rank | od/h264/resnet50 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_rank | od/h264/resnet50 | calibrated_mean | 4 | 0.0458 | -0.0345 | 0.0027 | 0.0028 | True |
| coco_rank | od/h264/resnet50 | fixed | 4 | 2.4286 | -5.1274 | 0.2073 | 0.2134 | False |
| coco_rank | od/h264/resnet50 | forced_303401 | 4 | -2.2458 | -11.8201 | 0.8115 | 0.7899 | False |
| coco_rank | od/h264/resnet50 | forced_303402 | 4 | -4.3793 | -10.6170 | 0.7294 | 0.6820 | False |
| coco_rank | od/h264/resnet50 | forced_303403 | 4 | -7.0933 | -10.0136 | 0.7084 | 0.6686 | True |
| coco_rank | od/h264/resnet50 | forced_ensemble | 4 | -1.1145 | -11.1728 | 0.7686 | 0.7190 | False |
| coco_rank | od/h264/resnet50 | forced_frozen_303401 | 4 | 2.4286 | -5.1274 | 0.2073 | 0.2134 | False |
| coco_rank | od/h264/resnet50 | forced_mean | 4 | -4.5183 | -10.7728 | 0.7497 | 0.7136 | False |
| coco_rank | od/h264/resnet50 | selected_303401 | 4 | -1.3614 | -15.7733 | 0.9814 | 0.9725 | False |
| coco_rank | od/h264/resnet50 | selected_303402 | 4 | -0.4231 | -15.2980 | 0.9551 | 0.9540 | False |
| coco_rank | od/h264/resnet50 | selected_303403 | 4 | -5.8159 | -15.6782 | 0.9704 | 0.9724 | False |
| coco_rank | od/h264/resnet50 | selected_ensemble | 4 | -4.9295 | -15.4457 | 0.9597 | 0.9630 | False |
| coco_rank | od/h264/resnet50 | selected_mean | 4 | -2.7143 | -15.5758 | 0.9690 | 0.9663 | False |
| coco_rank | od/h264/resnet50 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | forced_frozen_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h264/resnet50 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_rank | od/h265/resnet50 | calibrated_303401 | 4 | -0.3766 | -0.1821 | 0.0144 | 0.0138 | True |
| coco_rank | od/h265/resnet50 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_rank | od/h265/resnet50 | calibrated_303403 | 4 | -0.0205 | -0.0043 | 0.0003 | 0.0003 | True |
| coco_rank | od/h265/resnet50 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| coco_rank | od/h265/resnet50 | calibrated_mean | 4 | -0.1326 | -0.0622 | 0.0049 | 0.0047 | True |
| coco_rank | od/h265/resnet50 | fixed | 4 | -4.7696 | -0.2505 | 0.0328 | 0.0653 | True |
| coco_rank | od/h265/resnet50 | forced_303401 | 4 | -16.9596 | -7.9538 | 0.6550 | 0.7285 | False |
| coco_rank | od/h265/resnet50 | forced_303402 | 4 | -19.7311 | -4.2247 | 0.2667 | 0.3612 | False |
| coco_rank | od/h265/resnet50 | forced_303403 | 4 | -16.6385 | -8.0396 | 0.6768 | 0.7475 | False |
| coco_rank | od/h265/resnet50 | forced_ensemble | 4 | -23.1878 | -6.6424 | 0.4780 | 0.5757 | False |
| coco_rank | od/h265/resnet50 | forced_frozen_303401 | 4 | -4.7696 | -0.2505 | 0.0328 | 0.0653 | True |
| coco_rank | od/h265/resnet50 | forced_mean | 4 | -17.7355 | -6.7817 | 0.5326 | 0.6126 | False |
| coco_rank | od/h265/resnet50 | selected_303401 | 4 | -2.5400 | -12.5885 | 1.1042 | 1.1240 | False |
| coco_rank | od/h265/resnet50 | selected_303402 | 4 | -10.5569 | -12.6154 | 1.0541 | 1.1084 | True |
| coco_rank | od/h265/resnet50 | selected_303403 | 4 | -1.6126 | -13.3421 | 1.1464 | 1.1729 | False |
| coco_rank | od/h265/resnet50 | selected_ensemble | 4 | -5.5484 | -12.7266 | 1.0974 | 1.1295 | False |
| coco_rank | od/h265/resnet50 | selected_mean | 4 | -4.9836 | -12.8417 | 1.1014 | 1.1358 | False |
| coco_rank | od/h265/resnet50 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | forced_frozen_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| coco_rank | od/h265/resnet50 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/HOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/HOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/HOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/HOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/HOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/HOTA | fixed | 4 | -44.2259 | -12.2443 | 1.0856 | 1.0988 | False |
| ot_test | ot/h264/HOTA | forced_303401 | 4 | -44.2259 | -12.2443 | 1.0856 | 1.0988 | False |
| ot_test | ot/h264/HOTA | forced_303402 | 4 | -44.2259 | -12.2443 | 1.0856 | 1.0988 | False |
| ot_test | ot/h264/HOTA | forced_303403 | 4 | -44.2259 | -12.2443 | 1.0856 | 1.0988 | False |
| ot_test | ot/h264/HOTA | forced_ensemble | 4 | -44.2259 | -12.2443 | 1.0856 | 1.0988 | False |
| ot_test | ot/h264/HOTA | forced_mean | 4 | -44.2259 | -12.2443 | 1.0856 | 1.0988 | False |
| ot_test | ot/h264/HOTA | selected_303401 | 4 | 12.5345 | -17.1028 | 1.0245 | 1.0882 | False |
| ot_test | ot/h264/HOTA | selected_303402 | 4 | 11.9442 | -12.1055 | 0.8859 | 0.8649 | False |
| ot_test | ot/h264/HOTA | selected_303403 | 4 | 16408.6797 | -19.5984 | 1.3032 | 1.3391 | False |
| ot_test | ot/h264/HOTA | selected_ensemble | 4 | 16408.6797 | -19.5984 | 1.3032 | 1.3391 | False |
| ot_test | ot/h264/HOTA | selected_mean | 4 | 96.5131 | -16.0468 | 1.0655 | 1.0956 | False |
| ot_test | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/HOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/IDF1 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/IDF1 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/IDF1 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/IDF1 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/IDF1 | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ot_test | ot/h264/IDF1 | fixed | 4 | -45.9646 | -14.2387 | 0.9705 | 1.0063 | False |
| ot_test | ot/h264/IDF1 | forced_303401 | 4 | -45.9646 | -14.2387 | 0.9705 | 1.0063 | False |
| ot_test | ot/h264/IDF1 | forced_303402 | 4 | -45.9646 | -14.2387 | 0.9705 | 1.0063 | False |
| ot_test | ot/h264/IDF1 | forced_303403 | 4 | -45.9646 | -14.2387 | 0.9705 | 1.0063 | False |
| ot_test | ot/h264/IDF1 | forced_ensemble | 4 | -45.9646 | -14.2387 | 0.9705 | 1.0063 | False |
| ot_test | ot/h264/IDF1 | forced_mean | 4 | -45.9646 | -14.2387 | 0.9705 | 1.0063 | False |
| ot_test | ot/h264/IDF1 | selected_303401 | 4 | 66.0619 | -19.5803 | 1.1210 | 1.2004 | False |
| ot_test | ot/h264/IDF1 | selected_303402 | 4 | -23.5044 | -12.9338 | 0.6944 | 0.6898 | False |
| ot_test | ot/h264/IDF1 | selected_303403 | 4 | -98.8587 | -24.7577 | 1.8959 | 1.9739 | False |
| ot_test | ot/h264/IDF1 | selected_ensemble | 4 | -98.8587 | -24.7577 | 1.8959 | 1.9739 | False |
| ot_test | ot/h264/IDF1 | selected_mean | 4 | 3045.3341 | -20.1999 | 1.2273 | 1.2709 | False |
| ot_test | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/IDF1 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h264/MOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h264/MOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h264/MOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h264/MOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h264/MOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h264/MOTA | fixed | 4 | -14.7082 | -14.9218 | 1.2138 | 1.2464 | True |
| ot_test | ot/h264/MOTA | forced_303401 | 4 | -14.7082 | -14.9218 | 1.2138 | 1.2464 | True |
| ot_test | ot/h264/MOTA | forced_303402 | 4 | -14.7082 | -14.9218 | 1.2138 | 1.2464 | True |
| ot_test | ot/h264/MOTA | forced_303403 | 4 | -14.7082 | -14.9218 | 1.2138 | 1.2464 | True |
| ot_test | ot/h264/MOTA | forced_ensemble | 4 | -14.7082 | -14.9218 | 1.2138 | 1.2464 | True |
| ot_test | ot/h264/MOTA | forced_mean | 4 | -14.7082 | -14.9218 | 1.2138 | 1.2464 | True |
| ot_test | ot/h264/MOTA | selected_303401 | 4 | -11.7447 | -18.9319 | 1.8183 | 1.7969 | False |
| ot_test | ot/h264/MOTA | selected_303402 | 4 | -13.7624 | -13.2327 | 1.1119 | 1.1406 | True |
| ot_test | ot/h264/MOTA | selected_303403 | 4 | -10.1722 | -19.9621 | 1.9318 | 1.8867 | False |
| ot_test | ot/h264/MOTA | selected_ensemble | 4 | -10.1722 | -19.9621 | 1.9318 | 1.8867 | False |
| ot_test | ot/h264/MOTA | selected_mean | 4 | -13.1725 | -17.2390 | 1.6225 | 1.6290 | True |
| ot_test | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h264/MOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/HOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/HOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/HOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/HOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/HOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/HOTA | fixed | 4 | 490.6435 | -4.7964 | 0.5182 | 0.4130 | False |
| ot_test | ot/h265/HOTA | forced_303401 | 4 | 490.6435 | -4.7964 | 0.5182 | 0.4130 | False |
| ot_test | ot/h265/HOTA | forced_303402 | 4 | 490.6435 | -4.7964 | 0.5182 | 0.4130 | False |
| ot_test | ot/h265/HOTA | forced_303403 | 4 | 490.6435 | -4.7964 | 0.5182 | 0.4130 | False |
| ot_test | ot/h265/HOTA | forced_ensemble | 4 | 490.6435 | -4.7964 | 0.5182 | 0.4130 | False |
| ot_test | ot/h265/HOTA | forced_mean | 4 | 490.6435 | -4.7964 | 0.5182 | 0.4130 | False |
| ot_test | ot/h265/HOTA | selected_303401 | 4 | 240.9349 | -2.6666 | 0.2809 | 0.2049 | False |
| ot_test | ot/h265/HOTA | selected_303402 | 4 | 29.6460 | -4.1717 | 0.5170 | 0.4667 | False |
| ot_test | ot/h265/HOTA | selected_303403 | 4 | 70.5027 | -3.7906 | 0.4687 | 0.3858 | False |
| ot_test | ot/h265/HOTA | selected_ensemble | 4 | 118.6708 | -3.7864 | 0.4444 | 0.3599 | False |
| ot_test | ot/h265/HOTA | selected_mean | 4 | 72.3100 | -3.3967 | 0.4220 | 0.3527 | False |
| ot_test | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/HOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/IDF1 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/IDF1 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/IDF1 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/IDF1 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/IDF1 | calibrated_mean | 4 | -0.0000 | 0.0000 | -0.0000 | 0.0000 | True |
| ot_test | ot/h265/IDF1 | fixed | 4 | -72.2836 | -8.9299 | 1.3152 | 1.1793 | False |
| ot_test | ot/h265/IDF1 | forced_303401 | 4 | -72.2836 | -8.9299 | 1.3152 | 1.1793 | False |
| ot_test | ot/h265/IDF1 | forced_303402 | 4 | -72.2836 | -8.9299 | 1.3152 | 1.1793 | False |
| ot_test | ot/h265/IDF1 | forced_303403 | 4 | -72.2836 | -8.9299 | 1.3152 | 1.1793 | False |
| ot_test | ot/h265/IDF1 | forced_ensemble | 4 | -72.2836 | -8.9299 | 1.3152 | 1.1793 | False |
| ot_test | ot/h265/IDF1 | forced_mean | 4 | -72.2836 | -8.9299 | 1.3152 | 1.1793 | False |
| ot_test | ot/h265/IDF1 | selected_303401 | 4 | 9436.7773 | -7.5077 | 0.7903 | 0.6567 | False |
| ot_test | ot/h265/IDF1 | selected_303402 | 4 | 34.0011 | -8.3795 | 1.0950 | 1.0123 | False |
| ot_test | ot/h265/IDF1 | selected_303403 | 4 | 460.3942 | -8.3850 | 0.9864 | 0.8393 | False |
| ot_test | ot/h265/IDF1 | selected_ensemble | 4 | 2443.0060 | -8.5053 | 0.9702 | 0.8288 | False |
| ot_test | ot/h265/IDF1 | selected_mean | 4 | 160.7234 | -7.8930 | 0.9572 | 0.8368 | False |
| ot_test | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/IDF1 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/MOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/MOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/MOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/MOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/MOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_test | ot/h265/MOTA | fixed | 4 | 4.1215 | -7.7836 | 1.1122 | 1.0960 | False |
| ot_test | ot/h265/MOTA | forced_303401 | 4 | 4.1215 | -7.7836 | 1.1122 | 1.0960 | False |
| ot_test | ot/h265/MOTA | forced_303402 | 4 | 4.1215 | -7.7836 | 1.1122 | 1.0960 | False |
| ot_test | ot/h265/MOTA | forced_303403 | 4 | 4.1215 | -7.7836 | 1.1122 | 1.0960 | False |
| ot_test | ot/h265/MOTA | forced_ensemble | 4 | 4.1215 | -7.7836 | 1.1122 | 1.0960 | False |
| ot_test | ot/h265/MOTA | forced_mean | 4 | 4.1215 | -7.7836 | 1.1122 | 1.0960 | False |
| ot_test | ot/h265/MOTA | selected_303401 | 4 | 11.2820 | -9.5325 | 1.2824 | 1.2396 | False |
| ot_test | ot/h265/MOTA | selected_303402 | 4 | 4.0479 | -9.5910 | 1.3089 | 1.2860 | False |
| ot_test | ot/h265/MOTA | selected_303403 | 4 | 4.8231 | -9.5334 | 1.3052 | 1.2695 | False |
| ot_test | ot/h265/MOTA | selected_ensemble | 4 | 9.5435 | -9.1393 | 1.2503 | 1.2001 | False |
| ot_test | ot/h265/MOTA | selected_mean | 4 | 6.1497 | -9.5773 | 1.2987 | 1.2649 | False |
| ot_test | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_test | ot/h265/MOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/HOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/HOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/HOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/HOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/HOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/HOTA | fixed | 4 | 3.8342 | 2.9984 | -0.4616 | -0.4522 | True |
| ot_dev | ot/h264/HOTA | forced_303401 | 4 | 3.8342 | 2.9984 | -0.4616 | -0.4522 | True |
| ot_dev | ot/h264/HOTA | forced_303402 | 4 | 3.8342 | 2.9984 | -0.4616 | -0.4522 | True |
| ot_dev | ot/h264/HOTA | forced_303403 | 4 | 3.8342 | 2.9984 | -0.4616 | -0.4522 | True |
| ot_dev | ot/h264/HOTA | forced_ensemble | 4 | 3.8342 | 2.9984 | -0.4616 | -0.4522 | True |
| ot_dev | ot/h264/HOTA | forced_mean | 4 | 3.8342 | 2.9984 | -0.4616 | -0.4522 | True |
| ot_dev | ot/h264/HOTA | selected_303401 | 4 | 27.9158 | -11.4842 | 0.5254 | 0.5360 | False |
| ot_dev | ot/h264/HOTA | selected_303402 | 4 | -3.2521 | -9.0071 | 0.4817 | 0.4898 | False |
| ot_dev | ot/h264/HOTA | selected_303403 | 4 | 27.9158 | -11.4842 | 0.5254 | 0.5360 | False |
| ot_dev | ot/h264/HOTA | selected_ensemble | 4 | 27.9158 | -11.4842 | 0.5254 | 0.5360 | False |
| ot_dev | ot/h264/HOTA | selected_mean | 4 | 10.1178 | -10.5647 | 0.5110 | 0.5208 | False |
| ot_dev | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/HOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/IDF1 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/IDF1 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/IDF1 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/IDF1 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/IDF1 | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/IDF1 | fixed | 4 | -1.4761 | 4.4999 | -0.9659 | -0.9358 | False |
| ot_dev | ot/h264/IDF1 | forced_303401 | 4 | -1.4761 | 4.4999 | -0.9659 | -0.9358 | False |
| ot_dev | ot/h264/IDF1 | forced_303402 | 4 | -1.4761 | 4.4999 | -0.9659 | -0.9358 | False |
| ot_dev | ot/h264/IDF1 | forced_303403 | 4 | -1.4761 | 4.4999 | -0.9659 | -0.9358 | False |
| ot_dev | ot/h264/IDF1 | forced_ensemble | 4 | -1.4761 | 4.4999 | -0.9659 | -0.9358 | False |
| ot_dev | ot/h264/IDF1 | forced_mean | 4 | -1.4761 | 4.4999 | -0.9659 | -0.9358 | False |
| ot_dev | ot/h264/IDF1 | selected_303401 | 4 | -9.9982 | -13.0089 | 0.8533 | 0.8788 | True |
| ot_dev | ot/h264/IDF1 | selected_303402 | 4 | -10.8317 | -8.3213 | 0.5236 | 0.5428 | True |
| ot_dev | ot/h264/IDF1 | selected_303403 | 4 | -9.9982 | -13.0089 | 0.8533 | 0.8788 | True |
| ot_dev | ot/h264/IDF1 | selected_ensemble | 4 | -9.9982 | -13.0089 | 0.8533 | 0.8788 | True |
| ot_dev | ot/h264/IDF1 | selected_mean | 4 | -11.7657 | -11.3808 | 0.7438 | 0.7671 | True |
| ot_dev | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/IDF1 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/MOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/MOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/MOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/MOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/MOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h264/MOTA | fixed | 4 | -1.2181 | -3.9063 | 0.3772 | 0.3835 | True |
| ot_dev | ot/h264/MOTA | forced_303401 | 4 | -1.2181 | -3.9063 | 0.3772 | 0.3835 | True |
| ot_dev | ot/h264/MOTA | forced_303402 | 4 | -1.2181 | -3.9063 | 0.3772 | 0.3835 | True |
| ot_dev | ot/h264/MOTA | forced_303403 | 4 | -1.2181 | -3.9063 | 0.3772 | 0.3835 | True |
| ot_dev | ot/h264/MOTA | forced_ensemble | 4 | -1.2181 | -3.9063 | 0.3772 | 0.3835 | True |
| ot_dev | ot/h264/MOTA | forced_mean | 4 | -1.2181 | -3.9063 | 0.3772 | 0.3835 | True |
| ot_dev | ot/h264/MOTA | selected_303401 | 4 | 0.2872 | -1.8654 | 0.2973 | 0.2602 | True |
| ot_dev | ot/h264/MOTA | selected_303402 | 4 | 1.0923 | -5.0024 | 0.5877 | 0.5797 | False |
| ot_dev | ot/h264/MOTA | selected_303403 | 4 | 0.2872 | -1.8654 | 0.2973 | 0.2602 | True |
| ot_dev | ot/h264/MOTA | selected_ensemble | 4 | 0.2872 | -1.8654 | 0.2973 | 0.2602 | True |
| ot_dev | ot/h264/MOTA | selected_mean | 4 | 0.5683 | -2.8548 | 0.3940 | 0.3671 | True |
| ot_dev | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h264/MOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/HOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/HOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/HOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/HOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/HOTA | calibrated_mean | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/HOTA | fixed | 4 | 27.6587 | -5.2803 | 0.5862 | 0.5494 | False |
| ot_dev | ot/h265/HOTA | forced_303401 | 4 | 27.6587 | -5.2803 | 0.5862 | 0.5494 | False |
| ot_dev | ot/h265/HOTA | forced_303402 | 4 | 27.6587 | -5.2803 | 0.5862 | 0.5494 | False |
| ot_dev | ot/h265/HOTA | forced_303403 | 4 | 27.6587 | -5.2803 | 0.5862 | 0.5494 | False |
| ot_dev | ot/h265/HOTA | forced_ensemble | 4 | 27.6587 | -5.2803 | 0.5862 | 0.5494 | False |
| ot_dev | ot/h265/HOTA | forced_mean | 4 | 27.6587 | -5.2803 | 0.5862 | 0.5494 | False |
| ot_dev | ot/h265/HOTA | selected_303401 | 4 | 10.3967 | -1.9103 | 0.2605 | 0.2422 | False |
| ot_dev | ot/h265/HOTA | selected_303402 | 4 | 11.3435 | -1.4535 | 0.2152 | 0.1972 | False |
| ot_dev | ot/h265/HOTA | selected_303403 | 4 | 15.8097 | -1.3217 | 0.1751 | 0.1556 | False |
| ot_dev | ot/h265/HOTA | selected_ensemble | 4 | 11.3435 | -1.4535 | 0.2152 | 0.1972 | False |
| ot_dev | ot/h265/HOTA | selected_mean | 4 | 12.3369 | -1.5514 | 0.2170 | 0.1984 | False |
| ot_dev | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/HOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/IDF1 | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/IDF1 | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/IDF1 | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/IDF1 | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/IDF1 | calibrated_mean | 4 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | True |
| ot_dev | ot/h265/IDF1 | fixed | 4 | 6.7718 | -5.0383 | 0.8693 | 0.8394 | False |
| ot_dev | ot/h265/IDF1 | forced_303401 | 4 | 6.7718 | -5.0383 | 0.8693 | 0.8394 | False |
| ot_dev | ot/h265/IDF1 | forced_303402 | 4 | 6.7718 | -5.0383 | 0.8693 | 0.8394 | False |
| ot_dev | ot/h265/IDF1 | forced_303403 | 4 | 6.7718 | -5.0383 | 0.8693 | 0.8394 | False |
| ot_dev | ot/h265/IDF1 | forced_ensemble | 4 | 6.7718 | -5.0383 | 0.8693 | 0.8394 | False |
| ot_dev | ot/h265/IDF1 | forced_mean | 4 | 6.7718 | -5.0383 | 0.8693 | 0.8394 | False |
| ot_dev | ot/h265/IDF1 | selected_303401 | 4 | 2.5652 | -0.7766 | 0.2086 | 0.1914 | True |
| ot_dev | ot/h265/IDF1 | selected_303402 | 4 | 3.5857 | -0.0289 | 0.1234 | 0.1051 | True |
| ot_dev | ot/h265/IDF1 | selected_303403 | 4 | 7.3539 | 0.6789 | -0.0258 | -0.0486 | False |
| ot_dev | ot/h265/IDF1 | selected_ensemble | 4 | 3.5857 | -0.0289 | 0.1234 | 0.1051 | True |
| ot_dev | ot/h265/IDF1 | selected_mean | 4 | 4.3038 | -0.0031 | 0.1022 | 0.0829 | True |
| ot_dev | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/IDF1 | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/MOTA | calibrated_303401 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/MOTA | calibrated_303402 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/MOTA | calibrated_303403 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/MOTA | calibrated_ensemble | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ot_dev | ot/h265/MOTA | calibrated_mean | 4 | -0.0000 | 0.0000 | -0.0000 | 0.0000 | True |
| ot_dev | ot/h265/MOTA | fixed | 4 | 1.4113 | 0.3578 | -0.0363 | -0.0357 | True |
| ot_dev | ot/h265/MOTA | forced_303401 | 4 | 1.4113 | 0.3578 | -0.0363 | -0.0357 | True |
| ot_dev | ot/h265/MOTA | forced_303402 | 4 | 1.4113 | 0.3578 | -0.0363 | -0.0357 | True |
| ot_dev | ot/h265/MOTA | forced_303403 | 4 | 1.4113 | 0.3578 | -0.0363 | -0.0357 | True |
| ot_dev | ot/h265/MOTA | forced_ensemble | 4 | 1.4113 | 0.3578 | -0.0363 | -0.0357 | True |
| ot_dev | ot/h265/MOTA | forced_mean | 4 | 1.4113 | 0.3578 | -0.0363 | -0.0357 | True |
| ot_dev | ot/h265/MOTA | selected_303401 | 4 | 7.5942 | -0.5331 | 0.0275 | 0.0221 | False |
| ot_dev | ot/h265/MOTA | selected_303402 | 4 | 7.2505 | 0.1310 | -0.0149 | -0.0239 | False |
| ot_dev | ot/h265/MOTA | selected_303403 | 4 | 7.6389 | 0.1585 | -0.0215 | -0.0303 | False |
| ot_dev | ot/h265/MOTA | selected_ensemble | 4 | 7.2505 | 0.1310 | -0.0149 | -0.0239 | False |
| ot_dev | ot/h265/MOTA | selected_mean | 4 | 7.5103 | -0.0767 | -0.0031 | -0.0107 | False |
| ot_dev | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | calibrated_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | calibrated_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | calibrated_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | calibrated_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | calibrated_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | fixed | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | forced_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | forced_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | forced_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | forced_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | forced_mean | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | selected_303402 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | selected_303403 | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | selected_ensemble | 5 | N/A | N/A | N/A | N/A | False |
| ot_dev | ot/h265/MOTA | selected_mean | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h264/HOTA | adaptive | 4 | -56.5509 | -23.9235 | 1.6864 | 1.7057 | False |
| ByteTrack_ot-test | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ByteTrack_ot-test | ot/h264/HOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h264/IDF1 | adaptive | 4 | -50.5297 | -23.6236 | 1.9459 | 1.9405 | False |
| ByteTrack_ot-test | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| ByteTrack_ot-test | ot/h264/IDF1 | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h264/MOTA | adaptive | 4 | -18.5854 | -19.3468 | 1.9044 | 1.9037 | True |
| ByteTrack_ot-test | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-test | ot/h264/MOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h265/HOTA | adaptive | 4 | 25681.7926 | -16.0784 | 1.6540 | 1.5134 | False |
| ByteTrack_ot-test | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-test | ot/h265/HOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h265/IDF1 | adaptive | 4 | -65.7491 | -24.1688 | 3.3969 | 3.2686 | False |
| ByteTrack_ot-test | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-test | ot/h265/IDF1 | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h265/MOTA | adaptive | 4 | -11.1067 | -20.9922 | 2.5791 | 2.6267 | False |
| ByteTrack_ot-test | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-test | ot/h265/MOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-test | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h264/HOTA | adaptive | 4 | -15.0502 | -17.2250 | 0.8676 | 0.8940 | True |
| ByteTrack_ot-dev-A1 | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A1 | ot/h264/HOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h264/IDF1 | adaptive | 4 | -11.8970 | -15.1012 | 1.1206 | 1.2418 | True |
| ByteTrack_ot-dev-A1 | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A1 | ot/h264/IDF1 | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h264/MOTA | adaptive | 4 | -12.1889 | -13.4895 | 1.4862 | 1.5770 | True |
| ByteTrack_ot-dev-A1 | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A1 | ot/h264/MOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h265/HOTA | adaptive | 4 | -15.6700 | -1.4404 | 0.0409 | 0.1270 | False |
| ByteTrack_ot-dev-A1 | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A1 | ot/h265/HOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h265/IDF1 | adaptive | 4 | -22.9293 | -1.8514 | 0.2544 | 0.3135 | False |
| ByteTrack_ot-dev-A1 | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A1 | ot/h265/IDF1 | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h265/MOTA | adaptive | 4 | -1.3591 | -0.8914 | 0.2749 | 0.2555 | True |
| ByteTrack_ot-dev-A1 | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A1 | ot/h265/MOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A1 | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h264/HOTA | adaptive | 4 | -20.1108 | -17.9064 | 0.9517 | 0.9593 | True |
| ByteTrack_ot-dev-A2 | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A2 | ot/h264/HOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h264/IDF1 | adaptive | 4 | -20.4728 | -15.6379 | 1.2070 | 1.2621 | True |
| ByteTrack_ot-dev-A2 | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A2 | ot/h264/IDF1 | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h264/MOTA | adaptive | 4 | -13.9375 | -14.9396 | 1.6642 | 1.6981 | True |
| ByteTrack_ot-dev-A2 | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A2 | ot/h264/MOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h265/HOTA | adaptive | 4 | -23.5736 | -4.3766 | 0.3715 | 0.4139 | False |
| ByteTrack_ot-dev-A2 | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A2 | ot/h265/HOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h265/IDF1 | adaptive | 4 | -22.8970 | -3.3325 | 0.4273 | 0.4506 | False |
| ByteTrack_ot-dev-A2 | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A2 | ot/h265/IDF1 | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h265/MOTA | adaptive | 4 | 0.5746 | -0.1893 | 0.0962 | 0.1333 | True |
| ByteTrack_ot-dev-A2 | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| ByteTrack_ot-dev-A2 | ot/h265/MOTA | adaptive | 5 | N/A | N/A | N/A | N/A | False |
| ByteTrack_ot-dev-A2 | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-05_paper_rounded | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-05_paper_rounded | ot/h264/HOTA | selected_303401 | 4 | -6.2848 | -5.7786 | 0.3690 | 0.3701 | True |
| MOT20-05_paper_rounded | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-05_paper_rounded | ot/h264/HOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-05_paper_rounded | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT20-05_paper_rounded | ot/h265/HOTA | selected_303401 | 4 | -6.8227 | -5.9261 | 0.4137 | 0.4135 | True |
| MOT20-05_paper_rounded | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT20-05_paper_rounded | ot/h265/HOTA | selected_303401 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_0 | 4 | -38.1345 | -32.8615 | 1.4799 | 1.3712 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_1 | 4 | -63.6294 | -48.0535 | 1.3415 | 1.2657 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_16 | 4 | -38.0936 | -29.3852 | 0.6771 | 0.6403 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_2 | 4 | 284.1161 | -3.8137 | 0.3047 | 0.3132 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_32 | 4 | -100.0000 | -42.3065 | 1.0314 | 1.0211 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_4 | 4 | -27.3915 | -21.0338 | -0.7166 | -0.7849 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_8 | 4 | -46.6062 | -22.7169 | -0.7677 | -0.7935 | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/HOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_0 | 4 | -45.4399 | -49.1369 | 1.5231 | 1.3081 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_1 | 4 | -45.5154 | -44.6674 | 1.3351 | 1.2067 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_16 | 4 | -70.8679 | N/A | 1.9538 | 1.9272 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_2 | 4 | -41.8841 | -43.3840 | -0.1450 | -0.1906 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_32 | 4 | 35806639702104626604615367637916367825581746157465096406981148672.0000 | -70.2661 | 2.1890 | 2.2008 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_4 | 4 | -12.4881 | -33.0666 | -0.5488 | -0.5628 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_8 | 4 | -16.4990 | -37.6636 | -0.1024 | -0.0983 | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/IDF1 | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_0 | 4 | 86.2881 | N/A | 3.2108 | 3.0860 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_1 | 4 | 140.4821 | N/A | 2.6341 | 2.5545 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_16 | 4 | 25.6140 | N/A | 2.0736 | 2.0299 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_2 | 4 | 33.1450 | N/A | -2.4634 | -2.3274 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_32 | 4 | -38.2072 | N/A | 4.2560 | 4.2126 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_4 | 4 | -73.7277 | N/A | -4.0881 | -3.9569 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_8 | 4 | 8.4291 | N/A | -2.1358 | -2.1836 | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h264/MOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_0 | 4 | -14.7802 | -50.4389 | 2.3655 | 2.2611 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_1 | 4 | -69.2329 | -65.1597 | 2.1570 | 2.0909 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_16 | 4 | -84.3831 | -44.9186 | 0.7179 | 0.7343 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_2 | 4 | 1334.7745 | -23.3768 | 0.5558 | 0.6236 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_32 | 4 | 38.3411 | -65.9299 | 1.2520 | 1.2796 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_4 | 4 | 0.4133 | -42.8288 | -0.8486 | -0.8715 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_8 | 4 | 3.6720 | -48.1863 | -0.9495 | -0.9642 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/HOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_0 | 4 | -48.5118 | -56.1386 | 1.8610 | 1.7153 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_1 | 4 | -43.0676 | -51.6976 | 1.4673 | 1.3705 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_16 | 4 | -23.3749 | N/A | 1.5332 | 1.5790 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_2 | 4 | -80.5550 | -46.5909 | -0.7779 | -0.7210 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_32 | 4 | -99.8264 | -73.9266 | 1.9160 | 1.9680 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_4 | 4 | -33.6405 | -13.2249 | -1.1771 | -1.0803 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_8 | 4 | -29.7509 | -29.2901 | -0.9690 | -0.9008 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/IDF1 | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_0 | 4 | -99.9705 | N/A | 1.6717 | 2.1090 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_1 | 4 | -99.9814 | N/A | 0.2704 | 0.8729 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_16 | 4 | -99.9966 | N/A | -1.6375 | -1.0719 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_2 | 4 | -99.9959 | N/A | -7.9385 | -7.0740 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_32 | 4 | -89.5175 | N/A | 2.7496 | 3.0701 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_4 | 4 | -99.9939 | N/A | -8.7478 | -8.0209 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_8 | 4 | -99.9970 | N/A | -8.1803 | -7.6218 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h264/MOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_0 | 4 | -47.2883 | -24.1648 | 0.6590 | 0.6496 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_1 | 4 | -6.5419 | -22.0631 | 0.5766 | 0.5389 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_16 | 4 | -31.9523 | -34.8390 | 1.2933 | 1.3253 | True |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_2 | 4 | -55.5199 | -28.1375 | 0.7182 | 0.6900 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_32 | 4 | -19.8588 | -36.2940 | 1.2142 | 1.2403 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_4 | 4 | -19.6948 | -19.0917 | 0.6757 | 0.6847 | True |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_8 | 4 | -28.9916 | -34.1773 | 1.2465 | 1.2808 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/HOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_0 | 4 | 890.6933 | -36.4449 | 2.7738 | 2.6938 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_1 | 4 | -97.4338 | N/A | 3.1508 | 3.0884 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_16 | 4 | N/A | N/A | 4.5266 | 4.6456 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_2 | 4 | 386.4015 | N/A | 3.4361 | 3.3567 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_32 | 4 | N/A | N/A | 4.2804 | 4.3987 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_4 | 4 | 929.6065 | N/A | 2.7931 | 2.8280 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_8 | 4 | -51.5748 | -73.8273 | 4.1470 | 4.2877 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/IDF1 | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_0 | 4 | 48022.0183 | N/A | 1.7564 | 1.8641 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_1 | 4 | 3.2046 | N/A | 2.8102 | 2.8050 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_16 | 4 | N/A | N/A | 5.8996 | 6.1836 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_2 | 4 | 21.1899 | N/A | 3.9717 | 4.0864 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_32 | 4 | N/A | N/A | 5.5251 | 5.8084 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_4 | 4 | 2.1239 | N/A | 1.0591 | 1.4953 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_8 | 4 | -10.0764 | -47.2718 | 5.1601 | 5.4857 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h264/MOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_0 | 4 | -72.7957 | -17.1015 | 1.2162 | 1.3378 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_1 | 4 | -40.6486 | -2.5402 | 0.6823 | 0.7873 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_16 | 4 | 2.7648 | 1.5800 | 0.9171 | 0.8994 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_2 | 4 | -46.4594 | 11.9042 | 0.2470 | 0.3438 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_32 | 4 | 229.7811 | -47.3845 | 1.6666 | 1.6742 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_4 | 4 | -31.0719 | 0.6480 | 0.4836 | 0.5401 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_8 | 4 | -33.5393 | 2.3735 | 0.6791 | 0.6983 | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/HOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_0 | 4 | -95.1865 | -49.6141 | 3.5369 | 3.6523 | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_1 | 4 | -83.2075 | -35.1055 | 2.0501 | 2.1553 | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_16 | 4 | -55.1508 | -54.7798 | 2.4385 | 2.4193 | True |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_2 | 4 | -47.5540 | -49.2555 | 2.6606 | 2.7084 | True |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_32 | 4 | -48.5465 | -74.4673 | 2.9949 | 3.0266 | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_4 | 4 | -64.8068 | -55.0249 | 2.4025 | 2.4056 | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_8 | 4 | -54.9688 | -54.2891 | 2.5868 | 2.5803 | True |
| MOT17/fit/pooled_official | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/IDF1 | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | anchor | 4 | 0.0000 | N/A | 0.0000 | 0.0000 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_0 | 4 | 1927.1074 | N/A | 4.5153 | 4.4806 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_1 | 4 | 987.5875 | N/A | 1.6314 | 1.6509 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_16 | 4 | 15568.8978 | N/A | -1.0197 | -1.1716 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_2 | 4 | 2358.2573 | N/A | -1.2836 | -1.3066 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_32 | 4 | 1350.5716 | N/A | 2.2984 | 2.5343 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_4 | 4 | 59950.0160 | N/A | -2.6405 | -2.6834 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_8 | 4 | -100.0000 | N/A | -2.2535 | -2.4347 | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/pooled_official | ot/h265/MOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_0 | 4 | -98.4655 | -32.3655 | 1.7082 | 1.9966 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_1 | 4 | -65.2773 | -20.0545 | 1.1070 | 1.3811 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_16 | 4 | -23.8386 | -12.0554 | 1.1295 | 1.2995 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_2 | 4 | -83.5564 | -7.8936 | 0.4838 | 0.7196 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_32 | 4 | -95.6211 | -44.2725 | 1.9383 | 2.0813 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_4 | 4 | 9.7947 | -19.0803 | 0.7932 | 0.9902 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_8 | 4 | -56.7382 | -17.2716 | 1.0388 | 1.2589 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/HOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_0 | 4 | 1246.7717 | -54.1643 | 4.6234 | 4.9687 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_1 | 4 | 428.8679 | -49.1944 | 2.8763 | 3.2086 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_16 | 4 | -73.0737 | -68.8212 | 3.6721 | 3.8051 | True |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_2 | 4 | 2.7743 | -52.9071 | 3.7027 | 3.8945 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_32 | 4 | -60.1739 | -79.5588 | 3.9424 | 4.1371 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_4 | 4 | 2074.2506 | -61.5677 | 3.4135 | 3.5434 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_8 | 4 | -51.9998 | -60.8405 | 3.7183 | 3.9008 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/IDF1 | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | anchor | 4 | 0.0000 | N/A | 0.0000 | 0.0000 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_0 | 4 | -47.2974 | N/A | 4.0394 | 4.1020 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_1 | 4 | -82.2797 | N/A | -0.4059 | -0.0888 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_16 | 4 | 141499852610837807104.0000 | N/A | -6.0479 | -6.1705 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_2 | 4 | 779.2257 | N/A | -4.8243 | -4.4316 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_32 | 4 | -3.6755 | N/A | -1.3523 | -0.6729 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_4 | 4 | 7486.0749 | N/A | -6.9046 | -6.6546 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_8 | 4 | -100.0000 | N/A | -6.6978 | -6.6236 | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-02-FRCNN | ot/h265/MOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_0 | 4 | -87.8397 | -7.9158 | 0.0493 | 0.0099 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_1 | 4 | -87.3757 | -6.5799 | 0.0323 | -0.0064 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_16 | 4 | -86.5405 | -28.3931 | 0.7441 | 0.7382 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_2 | 4 | -91.0609 | -36.5417 | 0.1704 | 0.1190 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_32 | 4 | -92.8368 | -50.1143 | 1.1596 | 1.1705 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_4 | 4 | -95.7938 | -9.9773 | 0.2097 | 0.1671 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_8 | 4 | -98.2359 | -46.9768 | 0.2996 | 0.2371 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/HOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | anchor | 4 | 0.0000 | N/A | 0.0000 | 0.0000 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_0 | 4 | -54.4760 | N/A | 2.1396 | 2.0686 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_1 | 4 | -51.2124 | N/A | 1.6962 | 1.6138 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_16 | 4 | -62.2191 | N/A | 1.3309 | 1.3877 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_2 | 4 | 5.4651 | N/A | 2.4785 | 2.3816 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_32 | 4 | -57.8066 | N/A | 1.2126 | 1.2627 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_4 | 4 | -66.8818 | N/A | 1.8639 | 1.8239 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_8 | 4 | -60.0025 | N/A | 1.8562 | 1.8336 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/IDF1 | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | anchor | 4 | 0.0000 | N/A | 0.0000 | 0.0000 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_0 | 4 | 345.1546 | N/A | 2.7854 | 2.6941 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_1 | 4 | -3.9505 | N/A | 3.0170 | 2.9353 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_16 | 4 | -60.8445 | N/A | 5.1120 | 5.0736 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_2 | 4 | -29.5587 | N/A | 3.0805 | 2.9375 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_32 | 4 | -70.2349 | N/A | 6.9373 | 7.0352 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_4 | 4 | -36.4525 | N/A | 2.8377 | 2.7550 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_8 | 4 | -46.9604 | N/A | 3.1267 | 2.9290 | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_0 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_1 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_16 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_2 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_4 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/fit/MOT17-05-FRCNN | ot/h265/MOTA | full_lambda_8 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/pooled_official | ot/h264/HOTA | full_lambda_32 | 4 | -67.7536 | -29.3829 | 2.1615 | 2.0844 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/pooled_official | ot/h264/IDF1 | full_lambda_32 | 4 | -93.1173 | -20.0737 | 2.3089 | 2.2425 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/MOTA | full_lambda_32 | 4 | -35.4532 | -5.2940 | 1.6256 | 1.4798 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h264/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/HOTA | full_lambda_32 | 4 | 753252.3860 | -30.8967 | 2.4623 | 2.3984 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/IDF1 | full_lambda_32 | 4 | 259.9390 | -24.5219 | 2.5252 | 2.4823 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/MOTA | full_lambda_32 | 4 | -99.1149 | 17.7124 | 0.8562 | 0.8087 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h264/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/HOTA | full_lambda_32 | 4 | 112.7761 | -33.3452 | 0.9805 | 0.9541 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/IDF1 | full_lambda_32 | 4 | -91.5025 | N/A | 1.1680 | 1.1241 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/MOTA | anchor | 4 | 0.0000 | N/A | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/MOTA | full_lambda_32 | 4 | 41.4411 | N/A | 3.1137 | 3.0724 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h264/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/pooled_official | ot/h265/HOTA | full_lambda_32 | 4 | -8.5312 | -16.4884 | 1.1102 | 1.1144 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/pooled_official | ot/h265/IDF1 | full_lambda_32 | 4 | 1.6119 | -22.1177 | 1.6804 | 1.6572 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/MOTA | full_lambda_32 | 4 | -100.0000 | N/A | 0.1599 | 0.1287 | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/pooled_official | ot/h265/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/HOTA | full_lambda_32 | 4 | -19.7131 | -13.6093 | 1.2060 | 1.2376 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/IDF1 | full_lambda_32 | 4 | -13.3608 | -13.8386 | 1.7539 | 1.7430 | True |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/MOTA | full_lambda_32 | 4 | 919258039887.6642 | -7.1236 | 0.4036 | 0.4367 | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-04-FRCNN | ot/h265/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/HOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/HOTA | full_lambda_32 | 4 | -100.0000 | 4.9366 | 0.6420 | 0.6283 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/HOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/HOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/IDF1 | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/IDF1 | full_lambda_32 | 4 | -100.0000 | -16.6709 | 1.1547 | 1.1193 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/IDF1 | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/IDF1 | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/MOTA | anchor | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/MOTA | full_lambda_32 | 4 | 1486.6808 | N/A | -0.7420 | -0.8328 | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/MOTA | anchor | 5 | N/A | N/A | N/A | N/A | False |
| MOT17/validation_reused_cal/MOT17-11-FRCNN | ot/h265/MOTA | full_lambda_32 | 5 | N/A | N/A | N/A | N/A | False |

### MOT17 requested FIT numbers checked against this artifact

| Codec | Requested BD-rate % | Recomputed cubic % | Recomputed PCHIP % | Reliable |
|---|---:|---:|---:|---|
| h264 | -23.94 | -100.0000 | -42.3065 | False |
| h265 | -16.92 | 229.7811 | -47.3845 | False |

These input curves do not reproduce the requested FIT numbers. Particularly large cubic values reflect an unstable quality-to-log-rate polynomial; they are not evidence of a comparable bitrate gain.


### Largest interpolation discrepancies

| Dataset/codec/metric/arm | QPs | Cubic % | PCHIP % | Difference pp | Active QPs |
|---|---:|---:|---:|---:|---:|
| MOT17/fit/pooled_official/h264/IDF1/full_lambda_32 | 4 | 35806639702104626604615367637916367825581746157465096406981148672.0000 | -70.2661 | 35806639702104626604615367637916367825581746157465096406981148672.0000 | 4 |
| MOT17/validation_reused_cal/MOT17-04-FRCNN/h265/MOTA/full_lambda_32 | 4 | 919258039887.6642 | -7.1236 | 919258039894.7878 | 4 |
| MOT17/validation_reused_cal/MOT17-04-FRCNN/h264/HOTA/full_lambda_32 | 4 | 753252.3860 | -30.8967 | 753283.2827 | 4 |
| coco_historical/h264/resnet50/forced_mean | 4 | 30247.3367 | -10.5734 | 30257.9100 | 4 |
| ByteTrack_ot-test/h265/HOTA/adaptive | 4 | 25681.7926 | -16.0784 | 25697.8710 | 4 |
| ot_test/h264/HOTA/selected_303403 | 4 | 16408.6797 | -19.5984 | 16428.2781 | 4 |
| ot_test/h264/HOTA/selected_ensemble | 4 | 16408.6797 | -19.5984 | 16428.2781 | 4 |
| coco_historical/h264/resnet50/forced_303401 | 4 | 14478.8438 | -9.9931 | 14488.8370 | 4 |
| ot_test/h265/IDF1/selected_303401 | 4 | 9436.7773 | -7.5077 | 9444.2850 | 4 |
| ot_test/h264/IDF1/selected_mean | 4 | 3045.3341 | -20.1999 | 3065.5340 | 4 |
| ot_test/h265/IDF1/selected_ensemble | 4 | 2443.0060 | -8.5053 | 2451.5113 | 4 |
| MOT17/fit/MOT17-02-FRCNN/h265/IDF1/full_lambda_4 | 4 | 2074.2506 | -61.5677 | 2135.8183 | 4 |
| MOT17/fit/MOT17-02-FRCNN/h264/HOTA/full_lambda_2 | 4 | 1334.7745 | -23.3768 | 1358.1513 | 4 |
| MOT17/fit/MOT17-02-FRCNN/h265/IDF1/full_lambda_0 | 4 | 1246.7717 | -54.1643 | 1300.9361 | 4 |
| MOT17/fit/MOT17-05-FRCNN/h264/IDF1/full_lambda_0 | 4 | 890.6933 | -36.4449 | 927.1382 | 4 |
| ot_test/h265/HOTA/forced_mean | 4 | 490.6435 | -4.7964 | 495.4399 | 4 |
| ot_test/h265/HOTA/fixed | 4 | 490.6435 | -4.7964 | 495.4399 | 4 |
| ot_test/h265/HOTA/forced_303401 | 4 | 490.6435 | -4.7964 | 495.4399 | 4 |
| ot_test/h265/HOTA/forced_303402 | 4 | 490.6435 | -4.7964 | 495.4399 | 4 |
| ot_test/h265/HOTA/forced_303403 | 4 | 490.6435 | -4.7964 | 495.4399 | 4 |
| ot_test/h265/HOTA/forced_ensemble | 4 | 490.6435 | -4.7964 | 495.4399 | 4 |
| MOT17/fit/MOT17-02-FRCNN/h265/IDF1/full_lambda_1 | 4 | 428.8679 | -49.1944 | 478.0624 | 4 |
| ot_test/h265/IDF1/selected_303403 | 4 | 460.3942 | -8.3850 | 468.7792 | 4 |
| coco_historical/h264/resnet50/forced_303403 | 4 | 325.6424 | -8.5698 | 334.2122 | 4 |
| MOT17/fit/pooled_official/h264/HOTA/full_lambda_2 | 4 | 284.1161 | -3.8137 | 287.9298 | 4 |

### MOT20 macro and descriptive pooling

Official combined TrackEval cannot be obtained by averaging HOTA/IDF1. The following frame-weighted curves are descriptive only, include rounded MOT20-05, and compare different checkpoint versions. Four-sequence official pooling is unavailable.

| Codec/metric/arm | Macro cubic % | Macro PCHIP % | Descriptive pooled cubic % | Descriptive pooled PCHIP % |
|---|---:|---:|---:|---:|
| h264/HOTA/selected_303401 | 11.3885 | -11.4552 | -7.4164 | -8.2462 |
| h265/HOTA/selected_303401 | 81.5030 | -3.5010 | -2.1057 | -4.2566 |

## Task 3: measured computational costs

All stage timings below are seconds (mean / p50 / p90), grouped by task, codec, and arm; JSON retains every field and arm. End-to-end adds final receiver decoding and task evaluation to sender time. FPS is frames divided by sender time. The baseline denominator is the standard identity encoder, not unnecessary source-map/guard work.

| Dataset/task/codec/arm | Field | Mean | p50 | p90 | N |
|---|---|---:|---:|---:|---:|
| scaled_test/ar/h264/anchor | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h264/anchor | preprocess_seconds | 0.0064 | 0.0063 | 0.0067 | 512 |
| scaled_test/ar/h264/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | encode_seconds | 0.1186 | 0.1175 | 0.1312 | 512 |
| scaled_test/ar/h264/anchor | decode_seconds | 0.1054 | 0.1040 | 0.1124 | 512 |
| scaled_test/ar/h264/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h264/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | receiver_seconds | 0.2032 | 0.2030 | 0.2080 | 512 |
| scaled_test/ar/h264/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/anchor | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h264/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h264/calibrated_303401 | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h264/calibrated_303401 | preprocess_seconds | 0.0064 | 0.0063 | 0.0067 | 512 |
| scaled_test/ar/h264/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | encode_seconds | 0.1186 | 0.1175 | 0.1312 | 512 |
| scaled_test/ar/h264/calibrated_303401 | decode_seconds | 0.1054 | 0.1040 | 0.1124 | 512 |
| scaled_test/ar/h264/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h264/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | receiver_seconds | 0.2032 | 0.2030 | 0.2080 | 512 |
| scaled_test/ar/h264/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h264/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h264/fixed | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h264/fixed | preprocess_seconds | 0.0208 | 0.0205 | 0.0223 | 512 |
| scaled_test/ar/h264/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | encode_seconds | 0.1153 | 0.1142 | 0.1272 | 512 |
| scaled_test/ar/h264/fixed | decode_seconds | 0.1046 | 0.1038 | 0.1101 | 512 |
| scaled_test/ar/h264/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h264/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | receiver_seconds | 0.2073 | 0.2070 | 0.2124 | 512 |
| scaled_test/ar/h264/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/fixed | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h264/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h264/forced_303401 | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h264/forced_303401 | preprocess_seconds | 0.0162 | 0.0197 | 0.0215 | 512 |
| scaled_test/ar/h264/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0025 | 512 |
| scaled_test/ar/h264/forced_303401 | encode_seconds | 0.1165 | 0.1156 | 0.1280 | 512 |
| scaled_test/ar/h264/forced_303401 | decode_seconds | 0.1051 | 0.1039 | 0.1106 | 512 |
| scaled_test/ar/h264/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h264/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | receiver_seconds | 0.2061 | 0.2062 | 0.2115 | 512 |
| scaled_test/ar/h264/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h264/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h264/selected_303401 | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h264/selected_303401 | preprocess_seconds | 0.0139 | 0.0194 | 0.0214 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_preprocess_seconds | 0.1098 | 0.1090 | 0.1147 | 512 |
| scaled_test/ar/h264/selected_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0025 | 512 |
| scaled_test/ar/h264/selected_303401 | encode_seconds | 0.1174 | 0.1160 | 0.1308 | 512 |
| scaled_test/ar/h264/selected_303401 | decode_seconds | 0.1049 | 0.1038 | 0.1107 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_encode_seconds | 0.6980 | 0.6910 | 0.7724 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_probe_decode_seconds | 0.6284 | 0.6248 | 0.6554 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h264/selected_303401 | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_source_guard_seconds | 0.8837 | 0.8854 | 1.0641 | 512 |
| scaled_test/ar/h264/selected_303401 | receiver_seconds | 0.2050 | 0.2050 | 0.2104 | 512 |
| scaled_test/ar/h264/selected_303401 | deployed_sender_seconds | 3.2055 | 3.2308 | 3.4056 | 512 |
| scaled_test/ar/h264/selected_303401 | standalone_benchmark_seconds | 3.4105 | 3.4351 | 3.6144 | 512 |
| scaled_test/ar/h264/selected_303401 | aggregate_sender_fps | 4.9915 | - | - | - |
| scaled_test/ar/h264/selected_303401 | sender_slowdown_vs_b0 | 25.6524 | - | - | - |
| scaled_test/ar/h265/anchor | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h265/anchor | preprocess_seconds | 0.0064 | 0.0063 | 0.0067 | 512 |
| scaled_test/ar/h265/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | encode_seconds | 0.1598 | 0.1558 | 0.1911 | 512 |
| scaled_test/ar/h265/anchor | decode_seconds | 0.1367 | 0.1354 | 0.1436 | 512 |
| scaled_test/ar/h265/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h265/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | receiver_seconds | 0.2025 | 0.2020 | 0.2078 | 512 |
| scaled_test/ar/h265/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/anchor | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h265/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h265/calibrated_303401 | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h265/calibrated_303401 | preprocess_seconds | 0.0064 | 0.0063 | 0.0067 | 512 |
| scaled_test/ar/h265/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | encode_seconds | 0.1598 | 0.1558 | 0.1911 | 512 |
| scaled_test/ar/h265/calibrated_303401 | decode_seconds | 0.1367 | 0.1354 | 0.1436 | 512 |
| scaled_test/ar/h265/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h265/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | receiver_seconds | 0.2025 | 0.2020 | 0.2078 | 512 |
| scaled_test/ar/h265/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h265/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h265/fixed | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h265/fixed | preprocess_seconds | 0.0207 | 0.0205 | 0.0217 | 512 |
| scaled_test/ar/h265/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | encode_seconds | 0.1565 | 0.1532 | 0.1858 | 512 |
| scaled_test/ar/h265/fixed | decode_seconds | 0.1367 | 0.1353 | 0.1430 | 512 |
| scaled_test/ar/h265/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h265/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | receiver_seconds | 0.2063 | 0.2060 | 0.2106 | 512 |
| scaled_test/ar/h265/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/fixed | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h265/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h265/forced_303401 | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h265/forced_303401 | preprocess_seconds | 0.0142 | 0.0190 | 0.0204 | 512 |
| scaled_test/ar/h265/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0025 | 512 |
| scaled_test/ar/h265/forced_303401 | encode_seconds | 0.1590 | 0.1555 | 0.1888 | 512 |
| scaled_test/ar/h265/forced_303401 | decode_seconds | 0.1364 | 0.1354 | 0.1425 | 512 |
| scaled_test/ar/h265/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h265/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | receiver_seconds | 0.2049 | 0.2044 | 0.2104 | 512 |
| scaled_test/ar/h265/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/ar/h265/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/ar/h265/selected_303401 | support_seconds | 0.7137 | 0.7033 | 0.7596 | 512 |
| scaled_test/ar/h265/selected_303401 | preprocess_seconds | 0.0130 | 0.0069 | 0.0212 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_preprocess_seconds | 0.1097 | 0.1090 | 0.1145 | 512 |
| scaled_test/ar/h265/selected_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0025 | 512 |
| scaled_test/ar/h265/selected_303401 | encode_seconds | 0.1592 | 0.1557 | 0.1901 | 512 |
| scaled_test/ar/h265/selected_303401 | decode_seconds | 0.1371 | 0.1356 | 0.1438 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_encode_seconds | 0.9515 | 0.9372 | 1.1177 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_probe_decode_seconds | 0.8202 | 0.8167 | 0.8482 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/ar/h265/selected_303401 | diagnostic_teacher_seconds | 0.1696 | 0.1695 | 0.1745 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_source_guard_seconds | 0.8307 | 0.8922 | 1.0717 | 512 |
| scaled_test/ar/h265/selected_303401 | receiver_seconds | 0.2044 | 0.2042 | 0.2100 | 512 |
| scaled_test/ar/h265/selected_303401 | deployed_sender_seconds | 3.5978 | 3.6238 | 3.8238 | 512 |
| scaled_test/ar/h265/selected_303401 | standalone_benchmark_seconds | 3.8022 | 3.8276 | 4.0310 | 512 |
| scaled_test/ar/h265/selected_303401 | aggregate_sender_fps | 4.4472 | - | - | - |
| scaled_test/ar/h265/selected_303401 | sender_slowdown_vs_b0 | 21.6450 | - | - | - |
| scaled_test/od/h264/anchor | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h264/anchor | preprocess_seconds | 0.0027 | 0.0027 | 0.0029 | 512 |
| scaled_test/od/h264/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | encode_seconds | 0.0986 | 0.0973 | 0.1035 | 512 |
| scaled_test/od/h264/anchor | decode_seconds | 0.1006 | 0.0996 | 0.1060 | 512 |
| scaled_test/od/h264/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h264/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | receiver_seconds | 0.0884 | 0.0882 | 0.0907 | 512 |
| scaled_test/od/h264/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/anchor | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h264/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h264/calibrated_303401 | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h264/calibrated_303401 | preprocess_seconds | 0.0027 | 0.0027 | 0.0029 | 512 |
| scaled_test/od/h264/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | encode_seconds | 0.0986 | 0.0973 | 0.1035 | 512 |
| scaled_test/od/h264/calibrated_303401 | decode_seconds | 0.1006 | 0.0995 | 0.1060 | 512 |
| scaled_test/od/h264/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h264/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | receiver_seconds | 0.0884 | 0.0882 | 0.0907 | 512 |
| scaled_test/od/h264/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h264/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h264/fixed | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h264/fixed | preprocess_seconds | 0.0092 | 0.0091 | 0.0097 | 512 |
| scaled_test/od/h264/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | encode_seconds | 0.0959 | 0.0952 | 0.0995 | 512 |
| scaled_test/od/h264/fixed | decode_seconds | 0.0995 | 0.0984 | 0.1043 | 512 |
| scaled_test/od/h264/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h264/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | receiver_seconds | 0.0988 | 0.0987 | 0.1026 | 512 |
| scaled_test/od/h264/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/fixed | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h264/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h264/forced_303401 | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h264/forced_303401 | preprocess_seconds | 0.0091 | 0.0091 | 0.0096 | 512 |
| scaled_test/od/h264/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | controller_inference_seconds | 0.0019 | 0.0019 | 0.0021 | 512 |
| scaled_test/od/h264/forced_303401 | encode_seconds | 0.0957 | 0.0950 | 0.1000 | 512 |
| scaled_test/od/h264/forced_303401 | decode_seconds | 0.0994 | 0.0983 | 0.1040 | 512 |
| scaled_test/od/h264/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h264/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | receiver_seconds | 0.0982 | 0.0980 | 0.1017 | 512 |
| scaled_test/od/h264/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h264/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h264/selected_303401 | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h264/selected_303401 | preprocess_seconds | 0.0088 | 0.0090 | 0.0096 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_preprocess_seconds | 0.0483 | 0.0481 | 0.0501 | 512 |
| scaled_test/od/h264/selected_303401 | controller_inference_seconds | 0.0019 | 0.0019 | 0.0021 | 512 |
| scaled_test/od/h264/selected_303401 | encode_seconds | 0.0960 | 0.0951 | 0.1001 | 512 |
| scaled_test/od/h264/selected_303401 | decode_seconds | 0.0994 | 0.0984 | 0.1041 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_encode_seconds | 0.5772 | 0.5748 | 0.5977 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_probe_decode_seconds | 0.5979 | 0.5938 | 0.6182 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h264/selected_303401 | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_source_guard_seconds | 0.2084 | 0.1968 | 0.2615 | 512 |
| scaled_test/od/h264/selected_303401 | receiver_seconds | 0.0979 | 0.0981 | 0.1015 | 512 |
| scaled_test/od/h264/selected_303401 | deployed_sender_seconds | 1.5301 | 1.5163 | 1.6063 | 512 |
| scaled_test/od/h264/selected_303401 | standalone_benchmark_seconds | 1.6280 | 1.6148 | 1.7026 | 512 |
| scaled_test/od/h264/selected_303401 | aggregate_sender_fps | 0.6535 | - | - | - |
| scaled_test/od/h264/selected_303401 | sender_slowdown_vs_b0 | 15.1093 | - | - | - |
| scaled_test/od/h265/anchor | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h265/anchor | preprocess_seconds | 0.0027 | 0.0027 | 0.0029 | 512 |
| scaled_test/od/h265/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | encode_seconds | 0.1470 | 0.1446 | 0.1625 | 512 |
| scaled_test/od/h265/anchor | decode_seconds | 0.1241 | 0.1229 | 0.1299 | 512 |
| scaled_test/od/h265/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h265/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | receiver_seconds | 0.0896 | 0.0893 | 0.0923 | 512 |
| scaled_test/od/h265/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/anchor | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h265/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h265/calibrated_303401 | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h265/calibrated_303401 | preprocess_seconds | 0.0028 | 0.0027 | 0.0029 | 512 |
| scaled_test/od/h265/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | encode_seconds | 0.1470 | 0.1446 | 0.1625 | 512 |
| scaled_test/od/h265/calibrated_303401 | decode_seconds | 0.1242 | 0.1229 | 0.1299 | 512 |
| scaled_test/od/h265/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h265/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | receiver_seconds | 0.0896 | 0.0893 | 0.0923 | 512 |
| scaled_test/od/h265/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h265/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h265/fixed | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h265/fixed | preprocess_seconds | 0.0092 | 0.0091 | 0.0097 | 512 |
| scaled_test/od/h265/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | encode_seconds | 0.1470 | 0.1451 | 0.1632 | 512 |
| scaled_test/od/h265/fixed | decode_seconds | 0.1233 | 0.1223 | 0.1282 | 512 |
| scaled_test/od/h265/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h265/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | receiver_seconds | 0.0994 | 0.0993 | 0.1030 | 512 |
| scaled_test/od/h265/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/fixed | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h265/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h265/forced_303401 | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h265/forced_303401 | preprocess_seconds | 0.0091 | 0.0091 | 0.0096 | 512 |
| scaled_test/od/h265/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | controller_inference_seconds | 0.0019 | 0.0019 | 0.0020 | 512 |
| scaled_test/od/h265/forced_303401 | encode_seconds | 0.1460 | 0.1448 | 0.1595 | 512 |
| scaled_test/od/h265/forced_303401 | decode_seconds | 0.1233 | 0.1223 | 0.1286 | 512 |
| scaled_test/od/h265/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h265/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | receiver_seconds | 0.0989 | 0.0986 | 0.1022 | 512 |
| scaled_test/od/h265/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| scaled_test/od/h265/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| scaled_test/od/h265/selected_303401 | support_seconds | 0.0494 | 0.0493 | 0.0501 | 512 |
| scaled_test/od/h265/selected_303401 | preprocess_seconds | 0.0089 | 0.0091 | 0.0096 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_preprocess_seconds | 0.0485 | 0.0483 | 0.0501 | 512 |
| scaled_test/od/h265/selected_303401 | controller_inference_seconds | 0.0019 | 0.0019 | 0.0020 | 512 |
| scaled_test/od/h265/selected_303401 | encode_seconds | 0.1463 | 0.1448 | 0.1606 | 512 |
| scaled_test/od/h265/selected_303401 | decode_seconds | 0.1231 | 0.1223 | 0.1281 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_encode_seconds | 0.8814 | 0.8693 | 0.9602 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_probe_decode_seconds | 0.7408 | 0.7379 | 0.7627 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| scaled_test/od/h265/selected_303401 | diagnostic_teacher_seconds | 0.0470 | 0.0469 | 0.0473 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_source_guard_seconds | 0.2197 | 0.1999 | 0.2696 | 512 |
| scaled_test/od/h265/selected_303401 | receiver_seconds | 0.0986 | 0.0987 | 0.1021 | 512 |
| scaled_test/od/h265/selected_303401 | deployed_sender_seconds | 1.9888 | 1.9775 | 2.1148 | 512 |
| scaled_test/od/h265/selected_303401 | standalone_benchmark_seconds | 2.0874 | 2.0753 | 2.2147 | 512 |
| scaled_test/od/h265/selected_303401 | aggregate_sender_fps | 0.5028 | - | - | - |
| scaled_test/od/h265/selected_303401 | sender_slowdown_vs_b0 | 13.2839 | - | - | - |
| ot_test/ot/h264/anchor | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h264/anchor | preprocess_seconds | 0.3966 | 0.3942 | 0.4089 | 64 |
| ot_test/ot/h264/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | encode_seconds | 0.5239 | 0.5221 | 0.5916 | 64 |
| ot_test/ot/h264/anchor | decode_seconds | 0.3064 | 0.3064 | 0.3187 | 64 |
| ot_test/ot/h264/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h264/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | receiver_seconds | 2.5917 | 2.5997 | 2.6307 | 64 |
| ot_test/ot/h264/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/anchor | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h264/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h264/calibrated_303401 | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h264/calibrated_303401 | preprocess_seconds | 0.3966 | 0.3942 | 0.4089 | 64 |
| ot_test/ot/h264/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | encode_seconds | 0.5239 | 0.5221 | 0.5916 | 64 |
| ot_test/ot/h264/calibrated_303401 | decode_seconds | 0.3064 | 0.3064 | 0.3187 | 64 |
| ot_test/ot/h264/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h264/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | receiver_seconds | 2.5917 | 2.5997 | 2.6307 | 64 |
| ot_test/ot/h264/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h264/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h264/fixed | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h264/fixed | preprocess_seconds | 1.2903 | 1.2891 | 1.3098 | 64 |
| ot_test/ot/h264/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | encode_seconds | 0.5158 | 0.5168 | 0.5762 | 64 |
| ot_test/ot/h264/fixed | decode_seconds | 0.3066 | 0.3067 | 0.3159 | 64 |
| ot_test/ot/h264/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h264/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | receiver_seconds | 2.5725 | 2.5737 | 2.6252 | 64 |
| ot_test/ot/h264/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/fixed | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h264/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h264/forced_303401 | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h264/forced_303401 | preprocess_seconds | 1.2903 | 1.2891 | 1.3098 | 64 |
| ot_test/ot/h264/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0024 | 64 |
| ot_test/ot/h264/forced_303401 | encode_seconds | 0.5158 | 0.5168 | 0.5762 | 64 |
| ot_test/ot/h264/forced_303401 | decode_seconds | 0.3066 | 0.3067 | 0.3159 | 64 |
| ot_test/ot/h264/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h264/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | receiver_seconds | 2.5725 | 2.5737 | 2.6252 | 64 |
| ot_test/ot/h264/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h264/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h264/selected_303401 | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h264/selected_303401 | preprocess_seconds | 1.2452 | 1.3001 | 1.3257 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_preprocess_seconds | 6.8895 | 6.8888 | 6.9935 | 64 |
| ot_test/ot/h264/selected_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0024 | 64 |
| ot_test/ot/h264/selected_303401 | encode_seconds | 0.5119 | 0.5134 | 0.5661 | 64 |
| ot_test/ot/h264/selected_303401 | decode_seconds | 0.3063 | 0.3064 | 0.3157 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_encode_seconds | 3.1050 | 3.0809 | 3.4977 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_probe_decode_seconds | 1.8366 | 1.8343 | 1.8803 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h264/selected_303401 | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_source_guard_seconds | 14.8807 | 14.8402 | 15.1629 | 64 |
| ot_test/ot/h264/selected_303401 | receiver_seconds | 2.5728 | 2.5775 | 2.6384 | 64 |
| ot_test/ot/h264/selected_303401 | deployed_sender_seconds | 43.0148 | 42.8749 | 43.5890 | 64 |
| ot_test/ot/h264/selected_303401 | standalone_benchmark_seconds | 45.5876 | 45.4630 | 46.1538 | 64 |
| ot_test/ot/h264/selected_303401 | aggregate_sender_fps | 0.3720 | - | - | - |
| ot_test/ot/h264/selected_303401 | sender_slowdown_vs_b0 | 46.7295 | - | - | - |
| ot_test/ot/h265/anchor | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h265/anchor | preprocess_seconds | 0.3982 | 0.3970 | 0.4052 | 64 |
| ot_test/ot/h265/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | encode_seconds | 0.9475 | 0.9063 | 1.1543 | 64 |
| ot_test/ot/h265/anchor | decode_seconds | 0.3309 | 0.3310 | 0.3451 | 64 |
| ot_test/ot/h265/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h265/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | receiver_seconds | 2.6018 | 2.6075 | 2.6359 | 64 |
| ot_test/ot/h265/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/anchor | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h265/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h265/calibrated_303401 | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h265/calibrated_303401 | preprocess_seconds | 0.3982 | 0.3970 | 0.4052 | 64 |
| ot_test/ot/h265/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | encode_seconds | 0.9475 | 0.9063 | 1.1543 | 64 |
| ot_test/ot/h265/calibrated_303401 | decode_seconds | 0.3309 | 0.3310 | 0.3451 | 64 |
| ot_test/ot/h265/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h265/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | receiver_seconds | 2.6018 | 2.6075 | 2.6359 | 64 |
| ot_test/ot/h265/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h265/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h265/fixed | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h265/fixed | preprocess_seconds | 1.2888 | 1.2877 | 1.3131 | 64 |
| ot_test/ot/h265/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | encode_seconds | 0.9323 | 0.9002 | 1.1239 | 64 |
| ot_test/ot/h265/fixed | decode_seconds | 0.3287 | 0.3268 | 0.3430 | 64 |
| ot_test/ot/h265/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h265/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | receiver_seconds | 2.5938 | 2.6005 | 2.6324 | 64 |
| ot_test/ot/h265/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/fixed | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h265/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h265/forced_303401 | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h265/forced_303401 | preprocess_seconds | 1.2888 | 1.2877 | 1.3131 | 64 |
| ot_test/ot/h265/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0025 | 64 |
| ot_test/ot/h265/forced_303401 | encode_seconds | 0.9323 | 0.9002 | 1.1239 | 64 |
| ot_test/ot/h265/forced_303401 | decode_seconds | 0.3287 | 0.3268 | 0.3430 | 64 |
| ot_test/ot/h265/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h265/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | receiver_seconds | 2.5938 | 2.6005 | 2.6324 | 64 |
| ot_test/ot/h265/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_test/ot/h265/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_test/ot/h265/selected_303401 | support_seconds | 16.1241 | 16.0888 | 16.3450 | 64 |
| ot_test/ot/h265/selected_303401 | preprocess_seconds | 1.2383 | 1.2883 | 1.3250 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_preprocess_seconds | 6.8826 | 6.8832 | 6.9528 | 64 |
| ot_test/ot/h265/selected_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0025 | 64 |
| ot_test/ot/h265/selected_303401 | encode_seconds | 0.9333 | 0.8983 | 1.1239 | 64 |
| ot_test/ot/h265/selected_303401 | decode_seconds | 0.3289 | 0.3268 | 0.3446 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_encode_seconds | 5.6496 | 5.4121 | 6.8414 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_probe_decode_seconds | 1.9815 | 1.9730 | 2.0723 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_test/ot/h265/selected_303401 | diagnostic_teacher_seconds | 0.1765 | 0.1736 | 0.1899 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_source_guard_seconds | 14.9432 | 14.9599 | 15.1589 | 64 |
| ot_test/ot/h265/selected_303401 | receiver_seconds | 2.5938 | 2.6009 | 2.6339 | 64 |
| ot_test/ot/h265/selected_303401 | deployed_sender_seconds | 45.7599 | 45.5533 | 46.9485 | 64 |
| ot_test/ot/h265/selected_303401 | standalone_benchmark_seconds | 48.3537 | 48.1628 | 49.5265 | 64 |
| ot_test/ot/h265/selected_303401 | aggregate_sender_fps | 0.3497 | - | - | - |
| ot_test/ot/h265/selected_303401 | sender_slowdown_vs_b0 | 34.0048 | - | - | - |
| ot_dev/ot/h264/anchor | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h264/anchor | preprocess_seconds | 0.4008 | 0.3991 | 0.4100 | 64 |
| ot_dev/ot/h264/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | encode_seconds | 0.5557 | 0.5274 | 0.6419 | 64 |
| ot_dev/ot/h264/anchor | decode_seconds | 0.3072 | 0.3064 | 0.3192 | 64 |
| ot_dev/ot/h264/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h264/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | receiver_seconds | 2.6098 | 2.6245 | 2.6712 | 64 |
| ot_dev/ot/h264/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/anchor | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h264/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h264/calibrated_303401 | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h264/calibrated_303401 | preprocess_seconds | 0.4008 | 0.3991 | 0.4100 | 64 |
| ot_dev/ot/h264/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | encode_seconds | 0.5557 | 0.5274 | 0.6419 | 64 |
| ot_dev/ot/h264/calibrated_303401 | decode_seconds | 0.3072 | 0.3064 | 0.3192 | 64 |
| ot_dev/ot/h264/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h264/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | receiver_seconds | 2.6098 | 2.6245 | 2.6712 | 64 |
| ot_dev/ot/h264/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h264/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h264/fixed | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h264/fixed | preprocess_seconds | 1.2972 | 1.2947 | 1.3189 | 64 |
| ot_dev/ot/h264/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | encode_seconds | 0.5406 | 0.5277 | 0.6256 | 64 |
| ot_dev/ot/h264/fixed | decode_seconds | 0.3080 | 0.3055 | 0.3213 | 64 |
| ot_dev/ot/h264/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h264/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | receiver_seconds | 2.5932 | 2.6064 | 2.6638 | 64 |
| ot_dev/ot/h264/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/fixed | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h264/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h264/forced_303401 | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h264/forced_303401 | preprocess_seconds | 1.2972 | 1.2947 | 1.3189 | 64 |
| ot_dev/ot/h264/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0024 | 64 |
| ot_dev/ot/h264/forced_303401 | encode_seconds | 0.5406 | 0.5277 | 0.6256 | 64 |
| ot_dev/ot/h264/forced_303401 | decode_seconds | 0.3080 | 0.3055 | 0.3213 | 64 |
| ot_dev/ot/h264/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h264/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | receiver_seconds | 2.5932 | 2.6064 | 2.6638 | 64 |
| ot_dev/ot/h264/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h264/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h264/selected_303401 | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h264/selected_303401 | preprocess_seconds | 1.2161 | 1.2980 | 1.3174 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_preprocess_seconds | 6.9194 | 6.9183 | 6.9783 | 64 |
| ot_dev/ot/h264/selected_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0024 | 64 |
| ot_dev/ot/h264/selected_303401 | encode_seconds | 0.5399 | 0.5224 | 0.6256 | 64 |
| ot_dev/ot/h264/selected_303401 | decode_seconds | 0.3076 | 0.3061 | 0.3200 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_encode_seconds | 3.2724 | 3.1677 | 3.7782 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_probe_decode_seconds | 1.8463 | 1.8439 | 1.9024 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h264/selected_303401 | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_source_guard_seconds | 15.4530 | 15.5776 | 16.0130 | 64 |
| ot_dev/ot/h264/selected_303401 | receiver_seconds | 2.5943 | 2.6105 | 2.6638 | 64 |
| ot_dev/ot/h264/selected_303401 | deployed_sender_seconds | 43.9995 | 44.0093 | 44.9166 | 64 |
| ot_dev/ot/h264/selected_303401 | standalone_benchmark_seconds | 46.5937 | 46.5742 | 47.4950 | 64 |
| ot_dev/ot/h264/selected_303401 | aggregate_sender_fps | 0.3636 | - | - | - |
| ot_dev/ot/h264/selected_303401 | sender_slowdown_vs_b0 | 46.0009 | - | - | - |
| ot_dev/ot/h265/anchor | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h265/anchor | preprocess_seconds | 0.4009 | 0.3992 | 0.4085 | 64 |
| ot_dev/ot/h265/anchor | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | encode_seconds | 1.0334 | 1.0066 | 1.2631 | 64 |
| ot_dev/ot/h265/anchor | decode_seconds | 0.3335 | 0.3322 | 0.3520 | 64 |
| ot_dev/ot/h265/anchor | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h265/anchor | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | receiver_seconds | 2.6209 | 2.6271 | 2.6847 | 64 |
| ot_dev/ot/h265/anchor | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/anchor | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h265/anchor | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h265/calibrated_303401 | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h265/calibrated_303401 | preprocess_seconds | 0.4009 | 0.3992 | 0.4085 | 64 |
| ot_dev/ot/h265/calibrated_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | encode_seconds | 1.0334 | 1.0066 | 1.2631 | 64 |
| ot_dev/ot/h265/calibrated_303401 | decode_seconds | 0.3335 | 0.3322 | 0.3520 | 64 |
| ot_dev/ot/h265/calibrated_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h265/calibrated_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | receiver_seconds | 2.6209 | 2.6271 | 2.6847 | 64 |
| ot_dev/ot/h265/calibrated_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/calibrated_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h265/calibrated_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h265/fixed | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h265/fixed | preprocess_seconds | 1.3008 | 1.2933 | 1.3278 | 64 |
| ot_dev/ot/h265/fixed | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | controller_inference_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | encode_seconds | 1.0322 | 1.0074 | 1.2832 | 64 |
| ot_dev/ot/h265/fixed | decode_seconds | 0.3313 | 0.3291 | 0.3442 | 64 |
| ot_dev/ot/h265/fixed | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h265/fixed | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | receiver_seconds | 2.6164 | 2.6084 | 2.6711 | 64 |
| ot_dev/ot/h265/fixed | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/fixed | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h265/fixed | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h265/forced_303401 | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h265/forced_303401 | preprocess_seconds | 1.3008 | 1.2933 | 1.3278 | 64 |
| ot_dev/ot/h265/forced_303401 | deployed_preprocess_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0024 | 64 |
| ot_dev/ot/h265/forced_303401 | encode_seconds | 1.0322 | 1.0074 | 1.2832 | 64 |
| ot_dev/ot/h265/forced_303401 | decode_seconds | 0.3313 | 0.3291 | 0.3442 | 64 |
| ot_dev/ot/h265/forced_303401 | deployed_encode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | deployed_probe_decode_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | deployed_encode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h265/forced_303401 | deployed_source_guard_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | receiver_seconds | 2.6164 | 2.6084 | 2.6711 | 64 |
| ot_dev/ot/h265/forced_303401 | deployed_sender_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | standalone_benchmark_seconds | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/forced_303401 | aggregate_sender_fps | N/A | - | - | - |
| ot_dev/ot/h265/forced_303401 | sender_slowdown_vs_b0 | N/A | - | - | - |
| ot_dev/ot/h265/selected_303401 | support_seconds | 16.2174 | 16.2080 | 16.4209 | 64 |
| ot_dev/ot/h265/selected_303401 | preprocess_seconds | 1.2194 | 1.2959 | 1.3311 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_preprocess_seconds | 6.9238 | 6.9112 | 7.0279 | 64 |
| ot_dev/ot/h265/selected_303401 | controller_inference_seconds | 0.0023 | 0.0023 | 0.0024 | 64 |
| ot_dev/ot/h265/selected_303401 | encode_seconds | 1.0338 | 1.0074 | 1.2765 | 64 |
| ot_dev/ot/h265/selected_303401 | decode_seconds | 0.3316 | 0.3291 | 0.3456 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_encode_seconds | 6.2095 | 6.0715 | 7.5961 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_probe_decode_seconds | 1.9961 | 1.9884 | 2.0980 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_encode_count | 6.0000 | 6.0000 | 6.0000 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_decode_count | N/A | N/A | N/A | 0 |
| ot_dev/ot/h265/selected_303401 | diagnostic_teacher_seconds | 0.2885 | 0.2473 | 0.2542 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_source_guard_seconds | 13.4968 | 13.6912 | 16.0111 | 64 |
| ot_dev/ot/h265/selected_303401 | receiver_seconds | 2.6183 | 2.6255 | 2.6843 | 64 |
| ot_dev/ot/h265/selected_303401 | deployed_sender_seconds | 45.1344 | 45.6487 | 47.3679 | 64 |
| ot_dev/ot/h265/selected_303401 | standalone_benchmark_seconds | 47.7528 | 48.2766 | 49.9375 | 64 |
| ot_dev/ot/h265/selected_303401 | aggregate_sender_fps | 0.3545 | - | - | - |
| ot_dev/ot/h265/selected_303401 | sender_slowdown_vs_b0 | 31.4685 | - | - | - |

### Parameters, weights, and system memory

| Model | Parameters | FP32 parameter MB |
|---|---:|---:|
| Controller_frozen_9_conditions | 8661 | 0.0346 |
| Controller_frozen_6_conditions | 8565 | 0.0343 |
| FasterRCNN_ResNet50_FPN | 41755286 | 167.0211 |
| Swin3D_T | 28158070 | 112.6323 |
| R3D_18 | 33371472 | 133.4859 |

| CSV task/codec | Peak allocated GPU MB | Trial encodes | Sender seconds |
|---|---:|---:|---:|
| ar/h264 | 678.6867 | 5.0 | 0.6013 |
| ar/h265 | 678.6867 | 5.0 | 0.8115 |
| od/h264 | 424.4214 | 5.0 | 0.3090 |
| od/h265 | 424.4214 | 5.0 | 0.5067 |
| ot/h264 | 1250.6450 | 5.515625 | 16.2692 |
| ot/h265 | 1250.6450 | 5.515625 | 16.7331 |

Different experiment/policy: CSV has 5 or 5.515625 trials; later saved selected arms use 6. Latencies must not be conflated.

CPU RSS is not recorded in these experiment artifacts. Model FP32 weight bytes are not full-system VRAM. Bank timing subtotals omit unrecorded support/guard/CNN/receiver time; full ablation latency/FPS cannot be claimed from those subtotals.

### Ablation candidate-work comparison

| Bank/task/codec/arm | Encodes | Probe decodes | Guard candidates | Candidate subtotal seconds |
|---|---:|---:|---:|---:|
| scaled/ar/h264/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2395 |
| scaled/ar/h264/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2394 |
| scaled/ar/h264/B4 | 4.0000 | 4.0000 | 3.3423 | 0.9476 |
| scaled/ar/h264/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2275 |
| scaled/ar/h264/calibrated_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2275 |
| scaled/ar/h264/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2275 |
| scaled/ar/h264/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2275 |
| scaled/ar/h264/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2388 |
| scaled/ar/h264/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2338 |
| scaled/ar/h264/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2326 |
| scaled/ar/h264/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2322 |
| scaled/ar/h264/oracle | 13.0000 | 13.0000 | 10.3608 | 3.1068 |
| scaled/ar/h264/random_0 | 4.0000 | 4.0000 | 3.3381 | 0.9479 |
| scaled/ar/h264/selected_303401 | 4.0000 | 4.0000 | 3.5682 | 0.9487 |
| scaled/ar/h264/selected_303402 | 4.0000 | 4.0000 | 3.2585 | 0.9482 |
| scaled/ar/h264/selected_303403 | 4.0000 | 4.0000 | 3.4972 | 0.9484 |
| scaled/ar/h264/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 0.9487 |
| scaled/ar/h264/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 0.9482 |
| scaled/ar/h264/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 0.9484 |
| scaled/ar/h265/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.3127 |
| scaled/ar/h265/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.3125 |
| scaled/ar/h265/B4 | 4.0000 | 4.0000 | 3.3676 | 1.2382 |
| scaled/ar/h265/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2993 |
| scaled/ar/h265/calibrated_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2993 |
| scaled/ar/h265/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2993 |
| scaled/ar/h265/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2993 |
| scaled/ar/h265/fixed | 1.0000 | 1.0000 | 0.0000 | 0.3109 |
| scaled/ar/h265/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.3055 |
| scaled/ar/h265/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.3020 |
| scaled/ar/h265/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.3016 |
| scaled/ar/h265/oracle | 13.0000 | 13.0000 | 10.5114 | 4.0543 |
| scaled/ar/h265/random_0 | 4.0000 | 4.0000 | 3.3466 | 1.2370 |
| scaled/ar/h265/selected_303401 | 4.0000 | 4.0000 | 2.9773 | 1.2385 |
| scaled/ar/h265/selected_303402 | 4.0000 | 4.0000 | 3.1023 | 1.2388 |
| scaled/ar/h265/selected_303403 | 4.0000 | 4.0000 | 3.0256 | 1.2386 |
| scaled/ar/h265/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 1.2385 |
| scaled/ar/h265/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 1.2388 |
| scaled/ar/h265/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 1.2386 |
| scaled/od/h264/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2043 |
| scaled/od/h264/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2043 |
| scaled/od/h264/B4 | 4.0000 | 4.0000 | 3.5534 | 0.8141 |
| scaled/od/h264/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2000 |
| scaled/od/h264/calibrated_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2000 |
| scaled/od/h264/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2000 |
| scaled/od/h264/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2000 |
| scaled/od/h264/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2044 |
| scaled/od/h264/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2042 |
| scaled/od/h264/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2042 |
| scaled/od/h264/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2040 |
| scaled/od/h264/oracle | 13.0000 | 13.0000 | 11.2216 | 2.6561 |
| scaled/od/h264/random_0 | 4.0000 | 4.0000 | 3.5938 | 0.8140 |
| scaled/od/h264/selected_303401 | 4.0000 | 4.0000 | 3.9943 | 0.8131 |
| scaled/od/h264/selected_303402 | 4.0000 | 4.0000 | 3.9972 | 0.8134 |
| scaled/od/h264/selected_303403 | 4.0000 | 4.0000 | 3.9915 | 0.8125 |
| scaled/od/h264/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 0.8131 |
| scaled/od/h264/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 0.8134 |
| scaled/od/h264/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 0.8125 |
| scaled/od/h265/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2776 |
| scaled/od/h265/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2768 |
| scaled/od/h265/B4 | 4.0000 | 4.0000 | 3.6223 | 1.1089 |
| scaled/od/h265/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2733 |
| scaled/od/h265/calibrated_303401 | 1.0341 | 1.0341 | 0.0000 | 0.2827 |
| scaled/od/h265/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2733 |
| scaled/od/h265/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2733 |
| scaled/od/h265/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2774 |
| scaled/od/h265/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2767 |
| scaled/od/h265/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2762 |
| scaled/od/h265/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2765 |
| scaled/od/h265/oracle | 13.0000 | 13.0000 | 11.4403 | 3.6148 |
| scaled/od/h265/random_0 | 4.0000 | 4.0000 | 3.6506 | 1.1089 |
| scaled/od/h265/selected_303401 | 4.0000 | 4.0000 | 3.9915 | 1.1050 |
| scaled/od/h265/selected_303402 | 4.0000 | 4.0000 | 4.0000 | 1.1046 |
| scaled/od/h265/selected_303403 | 4.0000 | 4.0000 | 3.9915 | 1.1047 |
| scaled/od/h265/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 1.1050 |
| scaled/od/h265/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 1.1046 |
| scaled/od/h265/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 1.1047 |
| coco256_rank/od/h264/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2075 |
| coco256_rank/od/h264/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2074 |
| coco256_rank/od/h264/B4 | 4.0000 | 4.0000 | 3.5336 | 0.8245 |
| coco256_rank/od/h264/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2021 |
| coco256_rank/od/h264/calibrated_303401 | 1.0057 | 1.0057 | 0.0000 | 0.2035 |
| coco256_rank/od/h264/calibrated_303402 | 1.0057 | 1.0057 | 0.0000 | 0.2036 |
| coco256_rank/od/h264/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2024 |
| coco256_rank/od/h264/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2073 |
| coco256_rank/od/h264/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2075 |
| coco256_rank/od/h264/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2076 |
| coco256_rank/od/h264/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2075 |
| coco256_rank/od/h264/oracle | 13.0000 | 13.0000 | 11.1484 | 2.6918 |
| coco256_rank/od/h264/random_0 | 4.0000 | 4.0000 | 3.5303 | 0.8247 |
| coco256_rank/od/h264/selected_303401 | 4.0000 | 4.0000 | 3.9943 | 0.8242 |
| coco256_rank/od/h264/selected_303402 | 4.0000 | 4.0000 | 3.9972 | 0.8244 |
| coco256_rank/od/h264/selected_303403 | 4.0000 | 4.0000 | 3.9943 | 0.8248 |
| coco256_rank/od/h264/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 0.8242 |
| coco256_rank/od/h264/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 0.8244 |
| coco256_rank/od/h264/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 0.8248 |
| coco256_rank/od/h265/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2801 |
| coco256_rank/od/h265/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2793 |
| coco256_rank/od/h265/B4 | 4.0000 | 4.0000 | 3.5892 | 1.1170 |
| coco256_rank/od/h265/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2759 |
| coco256_rank/od/h265/calibrated_303401 | 1.0170 | 1.0170 | 0.0000 | 0.2815 |
| coco256_rank/od/h265/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2768 |
| coco256_rank/od/h265/calibrated_303403 | 1.0028 | 1.0028 | 0.0000 | 0.2777 |
| coco256_rank/od/h265/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2794 |
| coco256_rank/od/h265/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2791 |
| coco256_rank/od/h265/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2794 |
| coco256_rank/od/h265/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2788 |
| coco256_rank/od/h265/oracle | 13.0000 | 13.0000 | 11.3633 | 3.6403 |
| coco256_rank/od/h265/random_0 | 4.0000 | 4.0000 | 3.5771 | 1.1167 |
| coco256_rank/od/h265/selected_303401 | 4.0000 | 4.0000 | 3.9744 | 1.1177 |
| coco256_rank/od/h265/selected_303402 | 4.0000 | 4.0000 | 4.0000 | 1.1167 |
| coco256_rank/od/h265/selected_303403 | 4.0000 | 4.0000 | 4.0000 | 1.1166 |
| coco256_rank/od/h265/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 1.1177 |
| coco256_rank/od/h265/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 1.1167 |
| coco256_rank/od/h265/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 1.1166 |
| coco256_scaled_controller/od/h264/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2075 |
| coco256_scaled_controller/od/h264/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2074 |
| coco256_scaled_controller/od/h264/B4 | 4.0000 | 4.0000 | 3.5336 | 0.8245 |
| coco256_scaled_controller/od/h264/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2021 |
| coco256_scaled_controller/od/h264/calibrated_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2024 |
| coco256_scaled_controller/od/h264/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2024 |
| coco256_scaled_controller/od/h264/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2024 |
| coco256_scaled_controller/od/h264/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2073 |
| coco256_scaled_controller/od/h264/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2073 |
| coco256_scaled_controller/od/h264/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2074 |
| coco256_scaled_controller/od/h264/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2072 |
| coco256_scaled_controller/od/h264/oracle | 13.0000 | 13.0000 | 11.1484 | 2.6918 |
| coco256_scaled_controller/od/h264/random_0 | 4.0000 | 4.0000 | 3.5303 | 0.8247 |
| coco256_scaled_controller/od/h264/selected_303401 | 4.0000 | 4.0000 | 3.9943 | 0.8245 |
| coco256_scaled_controller/od/h264/selected_303402 | 4.0000 | 4.0000 | 3.9972 | 0.8245 |
| coco256_scaled_controller/od/h264/selected_303403 | 4.0000 | 4.0000 | 3.9915 | 0.8247 |
| coco256_scaled_controller/od/h264/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 0.8245 |
| coco256_scaled_controller/od/h264/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 0.8245 |
| coco256_scaled_controller/od/h264/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 0.8247 |
| coco256_scaled_controller/od/h265/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 0.2801 |
| coco256_scaled_controller/od/h265/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 0.2793 |
| coco256_scaled_controller/od/h265/B4 | 4.0000 | 4.0000 | 3.5892 | 1.1170 |
| coco256_scaled_controller/od/h265/anchor | 1.0000 | 1.0000 | 0.0000 | 0.2759 |
| coco256_scaled_controller/od/h265/calibrated_303401 | 1.0341 | 1.0341 | 0.0000 | 0.2863 |
| coco256_scaled_controller/od/h265/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2768 |
| coco256_scaled_controller/od/h265/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2768 |
| coco256_scaled_controller/od/h265/fixed | 1.0000 | 1.0000 | 0.0000 | 0.2794 |
| coco256_scaled_controller/od/h265/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 0.2800 |
| coco256_scaled_controller/od/h265/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 0.2794 |
| coco256_scaled_controller/od/h265/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 0.2796 |
| coco256_scaled_controller/od/h265/oracle | 13.0000 | 13.0000 | 11.3633 | 3.6403 |
| coco256_scaled_controller/od/h265/random_0 | 4.0000 | 4.0000 | 3.5771 | 1.1167 |
| coco256_scaled_controller/od/h265/selected_303401 | 4.0000 | 4.0000 | 3.9915 | 1.1175 |
| coco256_scaled_controller/od/h265/selected_303402 | 4.0000 | 4.0000 | 4.0000 | 1.1165 |
| coco256_scaled_controller/od/h265/selected_303403 | 4.0000 | 4.0000 | 3.9915 | 1.1170 |
| coco256_scaled_controller/od/h265/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 1.1175 |
| coco256_scaled_controller/od/h265/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 1.1165 |
| coco256_scaled_controller/od/h265/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 1.1170 |
| MOT20-02/ot/h264/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 2.0544 |
| MOT20-02/ot/h264/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 2.0502 |
| MOT20-02/ot/h264/B4 | 4.0000 | 4.0000 | 3.6078 | 7.3578 |
| MOT20-02/ot/h264/anchor | 1.0000 | 1.0000 | 0.0000 | 1.1859 |
| MOT20-02/ot/h264/calibrated_303401 | 1.0000 | 1.0000 | 0.0000 | 1.1859 |
| MOT20-02/ot/h264/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 1.1859 |
| MOT20-02/ot/h264/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 1.1859 |
| MOT20-02/ot/h264/fixed | 1.0000 | 1.0000 | 0.0000 | 2.0554 |
| MOT20-02/ot/h264/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 2.0554 |
| MOT20-02/ot/h264/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 2.0554 |
| MOT20-02/ot/h264/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 2.0554 |
| MOT20-02/ot/h264/oracle | 13.0000 | 13.0000 | 11.4688 | 25.8686 |
| MOT20-02/ot/h264/random_0 | 4.0000 | 4.0000 | 3.5469 | 7.3576 |
| MOT20-02/ot/h264/selected_303401 | 4.0000 | 4.0000 | 4.0000 | 7.3506 |
| MOT20-02/ot/h264/selected_303402 | 4.0000 | 4.0000 | 3.9688 | 7.3602 |
| MOT20-02/ot/h264/selected_303403 | 4.0000 | 4.0000 | 3.9688 | 7.3536 |
| MOT20-02/ot/h264/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 7.3506 |
| MOT20-02/ot/h264/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 7.3602 |
| MOT20-02/ot/h264/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 7.3536 |
| MOT20-02/ot/h265/B2_a5 | 1.0000 | 1.0000 | 0.0000 | 2.4697 |
| MOT20-02/ot/h265/B2_a6 | 1.0000 | 1.0000 | 0.0000 | 2.4629 |
| MOT20-02/ot/h265/B4 | 4.0000 | 4.0000 | 3.5484 | 9.0329 |
| MOT20-02/ot/h265/anchor | 1.0000 | 1.0000 | 0.0000 | 1.6086 |
| MOT20-02/ot/h265/calibrated_303401 | 1.0000 | 1.0000 | 0.0000 | 1.6086 |
| MOT20-02/ot/h265/calibrated_303402 | 1.0000 | 1.0000 | 0.0000 | 1.6086 |
| MOT20-02/ot/h265/calibrated_303403 | 1.0000 | 1.0000 | 0.0000 | 1.6086 |
| MOT20-02/ot/h265/fixed | 1.0000 | 1.0000 | 0.0000 | 2.4683 |
| MOT20-02/ot/h265/forced_303401 | 1.0000 | 1.0000 | 0.0000 | 2.4683 |
| MOT20-02/ot/h265/forced_303402 | 1.0000 | 1.0000 | 0.0000 | 2.4683 |
| MOT20-02/ot/h265/forced_303403 | 1.0000 | 1.0000 | 0.0000 | 2.4683 |
| MOT20-02/ot/h265/oracle | 13.0000 | 13.0000 | 11.2344 | 31.3037 |
| MOT20-02/ot/h265/random_0 | 4.0000 | 4.0000 | 3.5156 | 9.0235 |
| MOT20-02/ot/h265/selected_303401 | 4.0000 | 4.0000 | 3.9844 | 9.0202 |
| MOT20-02/ot/h265/selected_303402 | 4.0000 | 4.0000 | 3.9688 | 9.0248 |
| MOT20-02/ot/h265/selected_303403 | 4.0000 | 4.0000 | 4.0000 | 9.0399 |
| MOT20-02/ot/h265/unchecked_303401 | 4.0000 | 4.0000 | 0.0000 | 9.0202 |
| MOT20-02/ot/h265/unchecked_303402 | 4.0000 | 4.0000 | 0.0000 | 9.0248 |
| MOT20-02/ot/h265/unchecked_303403 | 4.0000 | 4.0000 | 0.0000 | 9.0399 |

## 6. Bugfix verification: live Before vs After replay

Before uses saved logits, quantile bounds and original shortlists/limits. After gates task logits with predicted risk/rate, recalibrates clipped mean UCB on CAL only, and uses repaired rank seeding and [.025,.05,.05] limits. Bank quality is negative GT-conditioned cost; ALL includes TRAIN and is a policy diagnostic, not held-out accuracy. Relative-floor diagnostics affect future supervision; frozen predictions and calibration keep their original units.

Status: **verified**. Torch uses two CPU threads and deterministic algorithms. The three frozen V34-R1 checkpoints produce exactly the same outputs under the historical and repaired default Controller. Saved task-logit checkpoints are also checked against their original architecture on every bank row.

Rates below are total selected packet bytes relative to total identity bytes. Cost excess is the mean raw primary-task cost difference. B3 uses predicted feasibility only; actual budget violations are retained in JSON. B5b applies the measured-rate fallback; Full uses measured costs.

| Bank/task/codec | Seed/policy | Rows | Before rate % | After rate % | Before identity % | After identity % | Before cost excess | After cost excess | Changed actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scaled/all/ar/h264 | 303401/B3 | 352 | -8.2641 | 0.0000 | 40.6250 | 100.0000 | 0.0379 | 0.0000 | 209 |
| scaled/all/ar/h264 | 303401/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/ar/h264 | 303401/Full | 352 | -2.7231 | -4.3369 | 41.4773 | 28.9773 | -0.3306 | -0.3279 | 70 |
| scaled/all/ar/h264 | 303402/B3 | 352 | -6.9081 | 0.0000 | 51.1364 | 100.0000 | 0.0085 | 0.0000 | 172 |
| scaled/all/ar/h264 | 303402/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/ar/h264 | 303402/Full | 352 | -2.4984 | -3.6600 | 42.3295 | 29.8295 | -0.2974 | -0.2957 | 61 |
| scaled/all/ar/h264 | 303403/B3 | 352 | -7.4301 | 0.0000 | 53.4091 | 100.0000 | 0.0275 | 0.0000 | 164 |
| scaled/all/ar/h264 | 303403/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/ar/h264 | 303403/Full | 352 | -2.9842 | -4.2316 | 42.3295 | 30.9659 | -0.3308 | -0.3291 | 57 |
| scaled/all/ar/h265 | 303401/B3 | 352 | -4.1562 | 0.0000 | 53.1250 | 100.0000 | 0.0257 | 0.0000 | 165 |
| scaled/all/ar/h265 | 303401/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/ar/h265 | 303401/Full | 352 | -1.4274 | -2.5601 | 53.9773 | 37.7841 | -0.2941 | -0.2920 | 69 |
| scaled/all/ar/h265 | 303402/B3 | 352 | -2.1858 | 0.0000 | 74.1477 | 100.0000 | 0.0662 | 0.0000 | 91 |
| scaled/all/ar/h265 | 303402/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/ar/h265 | 303402/Full | 352 | -1.4999 | -2.5923 | 50.0000 | 35.2273 | -0.3054 | -0.2970 | 66 |
| scaled/all/ar/h265 | 303403/B3 | 352 | -1.4725 | 0.0000 | 83.5227 | 100.0000 | 0.0058 | 0.0000 | 58 |
| scaled/all/ar/h265 | 303403/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/ar/h265 | 303403/Full | 352 | -1.2325 | -2.4998 | 53.9773 | 38.3523 | -0.2766 | -0.2745 | 70 |
| scaled/all/od/h264 | 303401/B3 | 352 | -18.4017 | -0.6136 | 0.0000 | 96.5909 | 0.0548 | 0.0000 | 352 |
| scaled/all/od/h264 | 303401/B5b | 352 | 0.0000 | -0.6136 | 100.0000 | 96.5909 | 0.0000 | 0.0000 | 12 |
| scaled/all/od/h264 | 303401/Full | 352 | -15.7913 | -15.7913 | 9.9432 | 9.9432 | -0.0471 | -0.0471 | 0 |
| scaled/all/od/h264 | 303402/B3 | 352 | -18.2396 | 0.0000 | 0.0000 | 100.0000 | 0.0441 | 0.0000 | 352 |
| scaled/all/od/h264 | 303402/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/od/h264 | 303402/Full | 352 | -15.6430 | -15.6430 | 9.9432 | 9.9432 | -0.0479 | -0.0479 | 0 |
| scaled/all/od/h264 | 303403/B3 | 352 | -18.3731 | -0.2429 | 0.0000 | 98.5795 | 0.0457 | -0.0004 | 352 |
| scaled/all/od/h264 | 303403/B5b | 352 | 0.0000 | -0.2429 | 100.0000 | 98.5795 | 0.0000 | -0.0004 | 5 |
| scaled/all/od/h264 | 303403/Full | 352 | -15.7235 | -15.7235 | 10.2273 | 10.2273 | -0.0467 | -0.0467 | 0 |
| scaled/all/od/h265 | 303401/B3 | 352 | -14.4447 | -1.3088 | 0.0000 | 90.3409 | 0.0607 | -0.0049 | 320 |
| scaled/all/od/h265 | 303401/B5b | 352 | -0.5021 | -1.3186 | 96.5909 | 90.3409 | 0.0002 | -0.0046 | 23 |
| scaled/all/od/h265 | 303401/Full | 352 | -12.4145 | -12.4145 | 11.0795 | 11.0795 | -0.0466 | -0.0466 | 0 |
| scaled/all/od/h265 | 303402/B3 | 352 | -14.6291 | 0.0000 | 0.0000 | 100.0000 | 0.0601 | 0.0000 | 352 |
| scaled/all/od/h265 | 303402/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/od/h265 | 303402/Full | 352 | -12.6321 | -12.6321 | 10.7955 | 10.7955 | -0.0474 | -0.0474 | 0 |
| scaled/all/od/h265 | 303403/B3 | 352 | -14.6667 | 0.0000 | 0.0000 | 100.0000 | 0.0623 | 0.0000 | 352 |
| scaled/all/od/h265 | 303403/B5b | 352 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| scaled/all/od/h265 | 303403/Full | 352 | -12.7090 | -12.7090 | 11.6477 | 11.6477 | -0.0440 | -0.0440 | 0 |
| MOT20-02/all/ot/h264 | 303401/B3 | 64 | -8.0476 | 0.0000 | 0.0000 | 100.0000 | 0.0103 | 0.0000 | 64 |
| MOT20-02/all/ot/h264 | 303401/B5b | 64 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| MOT20-02/all/ot/h264 | 303401/Full | 64 | -7.6599 | -7.6599 | 6.2500 | 6.2500 | 0.0015 | 0.0015 | 0 |
| MOT20-02/all/ot/h264 | 303402/B3 | 64 | -8.0476 | 0.0000 | 0.0000 | 100.0000 | 0.0103 | 0.0000 | 64 |
| MOT20-02/all/ot/h264 | 303402/B5b | 64 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| MOT20-02/all/ot/h264 | 303402/Full | 64 | -6.8831 | -6.8831 | 6.2500 | 6.2500 | 0.0002 | 0.0002 | 0 |
| MOT20-02/all/ot/h264 | 303403/B3 | 64 | -8.0476 | 0.0000 | 0.0000 | 100.0000 | 0.0103 | 0.0000 | 64 |
| MOT20-02/all/ot/h264 | 303403/B5b | 64 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| MOT20-02/all/ot/h264 | 303403/Full | 64 | -7.6765 | -7.6765 | 3.1250 | 3.1250 | 0.0004 | 0.0004 | 0 |
| MOT20-02/all/ot/h265 | 303401/B3 | 64 | -9.0318 | 0.0000 | 0.0000 | 100.0000 | 0.0100 | 0.0000 | 64 |
| MOT20-02/all/ot/h265 | 303401/B5b | 64 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| MOT20-02/all/ot/h265 | 303401/Full | 64 | -6.7964 | -6.7964 | 9.3750 | 9.3750 | -0.0001 | -0.0001 | 0 |
| MOT20-02/all/ot/h265 | 303402/B3 | 64 | -9.0318 | 0.0000 | 0.0000 | 100.0000 | 0.0100 | 0.0000 | 64 |
| MOT20-02/all/ot/h265 | 303402/B5b | 64 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| MOT20-02/all/ot/h265 | 303402/Full | 64 | -6.8108 | -6.8108 | 9.3750 | 9.3750 | -0.0010 | -0.0010 | 0 |
| MOT20-02/all/ot/h265 | 303403/B3 | 64 | -9.0318 | 0.0000 | 0.0000 | 100.0000 | 0.0100 | 0.0000 | 64 |
| MOT20-02/all/ot/h265 | 303403/B5b | 64 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0.0000 | 0 |
| MOT20-02/all/ot/h265 | 303403/Full | 64 | -6.3347 | -6.3347 | 9.3750 | 9.3750 | -0.0016 | -0.0016 | 0 |

### scaled_test AR: source-guard limit repair

Same recorded five-candidate shortlists and label-free costs; only limits change. Packet bytes replay exactly. Accuracy is available only when the chosen action has a saved task prediction; matched-subset accuracy is not full-cohort accuracy.

Limits change from `[0, 0.05, 0]` to `[0.025, 0.05, 0.05]`. Every historical decision is checked against its saved action and paid bytes.

| Codec/arm | Rows | Before rate % | After rate % | Before identity % | After identity % | Changed actions | After accuracy coverage % |
|---|---:|---:|---:|---:|---:|---:|---:|
| h264/selected_303401 | 512 | -2.7158 | -5.6821 | 47.4609 | 26.5625 | 152 | 72.8516 |
| h264/selected_303402 | 512 | -2.4692 | -5.0472 | 48.4375 | 28.5156 | 148 | 75.7812 |
| h264/selected_303403 | 512 | -2.3283 | -5.1657 | 49.6094 | 27.1484 | 160 | 72.2656 |
| h265/selected_303401 | 512 | -1.7330 | -4.0454 | 53.5156 | 33.2031 | 140 | 79.1016 |
| h265/selected_303402 | 512 | -1.9296 | -4.1623 | 50.1953 | 28.9062 | 141 | 80.4688 |
| h265/selected_303403 | 512 | -1.7663 | -4.1174 | 52.5391 | 30.8594 | 147 | 78.7109 |

Unobserved selected actions have no imputed accuracy. JSON retains matched-subset accuracy, coverage, all Before/After decisions, CAL bounds, normalization diagnostics, and exact prediction hashes. Mean UCB is a bound on clipped average residuals, not an individual-risk coverage guarantee.

Some frozen risk heads still reject every nonidentity action under B3 or B5b. The report records those identity-only results; applying the revised losses and normalization to new weights requires retraining.

The historical and repaired metric implementations agree exactly on all 12 published curve pairs. Published PCHIP values match exactly: `true`. Maximum point-metric difference from the published snapshot on this runtime: `7.89e-13`. The cubic and bootstrap source files are unchanged. The strict full-merge equality check is retained; `--verify-only` validates the original artifacts by hash.


## Unavailable evidence and limitations

- **k400_publication held-out B2/B4/B5a**: Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed
- **coco_historical held-out B2/B4/B5a**: Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed
- **scaled_test held-out B2/B4/B5a**: Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed
- **coco_rank held-out B2/B4/B5a**: Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed
- **ot_test held-out B2/B4/B5a**: Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed
- **ot_test proxy tracking accuracy**: Rows contain AR source KL, not a GT tracking-accuracy observation; use official sequence TrackEval metrics
- **ot_dev held-out B2/B4/B5a**: Saved arms do not measure all 13 actions per condition; exact unobserved candidate quality cannot be reconstructed
- **ot_dev proxy tracking accuracy**: Rows contain AR source KL, not a GT tracking-accuracy observation; use official sequence TrackEval metrics
- **B2_a5 requested coefficients**: The requested tuple is not action 5 in the saved bank. Tables report the actual measured action 5; the requested coefficient tuple has no saved measurement.
- **coco256_rank controller replay**: 168 of 256 bank source IDs lack saved inputs; partial controller curves are suppressed
- **coco256_scaled_controller controller replay**: 168 of 256 bank source IDs lack saved inputs; partial controller curves are suppressed
- **ByteTrack_ot-dev-A1 256-frame comparison**: Artifact has 2 windows, not 16; keep this run separate
- **ByteTrack_ot-dev-A2 256-frame comparison**: Artifact has 2 windows, not 16; keep this run separate
- **MOT20-05 IDF1/MOTA and exact precision**: LaTeX QP table gives only IDF1/MOTA deltas, no anchor values, and rounded rate/HOTA. Cannot verify claimed -12.88/-6.91 MOTA or exact unrounded HOTA.
- **MOT17 requested held-out 02/05/09/11, 120 windows**: Input report has no MOT17-09, uses MOT17-05 for FIT, flags independent_validation=false and expanded_evaluation_not_executed=true. Strict window-local ablation results are absent.
- **MOT20-01+03+05/h264/IDF1**: Complete comparable per-sequence QP curves absent
- **MOT20-01+03+05/h264/MOTA**: Complete comparable per-sequence QP curves absent
- **MOT20-01+03+05/h265/IDF1**: Complete comparable per-sequence QP curves absent
- **MOT20-01+03+05/h265/MOTA**: Complete comparable per-sequence QP curves absent
- **Official pooled MOT20-01+03+05 and +02**: Sequence-specific TrackEval sufficient statistics/tracks are absent; MOT20-02 bank contains only GT-conditioned window costs, not official HOTA/IDF1/MOTA. Averaging HOTA is not official TrackEval pooling.

All input paths, SHA-256 hashes, complete curves, per-seed metrics, code excerpts with line numbers, and missing-data evidence are in `audit_summary.json`. Re-run `python scripts/verify_and_ablate_v34.py` to regenerate both reports.
