# GCOF-PVS V5 Benchmark Layer

This directory owns V5 score evaluation, calibration, and result aggregation.
It does not train a model and does not read V4 aggregate summaries.

## Input Contract

Each JSON bundle uses schema `gcof-pvs-v5-score-bundle-v1`:

```json
{
  "schema": "gcof-pvs-v5-score-bundle-v1",
  "protocol": "shared",
  "variant": "FULL",
  "seed": 20260918,
  "testRead": false,
  "scenes": {
    "scene-a": {
      "calibration": [
        {
          "poseId": "0",
          "candidateIds": [0, 1],
          "scores": [0.91, 0.08],
          "targets": [1, 0],
          "visibleWeights": [1.0, 0.0]
        }
      ],
      "validation": []
    }
  }
}
```

The actual V5 bundle contains five scenes and non-empty validation rows. A LOSO
bundle additionally declares `heldOutScene` and the four `sourceScenes`.
`test` rows may be present for a final frozen replay, but calibration and model
selection reject any `testRead=true` or test selection provenance.

## Threshold Protocol

`source_global` scans one common set of score change points from the four
source calibration splits. It first selects the highest threshold for which
every source has weighted recall strictly above `0.99` and a one-sided 95%
pose-bootstrap LCB strictly above `0.99`. If that tier is empty, it selects
the highest common threshold for which every source point estimate still has
weighted recall strictly above `0.99`, and marks the result `mean_target`.
If even that tier is empty, the highest weighted-recall diagnostic point is
kept and marked `diagnostic`.

`target_calibrated` applies the same selector to the held-out scene's
calibration rows only. It updates no weights. Shared in-domain runs use this
scene-local calibration while keeping the same checkpoint across all five
scenes.

Validation and test only replay a `calibration`-frozen threshold. Test rows
are never used for model or threshold selection.

## Reported Metrics

Each scene is evaluated before aggregation. The result matrix reports weighted
recall and its LCB, ordinary recall, `FN/GT`, Bad Cull (`FN/candidate`), CNOR,
Useful Cull (`TN/candidate`), `FP/GT`, `pred/GT`, pose-macro PR-AUC, pose
positive prevalence, and AP lift. Aggregate precision, specificity, balanced
accuracy, F1, Jaccard, counts, and aggregate AP are retained as diagnostics.
Cross-scene summaries are arithmetic scene-equal means; candidates from large
scenes are never pooled into the main cross-scene conclusion.

The shared matrix is `FULL`, `GEOMETRY_FIELD`, `GENERIC_RELATION_28`, and
`PBCE_OBJECTIVE`, each with three seeds over five scenes. The LOSO matrix is
`FULL`, `GEOMETRY_FIELD`, and `GENERIC_RELATION_28`, each with three seeds over
five held-out folds and both threshold modes.

Example command:

```bash
python -m neural_instance_culling.benchmark.v5.cli \
  --protocol shared \
  --bundle shared_full_seed1.json \
  --bundle shared_full_seed2.json \
  --output shared_validation.json
```

The implementation was added on 2026-09-18 to make the V5 evaluation
provenance explicit and independent from the frozen V4 benchmark path.
