#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  attachWorkloadProvenance,
  attachRegionSamplingToResult,
  combineTimingResults,
  buildWorkload,
  formalGpuFailure,
  formalProtocolReady,
  parseArgs,
  readPoseIndexPlan,
  selectRegionSubposes,
  summarizeTimingRounds,
} from './run_geometry_shell_hzb_benchmark.mjs';

function writeTypedArray(filePath, values) {
  fs.writeFileSync(filePath, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

function makeSyntheticRegionWorkload({ sceneName = 'synthetic', cameraView = null } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'geometry-shell-hzb-runner-test-'));
  const shellDir = path.join(root, 'shell');
  const datasetDir = path.join(root, 'dataset');
  const regionDir = path.join(root, 'region');
  fs.mkdirSync(shellDir);
  fs.mkdirSync(datasetDir);
  fs.mkdirSync(regionDir);
  fs.writeFileSync(path.join(shellDir, 'shell_meta.json'), JSON.stringify({
    schema: 'geometry-shell-hzb-v2',
    sceneName,
    instanceCount: 4,
  }));
  fs.writeFileSync(path.join(datasetDir, 'dataset_meta.json'), JSON.stringify({
    poseCount: 1,
    numInstances: 4,
    splitIds: { test: 2 },
  }));
  const pose = Buffer.alloc(64);
  pose.writeFloatLE(0, 12);
  pose.writeFloatLE(0, 16);
  pose.writeFloatLE(0, 20);
  pose.writeFloatLE(0, 24);
  pose.writeFloatLE(0, 28);
  pose.writeFloatLE(-1, 32);
  if (cameraView) {
    pose.writeFloatLE(cameraView[0], 36);
    pose.writeFloatLE(cameraView[1], 40);
  }
  pose.writeUInt8(2, 44);
  fs.writeFileSync(path.join(datasetDir, 'poses.bin'), pose);
  writeTypedArray(path.join(datasetDir, 'candidate_offsets.bin'), new BigUint64Array([0n, 4n]));
  writeTypedArray(path.join(datasetDir, 'candidate_ids.bin'), new Uint32Array([0, 1, 2, 3]));
  writeTypedArray(path.join(datasetDir, 'query_center_world.bin'), new Float32Array([0, 0, 0]));
  fs.writeFileSync(path.join(regionDir, 'dataset_meta.json'), JSON.stringify({ viewcellCount: 1 }));
  writeTypedArray(path.join(regionDir, 'subpose_offsets.bin'), new BigUint64Array([0n, 4n]));
  writeTypedArray(path.join(regionDir, 'viewcell_centers.bin'), new Float32Array([0, 0, 0]));
  writeTypedArray(path.join(regionDir, 'subpose_pose_indices.bin'), new Uint32Array([10, 11, 12, 13]));
  writeTypedArray(path.join(regionDir, 'subpose_camera_pos.bin'), new Float32Array([
    0, 0, 0.1,
    1, 0, 0,
    -10, 0, 0,
    0, 0, 5,
  ]));
  writeTypedArray(path.join(regionDir, 'subpose_camera_forward.bin'), new Float32Array([
    0, 0, -1,
    0, 0, -1,
    0, 0, -1,
    0, 0, -1,
  ]));
  return { root, shellDir, datasetDir, regionDir };
}

const subposes = [
  { ordinal: 0, position: [0, 0, 0.1] },
  { ordinal: 1, position: [1, 0, 0] },
  { ordinal: 2, position: [-10, 0, 0] },
  { ordinal: 3, position: [0, 0, 5] },
];
assert.deepEqual(
  selectRegionSubposes(subposes, [0, 0, 0], 1).map((value) => value.ordinal),
  [0],
);
assert.deepEqual(
  selectRegionSubposes(subposes, [0, 0, 0], 0).map((value) => value.ordinal),
  [0, 1, 2, 3],
);
assert.deepEqual(
  selectRegionSubposes(subposes, [0, 0, 0], 5).map((value) => value.ordinal),
  [0, 2, 3, 1],
);
assert.deepEqual(
  selectRegionSubposes([
    { ordinal: 4, position: [1, 0, 0] },
    { ordinal: 2, position: [-1, 0, 0] },
  ], [0, 0, 0], 9).map((value) => value.ordinal),
  [2, 4],
);
assert.throws(() => selectRegionSubposes(subposes, [0, 0, 0], 2), /one of 0, 1, 5 or 9/);

const synthetic = makeSyntheticRegionWorkload();
try {
  const commonArgs = [
    'node', fileURLToPath(new URL('./run_geometry_shell_hzb_benchmark.mjs', import.meta.url)),
    '--shell-dir', synthetic.shellDir,
    '--dataset-dir', synthetic.datasetDir,
    '--region-dataset-dir', synthetic.regionDir,
    '--output', path.join(synthetic.root, 'out', 'region.json'),
    '--mode', 'Region66',
    '--limit', '1',
    '--width', '8',
    '--height', '8',
  ];
  const sampledOptions = parseArgs([...commonArgs, '--region-sample-count', '5']);
  assert.equal(sampledOptions.regionSampleCount, 5);
  assert.equal(sampledOptions.timingRounds, 5);
  const sampled = buildWorkload(sampledOptions);
  assert.equal(sampled.workload.aspectSource, 'fixed-aspect-fallback');
  assert.equal(sampled.workload.poseSelection.representative, false);
  const sampledRecord = sampled.workload.poses[0].regionSampling;
  assert.deepEqual(sampledRecord, {
    requestedCount: 5,
    availableSubposeCount: 4,
    selectedSubposeCount: 4,
    selectionMode: 'deterministic-fps-subset',
    strategy: 'canonical-nearest-then-farthest-point-world-position',
    labelSource: 'none',
    selectedSubposeOrdinals: [0, 2, 3, 1],
    selectedSourcePoseIndices: [10, 12, 13, 11],
  });
  assert.equal(sampled.workload.regionSampling.availableSubposeCount, 4);
  assert.equal(sampled.workload.regionSampling.selectedSubposeCount, 4);
  assert.equal(sampled.workload.regionSampling.labelSource, 'none');
  assert.equal(fs.existsSync(path.join(synthetic.datasetDir, 'visible_ids.bin')), false);
  const enriched = attachRegionSamplingToResult({
    schema: 'geometry-shell-hzb-browser-result-v2',
    workload: { poseCount: 1 },
    samples: [{ poseId: 0, visibleCount: 4 }],
  }, sampled.workload);
  assert.equal(enriched.regionSampling.selectedSubposeCount, 4);
  assert.equal(enriched.workload.regionSampling.selectionMode, 'deterministic-fps-subset');
  assert.equal(enriched.samples[0].regionSampling.availableSubposeCount, 4);
  const withProvenance = attachWorkloadProvenance({
    schema: 'geometry-shell-hzb-browser-result-v2',
    workload: {},
  }, sampled.workload);
  assert.equal(withProvenance.workload.mode, 'Region66');
  assert.equal(withProvenance.workload.split, 'test');

  const allOptions = parseArgs([...commonArgs, '--output', path.join(synthetic.root, 'out', 'all.json')]);
  const all = buildWorkload(allOptions);
  assert.equal(allOptions.regionSampleCount, 0);
  assert.deepEqual(all.workload.poses[0].regionSampling.selectedSubposeOrdinals, [0, 1, 2, 3]);
  assert.equal(all.workload.regionSampling.selectionMode, 'all');
  assert.equal(all.workload.regionSampling.strategy, 'all-subposes-in-source-order');
  assert.equal(all.workload.regionSampling.availableSubposeCount, 4);
  assert.equal(all.workload.regionSampling.selectedSubposeCount, 4);

  const planPath = path.join(synthetic.root, 'pose-index-plan.json');
  fs.writeFileSync(planPath, JSON.stringify({
    schema: 'geometry-shell-hzb-timing-pose-index-plan-v1',
    split: 'test',
    selection: 'candidate-count-stratified-evenly-spaced',
    poseIndices: [0],
  }));
  const plan = readPoseIndexPlan(planPath);
  assert.deepEqual(plan.poseIndices, [0]);
  const plannedOptions = parseArgs([
    ...commonArgs,
    '--output', path.join(synthetic.root, 'out', 'planned.json'),
    '--pose-index-plan', planPath,
    '--aspect', '2',
  ]);
  const planned = buildWorkload(plannedOptions);
  assert.equal(planned.workload.poseSelection.source, 'explicit-pose-index-plan');
  assert.deepEqual(planned.workload.poseSelection.selectedPoseIndices, [0]);
  assert.deepEqual(planned.workload.poseIndexPlan.poseIndices, [0]);
  assert.equal(planned.workload.provenance.configuration.depthBiasM, 0.001);
  assert.deepEqual(planned.workload.provenance.configuration.resolution, [8, 8]);
  assert.equal(planned.workload.provenance.aspect.fixedAspectUsed, true);
  assert.equal(
    planned.workload.provenance.timingDefinition.stages.total,
    'depthRaster + hzbBuild + aabbTest + compaction',
  );

  const roundResults = [
    { samples: [{ poseId: 0, candidateCount: 4, timings: {
      depthRasterMs: 1, hzbBuildMs: 2,
      aabbTestMs: 3, compactionMs: 4, totalMs: 10,
    } }] },
    { samples: [{ poseId: 0, candidateCount: 4, timings: {
      depthRasterMs: 2, hzbBuildMs: 3,
      aabbTestMs: 4, compactionMs: 5, totalMs: 14,
    } }] },
  ];
  const timingSummary = summarizeTimingRounds(roundResults);
  assert.equal(timingSummary.roundCount, 2);
  assert.equal(timingSummary.sampleCount, 2);
  assert.equal(timingSummary.stages.depthRaster.p50Ms, 1.5);
  assert.equal(timingSummary.stages.total.p95Ms, 13.8);
  const combined = combineTimingResults(roundResults, planned.workload);
  assert.equal(combined.summary.totalP50Ms, 12);
  assert.equal(combined.summary.totalP95Ms, 13.8);
  assert.equal(combined.timingRounds.length, 2);
  assert.equal(combined.timingRounds[1].samples[0].round, 2);

  assert.equal(
    formalGpuFailure({ requireHardwareGpu: true }, false),
    'Geometry-shell HZB hardware GPU gate or execution evidence failed.',
  );
  assert.equal(formalGpuFailure({ requireHardwareGpu: true }, false, 'existing failure'), 'existing failure');
  assert.equal(formalGpuFailure({ requireHardwareGpu: false }, false), null);
  assert.equal(
    formalProtocolReady(
      { timingRounds: 1, poseIndexPlan: '' },
      { poseSelection: { representative: true } },
    ),
    true,
  );
  assert.equal(
    formalProtocolReady(
      { timingRounds: 1, poseIndexPlan: planPath },
      { poseSelection: { representative: true } },
    ),
    false,
  );
  assert.equal(
    formalProtocolReady(
      { timingRounds: 5, poseIndexPlan: planPath },
      { poseSelection: { representative: true } },
    ),
    true,
  );

  assert.throws(
    () => parseArgs([...commonArgs, '--region-sample-count', '2']),
    /one of 0, 1, 5 or 9/,
  );
} finally {
  fs.rmSync(synthetic.root, { recursive: true, force: true });
}

const aspectSynthetic = makeSyntheticRegionWorkload({ cameraView: [2, 1] });
try {
  const options = parseArgs([
    'node', fileURLToPath(new URL('./run_geometry_shell_hzb_benchmark.mjs', import.meta.url)),
    '--shell-dir', aspectSynthetic.shellDir,
    '--dataset-dir', aspectSynthetic.datasetDir,
    '--region-dataset-dir', aspectSynthetic.regionDir,
    '--output', path.join(aspectSynthetic.root, 'out', 'aspect.json'),
    '--mode', 'Region66',
    '--width', '8',
    '--height', '8',
    '--aspect', '9',
  ]);
  const workload = buildWorkload(options).workload;
  assert.equal(workload.aspectSource, 'dataset.camera_view');
  assert.equal(workload.aspect, null);
  assert.equal(workload.poses[0].aspect, 2);
  assert.deepEqual(workload.poses[0].cameraView, [2, 1]);
  assert.equal(workload.provenance.aspect.fixedAspectUsed, false);
  assert.equal(workload.provenance.aspect.fixedAspectProvided, true);
  assert.equal(workload.provenance.poseAspects[0].aspect, 2);
  assert.deepEqual(workload.provenance.configuration.resolution, [8, 8]);
  assert.equal(workload.provenance.configuration.near, 0.05);
  assert.equal(workload.provenance.configuration.far, 20000);
} finally {
  fs.rmSync(aspectSynthetic.root, { recursive: true, force: true });
}

const ifcFallbackSynthetic = makeSyntheticRegionWorkload({
  sceneName: 'ifcbench_fantasy_metropolis_instanced_v2',
});
try {
  const options = parseArgs([
    'node', fileURLToPath(new URL('./run_geometry_shell_hzb_benchmark.mjs', import.meta.url)),
    '--shell-dir', ifcFallbackSynthetic.shellDir,
    '--dataset-dir', ifcFallbackSynthetic.datasetDir,
    '--region-dataset-dir', ifcFallbackSynthetic.regionDir,
    '--output', path.join(ifcFallbackSynthetic.root, 'out', 'ifc.json'),
    '--mode', 'Region66',
    '--width', '8',
    '--height', '8',
    '--aspect', '2',
  ]);
  assert.throws(
    () => buildWorkload(options),
    /IFCBench formal workloads require per-pose dataset camera_view aspect/,
  );
} finally {
  fs.rmSync(ifcFallbackSynthetic.root, { recursive: true, force: true });
}

console.log('Geometry-shell HZB runner tests passed.');
