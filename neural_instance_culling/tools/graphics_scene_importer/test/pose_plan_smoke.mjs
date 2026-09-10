#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  generatePosePlan,
  main as generatePosePlanMain,
  PosePlanBlockedError,
} from '../generate_pose_plan.mjs';
import { readGltfScene } from '../read_gltf.mjs';
import { createNonClosedShellScene } from './fixture_scene.mjs';

function runtimeMeta(records = []) {
  return {
    schemaVersion: 1,
    sceneName: 'pose_plan_fixture',
    sceneBounds: { min: [0, 0, 0], max: [16, 4, 16] },
    instanceCount: records.length,
    componentRecords: records,
  };
}

function obstacle(id, min, max, extra = {}) {
  return {
    componentGlobalId: id,
    bounds: { min, max },
    ...extra,
  };
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-graphics-scene-pose-plan-g1-'));
try {
  const source = runtimeMeta([
    obstacle(0, [7, 0, 7], [9, 4, 9], { staticPvsEligible: false, alwaysResident: true, alphaMode: 'UNKNOWN' }),
  ]);
  const first = generatePosePlan(source);
  const second = generatePosePlan(source);
  assert.deepEqual(first, second);
  assert.ok(first.rows.length > 0);
  assert.deepEqual(Object.keys(first.audit.spatialSplit.splitPoseCount).sort(), [
    'calibration', 'test', 'train', 'validation',
  ]);
  assert.equal(first.audit.legalRegion.staticPvsEligibleResourceCount, 0);
  assert.equal(first.audit.legalRegion.alwaysResidentResourceCount, 1);
  assert.equal(first.audit.legalRegion.collisionRejectedPositionCount > 0, true);
  assert.deepEqual(first.audit.legalRegion.viewcellHalfExtent, [0.5, 0.5, 0.25]);
  assert.equal(first.audit.legalRegion.viewcellOuterRadius, 0.75);
  assert.equal(first.audit.legalRegion.surfaceEpsilon, 0.05);
  assert.equal(first.audit.legalRegion.safetyRadius, 0.8);
  assert.equal(first.audit.legalRegion.cameraClearance, 0.8);

  const blockSplits = new Map();
  for (const [rowIndex, row] of first.rows.entries()) {
    assert.equal(row.pose_index, rowIndex);
    assert.equal(row.fov_y, 66);
    assert.equal(row.pvs_fov_y, 66);
    assert.equal(row.render_fov_y, 60);
    assert.deepEqual(row.viewcell_half_extent, [0.5, 0.5, 0.25]);
    assert.equal(row.viewcell_radius, 0.75);
    assert.equal(row.width, 512);
    assert.equal(row.height, 288);
    assert.equal(row.aspect, 512 / 288);
    assert.equal(Math.max(1, Math.round(row.height * row.aspect)), row.width);
    assert.equal(row.camera_pos.length, 3);
    assert.equal(row.camera_forward.length, 3);
    assert.ok(row.camera_forward.every(Number.isFinite));
    assert.ok(Math.abs(Math.hypot(...row.camera_forward) - 1) < 1e-6);
    const blockKey = row.spatial_block.join(',');
    if (blockSplits.has(blockKey)) assert.equal(blockSplits.get(blockKey), row.split);
    else blockSplits.set(blockKey, row.split);
    const [x, y, z] = row.camera_pos;
    assert.ok(x < 7 - first.audit.legalRegion.cameraClearance || x > 9 + first.audit.legalRegion.cameraClearance
      || z < 7 - first.audit.legalRegion.cameraClearance || z > 9 + first.audit.legalRegion.cameraClearance
      || y < -first.audit.legalRegion.cameraClearance || y > 4 + first.audit.legalRegion.cameraClearance);
  }
  assert.equal(blockSplits.size, first.audit.legalRegion.occupiedSpatialBlockCount);
  assert.equal(blockSplits.size >= 4, true);

  const runtimePath = path.join(root, 'runtimeVisibilityMeta.json');
  const outputPath = path.join(root, 'pose_plan.jsonl');
  const summaryPath = path.join(root, 'pose_plan_summary.json');
  fs.writeFileSync(runtimePath, `${JSON.stringify(source)}\n`, 'utf8');
  const cliResult = await generatePosePlanMain([
    '--runtime-meta', runtimePath,
    '--output', outputPath,
    '--summary', summaryPath,
  ]);
  assert.equal(cliResult.rows.length, first.rows.length);
  const outputRows = fs.readFileSync(outputPath, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
  assert.deepEqual(outputRows, first.rows);
  assert.equal(JSON.parse(fs.readFileSync(summaryPath, 'utf8')).schema, 'pvs-standard-graphics-pose-plan-audit-v1');

  const shellFixture = createNonClosedShellScene(path.join(root, 'shell'));
  const shellSource = await readGltfScene(shellFixture.gltfPath);
  const shellRuntime = runtimeMeta([
    obstacle(0, [0, 0, 0], [16, 4, 16]),
  ]);
  const shellRuntimePath = path.join(root, 'shell_runtimeVisibilityMeta.json');
  fs.writeFileSync(shellRuntimePath, `${JSON.stringify(shellRuntime)}\n`, 'utf8');
  assert.throws(
    () => generatePosePlan(shellRuntime),
    (error) => error instanceof PosePlanBlockedError && /no collision-free grid positions/.test(error.message),
  );
  const geometryResult = generatePosePlan(shellRuntime, {
    sourceScene: shellSource,
    sourceScenePath: shellFixture.gltfPath,
  });
  assert.equal(geometryResult.audit.legalRegion.mode, 'geometry');
  assert.equal(geometryResult.audit.legalRegion.sourceSceneTriangleCount, 8);
  assert.ok(geometryResult.rows.length > 0);
  assert.ok(geometryResult.rows.some((row) => (
    row.camera_pos[0] > 3 && row.camera_pos[0] < 13
      && row.camera_pos[2] > 3 && row.camera_pos[2] < 13
  )));
  const geometryOutput = path.join(root, 'geometry_pose_plan.jsonl');
  await generatePosePlanMain([
    '--runtime-meta', shellRuntimePath,
    '--source-scene', shellFixture.gltfPath,
    '--output', geometryOutput,
  ]);
  assert.equal(fs.readFileSync(geometryOutput, 'utf8').trim().split('\n').length, geometryResult.rows.length);

  assert.throws(
    () => generatePosePlan(runtimeMeta([{}])),
    (error) => error instanceof PosePlanBlockedError && /bounds/.test(error.message),
  );
  assert.throws(
    () => generatePosePlan(runtimeMeta([obstacle(0, [0, 0, 0], [16, 4, 16])])),
    (error) => error instanceof PosePlanBlockedError && /no collision-free grid positions/.test(error.message),
  );
  assert.throws(
    () => generatePosePlan({ sceneBounds: { min: [0, 0, 0], max: [16, 4, 16] }, componentRecords: [] }),
    (error) => error instanceof PosePlanBlockedError && /empty/.test(error.message),
  );
  console.log(JSON.stringify({ status: 'ok', poses: first.rows.length, output: outputPath }, null, 2));
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
