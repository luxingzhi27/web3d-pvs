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
  assert.deepEqual(Object.keys(first.audit.centerGroupSplit.splitPoseCount).sort(), [
    'calibration', 'test', 'train', 'validation',
  ]);
  assert.equal(first.audit.legalRegion.staticPvsEligibleResourceCount, 0);
  assert.equal(first.audit.legalRegion.alwaysResidentResourceCount, 1);
  assert.equal(first.audit.legalRegion.collisionRejectedPositionCount > 0, true);
  assert.equal(first.audit.legalRegion.viewcellShape, 'horizontal_disk');
  assert.equal(first.audit.legalRegion.viewcellRadius, 0.75);
  assert.equal(first.audit.legalRegion.surfaceEpsilon, 0.05);
  assert.equal(first.audit.legalRegion.safetyRadius, 0.8);
  assert.equal(first.audit.legalRegion.cameraClearance, 0.8);

  const centerSplits = new Map();
  for (const [rowIndex, row] of first.rows.entries()) {
    assert.equal(row.pose_index, rowIndex);
    assert.equal(row.fov_y, 66);
    assert.equal(row.pvs_fov_y, 66);
    assert.equal(row.render_fov_y, 60);
    assert.equal(row.viewcell_shape, 'horizontal_disk');
    assert.deepEqual(row.viewcell_half_extent, [0.75, 0.75, 0]);
    assert.equal(row.viewcell_radius, 0.75);
    assert.ok(Math.abs(row.pvs_back_offset - 0.75 / Math.tan(Math.PI / 6)) < 1e-12);
    assert.equal(row.width, 512);
    assert.equal(row.height, 288);
    assert.equal(row.aspect, 512 / 288);
    assert.equal(Math.max(1, Math.round(row.height * row.aspect)), row.width);
    assert.equal(row.camera_pos.length, 3);
    assert.equal(row.camera_forward.length, 3);
    assert.ok(row.camera_forward.every(Number.isFinite));
    assert.ok(Math.abs(Math.hypot(...row.camera_forward) - 1) < 1e-6);
    if (centerSplits.has(row.position_index)) assert.equal(centerSplits.get(row.position_index), row.split);
    else centerSplits.set(row.position_index, row.split);
    const [x, y, z] = row.camera_pos;
    assert.ok(x < 7 - first.audit.legalRegion.cameraClearance || x > 9 + first.audit.legalRegion.cameraClearance
      || z < 7 - first.audit.legalRegion.cameraClearance || z > 9 + first.audit.legalRegion.cameraClearance
      || y < -first.audit.legalRegion.cameraClearance || y > 4 + first.audit.legalRegion.cameraClearance);
  }
  assert.equal(centerSplits.size, first.audit.legalRegion.legalPositionCount);
  assert.equal(centerSplits.size >= 10, true);
  assert.equal(first.audit.centerGroupSplit.allOrientationsAtOneCenterStayInOneSplit, true);

  const denser = generatePosePlan(source, {
    gridDivisions: [20, 5, 20],
    splitSeed: 20260913,
  });
  assert.deepEqual(denser.audit.legalRegion.gridDivisions, [20, 5, 20]);
  assert.equal(denser.audit.centerGroupSplit.seed, 20260913);
  assert.notEqual(denser.audit.legalRegion.legalPositionCount, centerSplits.size);

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

  const groundPositions = new Float32Array([
    0, 0, 0,
    16, 0, 0,
    16, 0, 16,
    0, 0, 16,
  ]);
  const groundScene = {
    renderables: [
      {
        sourceNodePath: 'Content/Terrain/terrain_near',
        positions: groundPositions,
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        bounds: { min: [0, 0, 0], max: [16, 0, 16] },
      },
      {
        sourceNodePath: 'Content/Terrain/terrain_far',
        positions: new Float32Array([-100, -1, -100, 100, -1, -100, 0, -1, 100]),
        indices: new Uint32Array([0, 1, 2]),
        bounds: { min: [-100, -1, -100], max: [100, -1, 100] },
      },
    ],
  };
  const groundRuntime = runtimeMeta([
    obstacle(0, [0, 0, 0], [16, 0, 16], { sourceNodePath: 'Content/Terrain/terrain_near' }),
    obstacle(1, [-100, -1, -100], [100, -1, 100], { sourceNodePath: 'Content/Terrain/terrain_far' }),
    obstacle(2, [7, 0, 7], [9, 4, 9], { sourceNodePath: 'Content/Buildings/house' }),
  ]);
  groundRuntime.sceneBounds = { min: [-100, -1, -100], max: [100, 4, 100] };
  const groundResult = generatePosePlan(groundRuntime, {
    sourceScene: groundScene,
    placementMode: 'ground_surface_grid',
  });
  assert.equal(groundResult.audit.legalRegion.mode, 'ground_surface_grid');
  assert.deepEqual(groundResult.audit.legalRegion.gridDivisions, [24, 1, 24]);
  assert.equal(groundResult.audit.legalRegion.groundRenderableCount, 1);
  assert.equal(groundResult.audit.legalRegion.groundTriangleCount, 2);
  assert.equal(groundResult.audit.legalRegion.groundCameraHeight, 1.7);
  assert.equal(groundResult.audit.legalRegion.noGroundIntersectionPositionCount, 0);
  assert.equal(groundResult.audit.legalRegion.cameraDomainBounds.min[0], 0);
  assert.equal(groundResult.audit.legalRegion.cameraDomainBounds.max[0], 16);
  assert.ok(groundResult.rows.length > 0);
  assert.ok(groundResult.rows.every((row) => row.sample_category === 'ground_surface_grid'));
  assert.ok(groundResult.rows.every((row) => Math.abs(row.camera_pos[1] - 1.7) < 1e-6));
  assert.ok(groundResult.rows.every((row) => (
    row.camera_pos[0] < 7 - 0.8 || row.camera_pos[0] > 9 + 0.8
      || row.camera_pos[2] < 7 - 0.8 || row.camera_pos[2] > 9 + 0.8
  )));
  const denserGround = generatePosePlan(groundRuntime, {
    sourceScene: groundScene,
    placementMode: 'ground_surface_grid',
    groundGridDivisions: [32, 32],
    splitSeed: 20260913,
  });
  assert.deepEqual(denserGround.audit.legalRegion.gridDivisions, [32, 1, 32]);
  assert.equal(denserGround.audit.centerGroupSplit.seed, 20260913);

  assert.throws(
    () => generatePosePlan(groundRuntime, { placementMode: 'ground_surface_grid' }),
    (error) => error instanceof PosePlanBlockedError && /requires --source-scene/.test(error.message),
  );

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
