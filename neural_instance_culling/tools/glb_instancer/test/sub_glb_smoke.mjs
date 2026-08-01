import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { inspectGlbStructure, writePrototypeGlb } from '../src/glb.mjs';
import { buildSubGlbInstancedScene } from '../src/sub_pipeline.mjs';

function rotatePoint(point, quaternion) {
  const [qx, qy, qz, qw] = quaternion;
  const [x, y, z] = point;
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [
    x + qw * tx + (qy * tz - qz * ty),
    y + qw * ty + (qz * tx - qx * tz),
    z + qw * tz + (qx * ty - qy * tx),
  ];
}

function normalizedQuaternion(axis, angle) {
  const length = Math.hypot(...axis);
  const sine = Math.sin(angle * 0.5) / length;
  return [axis[0] * sine, axis[1] * sine, axis[2] * sine, Math.cos(angle * 0.5)];
}

function writeComponent(filePath, vertices, triangles, color) {
  writePrototypeGlb(filePath, {
    name: path.basename(filePath, '.glb'),
    positions: Float32Array.from(vertices.flat()),
    colors: Uint8Array.from(vertices.flatMap(() => color)),
    indices: Uint32Array.from(triangles.flat()),
    instances: [{ translation: [0, 0, 0], rotation: [0, 0, 0, 1] }],
  });
}

const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-sub-glb-instancer-smoke-'));
try {
  const sourceAssets = path.join(tempRoot, 'source', 'assets');
  const sourceGlbDir = path.join(sourceAssets, 'task-0', 'glb', 'LOD0');
  const outputAssets = path.join(tempRoot, 'output', 'assets');
  fs.mkdirSync(sourceGlbDir, { recursive: true });
  const prototype = [[0, 0, 0], [1.2, 0.1, 0], [0.2, 1.4, 0.3], [0.1, 0.2, 1.8]];
  const triangles = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
  writeComponent(path.join(sourceGlbDir, 'sub_10.glb'), prototype, triangles, [180, 90, 40, 255]);
  const quaternion = normalizedQuaternion([1, 2, 3], 0.73);
  const transformed = prototype.map((point) => {
    const rotated = rotatePoint(point, quaternion);
    return [rotated[0] + 5.2, rotated[1] - 2.4, rotated[2] + 8.1];
  });
  writeComponent(path.join(sourceGlbDir, 'sub_20.glb'), transformed, triangles, [180, 90, 40, 255]);
  writeComponent(
    path.join(sourceGlbDir, 'sub_30.glb'),
    [[-3, 0, 0], [-1, 0, 0], [-2, 2, 0]],
    [[0, 1, 2]],
    [20, 120, 220, 255],
  );
  fs.writeFileSync(path.join(sourceAssets, 'glbIndex.json'), JSON.stringify({
    version: 1,
    entries: [10, 20, 30].map((id) => ({
      globalId: id,
      taskId: 0,
      baseId: id,
      path: `task-0/glb/LOD0/sub_${id}.glb`,
    })),
  }));
  fs.writeFileSync(path.join(sourceAssets, 'sceneWeb.json'), JSON.stringify({
    materials: { proxy: [], useHdrJpg: false },
    groups: [{ idRange: [10, 30], instances: {} }],
    config: { sceneName: 'sub_source' },
  }));

  const result = await buildSubGlbInstancedScene({
    sourceAssets,
    outputAssets,
    sceneName: 'sub_instanced_smoke',
    lod: 'LOD0',
    report: '',
    fingerprintTolerance: 0.0001,
    rmsTolerance: 0.0001,
    maxTolerance: 0.0005,
    regionProgressEvery: 0,
    analyzeOnly: false,
    overwrite: true,
  });
  assert.equal(result.componentCount, 3);
  assert.equal(result.prototypeCount, 2);
  assert.equal(result.reusedComponentCount, 1);
  const sceneWeb = JSON.parse(fs.readFileSync(path.join(outputAssets, 'sceneWeb.json'), 'utf8'));
  assert.deepEqual(sceneWeb.groups[0].idRange, [10, 30]);
  assert.equal(sceneWeb.groups[0].instances['20'], 10);
  const glbIndex = JSON.parse(fs.readFileSync(path.join(outputAssets, 'glbIndex.json'), 'utf8'));
  const repeatedEntry = glbIndex.entries.find((entry) => entry.baseId === 10);
  assert.ok(repeatedEntry);
  const structure = inspectGlbStructure(path.join(outputAssets, repeatedEntry.path));
  assert.deepEqual(structure.extensionsUsed, ['EXT_mesh_gpu_instancing']);
  console.log(JSON.stringify({
    status: 'ok',
    components: result.componentCount,
    prototypes: result.prototypeCount,
    reusedComponents: result.reusedComponentCount,
  }, null, 2));
} finally {
  fs.rmSync(tempRoot, { recursive: true, force: true });
}
