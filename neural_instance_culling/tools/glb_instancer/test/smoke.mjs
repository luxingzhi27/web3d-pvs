import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { inspectGlbStructure, writePrototypeGlb } from '../src/glb.mjs';
import { buildInstancedScene } from '../src/pipeline.mjs';

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

function appendRegion(target, vertices, triangles, color) {
  const base = target.positions.length / 3;
  for (const vertex of vertices) target.positions.push(...vertex);
  for (let i = 0; i < vertices.length; i += 1) target.colors.push(...color);
  for (const triangle of triangles) target.indices.push(base + triangle[0], base + triangle[1], base + triangle[2]);
}

const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-glb-instancer-smoke-'));
try {
  const input = path.join(tempRoot, 'input.glb');
  const outputAssets = path.join(tempRoot, 'output', 'assets');
  const mesh = { positions: [], colors: [], indices: [] };
  const prototype = [[0, 0, 0], [1.2, 0.1, 0], [0.2, 1.4, 0.3], [0.1, 0.2, 1.8]];
  const triangles = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
  appendRegion(mesh, prototype, triangles, [180, 90, 40, 255]);
  const quaternion = normalizedQuaternion([1, 2, 3], 0.73);
  const translated = prototype.map((point) => {
    const rotated = rotatePoint(point, quaternion);
    return [rotated[0] + 5.2, rotated[1] - 2.4, rotated[2] + 8.1];
  });
  appendRegion(mesh, translated, triangles, [180, 90, 40, 255]);
  appendRegion(mesh, [[-3, 0, 0], [-1, 0, 0], [-2, 2, 0]], [[0, 1, 2]], [20, 120, 220, 255]);
  writePrototypeGlb(input, {
    name: 'smoke_input',
    positions: Float32Array.from(mesh.positions),
    colors: Uint8Array.from(mesh.colors),
    indices: Uint32Array.from(mesh.indices),
    instances: [{ translation: [0, 0, 0], rotation: [0, 0, 0, 1] }],
  });

  const result = await buildInstancedScene({
    input,
    outputAssets,
    sceneName: 'smoke_instanced',
    report: '',
    fingerprintTolerance: 0.0001,
    rmsTolerance: 0.0001,
    maxTolerance: 0.0005,
    triangleProgressEvery: 0,
    regionProgressEvery: 0,
    analyzeOnly: false,
    overwrite: true,
  });
  assert.equal(result.connectedRegionCount, 3);
  assert.equal(result.prototypeCount, 2);
  assert.equal(result.reusedRegionCount, 1);
  const sceneWeb = JSON.parse(fs.readFileSync(path.join(outputAssets, 'sceneWeb.json'), 'utf8'));
  assert.equal(Object.keys(sceneWeb.groups[0].instances).length, 1);
  const glbIndex = JSON.parse(fs.readFileSync(path.join(outputAssets, 'glbIndex.json'), 'utf8'));
  assert.equal(glbIndex.entries.length, 2);
  const repeatedEntry = glbIndex.entries.find((entry) => entry.baseId === 0);
  assert.ok(repeatedEntry);
  const structure = inspectGlbStructure(path.join(outputAssets, repeatedEntry.path));
  assert.deepEqual(structure.extensionsUsed, ['EXT_mesh_gpu_instancing']);
  console.log(JSON.stringify({ status: 'ok', connectedRegions: 3, prototypes: 2, repeatedStructure: structure }, null, 2));
} finally {
  fs.rmSync(tempRoot, { recursive: true, force: true });
}

