#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import * as THREE from '../../../slm2viewer/node_modules/three/build/three.module.js';
import {
  buildTriangleBvh,
  CompactTriangleStore,
  excludeDegenerateProbeUnits,
  generateExternalHitProbes,
  mergeExternalHitProbeShards,
  readColumnarProbeAsset,
  traceNearestExternalHit,
  visitRenderableTriangleBatches,
} from './generate_external_hit_probes.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..', '..');

function pad4(value) {
  return (value + 3) & ~3;
}

function writeGlb(filePath, json, binary = Buffer.alloc(0)) {
  const rawJson = Buffer.from(JSON.stringify(json), 'utf8');
  const jsonChunk = Buffer.alloc(pad4(rawJson.length), 0x20);
  rawJson.copy(jsonChunk);
  const binaryChunk = binary.length > 0 ? Buffer.alloc(pad4(binary.length)) : null;
  if (binaryChunk) binary.copy(binaryChunk);
  const totalLength = 12 + 8 + jsonChunk.length + (binaryChunk ? 8 + binaryChunk.length : 0);
  const output = Buffer.alloc(totalLength);
  output.writeUInt32LE(0x46546c67, 0);
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(totalLength, 8);
  output.writeUInt32LE(jsonChunk.length, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonChunk.copy(output, 20);
  if (binaryChunk) {
    const binaryHeader = 20 + jsonChunk.length;
    output.writeUInt32LE(binaryChunk.length, binaryHeader);
    output.writeUInt32LE(0x004e4942, binaryHeader + 4);
    binaryChunk.copy(output, binaryHeader + 8);
  }
  fs.writeFileSync(filePath, output);
}

function bounds(min, max) {
  const size = max.map((value, axis) => value - min[axis]);
  const center = min.map((value, axis) => (value + max[axis]) * 0.5);
  return { min, max, center, size };
}

function boxTriangles(min, max) {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const faces = [
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [3, 7, 6], [3, 6, 2],
    [0, 4, 7], [0, 7, 3],
    [1, 2, 6], [1, 6, 5],
  ];
  return faces.map(([a, b, c]) => ({ a: v[a], b: v[b], c: v[c] }));
}

function makeBoxGlb(filePath) {
  const boxes = [boxTriangles([0, 0, 0], [1, 1, 1]), boxTriangles([3, 0, 0], [4, 1, 1])];
  const positions = [];
  const indices = [];
  const positionOffsets = [];
  const indexOffsets = [];
  for (const triangles of boxes) {
    const positionOffset = positions.length * 4;
    for (const triangle of triangles) {
      for (const point of [triangle.a, triangle.b, triangle.c]) positions.push(...point);
    }
    const indexOffset = indices.length * 2;
    for (let index = 0; index < triangles.length * 3; index += 1) indices.push(index);
    positionOffsets.push(positionOffset);
    indexOffsets.push(indexOffset);
  }
  const positionBytes = Buffer.from(new Float32Array(positions).buffer);
  const indexBytes = Buffer.from(new Uint16Array(indices).buffer);
  const indexStart = pad4(positionBytes.length);
  const binary = Buffer.alloc(indexStart + indexBytes.length);
  positionBytes.copy(binary, 0);
  indexBytes.copy(binary, indexStart);
  const positionViewLength = boxes[0].length * 9 * 4;
  const indexViewLength = boxes[0].length * 3 * 2;
  writeGlb(filePath, {
    asset: { version: '2.0', generator: 'v5-external-hit-test' },
    scene: 0,
    scenes: [{ nodes: [0, 1] }],
    nodes: [{ mesh: 0 }, { mesh: 1 }],
    meshes: [
      { primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] },
      { primitives: [{ attributes: { POSITION: 2 }, indices: 3 }] },
    ],
    buffers: [{ byteLength: binary.length }],
    bufferViews: [
      { buffer: 0, byteOffset: positionOffsets[0], byteLength: positionViewLength },
      { buffer: 0, byteOffset: indexStart + indexOffsets[0], byteLength: indexViewLength },
      { buffer: 0, byteOffset: positionOffsets[1], byteLength: positionViewLength },
      { buffer: 0, byteOffset: indexStart + indexOffsets[1], byteLength: indexViewLength },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 36, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 36, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 36, type: 'VEC3' },
      { bufferView: 3, componentType: 5123, count: 36, type: 'SCALAR' },
    ],
  }, binary);
}

function writeSurfaceAsset(directory) {
  const values = new Float32Array(2 * 256 * 6);
  const normalizations = [
    { center: [0.5, 0.5, 0.5], halfDiagonal: Math.sqrt(3) * 0.5 },
    { center: [3.5, 0.5, 0.5], halfDiagonal: Math.sqrt(3) * 0.5 },
  ];
  const normalizedX = 0.5 / normalizations[0].halfDiagonal;
  for (let unitIndex = 0; unitIndex < 2; unitIndex += 1) {
    for (let pointIndex = 0; pointIndex < 256; pointIndex += 1) {
      const offset = (unitIndex * 256 + pointIndex) * 6;
      values[offset] = unitIndex === 0 ? normalizedX : -normalizedX;
      values[offset + 1] = 0;
      values[offset + 2] = 0;
      values[offset + 5] = 1;
    }
  }
  const header = Buffer.alloc(16);
  header.write('GPV5', 0, 'ascii');
  header.writeUInt16LE(1, 4);
  header.writeUInt16LE(6, 6);
  header.writeUInt32LE(2, 8);
  header.writeUInt32LE(256, 12);
  fs.writeFileSync(path.join(directory, 'surface_points_fp32.bin'), Buffer.concat([
    header,
    Buffer.from(values.buffer),
  ]));
  fs.writeFileSync(path.join(directory, 'surface_manifest.json'), JSON.stringify({
    schema: 'gcof-pvs-v5-local-surface-v1',
    version: 1,
    numUnits: 2,
    unitIds: [0, 1],
    files: { points: 'surface_points_fp32.bin' },
    records: normalizations.map((normalization, unitIndex) => ({
      componentGlobalId: unitIndex,
      normalization,
    })),
  }, null, 2));
}

function writeSceneInputs(directory) {
  const emptyStats = {
    visiblePoses: 0,
    visibleRatio: 0,
    weightMean: 0,
    weightP90: 0,
    weightMax: 0,
    priorityPrior: 0,
  };
  fs.writeFileSync(path.join(directory, 'runtimeVisibilityMeta.json'), JSON.stringify({
    schemaVersion: 1,
    componentCount: 2,
    instanceCount: 2,
    globalGlbCount: 1,
    componentRecords: [
      { componentGlobalId: 0, instanceId: 0, globalGlbId: 0, bounds: bounds([0, 0, 0], [1, 1, 1]), rvcStats: emptyStats },
      { componentGlobalId: 1, instanceId: 1, globalGlbId: 0, bounds: bounds([3, 0, 0], [4, 1, 1]), rvcStats: emptyStats },
    ],
    globalGlbRecords: [{ globalGlbId: 0, componentGlobalIds: [0, 1], aabb: bounds([0, 0, 0], [4, 1, 1]), rvcStats: emptyStats }],
  }, null, 2));
  fs.writeFileSync(path.join(directory, 'glbIndex.json'), JSON.stringify({
    version: 1,
    total: 1,
    entries: [{ globalId: 0, path: 'boxes.glb', componentGlobalIds: [0, 1] }],
  }, null, 2));
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'v5-external-hit-'));
try {
  const filteredInput = excludeDegenerateProbeUnits({
    components: [
      { componentGlobalId: 0, globalGlbId: 0 },
      { componentGlobalId: 1, globalGlbId: 0 },
      { componentGlobalId: 2, globalGlbId: 1 },
    ],
    componentById: new Map([
      [0, { componentGlobalId: 0 }],
      [1, { componentGlobalId: 1 }],
      [2, { componentGlobalId: 2 }],
    ]),
    componentIdsByGlb: new Map([[0, [0, 1]], [1, [2]]]),
  }, [1]);
  assert.deepEqual(filteredInput.components.map((unit) => unit.componentGlobalId), [0, 2]);
  assert.deepEqual(filteredInput.componentIdsByGlb.get(0), [0]);
  assert.deepEqual(filteredInput.componentIdsByGlb.get(1), [2]);

  const mixedGeometry = new THREE.BufferGeometry();
  mixedGeometry.setAttribute('position', new THREE.Float32BufferAttribute([
    0, 0, 0, 1e-8, 0, 0, 0, 1e-8, 0,
    0, 0, 0, 1, 0, 0, 0, 1, 0,
  ], 3));
  const mixedScene = new THREE.Scene();
  mixedScene.add(new THREE.Mesh(mixedGeometry, new THREE.MeshBasicMaterial()));
  let retainedTriangles = 0;
  const mixedStats = visitRenderableTriangleBatches(
    THREE,
    { scene: mixedScene },
    [0],
    (batch) => { retainedTriangles += batch.length; },
  );
  assert.equal(mixedStats.sourceTriangleCount, 2);
  assert.equal(mixedStats.degenerateTriangleCount, 1);
  assert.equal(retainedTriangles, 1, 'zero-area faces must not enter component matching or the BVH');
  mixedGeometry.dispose();

  makeBoxGlb(path.join(root, 'boxes.glb'));
  writeSceneInputs(root);
  writeSurfaceAsset(root);

  const unitTriangles = new Map([
    [0, boxTriangles([0, 0, 0], [1, 1, 1])],
    [1, boxTriangles([3, 0, 0], [4, 1, 1])],
  ]);
  const bvhIndex = buildTriangleBvh(THREE, unitTriangles);
  assert.equal('triangles' in bvhIndex, false, 'BVH index must not retain per-triangle JS objects');
  assert.equal(bvhIndex.memoryStats.retainedTriangleObjects, 0);
  const exactHit = traceNearestExternalHit({
    THREE,
    bvhIndex,
    startPoint: [1, 0.5, 0.5],
    targetCenter: [0.5, 0.5, 0.5],
    direction: [1, 0, 0],
    unitId: 0,
    radius: 1,
  });
  assert.ok(exactHit, 'a real triangle hit should be found');
  assert.equal(exactHit.triangleIndex >= 12, true, 'the hit should belong to the other box');
  assert.ok(Math.abs(exactHit.distance - 2) < 1e-5, `unexpected external hit distance ${exactHit.distance}`);
  assert.ok(Math.abs(exactHit.targetCenterDepth - 2.5) < 1e-5, `unexpected target-centered depth ${exactHit.targetCenterDepth}`);
  const shiftedStartHit = traceNearestExternalHit({
    THREE,
    bvhIndex,
    startPoint: [0.75, 0.5, 0.5],
    targetCenter: [0.5, 0.5, 0.5],
    direction: [1, 0, 0],
    unitId: 0,
    radius: 1,
  });
  assert.ok(Math.abs(shiftedStartHit.targetCenterDepth - exactHit.targetCenterDepth) < 1e-5, 'center depth must not depend on surface start');
  const reverseWindingHit = traceNearestExternalHit({
    THREE,
    bvhIndex: buildTriangleBvh(THREE, new Map([[1, [{
      a: [3, 0, 0], b: [3, 1, 0], c: [3, 0, 1],
    }]]])),
    startPoint: [0, 0.2, 0.2],
    targetCenter: [0, 0, 0],
    direction: [1, 0, 0],
    unitId: 0,
    radius: 1,
  });
  assert.ok(reverseWindingHit, 'formal DoubleSide tracing must accept reverse-wound faces');
  const censor = traceNearestExternalHit({
    THREE,
    bvhIndex: buildTriangleBvh(THREE, new Map([[1, boxTriangles([2000, 0, 0], [2001, 1, 1])]])),
    startPoint: [0, 0.5, 0.5],
    targetCenter: [0, 0, 0],
    direction: [1, 0, 0],
    unitId: 0,
    radius: 1,
  });
  assert.equal(censor, null, 'hits past 1024 radii must be right-censored');
  const selfOnly = traceNearestExternalHit({
    THREE,
    bvhIndex: buildTriangleBvh(THREE, new Map([[0, boxTriangles([0, 0, 0], [1, 1, 1])]])),
    startPoint: [1, 0.5, 0.5],
    targetCenter: [0.5, 0.5, 0.5],
    direction: [1, 0, 0],
    unitId: 0,
    radius: 1,
  });
  assert.equal(selfOnly, null, 'all triangles from the current unit must be skipped');

  // Scale smoke: the scene store grows in fixed typed-array chunks.  This
  // deliberately exercises more than one chunk without retaining 10k object
  // records, which is the failure mode seen on the real Viking scene.
  const scaleStore = new CompactTriangleStore({ chunkTriangleCapacity: 257 });
  const scaleTriangle = { a: [0, 0, 0], b: [1, 0, 0], c: [0, 1, 0] };
  for (let index = 0; index < 10000; index += 1) {
    scaleStore.appendTriangle(index % 3, scaleTriangle);
  }
  const scaleStorage = scaleStore.storageStats();
  assert.ok(scaleStorage.chunkCount > 1, 'scale smoke must allocate multiple typed-array chunks');
  assert.equal(scaleStorage.triangleObjectCount, 0);
  assert.equal(scaleStorage.typedArrayBytes, 10000 * (9 * 4 + 4));
  console.log(
    `[scale-smoke] triangles=${scaleStorage.triangleCount}`
    + ` chunks=${scaleStorage.chunkCount}`
    + ` typedArrayBytes=${scaleStorage.typedArrayBytes}`
    + ` retainedTriangleObjects=${scaleStorage.triangleObjectCount}`,
  );
  const scaleBvh = buildTriangleBvh(THREE, scaleStore);
  assert.equal(scaleBvh.memoryStats.retainedTriangleObjects, 0);
  assert.equal(scaleBvh.geometry.getAttribute('position').array instanceof Float32Array, true);
  assert.equal(scaleBvh.triangleUnitIds instanceof Uint32Array, true);
  scaleBvh.geometry.dispose();

  const common = {
    assetsDir: root,
    runtimeMetaPath: path.join(root, 'runtimeVisibilityMeta.json'),
    glbIndexPath: path.join(root, 'glbIndex.json'),
    surfaceManifestPath: path.join(root, 'surface_manifest.json'),
    sceneId: 'two_boxes',
    progressEvery: 0,
    dependencies: { THREE },
  };
  const full = await generateExternalHitProbes({
    ...common,
    outputDir: path.join(root, 'full'),
  });
  assert.equal(full.manifest.schema, 'parallel_external_hit_target_depth_current_status-v2');
  assert.equal(full.manifest.hitDistanceOrigin, 'target_center_directional_projection');
  assert.equal(full.manifest.numUnits, 2);
  assert.equal(full.manifest.rowCount, 2 * 16 * 36);
  assert.equal(full.geometryStats.retainedTriangleObjects, 0);
  assert.equal(full.geometryStats.storage, 'buffer_geometry_float32_positions_uint32_unit_ids');
  assert.deepEqual(Object.keys(full.manifest.files).sort(), [
    'directionIds', 'directions', 'hitDistances', 'maxDistances', 'startIds', 'unitIds',
  ]);
  const fullAsset = readColumnarProbeAsset(full.outputDir, full.manifest);
  assert.equal(fullAsset.values.unitIds.length, full.manifest.rowCount);
  assert.equal(fullAsset.values.directions.length, full.manifest.rowCount * 3);
  assert.equal(fullAsset.values.startIds[15], 15);
  assert.equal(fullAsset.values.startIds[16], 0);
  assert.equal(fullAsset.values.directionIds[15], 0);
  assert.equal(fullAsset.values.directionIds[16], 1);
  assert.equal(fullAsset.values.unitIds[0], 0);
  assert.equal(fullAsset.values.unitIds[16 * 36], 1);
  assert.ok(Array.from(fullAsset.values.hitDistances).some(Number.isFinite), 'two boxes should yield at least one external hit');
  assert.ok(Array.from(fullAsset.values.hitDistances).some(Number.isNaN), 'unobstructed directions should be censored');
  assert.ok(Array.from(fullAsset.values.maxDistances).every((value) => Math.abs(value - Math.sqrt(3) * 0.5 * 1024) < 1e-3));

  const shard0 = await generateExternalHitProbes({
    ...common,
    outputDir: path.join(root, 'shard0'),
    shardIndex: 0,
    shardCount: 2,
  });
  const shard1 = await generateExternalHitProbes({
    ...common,
    outputDir: path.join(root, 'shard1'),
    shardIndex: 1,
    shardCount: 2,
  });
  assert.equal(shard0.manifest.numUnits, 1);
  assert.equal(shard1.manifest.numUnits, 1);
  // Simulate a scene ID space with one degenerate placeholder.  Complete
  // shards cover the valid units [0,2], while sceneNumUnits remains three.
  for (const shard of [shard0, shard1]) {
    shard.manifest.sceneNumUnits = 3;
  }
  shard1.manifest.unitIds = [2];
  shard1.manifest.shard.selectedUnitIds = [2];
  fs.writeFileSync(
    path.join(shard1.outputDir, shard1.manifest.files.unitIds),
    Buffer.from(new Uint32Array(16 * 36).fill(2).buffer),
  );
  for (const shard of [shard0, shard1]) {
    fs.writeFileSync(
      shard.manifestPath,
      `${JSON.stringify(shard.manifest, null, 2)}\n`,
      'utf8',
    );
  }
  const merged = mergeExternalHitProbeShards({
    shardDirs: [shard1.outputDir, shard0.outputDir],
    outputDir: path.join(root, 'merged'),
  });
  const mergedAsset = readColumnarProbeAsset(merged.outputDir, merged.manifest);
  assert.equal(merged.manifest.numUnits, 2);
  assert.equal(merged.manifest.sceneNumUnits, 3);
  assert.deepEqual(merged.manifest.unitIds, [0, 2]);
  assert.deepEqual(Array.from(mergedAsset.values.unitIds.slice(0, 16 * 36)), new Array(16 * 36).fill(0));
  assert.deepEqual(Array.from(mergedAsset.values.unitIds.slice(16 * 36)), new Array(16 * 36).fill(2));
  assert.deepEqual(
    Array.from(mergedAsset.values.hitDistances),
    Array.from(fullAsset.values.hitDistances),
    'shard merge must preserve the full ray columns',
  );
  const resumed = await generateExternalHitProbes({ ...common, outputDir: full.outputDir });
  assert.equal(resumed.resumed, true, 'a complete output should be recoverable without recomputation');

  console.log('V5 external-hit probe tests passed.');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
