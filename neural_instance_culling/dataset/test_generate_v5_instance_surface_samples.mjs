#!/usr/bin/env node
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  POINTS_PER_UNIT,
  POINT_FEATURE_DIM,
  POINT_RECORD_BYTES,
  SURFACE_BINARY_HEADER_BYTES,
  collectGarbageIfAvailable,
  DEFAULT_GC_EVERY,
  generateV5InstanceSurfaceSamples,
  parseArgs,
  parseDenseFp32Header,
  sampleUnitSurfaceFeatures,
  stripGlbMaterials,
} from './generate_v5_instance_surface_samples.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUTPUT_FILES = [
  'surface_points_fp32.bin',
  'size_ratios_fp32.bin',
  'aabb_min_fp32.npy',
  'aabb_max_fp32.npy',
  'unit_ids_uint64.npy',
  'surface_manifest.json',
];

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

function writeTwoComponentGlb(filePath) {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    10, 0, 0,
    11, 0, 0,
    10, 1, 0,
  ]);
  const indices = new Uint16Array([0, 1, 2, 3, 4, 5]);
  const positionBytes = Buffer.from(positions.buffer);
  const indexOffset = pad4(positionBytes.length);
  const binary = Buffer.alloc(indexOffset + indices.byteLength);
  positionBytes.copy(binary, 0);
  Buffer.from(indices.buffer).copy(binary, indexOffset);
  writeGlb(filePath, {
    asset: { version: '2.0', generator: 'v5-surface-test' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    buffers: [{ byteLength: binary.length }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positionBytes.length },
      { buffer: 0, byteOffset: indexOffset, byteLength: indices.byteLength },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 6, type: 'VEC3', min: [0, 0, 0], max: [11, 1, 0] },
      { bufferView: 1, componentType: 5123, count: 6, type: 'SCALAR' },
    ],
  }, binary);
}

function writeTranslatedInstancedGlb(filePath) {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const indices = new Uint16Array([0, 1, 2]);
  const translations = new Float32Array([
    0, 0, 0,
    10, 0, 0,
  ]);
  const halfTurn = Math.sqrt(0.5);
  const rotations = new Float32Array([
    0, 0, 0, 1,
    0, 0, halfTurn, halfTurn,
  ]);
  const scales = new Float32Array([
    1, 1, 1,
    1, 1, 1,
  ]);
  const chunks = [
    Buffer.from(positions.buffer),
    Buffer.from(indices.buffer),
    Buffer.from(translations.buffer),
    Buffer.from(rotations.buffer),
    Buffer.from(scales.buffer),
  ];
  const offsets = [];
  let binaryLength = 0;
  for (const chunk of chunks) {
    binaryLength = pad4(binaryLength);
    offsets.push(binaryLength);
    binaryLength += chunk.length;
  }
  const binary = Buffer.alloc(binaryLength);
  chunks.forEach((chunk, index) => chunk.copy(binary, offsets[index]));
  writeGlb(filePath, {
    asset: { version: '2.0', generator: 'v5-surface-test-instanced' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    meshes: [{
      primitives: [{ attributes: { POSITION: 0 }, indices: 1 }],
    }],
    buffers: [{ byteLength: binary.length }],
    bufferViews: chunks.map((chunk, index) => ({
      buffer: 0,
      byteOffset: offsets[index],
      byteLength: chunk.length,
    })),
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: 2, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 2, type: 'VEC4' },
      { bufferView: 4, componentType: 5126, count: 2, type: 'VEC3' },
    ],
    extensionsUsed: ['EXT_mesh_gpu_instancing'],
    extensionsRequired: ['EXT_mesh_gpu_instancing'],
    nodes: [{
      mesh: 0,
      extensions: {
        EXT_mesh_gpu_instancing: {
          attributes: { TRANSLATION: 2, ROTATION: 3, SCALE: 4 },
        },
      },
    }],
  }, binary);
}

function writeEmptyGlb(filePath) {
  writeGlb(filePath, {
    asset: { version: '2.0', generator: 'v5-surface-test-empty' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{}],
  });
}

function writeMaterialReferenceGlb(filePath) {
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const positionBytes = Buffer.from(positions.buffer);
  writeGlb(filePath, {
    asset: { version: '2.0', generator: 'v5-surface-test-material' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{ source: 0 }],
    images: [{ bufferView: 1, mimeType: 'image/png' }],
    buffers: [{ byteLength: positionBytes.length }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positionBytes.length }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] }],
  }, positionBytes);
}

function bounds(min, max) {
  const size = max.map((value, axis) => value - min[axis]);
  const center = min.map((value, axis) => (value + max[axis]) * 0.5);
  return { min, max, center, size };
}

function readDenseFp32(filePath) {
  const bytes = fs.readFileSync(filePath);
  const header = parseDenseFp32Header(bytes);
  const payload = bytes.subarray(SURFACE_BINARY_HEADER_BYTES);
  assert.equal(payload.byteLength, header.numUnits * header.itemsPerUnit * header.featureDim * 4);
  return {
    bytes,
    header,
    values: new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4),
  };
}

function assertClose(actual, expected, epsilon = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} is not within ${epsilon} of ${expected}`);
}

function verifyWithCanonicalPythonLoader(outputDir) {
  const source = `
import json
import sys
import numpy as np
from pathlib import Path
from neural_instance_culling.dataset.v5.build_local_surface_points import load_local_surface_asset
from neural_instance_culling.dataset.v5.schemas import validate_local_surface_manifest

root = Path(sys.argv[1])
manifest = json.loads((root / "surface_manifest.json").read_text(encoding="utf-8"))
validate_local_surface_manifest(manifest)
asset = load_local_surface_asset(root)
assert asset.points.shape == (3, 256, 6)
assert asset.size_ratios.shape == (3, 3)
assert asset.aabb_min.shape == (3, 3)
assert asset.aabb_max.shape == (3, 3)
assert asset.unit_ids.tolist() == [0, 1, 2]
assert asset.degenerate_unit_ids == (2,)
assert np.all(asset.points[2] == 0.0)
assert np.allclose(np.linalg.norm(asset.points[:2, :, 3:], axis=2), 1.0, atol=1e-5)
print(json.dumps({"status": "ok", "shape": list(asset.points.shape)}))
`;
  const result = spawnSync(
    'conda',
    ['run', '-n', 'slm_pvs', 'python', '-c', source, outputDir],
    { cwd: REPO_ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, `canonical Python loader failed:\n${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /"status": "ok"/);
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'v5-instance-surface-'));
try {
  const assetsDir = path.join(root, 'assets');
  fs.mkdirSync(assetsDir, { recursive: true });
  writeTwoComponentGlb(path.join(assetsDir, 'multi.glb'));
  writeEmptyGlb(path.join(assetsDir, 'empty.glb'));
  const materialGlbPath = path.join(assetsDir, 'material.glb');
  writeMaterialReferenceGlb(materialGlbPath);
  const materialInput = fs.readFileSync(materialGlbPath);
  const materialStripped = stripGlbMaterials(materialInput);
  const materialJsonLength = materialStripped.readUInt32LE(12);
  const materialJson = JSON.parse(materialStripped.subarray(20, 20 + materialJsonLength).toString('utf8').trim());
  assert.equal(materialJson.meshes[0].primitives[0].material, undefined);
  assert.equal(materialJson.materials, undefined);
  assert.equal(materialJson.images, undefined);

  const instancedAssetsDir = path.join(root, 'instanced-assets');
  fs.mkdirSync(instancedAssetsDir, { recursive: true });
  writeTranslatedInstancedGlb(path.join(instancedAssetsDir, 'translated-instanced.glb'));

  const runtimeMetaPath = path.join(assetsDir, 'runtimeVisibilityMeta.json');
  const glbIndexPath = path.join(assetsDir, 'glbIndex.json');
  const emptyStats = {
    visiblePoses: 0,
    visibleRatio: 0,
    weightMean: 0,
    weightP90: 0,
    weightMax: 0,
    priorityPrior: 0,
  };
  fs.writeFileSync(runtimeMetaPath, JSON.stringify({
    schemaVersion: 1,
    componentCount: 3,
    instanceCount: 3,
    globalGlbCount: 2,
    componentRecords: [
      { componentGlobalId: 0, instanceId: 0, globalGlbId: 0, bounds: bounds([0, 0, 0], [1, 1, 0]), rvcStats: emptyStats },
      { componentGlobalId: 1, instanceId: 1, globalGlbId: 0, bounds: bounds([10, 0, 0], [11, 1, 0]), rvcStats: emptyStats },
      { componentGlobalId: 2, instanceId: 2, globalGlbId: 1, bounds: bounds([20, 0, 0], [21, 1, 0]), rvcStats: emptyStats },
    ],
    globalGlbRecords: [
      { globalGlbId: 0, componentGlobalIds: [0, 1], aabb: bounds([0, 0, 0], [11, 1, 0]), rvcStats: emptyStats },
      { globalGlbId: 1, componentGlobalIds: [2], aabb: bounds([20, 0, 0], [21, 1, 0]), rvcStats: emptyStats },
    ],
  }, null, 2));
  fs.writeFileSync(glbIndexPath, JSON.stringify({
    version: 1,
    ordering: 'test-order',
    lod: 'LOD0',
    total: 2,
    entries: [
      { globalId: 0, path: 'multi.glb', componentGlobalIds: [0, 1] },
      { globalId: 1, path: 'empty.glb', componentGlobalIds: [2] },
    ],
  }, null, 2));

  const outputA = path.join(root, 'surface-a');
  const first = await generateV5InstanceSurfaceSamples({
    assetsDir,
    runtimeMetaPath,
    glbIndexPath,
    outputDir: outputA,
    seed: 17,
    progressEvery: 0,
  });
  assert.deepEqual(fs.readdirSync(outputA).sort(), OUTPUT_FILES.slice().sort());
  assert.equal(first.manifest.schema, 'gcof-pvs-v5-local-surface-v1');
  assert.equal(first.manifest.storage, 'dense_fp32_header16');
  assert.deepEqual(first.manifest.unitIds, [0, 1, 2]);
  assert.deepEqual(first.manifest.degenerateUnitIds, [2]);
  assert.deepEqual(first.manifest.files, {
    points: 'surface_points_fp32.bin',
    sizeRatios: 'size_ratios_fp32.bin',
    aabbMin: 'aabb_min_fp32.npy',
    aabbMax: 'aabb_max_fp32.npy',
    unitIds: 'unit_ids_uint64.npy',
  });

  const points = readDenseFp32(path.join(outputA, first.manifest.files.points));
  assert.deepEqual(points.header, {
    magic: 'GPV5',
    version: 1,
    featureDim: POINT_FEATURE_DIM,
    numUnits: 3,
    itemsPerUnit: POINTS_PER_UNIT,
  });
  assert.equal(points.bytes.length, SURFACE_BINARY_HEADER_BYTES + 3 * POINT_RECORD_BYTES);
  const ratios = readDenseFp32(path.join(outputA, first.manifest.files.sizeRatios));
  assert.deepEqual(ratios.header, {
    magic: 'GPV5',
    version: 1,
    featureDim: 3,
    numUnits: 3,
    itemsPerUnit: 1,
  });
  assert.equal(ratios.values.length, 9);

  for (const unitIndex of [0, 1]) {
    const base = unitIndex * POINTS_PER_UNIT * POINT_FEATURE_DIM;
    for (let pointIndex = 0; pointIndex < POINTS_PER_UNIT; pointIndex += 1) {
      const offset = base + pointIndex * POINT_FEATURE_DIM;
      assert.ok(Math.abs(points.values[offset]) <= 0.71, 'component surfaces were mixed within the shared GLB');
      assert.ok(Math.abs(points.values[offset + 1]) <= 0.71);
      assertClose(points.values[offset + 2], 0);
      assertClose(points.values[offset + 3], 0);
      assertClose(points.values[offset + 4], 0);
      assertClose(points.values[offset + 5], 1);
    }
  }
  const emptyBase = 2 * POINTS_PER_UNIT * POINT_FEATURE_DIM;
  for (let index = emptyBase; index < points.values.length; index += 1) {
    assert.equal(points.values[index], 0, 'degenerate unit must keep a dense zero point row');
  }

  const directA = sampleUnitSurfaceFeatures(
    [{ a: [0, 0, 0], b: [1, 0, 0], c: [0, 1, 0] }],
    bounds([0, 0, 0], [1, 1, 0]),
    { seed: 17, componentGlobalId: 0 },
  );
  const directB = sampleUnitSurfaceFeatures(
    [{ a: [0, 0, 0], b: [1, 0, 0], c: [0, 1, 0] }],
    bounds([0, 0, 0], [1, 1, 0]),
    { seed: 17, componentGlobalId: 0 },
  );
  assert.deepEqual(Array.from(directA.features), Array.from(directB.features));
  assert.equal(directA.features.length, POINTS_PER_UNIT * POINT_FEATURE_DIM);

  const outputB = path.join(root, 'surface-b');
  await generateV5InstanceSurfaceSamples({
    assetsDir,
    runtimeMetaPath,
    glbIndexPath,
    outputDir: outputB,
    seed: 17,
    progressEvery: 0,
    gcEvery: 1,
  });
  for (const file of OUTPUT_FILES) {
    assert.deepEqual(fs.readFileSync(path.join(outputB, file)), fs.readFileSync(path.join(outputA, file)));
  }

  verifyWithCanonicalPythonLoader(outputA);

  const instancedStats = {
    visiblePoses: 0,
    visibleRatio: 0,
    weightMean: 0,
    weightP90: 0,
    weightMax: 0,
    priorityPrior: 0,
  };
  const instancedRuntimeMeta = {
    schemaVersion: 1,
    componentCount: 2,
    instanceCount: 2,
    globalGlbCount: 1,
    componentRecords: [
      { componentGlobalId: 0, instanceId: 0, globalGlbId: 0, bounds: bounds([0, 0, 0], [1, 1, 0]), rvcStats: instancedStats },
      { componentGlobalId: 1, instanceId: 1, globalGlbId: 0, bounds: bounds([9, 0, 0], [10, 1, 0]), rvcStats: instancedStats },
    ],
    globalGlbRecords: [
      { globalGlbId: 0, componentGlobalIds: [0, 1], aabb: bounds([0, 0, 0], [10, 1, 0]), rvcStats: instancedStats },
    ],
  };
  const instancedRuntimeMetaPath = path.join(instancedAssetsDir, 'runtimeVisibilityMeta.json');
  const instancedGlbIndexPath = path.join(instancedAssetsDir, 'glbIndex.json');
  fs.writeFileSync(instancedRuntimeMetaPath, JSON.stringify(instancedRuntimeMeta, null, 2));
  fs.writeFileSync(instancedGlbIndexPath, JSON.stringify({
    version: 1,
    ordering: 'test-order',
    lod: 'LOD0',
    total: 1,
    entries: [{ globalId: 0, path: 'translated-instanced.glb', componentGlobalIds: [0, 1] }],
  }, null, 2));
  const instancedOutput = path.join(root, 'surface-instanced');
  const instancedResult = await generateV5InstanceSurfaceSamples({
    assetsDir: instancedAssetsDir,
    runtimeMetaPath: instancedRuntimeMetaPath,
    glbIndexPath: instancedGlbIndexPath,
    outputDir: instancedOutput,
    seed: 17,
    progressEvery: 0,
    gcEvery: 1,
  });
  assert.deepEqual(instancedResult.manifest.unitIds, [0, 1]);
  assert.deepEqual(instancedResult.manifest.degenerateUnitIds, []);
  assert.equal(instancedResult.stats.matchModeCounts['instance-index-explicit'], 2);
  const instancedPoints = readDenseFp32(path.join(instancedOutput, instancedResult.manifest.files.points));
  for (let unitIndex = 0; unitIndex < 2; unitIndex += 1) {
    const base = unitIndex * POINTS_PER_UNIT * POINT_FEATURE_DIM;
    let nonZeroPointCount = 0;
    for (let pointIndex = 0; pointIndex < POINTS_PER_UNIT; pointIndex += 1) {
      const offset = base + pointIndex * POINT_FEATURE_DIM;
      if (Math.abs(instancedPoints.values[offset]) > 1e-6
        || Math.abs(instancedPoints.values[offset + 1]) > 1e-6) nonZeroPointCount += 1;
      assert.ok(instancedPoints.values[offset + 5] > 0.99, 'instanced surface normal must remain +Z');
    }
    assert.ok(nonZeroPointCount > 0, 'translated instance must have sampled surface points');
  }

  const streamedAssetsDir = path.join(root, 'streamed-assets');
  fs.mkdirSync(streamedAssetsDir, { recursive: true });
  writeTwoComponentGlb(path.join(streamedAssetsDir, 'single-component.glb'));
  const streamedStats = {
    visiblePoses: 0,
    visibleRatio: 0,
    weightMean: 0,
    weightP90: 0,
    weightMax: 0,
    priorityPrior: 0,
  };
  const streamedRuntimeMetaPath = path.join(streamedAssetsDir, 'runtimeVisibilityMeta.json');
  const streamedGlbIndexPath = path.join(streamedAssetsDir, 'glbIndex.json');
  fs.writeFileSync(streamedRuntimeMetaPath, JSON.stringify({
    schemaVersion: 1,
    componentCount: 1,
    instanceCount: 1,
    globalGlbCount: 1,
    componentRecords: [
      { componentGlobalId: 0, instanceId: 0, globalGlbId: 0, bounds: bounds([0, 0, 0], [11, 1, 0]), rvcStats: streamedStats },
    ],
    globalGlbRecords: [
      { globalGlbId: 0, componentGlobalIds: [0], aabb: bounds([0, 0, 0], [11, 1, 0]), rvcStats: streamedStats },
    ],
  }, null, 2));
  fs.writeFileSync(streamedGlbIndexPath, JSON.stringify({
    version: 1,
    ordering: 'test-order',
    lod: 'LOD0',
    total: 1,
    entries: [{ globalId: 0, path: 'single-component.glb', componentGlobalIds: [0] }],
  }, null, 2));
  const streamedResult = await generateV5InstanceSurfaceSamples({
    assetsDir: streamedAssetsDir,
    runtimeMetaPath: streamedRuntimeMetaPath,
    glbIndexPath: streamedGlbIndexPath,
    outputDir: path.join(root, 'surface-streamed'),
    seed: 17,
    progressEvery: 0,
    gcEvery: 1,
  });
  assert.deepEqual(streamedResult.manifest.degenerateUnitIds, []);
  assert.equal(streamedResult.stats.matchModeCounts['single-component-stream'], 2);

  assert.equal(DEFAULT_GC_EVERY, 10);
  assert.equal(parseArgs(['node', 'surface.mjs', '--gc-every', '3']).gcEvery, 3);
  const originalGc = globalThis.gc;
  let gcCalls = 0;
  globalThis.gc = () => { gcCalls += 1; };
  try {
    assert.equal(collectGarbageIfAvailable(), true);
    assert.equal(gcCalls, 1);
  } finally {
    if (originalGc === undefined) delete globalThis.gc;
    else globalThis.gc = originalGc;
  }
  console.log('V5 instance surface sample tests passed.');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
