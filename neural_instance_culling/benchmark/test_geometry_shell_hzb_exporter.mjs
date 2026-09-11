#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { MeshoptDecoder } from 'meshoptimizer/decoder';
import { buildExport, resolveAssetsDir } from './geometry_shell_hzb_exporter.mjs';

function pad4(value) {
  return (value + 3) & ~3;
}

function writeTinyGlb(filePath, { alphaMode = null, translation = [0, 0, -5] } = {}) {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const indices = new Uint16Array([0, 1, 2]);
  const binary = Buffer.alloc(pad4(positions.byteLength) + pad4(indices.byteLength));
  Buffer.from(positions.buffer).copy(binary, 0);
  Buffer.from(indices.buffer).copy(binary, pad4(positions.byteLength));
  const material = alphaMode ? { alphaMode } : {};
  const json = {
    asset: { version: '2.0', generator: 'geometry-shell-hzb-test' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, translation }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    materials: [material],
    buffers: [{ byteLength: binary.byteLength }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
      { buffer: 0, byteOffset: pad4(positions.byteLength), byteLength: indices.byteLength },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
  };
  const rawJson = Buffer.from(JSON.stringify(json), 'utf8');
  const jsonChunk = Buffer.alloc(pad4(rawJson.length), 0x20);
  rawJson.copy(jsonChunk);
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binary.length;
  const output = Buffer.alloc(totalLength);
  output.writeUInt32LE(0x46546c67, 0);
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(totalLength, 8);
  output.writeUInt32LE(jsonChunk.length, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonChunk.copy(output, 20);
  const binaryHeader = 20 + jsonChunk.length;
  output.writeUInt32LE(binary.length, binaryHeader);
  output.writeUInt32LE(0x004e4942, binaryHeader + 4);
  binary.copy(output, binaryHeader + 8);
  fs.writeFileSync(filePath, output);
}

function readTyped(filePath, Type) {
  const bytes = fs.readFileSync(filePath);
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'geometry-shell-hzb-exporter-'));
const assetsDir = path.join(root, 'assets');
const outputDir = path.join(root, 'output');
fs.mkdirSync(path.join(assetsDir, 'task-0', 'glb', 'LOD0'), { recursive: true });
writeTinyGlb(path.join(assetsDir, 'task-0/glb/LOD0/sub_0.glb'));
writeTinyGlb(path.join(assetsDir, 'task-0/glb/LOD0/sub_1.glb'), { alphaMode: 'BLEND' });
writeTinyGlb(path.join(assetsDir, 'task-0/glb/LOD0/sub_2.glb'));
fs.writeFileSync(path.join(assetsDir, 'sceneWeb.json'), JSON.stringify({
  config: { sceneName: 'geometry-shell-test' },
  groups: [{ idRange: [0, 2], instances: {} }],
}));
fs.writeFileSync(path.join(assetsDir, 'glbIndex.json'), JSON.stringify({
  version: 1,
  ordering: 'task-asc,baseId-asc',
  lod: 'LOD0',
  total: 3,
  entries: [
  { globalId: 0, taskId: 0, baseId: 0, path: 'task-0/glb/LOD0/sub_0.glb' },
  { globalId: 1, taskId: 0, baseId: 1, path: 'task-0/glb/LOD0/sub_1.glb' },
  { globalId: 2, taskId: 0, baseId: 2, path: 'task-0/glb/LOD0/sub_2.glb' },
  ],
}));
const emptyStats = { visiblePoses: 0, visibleRatio: 0, weightMean: 0, weightP90: 0, weightMax: 0, priorityPrior: 0 };
fs.writeFileSync(path.join(assetsDir, 'runtimeVisibilityMeta.json'), JSON.stringify({
  schemaVersion: 1,
  componentCount: 3,
  instanceCount: 3,
  globalGlbCount: 3,
  componentRecords: [
    { componentGlobalId: 0, instanceId: 0, globalGlbId: 0, bounds: { center: [0.5, 0.5, -5], size: [1, 1, 0] }, rvcStats: emptyStats },
    { componentGlobalId: 1, instanceId: 1, globalGlbId: 1, bounds: { center: [0.5, 0.5, -5], size: [1, 1, 0] }, rvcStats: emptyStats },
    { componentGlobalId: 2, instanceId: 2, globalGlbId: 2, occluder: false, bounds: { center: [0.5, 0.5, -5], size: [1, 1, 0] }, rvcStats: emptyStats },
  ],
  globalGlbRecords: [
    { globalGlbId: 0, componentGlobalIds: [0], aabb: { min: [0, 0, -5], max: [1, 1, -5] }, rvcStats: emptyStats },
    { globalGlbId: 1, componentGlobalIds: [1], aabb: { min: [0, 0, -5], max: [1, 1, -5] }, rvcStats: emptyStats },
    { globalGlbId: 2, componentGlobalIds: [2], aabb: { min: [0, 0, -5], max: [1, 1, -5] }, rvcStats: emptyStats },
  ],
}));

assert.equal(resolveAssetsDir({ sceneRoot: root }), assetsDir);
const result = await buildExport({
  assetsDir,
  sceneRoot: '',
  dataRoot: '',
  outputDir,
  variant: 'lossless',
  budgetBytes: 0,
  budgetFromDir: '',
  importancePosePlan: '',
  importancePoseCount: 128,
  maxGlbs: 0,
  progressEvery: 0,
  overwrite: false,
});
await MeshoptDecoder.ready;
assert.equal(result.meta.schema, 'geometry-shell-hzb-v2');
assert.equal(result.meta.prototypeCount, 1);
assert.equal(result.meta.stats.sourcePrimitiveCount, 3);
assert.equal(result.meta.stats.opaquePrimitiveCount, 1);
assert.equal(result.meta.stats.excludedByReason['alpha-mode-blend'], 1);
assert.equal(result.meta.stats.excludedByReason['runtime-meta-non-occluder'], 1);
assert.equal(result.meta.queryContract.sampleCount, 1);
assert.match(result.meta.queryContract.mipDimensions, /explicit/);
assert.equal(result.meta.queryContract.candidateTest, 'all-candidate-conservative-aabb-hzb');
assert.equal(fs.existsSync(path.join(outputDir, 'instance_occluder_uint32.bin')), false);
assert.equal(fs.existsSync(path.join(outputDir, 'shell_instance_component_ids_uint32.bin')), false);
assert.equal(result.meta.primitiveAudits, undefined);
const losslessOfflineReport = JSON.parse(fs.readFileSync(result.offlineReportPath, 'utf8'));
assert.equal(losslessOfflineReport.runtimeAsset, false);
assert.equal(losslessOfflineReport.schema, 'geometry-shell-hzb-offline-report-v1');
assert.equal(Array.isArray(losslessOfflineReport.primitiveAudits), true);
assert.equal(path.dirname(result.offlineReportPath), path.dirname(outputDir));
assert.notEqual(result.offlineReportPath, path.join(outputDir, 'shell_meta.json'));

const prototype = result.meta.prototypes[0];
const positionSegment = result.meta.streams.positions.segments[prototype.positionSegment];
const positionSource = new Uint8Array(fs.readFileSync(path.join(outputDir, result.meta.streams.positions.file)));
const decodedPosition = new Uint8Array(positionSegment.decodedByteLength);
MeshoptDecoder.decodeGltfBuffer(
  decodedPosition,
  positionSegment.count,
  positionSegment.stride,
  positionSource.subarray(positionSegment.offset, positionSegment.offset + positionSegment.byteLength),
  'ATTRIBUTES',
);
assert.deepEqual(Array.from(new Float32Array(decodedPosition.buffer)), [0, 0, 0, 1, 0, 0, 0, 1, 0]);

const indexSegment = result.meta.streams.indices.segments[prototype.indexSegment];
const indexSource = new Uint8Array(fs.readFileSync(path.join(outputDir, result.meta.streams.indices.file)));
const decodedIndex = new Uint8Array(indexSegment.decodedByteLength);
MeshoptDecoder.decodeGltfBuffer(
  decodedIndex,
  indexSegment.count,
  indexSegment.stride,
  indexSource.subarray(indexSegment.offset, indexSegment.offset + indexSegment.byteLength),
  'TRIANGLES',
);
assert.deepEqual(Array.from(new Uint32Array(decodedIndex.buffer)), [0, 1, 2]);

const transformSegment = result.meta.streams.transforms.segments[prototype.transformSegment];
const transformSource = new Uint8Array(fs.readFileSync(path.join(outputDir, result.meta.streams.transforms.file)));
const decodedTransform = new Uint8Array(transformSegment.decodedByteLength);
MeshoptDecoder.decodeGltfBuffer(
  decodedTransform,
  transformSegment.count,
  transformSegment.stride,
  transformSource.subarray(transformSegment.offset, transformSegment.offset + transformSegment.byteLength),
  'ATTRIBUTES',
);
const transform = new Float32Array(decodedTransform.buffer);
assert.deepEqual(Array.from(transform.slice(12, 15)), [0, 0, -5]);
assert.ok(result.meta.geometryPolicy.removedAttributes.includes('NORMAL'));
assert.ok(result.meta.geometryPolicy.removedAttributes.includes('texture'));

const importancePosePlan = path.join(root, 'importance_pose_plan.jsonl');
fs.writeFileSync(importancePosePlan, `${JSON.stringify({
  split: 'train',
  camera_pos: [0, 0, 0],
  camera_forward: [0, 0, -1],
  fov_y: 60,
  aspect: 1,
})}\n`);
const equalOutputDir = path.join(root, 'equal-output');
const equalResult = await buildExport({
  assetsDir,
  sceneRoot: '',
  dataRoot: '',
  outputDir: equalOutputDir,
  variant: 'equal-asset',
  budgetBytes: result.runtimeAssetBytes + 10000,
  budgetFromDir: '',
  importancePosePlan,
  importancePoseCount: 1,
  maxGlbs: 0,
  progressEvery: 0,
  overwrite: false,
});
assert.equal(equalResult.meta.selection.mode, 'complete-primitive-deletion-only');
assert.equal(equalResult.meta.selection.selectedPrimitiveCount, 1);
assert.equal(equalResult.meta.prototypeCount, 1);
assert.equal(equalResult.meta.prototypes[0].triangleCount, 1);
assert.equal(equalResult.meta.primitiveAudits, undefined);
assert.equal(equalResult.runtimeAssetBytes, equalResult.meta.selection.totalAssetBytes);
assert.ok(equalResult.runtimeAssetBytes <= equalResult.meta.selection.budgetBytes);
assert.equal(
  fs.readdirSync(equalOutputDir).some((entry) => entry.includes('offline')),
  false,
);
const equalOfflineReport = JSON.parse(fs.readFileSync(equalResult.offlineReportPath, 'utf8'));
assert.equal(equalOfflineReport.runtimeAsset, false);
assert.equal(equalOfflineReport.primitiveAudits.length, 3);

fs.rmSync(root, { recursive: true, force: true });
console.log('Geometry-shell HZB exporter tests passed.');
