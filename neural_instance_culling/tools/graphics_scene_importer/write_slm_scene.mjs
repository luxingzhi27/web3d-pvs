import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { MeshoptEncoder } from 'meshoptimizer/encoder';

import { auditSourceScene, auditUnits } from './audit_scene.mjs';
import { partitionScene, TARGET_UNIT_BYTES } from './partition_units.mjs';
import { readGltfScene } from './read_gltf.mjs';

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;
const EXTENSION = 'EXT_meshopt_compression';

function pad4(value) {
  return (value + 3) & ~3;
}

function bytesOf(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function appendAligned(parts, state, values) {
  const offset = pad4(state.length);
  if (offset > state.length) parts.push(Buffer.alloc(offset - state.length));
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  parts.push(bytes);
  state.length = offset + bytes.length;
  return offset;
}

function boundsAccessor(bounds) {
  return bounds ? { min: bounds.min.map(Number), max: bounds.max.map(Number) } : {};
}

function writeGlb(filePath, json, binary) {
  const jsonRaw = Buffer.from(JSON.stringify(json), 'utf8');
  const jsonChunk = Buffer.alloc(pad4(jsonRaw.length), 0x20);
  jsonRaw.copy(jsonChunk);
  const binaryChunkLength = pad4(binary.length);
  const totalLength = 12 + 8 + jsonChunk.length + 8 + binaryChunkLength;
  const output = Buffer.alloc(totalLength);
  output.writeUInt32LE(GLB_MAGIC, 0);
  output.writeUInt32LE(GLB_VERSION, 4);
  output.writeUInt32LE(totalLength, 8);
  output.writeUInt32LE(jsonChunk.length, 12);
  output.writeUInt32LE(JSON_CHUNK_TYPE, 16);
  jsonChunk.copy(output, 20);
  const binaryHeader = 20 + jsonChunk.length;
  output.writeUInt32LE(binaryChunkLength, binaryHeader);
  output.writeUInt32LE(BIN_CHUNK_TYPE, binaryHeader + 4);
  binary.copy(output, binaryHeader + 8);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, output);
}

async function encodeAttribute(values, count, stride) {
  if (!MeshoptEncoder.supported) throw new Error('meshopt encoder is unavailable');
  await MeshoptEncoder.ready;
  return MeshoptEncoder.encodeGltfBuffer(bytesOf(values), count, stride, 'ATTRIBUTES');
}

async function encodeIndices(indices, vertexCount) {
  if (!MeshoptEncoder.supported) throw new Error('meshopt encoder is unavailable');
  await MeshoptEncoder.ready;
  const typed = vertexCount <= 65535 ? new Uint16Array(indices) : new Uint32Array(indices);
  return {
    typed,
    encoded: MeshoptEncoder.encodeGltfBuffer(bytesOf(typed), typed.length, typed.BYTES_PER_ELEMENT, 'TRIANGLES'),
  };
}

function writeSharedImages(outputAssets, imageResources, units) {
  const referenced = [...new Set(
    units
      .map((unit) => unit.materialTexture?.imageIndex)
      .filter((index) => Number.isInteger(index) && index >= 0),
  )].sort((left, right) => left - right);
  const byImage = new Map();
  const records = [];
  for (const imageIndex of referenced) {
    const image = imageResources[imageIndex];
    if (!image || !image.available || !image.bytes) {
      throw new Error(`unit references unavailable shared image ${imageIndex}`);
    }
    const relativePath = `task-0/glb/LOD0/shared/images/image_${imageIndex}.${image.extension}`;
    const filePath = path.join(outputAssets, relativePath);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, image.bytes);
    const descriptor = {
      sourceImageIndex: imageIndex,
      path: relativePath,
      uri: `shared/images/image_${imageIndex}.${image.extension}`,
      mimeType: image.mimeType,
      byteSize: image.bytes.length,
    };
    byImage.set(imageIndex, descriptor);
    records.push(descriptor);
  }
  return { byImage, records };
}

async function addAttribute({ json, parts, state, geometry, name, accessorType }) {
  const attribute = geometry.attributes?.[name];
  if (!attribute) return null;
  const stride = attribute.components * 4;
  const encoded = await encodeAttribute(attribute.values, geometry.vertexCount, stride);
  const compressedOffset = appendAligned(parts, state, encoded);
  const bufferView = json.bufferViews.length;
  json.bufferViews.push({
    buffer: 0,
    byteOffset: 0,
    byteLength: geometry.vertexCount * stride,
    target: 34962,
    extensions: {
      [EXTENSION]: {
        buffer: 0,
        byteOffset: compressedOffset,
        byteLength: encoded.length,
        byteStride: stride,
        count: geometry.vertexCount,
        mode: 'ATTRIBUTES',
      },
    },
  });
  const accessor = json.accessors.length;
  json.accessors.push({
    bufferView,
    componentType: 5126,
    count: geometry.vertexCount,
    type: accessorType,
  });
  return accessor;
}

export async function writeUnitGlb(filePath, unit, sharedImages = new Map()) {
  const geometry = unit.geometry;
  const sourceTexture = unit.materialTexture;
  const sharedImage = sourceTexture ? sharedImages.get(sourceTexture.imageIndex) : null;
  if (sourceTexture && !sharedImage) {
    throw new Error(`unit ${unit.unitId} has no shared image export for image ${sourceTexture.imageIndex}`);
  }
  const material = {
    ...unit.materialOutput,
    ...(sourceTexture
      ? {
        pbrMetallicRoughness: {
          ...unit.materialOutput.pbrMetallicRoughness,
          baseColorTexture: {
            index: 0,
            texCoord: sourceTexture.texCoord,
            ...(sourceTexture.textureTransform
              ? { extensions: { KHR_texture_transform: sourceTexture.textureTransform } }
              : {}),
          },
        },
      }
      : {}),
  };
  const parts = [];
  const state = { length: 0 };
  const json = {
    asset: { version: '2.0', generator: 'slm-graphics-scene-importer-g1' },
    scene: 0,
    scenes: [{ name: `unit_${unit.unitId}`, nodes: [0] }],
    nodes: [{ name: `unit_${unit.unitId}`, mesh: 0 }],
    meshes: [{
      name: `unit_${unit.unitId}`,
      primitives: [{ attributes: {}, indices: null, material: 0, mode: 4 }],
    }],
    materials: [material],
    samplers: sourceTexture ? [sourceTexture.sampler] : [],
    images: sourceTexture ? [{ uri: sharedImage.uri, mimeType: sharedImage.mimeType }] : [],
    textures: sourceTexture ? [{ name: sourceTexture.name || undefined, sampler: 0, source: 0 }] : [],
    accessors: [],
    bufferViews: [],
    buffers: [{ byteLength: 0 }],
    extensionsUsed: [
      EXTENSION,
      ...(sourceTexture?.textureTransform ? ['KHR_texture_transform'] : []),
    ],
    extensionsRequired: [EXTENSION],
  };
  const positionEncoded = await encodeAttribute(geometry.positions, geometry.vertexCount, 12);
  const positionOffset = appendAligned(parts, state, positionEncoded);
  const positionView = json.bufferViews.length;
  json.bufferViews.push({
    buffer: 0,
    byteOffset: 0,
    byteLength: geometry.vertexCount * 12,
    target: 34962,
    extensions: {
      [EXTENSION]: {
        buffer: 0,
        byteOffset: positionOffset,
        byteLength: positionEncoded.length,
        byteStride: 12,
        count: geometry.vertexCount,
        mode: 'ATTRIBUTES',
      },
    },
  });
  const positionAccessor = json.accessors.length;
  json.accessors.push({
    bufferView: positionView,
    componentType: 5126,
    count: geometry.vertexCount,
    type: 'VEC3',
    ...boundsAccessor(geometry.bounds),
  });
  json.meshes[0].primitives[0].attributes.POSITION = positionAccessor;

  const indexData = await encodeIndices(geometry.indices, geometry.vertexCount);
  const indexOffset = appendAligned(parts, state, indexData.encoded);
  const indexView = json.bufferViews.length;
  json.bufferViews.push({
    buffer: 0,
    byteOffset: 0,
    byteLength: indexData.typed.byteLength,
    target: 34963,
    extensions: {
      [EXTENSION]: {
        buffer: 0,
        byteOffset: indexOffset,
        byteLength: indexData.encoded.length,
        byteStride: indexData.typed.BYTES_PER_ELEMENT,
        count: indexData.typed.length,
        mode: 'TRIANGLES',
      },
    },
  });
  const indexAccessor = json.accessors.length;
  json.accessors.push({
    bufferView: indexView,
    componentType: indexData.typed.BYTES_PER_ELEMENT === 2 ? 5123 : 5125,
    count: indexData.typed.length,
    type: 'SCALAR',
    min: [0],
    max: [Math.max(0, geometry.vertexCount - 1)],
  });
  json.meshes[0].primitives[0].indices = indexAccessor;

  for (const name of ['NORMAL', 'TANGENT', 'TEXCOORD_0', 'TEXCOORD_1', 'COLOR_0']) {
    const attribute = geometry.attributes?.[name];
    if (!attribute) continue;
    const accessor = await addAttribute({
      json,
      parts,
      state,
      geometry,
      name,
      accessorType: `VEC${attribute.components}`,
    });
    if (accessor != null) json.meshes[0].primitives[0].attributes[name] = accessor;
  }
  const binary = Buffer.concat(parts);
  json.buffers[0].byteLength = binary.length;
  writeGlb(filePath, json, binary);
  return {
    filePath,
    glbBytes: fs.statSync(filePath).size,
    compressedGeometryBytes: state.length,
    encodedPositionBytes: positionEncoded.length,
    encodedIndexBytes: indexData.encoded.length,
  };
}

function prepareOutput(outputAssets, overwrite) {
  if (fs.existsSync(outputAssets)) {
    if (!overwrite) throw new Error(`output exists; pass --overwrite: ${outputAssets}`);
    fs.rmSync(outputAssets, { recursive: true, force: true });
  }
  fs.mkdirSync(path.join(outputAssets, 'task-0', 'glb', 'LOD0'), { recursive: true });
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function zeroRvcStats() {
  return {
    visiblePoses: 0,
    visibleRatio: 0,
    weightMean: 0,
    weightP90: 0,
    weightMax: 0,
    priorityPrior: 0,
  };
}

function sceneBoundsOrEmpty(sceneBounds) {
  return sceneBounds || {
    min: [0, 0, 0],
    max: [0, 0, 0],
    center: [0, 0, 0],
    size: [0, 0, 0],
  };
}

function unitRecord(unit, writeResult, relativePath) {
  return {
    unitId: unit.unitId,
    componentGlobalId: unit.unitId,
    globalGlbId: unit.unitId,
    sourceNodeIndex: unit.sourceNodeIndex,
    sourceNodePath: unit.sourceNodePath,
    sourceMeshIndex: unit.sourceMeshIndex,
    sourcePrimitiveIndex: unit.sourcePrimitiveIndex,
    sourceMaterialIndex: unit.sourceMaterialIndex,
    sourceTriangleCount: unit.sourceTriangleIndices.length,
    sourceComponentOrdinals: unit.sourceComponentOrdinals,
    partitionReason: unit.partitionReason,
    triangleCount: unit.triangleCount,
    vertexCount: unit.vertexCount,
    targetUnitBytes: unit.targetUnitBytes,
    compressedGeometryBytes: writeResult.compressedGeometryBytes,
    glbBytes: writeResult.glbBytes,
    path: relativePath,
    alphaMode: unit.material.alphaMode,
    staticPvsEligible: unit.material.staticPvsEligible,
    alwaysResident: unit.material.alwaysResident,
    occluder: unit.material.occluder,
    baseColorTexture: unit.materialTexture
      ? {
        sourceImageIndex: unit.materialTexture.imageIndex,
        texCoord: unit.materialTexture.texCoord,
      }
      : null,
    bounds: unit.bounds,
  };
}

export async function writeSlmScene(scene, outputAssets, options = {}) {
  const resolvedOutput = path.resolve(outputAssets);
  const targetUnitBytes = Number(options.targetBytes ?? TARGET_UNIT_BYTES);
  const conversion = await partitionScene(scene, { targetBytes: targetUnitBytes });
  const sourceAudit = auditSourceScene({
    ...scene,
    renderables: (scene.renderables || []).filter((item) => item.material?.alphaMode !== 'BLEND'),
    excluded: conversion.excluded,
  });
  prepareOutput(resolvedOutput, Boolean(options.overwrite));
  const sharedImages = writeSharedImages(resolvedOutput, conversion.imageResources, conversion.units);

  const unitRecords = [];
  for (const unit of conversion.units) {
    const relativePath = `task-0/glb/LOD0/sub_${unit.unitId}.glb`;
    const result = await writeUnitGlb(path.join(resolvedOutput, relativePath), unit, sharedImages.byImage);
    unitRecords.push(unitRecord(unit, result, relativePath));
  }
  const sceneBounds = sceneBoundsOrEmpty(scene.sceneBounds);
  const sceneName = options.sceneName || path.basename(scene.source?.filePath || 'graphics_scene').replace(/\.(gltf|glb)$/i, '');
  const runtimeRecords = unitRecords.map((unit) => ({
    componentGlobalId: unit.unitId,
    instanceId: unit.unitId,
    globalGlbId: unit.unitId,
    taskId: 0,
    baseId: unit.unitId,
    sourceNodePath: unit.sourceNodePath,
    sourcePrimitiveIndex: unit.sourcePrimitiveIndex,
    alphaMode: unit.alphaMode,
    staticPvsEligible: unit.staticPvsEligible,
    alwaysResident: unit.alwaysResident,
    occluder: unit.occluder,
    bounds: unit.bounds,
    rvcStats: zeroRvcStats(),
  }));
  const glbEntries = unitRecords.map((unit) => ({
    globalId: unit.unitId,
    taskId: 0,
    baseId: unit.unitId,
    path: unit.path,
    byteSize: unit.glbBytes,
    componentGlobalIds: [unit.unitId],
  }));
  const globalRecords = unitRecords.map((unit) => ({
    globalGlbId: unit.unitId,
    taskId: 0,
    baseId: unit.unitId,
    componentGlobalIds: [unit.unitId],
    aabb: unit.bounds,
    byteSize: unit.glbBytes,
    rvcStats: zeroRvcStats(),
  }));
  const conversionManifest = {
    schema: 'pvs-standard-graphics-scene-conversion-v1',
    generatedBy: 'slm-graphics-scene-importer-g1',
    sceneName,
    source: scene.source,
    targetUnitBytes,
    compression: EXTENSION,
    compressionParameters: { mode: 'ATTRIBUTES/TRIANGLES', encoder: 'meshoptimizer', version: 0 },
    unitSemantics: 'one renderable unit -> one componentGlobalId -> one GLB/resource',
    oneUnitPerResource: true,
    unitCount: unitRecords.length,
    instanceCount: unitRecords.length,
    globalGlbCount: unitRecords.length,
    sceneBounds,
    sourceAudit,
    sharedImages: sharedImages.records,
    excludedPrimitives: conversion.excluded,
    units: unitRecords,
    unitRecords,
  };
  const sceneWeb = {
    materials: { proxy: [], useHdrJpg: false },
    groups: [{ idRange: [0, Math.max(-1, unitRecords.length - 1)], instances: {} }],
    groupSemantics: 'Each deterministic renderable unit is an independent resource; no prototype reuse is performed.',
    config: {
      sceneName,
      source: path.basename(scene.source?.filePath || ''),
      renderableUnitSemantics: 'source node/primitive or deterministic Morton cluster',
      bounds: { center: sceneBounds.center, size: sceneBounds.size },
    },
  };
  const glbIndex = {
    version: 1,
    ordering: 'unitId-ascending',
    lod: 'LOD0',
    total: glbEntries.length,
    entries: glbEntries,
  };
  const runtimeVisibilityMeta = {
    schemaVersion: 1,
    idSpaces: {
      componentGlobalId: 'dense deterministic renderable unit id',
      globalGlbId: 'one-to-one dense resource id equal to componentGlobalId',
    },
    sceneName,
    sceneBounds,
    componentCount: unitRecords.length,
    globalGlbCount: unitRecords.length,
    instanceCount: unitRecords.length,
    componentRecords: runtimeRecords,
    globalGlbRecords: globalRecords,
    buildStats: {
      sourceNodeCount: sourceAudit.nodeCount,
      sourcePrimitiveCount: sourceAudit.primitiveCount,
      outputUnitCount: unitRecords.length,
      outputGlbBytes: unitRecords.reduce((sum, unit) => sum + unit.glbBytes, 0),
    },
  };
  const unitAudit = auditUnits({ ...conversionManifest, units: unitRecords });
  writeJson(path.join(resolvedOutput, 'sceneWeb.json'), sceneWeb);
  writeJson(path.join(resolvedOutput, 'glbIndex.json'), glbIndex);
  writeJson(path.join(resolvedOutput, 'runtimeVisibilityMeta.json'), runtimeVisibilityMeta);
  writeJson(path.join(resolvedOutput, 'conversionManifest.json'), conversionManifest);
  writeJson(path.join(resolvedOutput, 'scene_source_audit.json'), sourceAudit);
  writeJson(path.join(resolvedOutput, 'scene_audit.json'), unitAudit);
  return {
    outputAssets: resolvedOutput,
    sceneName,
    unitCount: unitRecords.length,
    glbCount: glbEntries.length,
    manifest: conversionManifest,
    audit: unitAudit,
  };
}

export const buildSlmScene = writeSlmScene;

function parseArgs(argv) {
  const values = { overwrite: false, targetBytes: TARGET_UNIT_BYTES };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--input') values.input = argv[++index];
    else if (argument === '--output-assets') values.outputAssets = argv[++index];
    else if (argument === '--scene-name') values.sceneName = argv[++index];
    else if (argument === '--target-unit-kib') values.targetBytes = Number(argv[++index]) * 1024;
    else if (argument === '--overwrite') values.overwrite = true;
    else if (argument === '--help') values.help = true;
    else throw new Error(`unknown argument: ${argument}`);
  }
  return values;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log('Usage: node write_slm_scene.mjs --input scene.gltf|scene.glb --output-assets DIR [--target-unit-kib 128] [--overwrite]');
    return;
  }
  if (!args.input || !args.outputAssets) throw new Error('--input and --output-assets are required');
  const scene = await readGltfScene(args.input);
  const result = await writeSlmScene(scene, args.outputAssets, args);
  console.log(JSON.stringify({ outputAssets: result.outputAssets, sceneName: result.sceneName, unitCount: result.unitCount }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
