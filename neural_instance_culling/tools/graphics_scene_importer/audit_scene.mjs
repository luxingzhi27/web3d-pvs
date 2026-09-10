import fs from 'node:fs';
import path from 'node:path';

export const SOURCE_AUDIT_SCHEMA = 'pvs-standard-graphics-scene-source-audit-v1';
export const UNIT_AUDIT_SCHEMA = 'pvs-standard-graphics-scene-unit-audit-v1';

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function distribution(values) {
  const numeric = values.filter((value) => Number.isFinite(value)).map(Number);
  return {
    count: numeric.length,
    min: numeric.length ? numeric.reduce((minimum, value) => Math.min(minimum, value), Infinity) : null,
    p50: percentile(numeric, 0.5),
    p95: percentile(numeric, 0.95),
    max: numeric.length ? numeric.reduce((maximum, value) => Math.max(maximum, value), -Infinity) : null,
    sum: numeric.length ? numeric.reduce((sum, value) => sum + value, 0) : 0,
  };
}

function zeroCounts() {
  return { OPAQUE: 0, MASK: 0, BLEND: 0, UNKNOWN: 0 };
}

function increment(map, key) {
  const normalized = map[key] == null ? 'UNKNOWN' : key;
  map[normalized] = (map[normalized] || 0) + 1;
}

function boundsSize(bounds) {
  return bounds?.size || null;
}

function sceneExtent(sceneBounds) {
  if (!sceneBounds) return null;
  return {
    min: sceneBounds.min,
    max: sceneBounds.max,
    center: sceneBounds.center,
    size: sceneBounds.size,
  };
}

export function auditSourceScene(scene) {
  const source = scene.source || {};
  const alphaCounts = zeroCounts();
  for (const item of scene.renderables || []) increment(alphaCounts, item.material?.alphaMode);
  for (const item of scene.excluded || []) increment(alphaCounts, item.material?.alphaMode);
  const excludedByReason = {};
  for (const item of scene.excluded || []) {
    excludedByReason[item.reason] = (excludedByReason[item.reason] || 0) + 1;
  }
  const blendExcluded = (scene.excluded || []).filter((item) => item.material?.alphaMode === 'BLEND');
  const maskTextureExcluded = (scene.excluded || []).filter(
    (item) => item.reason === 'mask-baseColor-alpha-texture-unavailable',
  );
  const blendExcludedTriangles = blendExcluded.reduce(
    (sum, item) => sum + Number(item.sourceTriangleCount || 0),
    0,
  );
  const renderableTriangles = (scene.renderables || []).reduce((sum, item) => sum + item.triangleCount, 0);
  const renderableVertices = (scene.renderables || []).reduce((sum, item) => sum + item.vertexCount, 0);
  return {
    schema: SOURCE_AUDIT_SCHEMA,
    sourcePath: source.filePath || null,
    sourceFormat: source.format || null,
    sourceBytes: source.sourceBytes ?? null,
    sceneIndex: source.sceneIndex ?? null,
    coordinateUnit: source.coordinateUnit || 'unspecified',
    nodeCount: source.nodeCount ?? 0,
    meshCount: source.meshCount ?? 0,
    primitiveCount: source.primitiveCount ?? 0,
    animationCount: source.animationCount ?? 0,
    skinCount: source.skinCount ?? 0,
    extensionsUsed: source.extensionsUsed || [],
    traversedNodeCount: (scene.nodes || []).length,
    staticRenderablePrimitiveCount: (scene.renderables || []).length,
    excludedPrimitiveCount: (scene.excluded || []).length,
    staticRenderableVertexCount: renderableVertices,
    staticRenderableTriangleCount: renderableTriangles,
    materialAlphaModeCounts: alphaCounts,
    excludedByReason,
    blendExcludedFromPvsCount: blendExcluded.length,
    blendAlwaysResidentPrimitiveCount: blendExcluded.length,
    blendAlwaysResidentTriangleCount: blendExcludedTriangles,
    blendAlwaysResidentPolicy: 'BLEND resources are always resident at runtime and excluded from static PVS units/resources',
    maskTextureExcludedFromPvsCount: maskTextureExcluded.length,
    sceneExtent: sceneExtent(scene.sceneBounds),
    policy: {
      opaque: 'eligible for static PVS and geometry-shell occlusion',
      mask: 'eligible for static PVS but not a geometry-only occluder',
      blend: 'excluded from static PVS candidates and always resident',
      unknown: 'non-occluder; alpha semantics require review before formal use',
      dynamic: 'skinned, animated, or morphed primitives are excluded from static PVS output',
    },
  };
}

export function auditUnits(conversion) {
  const units = conversion.units || [];
  const alphaCounts = zeroCounts();
  const unitBytes = [];
  const compressedBytes = [];
  const triangles = [];
  const vertices = [];
  const aabbSizes = [];
  let eligibleBytes = 0;
  let blendBytes = 0;
  for (const unit of units) {
    const mode = unit.material?.alphaMode || unit.alphaMode || 'UNKNOWN';
    increment(alphaCounts, mode);
    unitBytes.push(Number(unit.glbBytes ?? unit.outputBytes ?? 0));
    compressedBytes.push(Number(unit.compression?.encodedGeometryBytes ?? unit.compressedGeometryBytes ?? 0));
    triangles.push(Number(unit.triangleCount ?? unit.geometry?.triangleCount ?? 0));
    vertices.push(Number(unit.vertexCount ?? unit.geometry?.vertexCount ?? 0));
    if (unit.bounds?.size) aabbSizes.push(Math.max(...unit.bounds.size.map(Number)));
    if (mode === 'BLEND') blendBytes += Number(unit.glbBytes ?? unit.outputBytes ?? 0);
    else eligibleBytes += Number(unit.glbBytes ?? unit.outputBytes ?? 0);
  }
  const sourceAudit = conversion.sourceAudit || {};
  return {
    schema: UNIT_AUDIT_SCHEMA,
    sceneName: conversion.sceneName || null,
    sourcePath: sourceAudit.sourcePath || null,
    targetUnitBytes: conversion.targetUnitBytes ?? null,
    compression: conversion.compression || 'EXT_meshopt_compression',
    unitSemantics: 'one deterministic renderable unit maps to one componentGlobalId and one GLB/resource',
    unitCount: units.length,
    oneUnitPerResource: true,
    instanceCount: units.length,
    globalGlbCount: units.length,
    sourceNodeCount: sourceAudit.nodeCount ?? null,
    sourcePrimitiveCount: sourceAudit.primitiveCount ?? null,
    materialAlphaModeCounts: alphaCounts,
    eligibleStaticPvsBytes: eligibleBytes,
    alwaysResidentBlendBytes: blendBytes || null,
    blendExcludedFromPvsCount: sourceAudit.blendExcludedFromPvsCount ?? 0,
    blendAlwaysResidentPrimitiveCount: sourceAudit.blendAlwaysResidentPrimitiveCount ?? 0,
    blendAlwaysResidentTriangleCount: sourceAudit.blendAlwaysResidentTriangleCount ?? 0,
    maskTextureExcludedFromPvsCount: sourceAudit.maskTextureExcludedFromPvsCount ?? 0,
    unitGlbBytes: distribution(unitBytes),
    compressedGeometryBytes: distribution(compressedBytes),
    triangleCount: distribution(triangles),
    vertexCount: distribution(vertices),
    aabbLongestSide: distribution(aabbSizes),
    sourceAudit,
  };
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

export function auditOutputAssets(assetsDir) {
  const root = path.resolve(assetsDir);
  const manifest = readJson(path.join(root, 'conversionManifest.json'));
  const index = readJson(path.join(root, 'glbIndex.json'));
  const runtime = readJson(path.join(root, 'runtimeVisibilityMeta.json'));
  const unitFiles = (index.entries || []).map((entry) => path.join(root, entry.path));
  const missing = unitFiles.filter((filePath) => !fs.existsSync(filePath));
  if (missing.length) throw new Error(`converted scene is missing ${missing.length} GLB resources`);
  if (index.entries.length !== manifest.unitCount || runtime.instanceCount !== manifest.unitCount) {
    throw new Error('converted scene metadata does not agree on unit count');
  }
  return auditUnits({
    ...manifest,
    units: manifest.units || manifest.unitRecords || [],
  });
}

export async function main(argv = process.argv.slice(2)) {
  const inputIndex = argv.indexOf('--input');
  const assetsIndex = argv.indexOf('--assets');
  const outputIndex = argv.indexOf('--output');
  if (inputIndex >= 0) {
    const { readGltfScene } = await import('./read_gltf.mjs');
    const scene = await readGltfScene(argv[inputIndex + 1]);
    const output = argv[outputIndex + 1];
    if (!output) throw new Error('audit source mode requires --output');
    fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
    fs.writeFileSync(path.resolve(output), `${JSON.stringify(auditSourceScene(scene), null, 2)}\n`);
    return;
  }
  if (assetsIndex >= 0) {
    const output = argv[outputIndex + 1];
    if (!output) throw new Error('audit asset mode requires --output');
    fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
    fs.writeFileSync(path.resolve(output), `${JSON.stringify(auditOutputAssets(argv[assetsIndex + 1]), null, 2)}\n`);
    return;
  }
  throw new Error('audit_scene requires --input <scene.gltf|scene.glb> or --assets <assets-dir> and --output <path>');
}

import { pathToFileURL } from 'node:url';

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
