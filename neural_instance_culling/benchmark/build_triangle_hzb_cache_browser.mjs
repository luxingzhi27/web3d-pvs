#!/usr/bin/env node
/*
 * Build a warm-cache triangle HZB with the same browser rasterizer used by the
 * retained scene assets.  The target GLBs must already be local and decoded by
 * Chrome; this tool is therefore an L2 geometry baseline, not a cold-start
 * visibility method.
 *
 * The browser renders linear view depth into RGBA8, constructs a min-depth
 * pyramid from level zero, and posts one flattened Float32 block per pose.
 * Node writes the blocks at fixed offsets and emits the metadata consumed by
 * benchmark/triangle_hzb.py.
 */

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SLM2_ROOT = path.join(REPO_ROOT, 'slm2viewer');
const CACHE_SCHEMA = 'neuralstreamweb3d-triangle-hzb-cache-v1';
const DEPTH_ENCODING = 'linear_view_depth_normalized_by_camera_far';
const POSE_STRIDE_BYTES = 64;
const MODEL_FOV_Y_DEG = 66;

function parseArgs(argv) {
  const args = {
    assetsDir: '',
    datasetDir: '',
    output: '',
    split: 'test',
    poseStart: 0,
    poseCount: 0,
    maxPoses: 0,
    width: 512,
    height: 288,
    fovYDeg: MODEL_FOV_Y_DEG,
    aspect: 0,
    cameraFar: 0,
    glbIdList: '',
    chromeExe: process.env.CHROME_EXE || '',
    port: 0,
    timeoutMs: 60 * 60 * 1000,
    force: false,
    requireHardwareGpu: true,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--force') {
      args.force = true;
      continue;
    }
    if (key === '--require-hardware-gpu') {
      args.requireHardwareGpu = true;
      continue;
    }
    if (key === '--allow-software-gpu') {
      args.requireHardwareGpu = false;
      continue;
    }
    const value = argv[index + 1];
    index += 1;
    if (key === '--assets-dir') args.assetsDir = path.resolve(value);
    else if (key === '--dataset-dir') args.datasetDir = path.resolve(value);
    else if (key === '--output') args.output = path.resolve(value);
    else if (key === '--split') args.split = String(value);
    else if (key === '--pose-start') args.poseStart = Number(value);
    else if (key === '--pose-count') args.poseCount = Number(value);
    else if (key === '--max-poses') args.maxPoses = Number(value);
    else if (key === '--width') args.width = Number(value);
    else if (key === '--height') args.height = Number(value);
    else if (key === '--fov-y-deg') args.fovYDeg = Number(value);
    else if (key === '--aspect') args.aspect = Number(value);
    else if (key === '--camera-far') args.cameraFar = Number(value);
    else if (key === '--glb-id-list') args.glbIdList = String(value);
    else if (key === '--chrome-exe') args.chromeExe = path.resolve(value);
    else if (key === '--port') args.port = Number(value);
    else if (key === '--timeout-ms') args.timeoutMs = Number(value);
    else throw new Error(`unknown argument: ${key}`);
  }
  if (!args.assetsDir) throw new Error('--assets-dir is required');
  if (!args.datasetDir) throw new Error('--dataset-dir is required');
  if (!args.output) throw new Error('--output is required');
  if (!fs.existsSync(args.assetsDir)) throw new Error(`assets directory does not exist: ${args.assetsDir}`);
  if (!fs.existsSync(args.datasetDir)) throw new Error(`dataset directory does not exist: ${args.datasetDir}`);
  if (!Number.isInteger(args.width) || args.width < 8) throw new Error('--width must be an integer >= 8');
  if (!Number.isInteger(args.height) || args.height < 8) throw new Error('--height must be an integer >= 8');
  if (Math.abs(args.fovYDeg - MODEL_FOV_Y_DEG) > 1e-6) {
    throw new Error(`triangle HZB query FOV must be exactly ${MODEL_FOV_Y_DEG} degrees`);
  }
  if (!Number.isFinite(args.aspect) || args.aspect < 0) throw new Error('--aspect must be non-negative');
  if (!Number.isInteger(args.poseStart) || args.poseStart < 0) throw new Error('--pose-start must be non-negative');
  if (!Number.isInteger(args.poseCount) || args.poseCount < 0) throw new Error('--pose-count must be non-negative');
  if (!Number.isInteger(args.maxPoses) || args.maxPoses < 0) throw new Error('--max-poses must be non-negative');
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function findChrome(explicit) {
  const candidates = [
    explicit,
    process.env.CHROME_EXE,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/snap/bin/chromium',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error('Chrome/Chromium executable not found; pass --chrome-exe or set CHROME_EXE');
}

function classifyGpuBackend(gpuBackend) {
  const value = gpuBackend && typeof gpuBackend === 'object' ? gpuBackend : {};
  const text = [value.vendor, value.renderer, value.version].filter(Boolean).join(' ');
  const softwarePattern = /swiftshader|llvmpipe|softpipe|swrast|software(?:\s+webgl|\s+rasterizer)?|no-webgl/i;
  return {
    vendor: String(value.vendor || ''),
    renderer: String(value.renderer || ''),
    version: String(value.version || ''),
    hardware: Boolean(text) && !softwarePattern.test(text),
    softwareMarkers: text.match(softwarePattern)?.[0] || null,
  };
}

function contentType(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === '.js' || extension === '.mjs') return 'text/javascript; charset=utf-8';
  if (extension === '.json') return 'application/json; charset=utf-8';
  if (extension === '.wasm') return 'application/wasm';
  if (extension === '.glb') return 'model/gltf-binary';
  if (extension === '.html') return 'text/html; charset=utf-8';
  return 'application/octet-stream';
}

function safeJoin(root, relativePath) {
  const resolved = path.resolve(root, relativePath.replace(/^[/\\]+/, ''));
  const rootResolved = path.resolve(root);
  if (resolved !== rootResolved && !resolved.startsWith(rootResolved + path.sep)) {
    throw new Error(`path escapes root: ${relativePath}`);
  }
  return resolved;
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on('data', (chunk) => chunks.push(chunk));
    request.on('end', () => resolve(Buffer.concat(chunks)));
    request.on('error', reject);
  });
}

function writeJson(response, value, status = 200) {
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  response.end(JSON.stringify(value));
}

function sceneDiagonal(runtimeMeta) {
  const size = runtimeMeta.sceneBounds?.size || [1, 1, 1];
  return Math.sqrt(size.reduce((sum, value) => sum + Number(value) ** 2, 0));
}

function parseSplitIds(datasetMeta, splitValue) {
  const names = String(splitValue || 'all')
    .split(/[,+]/)
    .map((value) => value.trim())
    .filter(Boolean);
  if (names.length === 0 || names.includes('all')) {
    if (names.length > 1 || (names.length === 1 && names[0] !== 'all')) {
      throw new Error("split 'all' cannot be combined with another split");
    }
    return { names: ['all'], ids: null };
  }
  const splitIds = datasetMeta.splitIds || {};
  const ids = names.map((name) => {
    if (!Object.prototype.hasOwnProperty.call(splitIds, name)) {
      throw new Error(`dataset splitIds has no '${name}'`);
    }
    return Number(splitIds[name]);
  });
  return { names, ids: new Set(ids) };
}

function readPoseRecords(datasetDir, datasetMeta, args) {
  const posePath = path.join(datasetDir, 'poses.bin');
  const bytes = fs.readFileSync(posePath);
  const stride = Number(datasetMeta.poseStrideBytes || POSE_STRIDE_BYTES);
  if (stride !== POSE_STRIDE_BYTES || bytes.byteLength % stride !== 0) {
    throw new Error(`triangle HZB generator requires directional 64-byte poses.bin, got stride=${stride}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const selectedSplits = parseSplitIds(datasetMeta, args.split);
  const selected = [];
  for (let poseIndex = 0; poseIndex < bytes.byteLength / stride; poseIndex += 1) {
    const offset = poseIndex * stride;
    const splitId = view.getUint8(offset + 44);
    if (selectedSplits.ids !== null && !selectedSplits.ids.has(splitId)) continue;
    selected.push({
      ordinal: selected.length,
      poseIndex,
      cameraWorld: [view.getFloat32(offset + 12, true), view.getFloat32(offset + 16, true), view.getFloat32(offset + 20, true)],
      cameraForward: [view.getFloat32(offset + 24, true), view.getFloat32(offset + 28, true), view.getFloat32(offset + 32, true)],
      tanX: view.getFloat32(offset + 36, true),
      tanY: view.getFloat32(offset + 40, true),
      splitId: view.getUint8(offset + 44),
    });
  }
  const start = Math.min(args.poseStart, selected.length);
  const endByCount = args.poseCount > 0 ? Math.min(selected.length, start + args.poseCount) : selected.length;
  const end = args.maxPoses > 0 ? Math.min(endByCount, start + args.maxPoses) : endByCount;
  const result = selected.slice(start, end).map((row, ordinal) => {
    const poseAspect = row.tanY > 0 ? row.tanX / row.tanY : 0;
    if (!Number.isFinite(poseAspect) || poseAspect <= 0) throw new Error(`pose ${row.poseIndex} has invalid tan FOV values`);
    return { ...row, ordinal, aspect: args.aspect > 0 ? args.aspect : poseAspect };
  });
  if (result.length === 0) throw new Error(`no poses selected from split '${args.split}'`);
  return { poses: result, splitNames: selectedSplits.names };
}

function readGlbEntries(assetsDir, glbIdList) {
  const indexPath = path.join(assetsDir, 'glbIndex.json');
  const index = readJson(indexPath);
  const entries = Array.isArray(index.entries) ? index.entries : [];
  if (entries.length === 0) throw new Error(`glbIndex.json has no entries: ${indexPath}`);
  let selectedIds = entries.map((entry) => Number(entry.globalId));
  if (glbIdList) {
    const text = fs.existsSync(glbIdList) ? fs.readFileSync(glbIdList, 'utf8') : glbIdList;
    selectedIds = text.split(/[\\s,]+/).filter(Boolean).map(Number);
  }
  const selectedSet = new Set(selectedIds);
  const selected = entries.filter((entry) => selectedSet.has(Number(entry.globalId)));
  if (selected.length !== selectedSet.size) throw new Error('glb id list contains an id missing from glbIndex.json');
  for (const entry of selected) {
    const filePath = safeJoin(assetsDir, entry.path);
    if (!fs.existsSync(filePath)) throw new Error(`missing GLB: ${filePath}`);
  }
  return { index, selected };
}

function buildLevelDescriptors(width, height) {
  const descriptors = [];
  let level = 0;
  let currentWidth = width;
  let currentHeight = height;
  let offset = 0;
  while (true) {
    const count = currentWidth * currentHeight;
    descriptors.push({ level, width: currentWidth, height: currentHeight, offset, count });
    offset += count;
    if (currentWidth === 1 && currentHeight === 1) break;
    currentWidth = Math.max(1, Math.ceil(currentWidth / 2));
    currentHeight = Math.max(1, Math.ceil(currentHeight / 2));
    level += 1;
  }
  return { descriptors, valueCount: offset };
}

function rendererHtml() {
  return `<!doctype html><html><head><meta charset="utf-8"><title>Triangle HZB builder</title><style>html,body{margin:0;background:#111;color:#ddd;font:12px monospace}#status{padding:8px;white-space:pre-wrap}</style></head><body><div id="status">booting</div><script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script><script type="module" src="/renderer.js"></script></body></html>`;
}

function rendererJs() {
  return String.raw`
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const statusElement = document.getElementById('status');
const manifest = await (await fetch('/manifest', { cache: 'no-store' })).json();
function status(message) {
  statusElement.textContent = message;
  console.log('[triangle-hzb]', message);
  fetch('/log', { method: 'POST', body: message }).catch(() => {});
}

function makeDepthMaterial(instanced) {
  const vertexShader = instanced ? [
    'varying float vLinearDepth;',
    'uniform float cameraFar;',
    'void main() {',
    '  vec4 viewPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);',
    '  vLinearDepth = clamp(-viewPosition.z / cameraFar, 0.0, 0.99999994);',
    '  gl_Position = projectionMatrix * viewPosition;',
    '}',
  ].join('\n') : [
    'varying float vLinearDepth;',
    'uniform float cameraFar;',
    'void main() {',
    '  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);',
    '  vLinearDepth = clamp(-viewPosition.z / cameraFar, 0.0, 0.99999994);',
    '  gl_Position = projectionMatrix * viewPosition;',
    '}',
  ].join('\n');
  const fragmentShader = [
    'varying float vLinearDepth;',
    'vec4 packDepth(const in float depth) {',
    '  const vec4 bitShift = vec4(1.0, 255.0, 65025.0, 160581375.0);',
    '  const vec4 bitMask = vec4(0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);',
    '  vec4 result = fract(depth * bitShift);',
    '  result -= result.yzww * bitMask;',
    '  return result;',
    '}',
    'void main() { gl_FragColor = packDepth(vLinearDepth); }',
  ].join('\n');
  return new THREE.ShaderMaterial({
    uniforms: { cameraFar: { value: Number(manifest.cameraFar) } },
    vertexShader,
    fragmentShader,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
    toneMapped: false,
    fog: false,
  });
}

function decodeDepth(pixels, output) {
  const inv255 = 1 / 255;
  const inv65025 = 1 / 65025;
  const inv160581375 = 1 / 160581375;
  for (let index = 0, pixel = 0; index < pixels.length; index += 4, pixel += 1) {
    output[pixel] = Math.min(1.0, pixels[index] * inv255
      + pixels[index + 1] * inv255 * inv255
      + pixels[index + 2] * inv255 * inv65025
      + pixels[index + 3] * inv255 * inv160581375);
  }
}

function buildMinPyramid(levelZero, width, height) {
  const levels = [levelZero];
  let current = levelZero;
  let currentWidth = width;
  let currentHeight = height;
  while (currentWidth > 1 || currentHeight > 1) {
    const nextWidth = Math.max(1, Math.ceil(currentWidth / 2));
    const nextHeight = Math.max(1, Math.ceil(currentHeight / 2));
    const next = new Float32Array(nextWidth * nextHeight);
    for (let y = 0; y < nextHeight; y += 1) {
      for (let x = 0; x < nextWidth; x += 1) {
        let value = 1.0;
        for (let oy = 0; oy < 2; oy += 1) {
          for (let ox = 0; ox < 2; ox += 1) {
            const sx = Math.min(currentWidth - 1, x * 2 + ox);
            const sy = Math.min(currentHeight - 1, y * 2 + oy);
            value = Math.min(value, current[sy * currentWidth + sx]);
          }
        }
        next[y * nextWidth + x] = value;
      }
    }
    levels.push(next);
    current = next;
    currentWidth = nextWidth;
    currentHeight = nextHeight;
  }
  let total = 0;
  for (const level of levels) total += level.length;
  const flattened = new Float32Array(total);
  let offset = 0;
  for (const level of levels) {
    flattened.set(level, offset);
    offset += level.length;
  }
  return flattened;
}

function setCamera(camera, pose) {
  camera.fov = Number(manifest.fovYDeg);
  camera.aspect = Number(pose.aspect);
  camera.near = 0.05;
  camera.far = Number(manifest.cameraFar);
  camera.position.set(pose.cameraWorld[0], pose.cameraWorld[1], pose.cameraWorld[2]);
  camera.up.set(0, 1, 0);
  camera.lookAt(
    pose.cameraWorld[0] + pose.cameraForward[0],
    pose.cameraWorld[1] + pose.cameraForward[1],
    pose.cameraWorld[2] + pose.cameraForward[2],
  );
  camera.updateProjectionMatrix();
}

async function postPose(ordinal, flattened) {
  const response = await fetch('/pose-hzb?ordinal=' + encodeURIComponent(ordinal), {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: flattened.buffer,
  });
  if (!response.ok) throw new Error('Node rejected HZB pose ' + ordinal);
}

async function main() {
  const renderer = new THREE.WebGLRenderer({
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
  });
  renderer.setSize(Number(manifest.width), Number(manifest.height), false);
  renderer.setPixelRatio(1);
  renderer.setClearColor(0xffffff, 1.0);
  renderer.toneMapping = THREE.NoToneMapping;
  document.body.appendChild(renderer.domElement);
  const target = new THREE.WebGLRenderTarget(Number(manifest.width), Number(manifest.height), {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: true,
    stencilBuffer: false,
  });
  const pixels = new Uint8Array(Number(manifest.width) * Number(manifest.height) * 4);
  const levelZero = new Float32Array(Number(manifest.width) * Number(manifest.height));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(Number(manifest.fovYDeg), Number(manifest.poses[0].aspect), 0.05, Number(manifest.cameraFar));
  const dracoLoader = new DRACOLoader().setDecoderPath('/node_modules/three/examples/jsm/libs/draco/gltf/');
  const loader = new GLTFLoader().setDRACOLoader(dracoLoader).setMeshoptDecoder(MeshoptDecoder);
  const loadStarted = performance.now();
  for (let index = 0; index < manifest.glbEntries.length; index += 1) {
    const entry = manifest.glbEntries[index];
    const gltf = await loader.loadAsync('/assets/' + entry.path.replace(/\\/g, '/'));
    gltf.scene.traverse((object) => {
      object.frustumCulled = false;
      if (object.isInstancedMesh) object.material = makeDepthMaterial(true);
      else if (object.isMesh) object.material = makeDepthMaterial(false);
    });
    gltf.scene.updateMatrixWorld(true);
    scene.add(gltf.scene);
    if ((index + 1) % 25 === 0 || index + 1 === manifest.glbEntries.length) {
      status('loaded ' + (index + 1) + '/' + manifest.glbEntries.length + ' GLBs');
    }
  }
  const loadElapsedMs = performance.now() - loadStarted;
  const gl = renderer.getContext();
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuBackend = {
    api: 'WebGL',
    vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
  };
  const rendererName = gpuBackend.renderer || 'unknown';
  const started = performance.now();
  for (let index = 0; index < manifest.poses.length; index += 1) {
    const pose = manifest.poses[index];
    setCamera(camera, pose);
    if (index === 0) {
      scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      const debugBox = new THREE.Box3().setFromObject(scene);
      const debugCenter = debugBox.getCenter(new THREE.Vector3()).project(camera);
      const debugDirection = camera.getWorldDirection(new THREE.Vector3());
      status('camera=' + camera.position.toArray().map((value) => value.toFixed(2)).join(',')
        + ' direction=' + debugDirection.toArray().map((value) => value.toFixed(3)).join(',')
        + ' aspect=' + camera.aspect.toFixed(3) + ' fov=' + camera.fov.toFixed(1)
        + '; first pose scene box [' + debugBox.min.toArray().map((value) => value.toFixed(1)).join(',')
        + ']..[' + debugBox.max.toArray().map((value) => value.toFixed(1)).join(',')
        + '], centerNdc=' + debugCenter.toArray().map((value) => value.toFixed(3)).join(','));
    }
    renderer.setRenderTarget(target);
    renderer.setClearColor(0xffffff, 1.0);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(target, 0, 0, Number(manifest.width), Number(manifest.height), pixels);
    decodeDepth(pixels, levelZero);
    if (index === 0) {
      let minDepth = 1.0;
      let maxDepth = 0.0;
      for (const value of levelZero) {
        minDepth = Math.min(minDepth, value);
        maxDepth = Math.max(maxDepth, value);
      }
      status('first pose depth range [' + minDepth.toFixed(6) + ', ' + maxDepth.toFixed(6)
        + '], drawCalls=' + renderer.info.render.calls + ', sceneChildren=' + scene.children.length);
    }
    const flattened = buildMinPyramid(levelZero, Number(manifest.width), Number(manifest.height));
    await postPose(index, flattened);
    if ((index + 1) % 8 === 0 || index + 1 === manifest.poses.length) {
      status('rendered ' + (index + 1) + '/' + manifest.poses.length + ' poses');
    }
  }
  await fetch('/done', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema: 'neuralstreamweb3d-triangle-hzb-browser-build-v1',
      status: 'rendered',
      width: Number(manifest.width),
      height: Number(manifest.height),
      fovYDeg: Number(manifest.fovYDeg),
      cameraFar: Number(manifest.cameraFar),
      poseCount: manifest.poses.length,
      glbCount: manifest.glbEntries.length,
      loadElapsedMs,
      renderElapsedMs: performance.now() - started,
      totalElapsedMs: performance.now() - loadStarted,
      gpuBackend,
      webglRenderer: rendererName,
    }),
  });
}

main().catch(async (error) => {
  await fetch('/done', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema: 'neuralstreamweb3d-triangle-hzb-browser-build-v1',
      status: 'failed',
      error: String(error && error.stack ? error.stack : error),
    }),
  }).catch(() => {});
});
`;
}

async function main() {
  const args = parseArgs(process.argv);
  const datasetMeta = readJson(path.join(args.datasetDir, 'dataset_meta.json'));
  const runtimeMeta = readJson(path.resolve(datasetMeta.runtimeMeta || path.join(args.assetsDir, 'runtimeVisibilityMeta.json')));
  const poseSelection = readPoseRecords(args.datasetDir, datasetMeta, args);
  const poses = poseSelection.poses;
  const { index: glbIndex, selected: glbEntries } = readGlbEntries(args.assetsDir, args.glbIdList);
  const { descriptors, valueCount } = buildLevelDescriptors(args.width, args.height);
  const cameraFar = args.cameraFar > 0 ? args.cameraFar : sceneDiagonal(runtimeMeta) * 2.0;
  if (!Number.isFinite(cameraFar) || cameraFar <= 0) throw new Error('cameraFar must be positive');
  const outputDir = path.dirname(args.output);
  fs.mkdirSync(outputDir, { recursive: true });
  const metaPath = args.output.replace(/\.bin$/i, '.json');
  if (!args.force && (fs.existsSync(args.output) || fs.existsSync(metaPath))) {
    throw new Error(`output exists; pass --force to replace: ${args.output}`);
  }
  const partialPath = `${args.output}.partial`;
  fs.rmSync(partialPath, { force: true });
  const valuesFile = fs.openSync(partialPath, 'w');
  fs.ftruncateSync(valuesFile, poses.length * valueCount * 4);
  const manifest = {
    schema: 'neuralstreamweb3d-triangle-hzb-browser-build-manifest-v1',
    cacheSchema: CACHE_SCHEMA,
    depthEncoding: DEPTH_ENCODING,
    assetsDir: args.assetsDir,
    datasetDir: args.datasetDir,
    glbIndex: path.join(args.assetsDir, 'glbIndex.json'),
    split: args.split,
    selectedSplits: poseSelection.splitNames,
    width: args.width,
    height: args.height,
    fovYDeg: args.fovYDeg,
    frontendRenderFovYDeg: 60,
    aspect: args.aspect > 0 ? args.aspect : 'pose_record_tan_x_over_tan_y',
    cameraFar,
    glbEntries,
    poses,
  };
  let doneResolve;
  let doneReject;
  const donePromise = new Promise((resolve, reject) => {
    doneResolve = resolve;
    doneReject = reject;
  });
  const written = new Set();
  const logs = [];
  const server = http.createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (request.method === 'GET' && url.pathname === '/') {
        response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
        response.end(rendererHtml());
        return;
      }
      if (request.method === 'GET' && url.pathname === '/renderer.js') {
        response.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store' });
        response.end(rendererJs());
        return;
      }
      if (request.method === 'GET' && url.pathname === '/manifest') {
        writeJson(response, manifest);
        return;
      }
      if (request.method === 'GET' && url.pathname.startsWith('/node_modules/')) {
        const file = safeJoin(path.join(SLM2_ROOT, 'node_modules'), url.pathname.slice('/node_modules/'.length));
        if (!fs.existsSync(file)) throw new Error(`missing dependency file: ${file}`);
        response.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'public, max-age=3600' });
        fs.createReadStream(file).pipe(response);
        return;
      }
      if (request.method === 'GET' && url.pathname.startsWith('/assets/')) {
        const file = safeJoin(args.assetsDir, url.pathname.slice('/assets/'.length));
        if (!fs.existsSync(file)) throw new Error(`missing scene asset: ${file}`);
        response.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'public, max-age=3600' });
        fs.createReadStream(file).pipe(response);
        return;
      }
      if (request.method === 'POST' && url.pathname === '/pose-hzb') {
        const ordinal = Number(url.searchParams.get('ordinal'));
        const body = await readBody(request);
        if (!Number.isInteger(ordinal) || ordinal < 0 || ordinal >= poses.length) throw new Error(`invalid pose ordinal: ${ordinal}`);
        if (body.byteLength !== valueCount * 4) throw new Error(`pose ${ordinal} has ${body.byteLength} bytes; expected ${valueCount * 4}`);
        if (written.has(ordinal)) throw new Error(`pose ${ordinal} was posted twice`);
        fs.writeSync(valuesFile, body, 0, body.byteLength, ordinal * valueCount * 4);
        written.add(ordinal);
        writeJson(response, { ok: true });
        return;
      }
      if (request.method === 'POST' && url.pathname === '/log') {
        const message = (await readBody(request)).toString('utf8');
        logs.push({ time: new Date().toISOString(), message });
        console.log('[browser]', message);
        writeJson(response, { ok: true });
        return;
      }
      if (request.method === 'POST' && url.pathname === '/done') {
        const body = JSON.parse((await readBody(request)).toString('utf8'));
        if (args.requireHardwareGpu) {
          const gpuGate = classifyGpuBackend(body.gpuBackend);
          body.gpuGate = { required: true, ...gpuGate };
          if (!gpuGate.hardware && !body.error) {
            body.error = `hardware GPU required, browser reported ${gpuGate.renderer || 'no WebGL renderer'}`;
            body.status = 'failed_hardware_gpu_gate';
          }
        }
        if (body.status === 'failed' || body.error) doneReject(new Error(body.error || 'browser HZB build failed'));
        else doneResolve(body);
        writeJson(response, { ok: true });
        return;
      }
      writeJson(response, { error: 'not found' }, 404);
    } catch (error) {
      writeJson(response, { error: String(error && error.message ? error.message : error) }, 500);
    }
  });
  await new Promise((resolve) => server.listen(args.port, '127.0.0.1', resolve));
  const port = server.address().port;
  const chromeExe = findChrome(args.chromeExe);
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'triangle-hzb-chrome-'));
  const chromeArgs = [
    '--headless=new',
    '--no-first-run',
    '--disable-background-networking',
    '--disable-extensions',
    '--disable-dev-shm-usage',
    '--hide-scrollbars',
    '--mute-audio',
    '--enable-gpu',
    '--enable-webgl',
    '--use-angle=vulkan',
    '--enable-accelerated-2d-canvas',
    '--enable-zero-copy',
    '--ignore-gpu-blocklist',
    ...(args.requireHardwareGpu ? ['--disable-software-rasterizer'] : []),
    '--disable-gpu-sandbox',
    '--no-sandbox',
    `--user-data-dir=${userDataDir}`,
    `http://127.0.0.1:${port}/`,
  ];
  console.log(JSON.stringify({
    schema: manifest.schema,
    split: args.split,
    selectedSplits: poseSelection.splitNames,
    poseCount: poses.length,
    glbCount: glbEntries.length,
    width: args.width,
    height: args.height,
    fovYDeg: args.fovYDeg,
    cameraFar,
    hardwareGpuRequired: args.requireHardwareGpu,
    valueCountPerPose: valueCount,
    output: args.output,
  }, null, 2));
  console.log(`[triangle-hzb] launching ${chromeExe}`);
  const chrome = spawn(chromeExe, chromeArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
  chrome.stdout.on('data', (chunk) => process.stdout.write(chunk));
  chrome.stderr.on('data', (chunk) => process.stderr.write(chunk));
  const timer = setTimeout(() => doneReject(new Error(`browser HZB build timed out after ${args.timeoutMs} ms`)), args.timeoutMs);
  let browserResult;
  try {
    browserResult = await donePromise;
    clearTimeout(timer);
    if (written.size !== poses.length) throw new Error(`browser completed with ${written.size}/${poses.length} pose blocks`);
    fs.closeSync(valuesFile);
    fs.renameSync(partialPath, args.output);
    const cacheMetadata = {
      schema: CACHE_SCHEMA,
      depthEncoding: DEPTH_ENCODING,
      sceneName: path.basename(args.assetsDir),
      assetsDir: args.assetsDir,
      datasetDir: args.datasetDir,
      glbIndex: path.join(args.assetsDir, 'glbIndex.json'),
      geometryScope: glbEntries.length === Number(glbIndex.total) ? 'complete_glb_inventory' : 'explicit_glb_subset_non_formal',
      formalReady: glbEntries.length === Number(glbIndex.total),
      width: args.width,
      height: args.height,
      cameraFovYDeg: args.fovYDeg,
      frontendRenderFovYDeg: 60,
      aspect: args.aspect > 0 ? args.aspect : 'per_pose_tan_x_over_tan_y',
      cameraFar,
      cameraFarSource: args.cameraFar > 0 ? 'command_line' : 'scene_diagonal_x2',
      levelDescriptors: descriptors,
      valueCount: poses.length * valueCount,
      poseCount: poses.length,
      glbCount: glbEntries.length,
      sourceGlbCount: Number(glbIndex.total),
      poses: poses.map((pose, ordinal) => ({
        poseIndex: pose.poseIndex,
        cameraWorld: pose.cameraWorld,
        cameraForward: pose.cameraForward,
        valueOffset: ordinal * valueCount,
        valueCount,
        splitId: pose.splitId,
      })),
      browserBuild: browserResult,
      browserLogs: logs,
    };
    fs.writeFileSync(metaPath, JSON.stringify(cacheMetadata, null, 2), 'utf8');
    fs.writeFileSync(`${args.output}.build_summary.json`, JSON.stringify({ ...cacheMetadata, output: args.output }, null, 2), 'utf8');
    console.log(`triangle HZB cache written: ${args.output}`);
    console.log(`triangle HZB metadata written: ${metaPath}`);
  } finally {
    clearTimeout(timer);
    chrome.kill();
    server.close();
    try {
      fs.closeSync(valuesFile);
    } catch (_) {
      // The file may already be closed after a successful rename.
    }
    fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 });
    if (!browserResult) fs.rmSync(partialPath, { force: true });
  }
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
