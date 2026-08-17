#!/usr/bin/env node
/*
 * Hardware-WebGL triangle depth peeling for the bounded-relation experiment.
 *
 * Input is an instance-ID render manifest with `poses`, `glbEntries`,
 * `instanceBindings`, `assetsDir` and a model-input FOV of 66 degrees.  The
 * browser emits fixed-size ID/depth blocks; Python performs the relation
 * aggregation so the browser never runs a graph or a neighbour query.
 */
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SLM2_ROOT = path.join(REPO_ROOT, 'slm2viewer');
const CACHE_SCHEMA = 'triangle-depth-layer-cache-v3';
const LINEAR_DEPTH_ENCODING = 'camera_forward_axial_depth_divided_by_camera_far';
const MANIFEST_SCHEMAS = new Set([
  'triangle-depth-layer-render-manifest-v1',
  'triangle-depth-layer-render-manifest-v2',
]);
const MODEL_FOV = 66;

function parseArgs(argv) {
  const args = {
    manifest: null,
    output: null,
    chromeExe: process.env.CHROME_EXE || null,
    width: 320,
    height: 180,
    maxLayers: 6,
    port: 0,
    timeoutMs: 30 * 60 * 1000,
    requireHardwareGpu: true,
    resume: false,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--require-hardware-gpu') { args.requireHardwareGpu = true; continue; }
    if (key === '--allow-software-gpu') { args.requireHardwareGpu = false; continue; }
    if (key === '--resume') { args.resume = true; continue; }
    const value = argv[index + 1];
    index += 1;
    if (key === '--manifest') args.manifest = path.resolve(value);
    else if (key === '--output') args.output = path.resolve(value);
    else if (key === '--chrome-exe') args.chromeExe = path.resolve(value);
    else if (key === '--width') args.width = Number(value);
    else if (key === '--height') args.height = Number(value);
    else if (key === '--max-layers') args.maxLayers = Number(value);
    else if (key === '--port') args.port = Number(value);
    else if (key === '--timeout-ms') args.timeoutMs = Number(value);
    else throw new Error(`unknown argument ${key}`);
  }
  if (!args.manifest || !args.output) throw new Error('--manifest and --output are required');
  if (!Number.isInteger(args.width) || !Number.isInteger(args.height) || args.width <= 0 || args.height <= 0) {
    throw new Error('width and height must be positive integers');
  }
  if (!Number.isInteger(args.maxLayers) || args.maxLayers < 2 || args.maxLayers > 16) {
    throw new Error('max-layers must be between 2 and 16');
  }
  return args;
}

function readJson(file) { return JSON.parse(fs.readFileSync(file, 'utf8')); }

function findChrome(explicit) {
  const candidates = [explicit, process.env.CHROME_BIN, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error('Chrome/Chromium executable not found; pass --chrome-exe');
}

function gpuIsHardware(value) {
  const text = [value?.vendor, value?.renderer, value?.version].filter(Boolean).join(' ');
  return Boolean(text) && !/swiftshader|llvmpipe|softpipe|swrast|software|no-webgl/i.test(text);
}

function captureCommand(command, commandArgs) {
  try {
    return {
      available: true,
      output: execFileSync(command, commandArgs, {
        encoding: 'utf8',
        timeout: 10000,
        stdio: ['ignore', 'pipe', 'pipe'],
      }).trim(),
    };
  } catch (error) {
    return {
      available: false,
      output: '',
      error: String(error?.message || error),
    };
  }
}

function captureHostGpuEvidence() {
  return {
    capturedAt: new Date().toISOString(),
    nvidiaSmi: captureCommand('nvidia-smi', [
      '--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total',
      '--format=csv,noheader,nounits',
    ]),
    nvidiaSmiPmon: captureCommand('nvidia-smi', ['pmon', '-c', '1', '-s', 'um']),
  };
}

function writeGpuEvidence(outputPath, value) {
  fs.writeFileSync(`${outputPath}.gpu_evidence.json`, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function bodyBuffer(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on('data', (chunk) => chunks.push(chunk));
    request.on('end', () => resolve(Buffer.concat(chunks)));
    request.on('error', reject);
  });
}

function writeJson(response, value, status = 200) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(value));
}

function contentType(file) {
  const extension = path.extname(file).toLowerCase();
  if (extension === '.js' || extension === '.mjs') return 'text/javascript; charset=utf-8';
  if (extension === '.json') return 'application/json; charset=utf-8';
  if (extension === '.wasm') return 'application/wasm';
  if (extension === '.glb') return 'model/gltf-binary';
  if (extension === '.bin') return 'application/octet-stream';
  return 'application/octet-stream';
}

function rendererHtml() {
  return `<!doctype html><html><body style="margin:0;background:#111;color:#ddd;font:12px monospace"><pre id="status">booting</pre><script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script><script type="module" src="/renderer.js"></script></body></html>`;
}

function rendererJs() {
  return String.raw`
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const manifest = await (await fetch('/manifest', { cache: 'no-store' })).json();
const statusNode = document.getElementById('status');
function status(message) {
  statusNode.textContent = message;
  console.log('[triangle-depth-layer]', message);
  fetch('/log', { method: 'POST', body: message }).catch(() => {});
}

function packDepth(depth) {
  const bitShift = [1.0, 255.0, 65025.0, 160581375.0];
  const bitMask = [0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0];
  const value = Math.min(0.99999994, Math.max(0.0, depth));
  const result = bitShift.map((v) => (value * v) % 1.0);
  for (let i = 0; i < 4; i += 1) result[i] -= (i < 3 ? result[i + 1] : result[3]) * bitMask[i];
  return result.map((v) => Math.max(0, Math.min(255, Math.floor(v * 255.0 + 0.5))));
}

function shaderMaterial(kind, componentId = 0) {
  const instanced = componentId === null;
  const vertex = [
    'precision highp float;',
    instanced ? 'in float componentId;' : 'uniform float componentId;',
    'flat out highp float vComponentId;',
    'out highp float vLinearDepth;',
    'uniform float cameraFar;',
    'void main() {',
    '  vec4 viewPosition = modelViewMatrix * ' + (instanced ? 'instanceMatrix * ' : '') + 'vec4(position, 1.0);',
    '  vComponentId = componentId;',
    '  vLinearDepth = clamp(-viewPosition.z / cameraFar, 0.0, 0.99999994);',
    '  gl_Position = projectionMatrix * viewPosition;',
    '}',
  ].join('\n');
  const fragment = [
    'precision highp float;',
    instanced ? 'flat in highp float vComponentId;' : 'uniform highp float componentId;',
    'in highp float vLinearDepth;',
    'out vec4 outColor;',
    'uniform sampler2D previousDepth;',
    'uniform vec2 targetSize;',
    'uniform float peelEnabled;',
    'uniform float peelEpsilon;',
    'float unpackDepth(vec4 value) {',
    '  return dot(value, vec4(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));',
    '}',
    'vec4 packDepth(float value) {',
    '  const vec4 bitShift = vec4(1.0, 255.0, 65025.0, 160581375.0);',
    '  const vec4 bitMask = vec4(0.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);',
    '  vec4 result = fract(value * bitShift);',
    '  result -= result.yzww * bitMask;',
    '  return result;',
    '}',
    'void main() {',
    '  vec2 uv = gl_FragCoord.xy / targetSize;',
    '  if (peelEnabled > 0.5 && vLinearDepth <= unpackDepth(texture(previousDepth, uv)) + peelEpsilon) discard;',
    kind === 'depth'
      ? '  outColor = packDepth(vLinearDepth);'
      : [
          '  float id = ' + (instanced ? 'vComponentId' : 'componentId') + ' + 1.0;',
          '  float r = mod(id, 256.0);',
          '  float g = mod(floor(id / 256.0), 256.0);',
          '  float b = mod(floor(id / 65536.0), 256.0);',
          '  outColor = vec4(r, g, b, 255.0) / 255.0;',
        ].join('\n'),
    '}',
  ].join('\n');
  return new THREE.ShaderMaterial({
    uniforms: {
      componentId: { value: Number(componentId || 0) },
      cameraFar: { value: Number(manifest.cameraFar) },
      previousDepth: { value: null },
      targetSize: { value: new THREE.Vector2(Number(manifest.width), Number(manifest.height)) },
      peelEnabled: { value: 0.0 },
      peelEpsilon: { value: 2e-5 },
    },
    vertexShader: vertex,
    fragmentShader: fragment,
    glslVersion: THREE.GLSL3,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    toneMapped: false,
  });
}

function decodeIds(pixels) {
  const ids = new Uint32Array(pixels.length / 4);
  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    const value = pixels[i] + pixels[i + 1] * 256 + pixels[i + 2] * 65536;
    ids[p] = pixels[i + 3] === 0 ? 0xffffffff : value - 1;
  }
  return ids;
}

function decodeDepth(pixels) {
  const depth = new Float32Array(pixels.length / 4);
  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    depth[p] = Math.min(1.0, pixels[i] / 255.0 + pixels[i + 1] / 65025.0 + pixels[i + 2] / 16581375.0 + pixels[i + 3] / 4228250625.0);
  }
  return depth;
}

function setCamera(camera, pose) {
  camera.fov = Number(manifest.modelInputFovYDeg);
  camera.aspect = Number(pose.aspect || manifest.aspect || 16 / 9);
  camera.near = 0.05;
  camera.far = Number(manifest.cameraFar);
  camera.position.set(...pose.cameraWorld);
  camera.up.set(0, 1, 0);
  camera.lookAt(new THREE.Vector3(
    pose.cameraWorld[0] + pose.cameraForward[0],
    pose.cameraWorld[1] + pose.cameraForward[1],
    pose.cameraWorld[2] + pose.cameraForward[2],
  ));
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
}

async function main() {
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
  renderer.setSize(Number(manifest.width), Number(manifest.height), false);
  renderer.setPixelRatio(1);
  renderer.setClearColor(0xffffff, 1.0);
  document.body.appendChild(renderer.domElement);
  const width = Number(manifest.width), height = Number(manifest.height);
  const depthTarget = new THREE.WebGLRenderTarget(width, height, { format: THREE.RGBAFormat, type: THREE.UnsignedByteType, depthBuffer: true, stencilBuffer: false });
  const idTarget = new THREE.WebGLRenderTarget(width, height, { format: THREE.RGBAFormat, type: THREE.UnsignedByteType, depthBuffer: true, stencilBuffer: false });
  const depthPixels = new Uint8Array(width * height * 4);
  const idPixels = new Uint8Array(width * height * 4);
  const scene = new THREE.Scene();
  const entries = [];
  const loader = new GLTFLoader().setDRACOLoader(new DRACOLoader().setDecoderPath('/node_modules/three/examples/jsm/libs/draco/gltf/')).setMeshoptDecoder(MeshoptDecoder);
  const bindingByGlb = manifest.instanceBindings.byGlobalGlbId || {};
  for (let index = 0; index < manifest.glbEntries.length; index += 1) {
    const entry = manifest.glbEntries[index];
    const gid = Number(entry.globalId);
    const binding = bindingByGlb[String(gid)];
    if (!binding) throw new Error('missing instance binding for GLB ' + gid);
    const url = '/assets/' + String(entry.path).replace(/^\/+/, '').replaceAll('\\\\', '/');
    const gltf = await loader.loadAsync(url);
    gltf.scene.traverse((object) => {
      object.frustumCulled = false;
      if (object.isInstancedMesh) {
        if (object.count !== binding.componentGlobalIds.length) throw new Error('instance count mismatch for GLB ' + gid);
        object.geometry.setAttribute('componentId', new THREE.InstancedBufferAttribute(new Float32Array(binding.componentGlobalIds.map(Number)), 1));
        const depthMaterial = shaderMaterial('depth', null);
        const idMaterial = shaderMaterial('id', null);
        object.material = idMaterial;
        entries.push({ object, depthMaterial, idMaterial, componentId: null, globalGlbId: gid });
      } else if (object.isMesh) {
        if (binding.componentGlobalIds.length !== 1) throw new Error('static GLB must bind exactly one component: ' + gid);
        const depthMaterial = shaderMaterial('depth', Number(binding.componentGlobalIds[0]));
        const idMaterial = shaderMaterial('id', Number(binding.componentGlobalIds[0]));
        object.material = idMaterial;
        entries.push({ object, depthMaterial, idMaterial, componentId: Number(binding.componentGlobalIds[0]), globalGlbId: gid });
      }
    });
    scene.add(gltf.scene);
    if ((index + 1) % 25 === 0 || index + 1 === manifest.glbEntries.length) status('loaded ' + (index + 1) + '/' + manifest.glbEntries.length + ' GLBs');
  }
  const gl = renderer.getContext();
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuBackend = {
    api: 'WebGL',
    vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
  };
  const camera = new THREE.PerspectiveCamera(Number(manifest.modelInputFovYDeg), 16 / 9, 0.05, Number(manifest.cameraFar));
  const initialPacked = new Uint8Array(width * height * 4);
  for (let i = 0; i < initialPacked.length; i += 4) { initialPacked[i] = 255; initialPacked[i + 1] = 255; initialPacked[i + 2] = 255; initialPacked[i + 3] = 255; }
  const referenceIdPixels = new Uint8Array(width * height * 4);
  let firstLayerReferenceChecks = Number(manifest.resumeStart || 0);
  let firstLayerReferenceMismatches = 0;
  const previousTexture = new THREE.DataTexture(initialPacked, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  previousTexture.needsUpdate = true;
  previousTexture.flipY = false;
  function debugComponentPose(pose) {
    const requestedComponentId = Number(manifest.debugComponentId);
    const requestedPoseIndex = Number(manifest.debugPoseIndex);
    const sourcePoseIndex = Number(pose.sourcePoseIndex ?? pose.poseIndex);
    if (!Number.isInteger(requestedComponentId) || !Number.isInteger(requestedPoseIndex) || sourcePoseIndex !== requestedPoseIndex) return;
    let entry = entries.find((candidate) => candidate.componentId === requestedComponentId);
    let instanceIndex = null;
    if (!entry) {
      for (const candidate of entries) {
        if (!candidate.object.isInstancedMesh) continue;
        const componentAttribute = candidate.object.geometry.getAttribute('componentId');
        if (!componentAttribute) continue;
        for (let index = 0; index < componentAttribute.count; index += 1) {
          if (Number(componentAttribute.getX(index)) === requestedComponentId) {
            entry = candidate;
            instanceIndex = index;
            break;
          }
        }
        if (entry) break;
      }
    }
    if (!entry) {
      fetch('/debug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ poseIndex: Number(pose.poseIndex), componentId: requestedComponentId, error: 'component is not a static mesh entry' }) }).catch(() => {});
      return;
    }
    entry.object.updateWorldMatrix(true, false);
    if (manifest.debugHideAll === true) {
      for (const candidate of entries) candidate.object.visible = false;
    }
    if (manifest.debugHideComponentId === requestedComponentId) entry.object.visible = false;
    entry.object.geometry.computeBoundingBox();
    const objectMatrix = entry.object.matrixWorld.clone();
    if (instanceIndex !== null) {
      const instanceMatrix = new THREE.Matrix4();
      entry.object.getMatrixAt(instanceIndex, instanceMatrix);
      objectMatrix.multiply(instanceMatrix);
    }
    const box = entry.object.geometry.boundingBox.clone().applyMatrix4(objectMatrix);
    const corners = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const projected = new THREE.Vector3(x, y, z).project(camera);
      corners.push({ x: projected.x, y: projected.y, z: projected.z });
    }
    const debug = {
      poseIndex: sourcePoseIndex,
      renderPoseId: Number(pose.renderPoseId ?? pose.poseIndex),
      sourcePoseIndex,
      componentId: requestedComponentId,
      globalGlbId: entry.globalGlbId,
      instanceIndex,
      cameraWorld: camera.position.toArray(),
      cameraForward: new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).toArray(),
      cameraAspect: camera.aspect,
      cameraFovYDeg: camera.fov,
      cameraNear: camera.near,
      cameraFar: camera.far,
      cameraMatrixWorld: camera.matrixWorld.toArray(),
      cameraMatrixWorldInverse: camera.matrixWorldInverse.toArray(),
      cameraProjectionMatrix: camera.projectionMatrix.toArray(),
      objectMatrixWorld: objectMatrix.toArray(),
      worldBounds: { min: box.min.toArray(), max: box.max.toArray() },
      projectedBounds: {
        minX: Math.min(...corners.map((value) => value.x)),
        maxX: Math.max(...corners.map((value) => value.x)),
        minY: Math.min(...corners.map((value) => value.y)),
        maxY: Math.max(...corners.map((value) => value.y)),
        minZ: Math.min(...corners.map((value) => value.z)),
        maxZ: Math.max(...corners.map((value) => value.z)),
      },
      projectedCorners: corners,
    };
    if (entry.object.isInstancedMesh) {
      const componentAttribute = entry.object.geometry.getAttribute('componentId');
      const siblings = [];
      for (let index = 0; index < entry.object.count; index += 1) {
        const siblingMatrix = new THREE.Matrix4();
        entry.object.getMatrixAt(index, siblingMatrix);
        const siblingWorldMatrix = entry.object.matrixWorld.clone().multiply(siblingMatrix);
        const siblingBox = entry.object.geometry.boundingBox.clone().applyMatrix4(siblingWorldMatrix);
        const siblingCorners = [];
        for (const x of [siblingBox.min.x, siblingBox.max.x]) for (const y of [siblingBox.min.y, siblingBox.max.y]) for (const z of [siblingBox.min.z, siblingBox.max.z]) {
          const projected = new THREE.Vector3(x, y, z).project(camera);
          siblingCorners.push(projected);
        }
        siblings.push({
          index,
          componentId: componentAttribute ? Number(componentAttribute.getX(index)) : null,
          projectedBounds: {
            minX: Math.min(...siblingCorners.map((value) => value.x)),
            maxX: Math.max(...siblingCorners.map((value) => value.x)),
            minY: Math.min(...siblingCorners.map((value) => value.y)),
            maxY: Math.max(...siblingCorners.map((value) => value.y)),
          },
        });
      }
      debug.instancedProjectionSummary = siblings;
    }
    console.log('[triangle-depth-debug]', JSON.stringify(debug));
    fetch('/debug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(debug) }).catch(() => {});
  }
  function setMode(mode, peelEnabled) {
    for (const entry of entries) {
      const material = mode === 'depth' ? entry.depthMaterial : entry.idMaterial;
      if (entry.componentId !== null) material.uniforms.componentId.value = entry.componentId;
      material.uniforms.previousDepth.value = previousTexture;
      material.uniforms.peelEnabled.value = peelEnabled ? 1.0 : 0.0;
      // Several static meshes share the same ShaderMaterial program. Three.js
      // otherwise may keep the previous object's componentId uniform in the
      // WebGL uniform cache, shifting IDs at adjacent draw calls.
      material.uniformsNeedUpdate = true;
      entry.object.material = material;
    }
  }
  async function postPose(ordinal, ids, depths) {
    const body = new Uint8Array(ids.byteLength + depths.byteLength);
    body.set(new Uint8Array(ids.buffer), 0);
    body.set(new Uint8Array(depths.buffer), ids.byteLength);
    const response = await fetch('/pose-layers?ordinal=' + encodeURIComponent(ordinal), { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: body.buffer });
    if (!response.ok) throw new Error('server rejected pose ' + ordinal);
  }
  async function debugProbePixel(pose) {
    const requestedPoseIndex = Number(manifest.debugProbePoseIndex ?? manifest.debugPoseIndex);
    const probeX = Number(manifest.debugProbeX);
    const probeY = Number(manifest.debugProbeY);
    const sourcePoseIndex = Number(pose.sourcePoseIndex ?? pose.poseIndex);
    if (!Number.isInteger(requestedPoseIndex) || sourcePoseIndex !== requestedPoseIndex || !Number.isInteger(probeX) || !Number.isInteger(probeY)) return;
    const pixel = new Uint8Array(4);
    const previousVisibility = entries.map((entry) => entry.object.visible);
    for (const entry of entries) entry.object.visible = false;
    setMode('id', false);
    const hits = [];
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      entry.object.visible = true;
      renderer.setRenderTarget(idTarget);
      renderer.setClearColor(0x000000, 0.0);
      renderer.clear(true, true, true);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(idTarget, probeX, probeY, 1, 1, pixel);
      const encoded = pixel[0] + pixel[1] * 256 + pixel[2] * 65536;
      const componentId = pixel[3] === 0 ? null : encoded - 1;
      if (componentId !== null) hits.push({ entryIndex: index, globalGlbId: entry.globalGlbId, entryComponentId: entry.componentId, materialComponentId: entry.idMaterial.uniforms.componentId?.value ?? null, renderedComponentId: componentId, rgba: Array.from(pixel) });
      entry.object.visible = false;
    }
    entries.forEach((entry, index) => { entry.object.visible = previousVisibility[index]; });
    const payload = { poseIndex: sourcePoseIndex, renderPoseId: Number(pose.renderPoseId ?? pose.poseIndex), sourcePoseIndex, x: probeX, y: probeY, hits };
    console.log('[triangle-depth-probe]', JSON.stringify(payload));
    await fetch('/debug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).catch(() => {});
  }
  const started = performance.now();
  for (let poseOrdinal = Number(manifest.resumeStart || 0); poseOrdinal < manifest.poses.length; poseOrdinal += 1) {
    const pose = manifest.poses[poseOrdinal];
    setCamera(camera, pose);
    debugComponentPose(pose);
    await debugProbePixel(pose);
    const poseIds = new Uint32Array(Number(manifest.maxLayers) * width * height);
    const poseDepths = new Float32Array(Number(manifest.maxLayers) * width * height);
    previousTexture.image.data = initialPacked;
    previousTexture.needsUpdate = true;
    for (let layer = 0; layer < Number(manifest.maxLayers); layer += 1) {
      setMode('depth', layer > 0);
      renderer.setRenderTarget(depthTarget);
      renderer.setClearColor(0xffffff, 1.0);
      renderer.clear(true, true, true);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(depthTarget, 0, 0, width, height, depthPixels);
      const decodedDepth = decodeDepth(depthPixels);
      setMode('id', layer > 0);
      renderer.setRenderTarget(idTarget);
      renderer.setClearColor(0x000000, 0.0);
      renderer.clear(true, true, true);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(idTarget, 0, 0, width, height, idPixels);
      const decodedIds = decodeIds(idPixels);
      if (layer === 0) {
        // Layer zero is the ordinary, unpeeled Color-ID render. Repeat that
        // pass once so the formal cache proves that its first layer is not a
        // side effect of the peeling path.
        referenceIdPixels.set(idPixels);
        setMode('id', false);
        renderer.setRenderTarget(idTarget);
        renderer.setClearColor(0x000000, 0.0);
        renderer.clear(true, true, true);
        renderer.render(scene, camera);
        renderer.readRenderTargetPixels(idTarget, 0, 0, width, height, idPixels);
        firstLayerReferenceChecks += 1;
        let mismatch = false;
        for (let pixel = 0; pixel < idPixels.length; pixel += 1) {
          if (idPixels[pixel] !== referenceIdPixels[pixel]) { mismatch = true; break; }
        }
        if (mismatch) firstLayerReferenceMismatches += 1;
        if (mismatch) throw new Error('independent unpeeled Color-ID pass disagrees with depth-layer zero');
      }
      poseIds.set(decodedIds, layer * width * height);
      poseDepths.set(decodedDepth, layer * width * height);
      previousTexture.image.data = new Uint8Array(depthPixels);
      previousTexture.needsUpdate = true;
    }
    await postPose(poseOrdinal, poseIds, poseDepths);
    if ((poseOrdinal + 1) % 8 === 0 || poseOrdinal + 1 === manifest.poses.length) status('rendered ' + (poseOrdinal + 1) + '/' + manifest.poses.length + ' poses');
  }
  await fetch('/done', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ schema: manifest.cacheSchema, status: 'rendered', gpuBackend, renderElapsedMs: performance.now() - started, firstLayerReference: { checks: firstLayerReferenceChecks, mismatches: firstLayerReferenceMismatches } }) });
}

main().catch(async (error) => {
  await fetch('/done', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ schema: manifest?.cacheSchema, status: 'failed', error: String(error && error.stack ? error.stack : error) }) }).catch(() => {});
});
`;
}

async function main() {
  const args = parseArgs(process.argv);
  const manifest = readJson(args.manifest);
  if (!MANIFEST_SCHEMAS.has(manifest.schema)) throw new Error(`unsupported triangle manifest schema ${manifest.schema}`);
  if (Number(manifest.modelInputFovYDeg) !== MODEL_FOV) throw new Error('triangle cache must use model input FOV 66');
  if (!Array.isArray(manifest.poses) || !Array.isArray(manifest.glbEntries)) throw new Error('manifest poses/glbEntries are required');
  if (!Number.isFinite(Number(manifest.cameraFar)) || Number(manifest.cameraFar) <= 0) throw new Error('manifest cameraFar must be finite and positive');
  manifest.width = args.width;
  manifest.height = args.height;
  manifest.maxLayers = args.maxLayers;
  manifest.requireHardwareGpu = args.requireHardwareGpu;
  manifest.cacheSchema = CACHE_SCHEMA;
  const outputDir = path.dirname(args.output);
  fs.mkdirSync(outputDir, { recursive: true });
  const partial = `${args.output}.partial`;
  const pixelCount = args.width * args.height;
  const bytesPerPose = args.maxLayers * pixelCount * 4;
  const poseIndexPath = `${args.output}.pose_indices.partial`;
  const sourcePoseIndexPath = `${args.output}.source_pose_indices.partial`;
  const renderPoseIdPath = `${args.output}.render_pose_ids.partial`;
  const idsPath = `${args.output}.instance_ids.partial`;
  const depthPath = `${args.output}.linear_depth.partial`;
  const expectedBinaryBytes = manifest.poses.length * bytesPerPose;
  const indexPaths = [poseIndexPath, sourcePoseIndexPath, renderPoseIdPath];
  let resumeStart = 0;
  if (args.resume) {
    const indexSizes = indexPaths.map((file) => {
      if (!fs.existsSync(file)) throw new Error(`cannot resume: missing ${file}`);
      const size = fs.statSync(file).size;
      if (size % 4 !== 0) throw new Error(`cannot resume: ${file} is not aligned to uint32 records`);
      return size / 4;
    });
    if (!indexSizes.every((value) => value === indexSizes[0])) {
      throw new Error(`cannot resume: index prefixes have different lengths ${indexSizes.join(',')}`);
    }
    resumeStart = indexSizes[0];
    if (resumeStart > manifest.poses.length) {
      throw new Error(`cannot resume: prefix ${resumeStart} exceeds manifest pose count ${manifest.poses.length}`);
    }
    for (const file of [idsPath, depthPath]) {
      if (!fs.existsSync(file) || fs.statSync(file).size !== expectedBinaryBytes) {
        throw new Error(`cannot resume: ${file} must be a preallocated ${expectedBinaryBytes}-byte file`);
      }
    }
    const renderPrefix = fs.readFileSync(renderPoseIdPath);
    const sourcePrefix = fs.readFileSync(sourcePoseIndexPath);
    const explicitPrefix = fs.readFileSync(poseIndexPath);
    for (let index = 0; index < resumeStart; index += 1) {
      const record = manifest.poses[index];
      const expectedRenderId = Number(record.renderPoseId ?? index);
      const expectedSourceId = Number(record.sourcePoseIndex ?? record.poseIndex);
      if (
        renderPrefix.readUInt32LE(index * 4) !== expectedRenderId
        || explicitPrefix.readUInt32LE(index * 4) !== expectedRenderId
        || sourcePrefix.readUInt32LE(index * 4) !== expectedSourceId
      ) {
        throw new Error(`cannot resume: prefix identity mismatch at pose ordinal ${index}`);
      }
    }
  }
  manifest.resumeStart = resumeStart;
  const poseIndexFd = fs.openSync(poseIndexPath, args.resume ? 'a' : 'w');
  const sourcePoseIndexFd = fs.openSync(sourcePoseIndexPath, args.resume ? 'a' : 'w');
  const renderPoseIdFd = fs.openSync(renderPoseIdPath, args.resume ? 'a' : 'w');
  const idsFd = fs.openSync(idsPath, args.resume ? 'r+' : 'w');
  const depthFd = fs.openSync(depthPath, args.resume ? 'r+' : 'w');
  if (!args.resume) {
    fs.ftruncateSync(idsFd, expectedBinaryBytes);
    fs.ftruncateSync(depthFd, expectedBinaryBytes);
  }
  const written = new Set(Array.from({ length: resumeStart }, (_value, index) => index));
  let doneResolve;
  let doneReject;
  const donePromise = new Promise((resolve, reject) => { doneResolve = resolve; doneReject = reject; });
  const server = http.createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (request.method === 'GET' && url.pathname === '/') { response.writeHead(200, { 'Content-Type': 'text/html' }); response.end(rendererHtml()); return; }
      if (request.method === 'GET' && url.pathname === '/favicon.ico') { response.writeHead(204); response.end(); return; }
      if (request.method === 'GET' && url.pathname === '/renderer.js') { response.writeHead(200, { 'Content-Type': 'text/javascript' }); response.end(rendererJs()); return; }
      if (request.method === 'GET' && url.pathname === '/manifest') { writeJson(response, manifest); return; }
      if (request.method === 'GET' && url.pathname.startsWith('/node_modules/')) {
        const file = path.resolve(SLM2_ROOT, 'node_modules', url.pathname.slice('/node_modules/'.length));
        if (!file.startsWith(path.resolve(SLM2_ROOT, 'node_modules')) || !fs.existsSync(file)) throw new Error(`missing dependency ${file}`);
        response.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'no-store' });
        fs.createReadStream(file).pipe(response); return;
      }
      if (request.method === 'GET' && url.pathname.startsWith('/assets/')) {
        const file = path.resolve(manifest.assetsDir, url.pathname.slice('/assets/'.length));
        if (!file.startsWith(path.resolve(manifest.assetsDir)) || !fs.existsSync(file)) throw new Error(`missing asset ${file}`);
        response.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'no-store' });
        fs.createReadStream(file).pipe(response); return;
      }
      if (request.method === 'POST' && url.pathname === '/log') { console.log('[browser]', (await bodyBuffer(request)).toString('utf8')); writeJson(response, { ok: true }); return; }
      if (request.method === 'POST' && url.pathname === '/debug') { console.log('[browser:debug]', (await bodyBuffer(request)).toString('utf8')); writeJson(response, { ok: true }); return; }
      if (request.method === 'POST' && url.pathname === '/pose-layers') {
        const ordinal = Number(url.searchParams.get('ordinal'));
        const body = await bodyBuffer(request);
        if (!Number.isInteger(ordinal) || ordinal < 0 || ordinal >= manifest.poses.length) throw new Error('invalid pose ordinal');
        if (body.byteLength !== bytesPerPose * 2) throw new Error(`pose body has ${body.byteLength}, expected ${bytesPerPose * 2}`);
        if (written.has(ordinal)) throw new Error(`pose ${ordinal} posted twice`);
        fs.writeSync(idsFd, body, 0, bytesPerPose, ordinal * bytesPerPose);
        fs.writeSync(depthFd, body, bytesPerPose, bytesPerPose, ordinal * bytesPerPose);
        const record = manifest.poses[ordinal];
        const sourcePoseIndex = Number(record.sourcePoseIndex ?? record.poseIndex);
        const renderPoseId = Number(record.renderPoseId ?? ordinal);
        if (!Number.isInteger(sourcePoseIndex) || sourcePoseIndex < 0 || !Number.isInteger(renderPoseId) || renderPoseId < 0) throw new Error('manifest pose IDs must be non-negative integers');
        const poseIndex = Buffer.alloc(4); poseIndex.writeUInt32LE(renderPoseId, 0); fs.writeSync(poseIndexFd, poseIndex);
        const sourcePose = Buffer.alloc(4); sourcePose.writeUInt32LE(sourcePoseIndex, 0); fs.writeSync(sourcePoseIndexFd, sourcePose);
        const renderPose = Buffer.alloc(4); renderPose.writeUInt32LE(renderPoseId, 0); fs.writeSync(renderPoseIdFd, renderPose);
        written.add(ordinal); writeJson(response, { ok: true }); return;
      }
      if (request.method === 'POST' && url.pathname === '/done') {
        const value = JSON.parse((await bodyBuffer(request)).toString('utf8'));
        if (value.status !== 'rendered' || (args.requireHardwareGpu && !gpuIsHardware(value.gpuBackend))) {
          doneReject(new Error(value.error || `hardware GPU gate failed: ${value.gpuBackend?.renderer || 'unknown'}`));
        } else doneResolve(value);
        writeJson(response, { ok: true }); return;
      }
      writeJson(response, { error: 'not found' }, 404);
    } catch (error) { writeJson(response, { error: String(error?.stack || error) }, 500); }
  });
  await new Promise((resolve) => server.listen(args.port, '127.0.0.1', resolve));
  const port = server.address().port;
  const chrome = findChrome(args.chromeExe);
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
    '--disable-gpu-sandbox',
    '--no-sandbox',
    ...(args.requireHardwareGpu ? ['--disable-software-rasterizer'] : []),
  ];
  const hostGpuBefore = captureHostGpuEvidence();
  const chromeStdoutPath = `${args.output}.chrome_stdout.log`;
  const chromeStderrPath = `${args.output}.chrome_stderr.log`;
  const chromeStdout = fs.createWriteStream(chromeStdoutPath, { flags: 'w' });
  const chromeStderr = fs.createWriteStream(chromeStderrPath, { flags: 'w' });
  console.log(JSON.stringify({ schema: CACHE_SCHEMA, poseCount: manifest.poses.length, resumeStart, width: args.width, height: args.height, maxLayers: args.maxLayers, modelInputFovYDeg: MODEL_FOV, hardwareGpuRequired: args.requireHardwareGpu, output: args.output }, null, 2));
  let userDataDir = null;
  let chromeProcess = null;
  let actualChromeArgs = chromeArgs;
  const timer = setTimeout(() => doneReject(new Error(`browser timed out after ${args.timeoutMs} ms`)), args.timeoutMs);
  let result = null;
  try {
    userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'triangle-depth-chrome-'));
    actualChromeArgs = [
      ...chromeArgs,
      `--user-data-dir=${userDataDir}`,
      `http://127.0.0.1:${port}/`,
    ];
    chromeProcess = spawn(chrome, actualChromeArgs, {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    chromeProcess?.stdout?.on('data', (chunk) => { process.stdout.write(chunk); chromeStdout.write(chunk); });
    chromeProcess?.stderr?.on('data', (chunk) => { process.stderr.write(chunk); chromeStderr.write(chunk); });
    result = await donePromise;
    clearTimeout(timer);
    const gpuGate = {
      required: args.requireHardwareGpu,
      hardware: gpuIsHardware(result.gpuBackend),
      software: !gpuIsHardware(result.gpuBackend),
      vendor: String(result.gpuBackend?.vendor || ''),
      renderer: String(result.gpuBackend?.renderer || ''),
      version: String(result.gpuBackend?.version || ''),
    };
    result.gpuGate = gpuGate;
    if (args.requireHardwareGpu && !gpuGate.hardware) {
      throw new Error(`hardware GPU required, browser reported ${gpuGate.renderer || 'no WebGL renderer'}`);
    }
    const hostGpuDuring = captureHostGpuEvidence();
    if (args.requireHardwareGpu && (!hostGpuDuring.nvidiaSmi.available || !hostGpuDuring.nvidiaSmiPmon.available)) {
      throw new Error('hardware GPU required, but nvidia-smi/pmon evidence was unavailable');
    }
    if (written.size !== manifest.poses.length) throw new Error(`only ${written.size}/${manifest.poses.length} poses were written`);
    fs.closeSync(poseIndexFd); fs.closeSync(sourcePoseIndexFd); fs.closeSync(renderPoseIdFd); fs.closeSync(idsFd); fs.closeSync(depthFd);
    const poseIndexFinal = args.output + '.pose_indices.bin';
    const idsFinal = args.output + '.instance_ids.bin';
    const depthFinal = args.output + '.linear_depth.bin';
    const sourcePoseIndexFinal = args.output + '.source_pose_indices.bin';
    const renderPoseIdFinal = args.output + '.render_pose_ids.bin';
    fs.renameSync(poseIndexPath, poseIndexFinal); fs.renameSync(sourcePoseIndexPath, sourcePoseIndexFinal); fs.renameSync(renderPoseIdPath, renderPoseIdFinal); fs.renameSync(idsPath, idsFinal); fs.renameSync(depthPath, depthFinal);
    const meta = { schema: CACHE_SCHEMA, modelInputFovYDeg: MODEL_FOV, width: args.width, height: args.height, maxLayers: args.maxLayers, poseCount: manifest.poses.length, resumedPrefixPoseCount: resumeStart, assetsDir: manifest.assetsDir, sourceManifest: args.manifest, candidateIdentity: manifest.candidateIdentity || null, poseIdSemantics: 'poseIndices stores renderPoseId; sourcePoseIndices stores canonical PoseCSR sourcePoseIndex; renderPoseIds repeats the explicit render IDs for audit', linearDepthEncoding: LINEAR_DEPTH_ENCODING, cameraFarMeters: Number(manifest.cameraFar), decodedDepthContract: 'dataset builders reconstruct per-pixel camera-ray range in meters before relation or survival supervision', firstLayerReference: result.firstLayerReference || null, gpuBackend: result.gpuBackend, gpuGate, files: { poseIndices: path.basename(poseIndexFinal), renderPoseIds: path.basename(renderPoseIdFinal), sourcePoseIndices: path.basename(sourcePoseIndexFinal), instanceIds: path.basename(idsFinal), linearDepth: path.basename(depthFinal) } };
    fs.writeFileSync(path.join(outputDir, 'layer_cache_meta.json'), JSON.stringify(meta, null, 2));
    writeGpuEvidence(args.output, {
      schema: 'triangle-depth-layer-gpu-evidence-v1',
      formalReady: Boolean(args.requireHardwareGpu && gpuGate.hardware && hostGpuDuring.nvidiaSmi.available && hostGpuDuring.nvidiaSmiPmon.available),
      output: args.output,
      manifest: args.manifest,
      width: args.width,
      height: args.height,
      maxLayers: args.maxLayers,
      modelInputFovYDeg: MODEL_FOV,
      resumedPrefixPoseCount: resumeStart,
      chrome: { executablePath: chrome, args: actualChromeArgs, stdout: chromeStdoutPath, stderr: chromeStderrPath },
      gpuBackend: result.gpuBackend || null,
      gpuGate,
      firstLayerReference: result.firstLayerReference || null,
      hostGpuBefore,
      hostGpuDuring,
      capturedAt: new Date().toISOString(),
    });
    console.log(`triangle depth layer cache written: ${outputDir}`);
  } catch (error) {
    const hostGpuDuring = captureHostGpuEvidence();
    writeGpuEvidence(args.output, {
      schema: 'triangle-depth-layer-gpu-evidence-v1',
      formalReady: false,
      output: args.output,
      manifest: args.manifest,
      width: args.width,
      height: args.height,
      maxLayers: args.maxLayers,
      modelInputFovYDeg: MODEL_FOV,
      chrome: { executablePath: chrome, args: actualChromeArgs, stdout: chromeStdoutPath, stderr: chromeStderrPath },
      gpuBackend: result?.gpuBackend || null,
      gpuGate: result?.gpuGate || null,
      hostGpuBefore,
      hostGpuDuring,
      error: String(error?.stack || error),
      capturedAt: new Date().toISOString(),
    });
    throw error;
  } finally {
    clearTimeout(timer);
    if (chromeProcess && !chromeProcess.killed) chromeProcess.kill('SIGTERM');
    if (userDataDir) {
      try { fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }); } catch {}
    }
    server.close();
    chromeStdout.end(); chromeStderr.end();
    for (const fd of [poseIndexFd, sourcePoseIndexFd, renderPoseIdFd, idsFd, depthFd]) { try { fs.closeSync(fd); } catch {} }
  }
}

main().catch((error) => { console.error(error?.stack || error); process.exitCode = 1; });
