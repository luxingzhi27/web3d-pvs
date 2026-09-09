#!/usr/bin/env node
/*
 * Render and validate the component-ID manifest used by image evaluation.
 *
 * The browser path binds component IDs and visibility masks to each loaded
 * InstancedMesh.  --validate-only is intentionally dependency and Chrome
 * independent; --synthetic-render-smoke exercises the shader without GLB
 * loading.  The manifest still remains non-formal until the full benchmark is
 * reviewed on the real scene split.
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
const INSTANCE_RENDER_MANIFEST_SCHEMA = 'local-true-component-id-render-manifest-v3';
const FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA = 'local-true-component-id-formal-render-manifest-v2';
const INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA = 'local-true-component-id-render-batch-manifest-v2';
const INSTANCE_BINDING_SCHEMA = 'component-instance-binding-preflight-v1';
const INSTANCE_ID_ENCODING = 'componentGlobalId + 1, RGB24, 0 background';
const PREDICTION_COMPONENT_IDS_BY_KEY_FIELD = 'predictionComponentIdsByKey';
const PREDICTION_KEY_FIELD = 'predictionKey';
const RENDER_FOV_Y_DEG = 60;
const MODEL_INPUT_FOV_Y_DEG = 66;

function parseArgs(argv) {
  const args = {
    manifest: null,
    outputDir: null,
    chromeExe: process.env.CHROME_EXE || null,
    port: 0,
    timeoutMs: 30 * 60 * 1000,
    validateOnly: false,
    syntheticRenderSmoke: false,
    chromeArgs: [],
    browserMode: process.env.PVS_BROWSER_MODE || 'headless',
    // Formal image rendering must fail closed when Chrome selects SwiftShader
    // or another software backend. Use --allow-software-gpu only for explicit
    // semantic debugging.
    requireHardwareGpu: true,
    display: process.env.DISPLAY || '',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    if (key === '--validate-only') {
      args.validateOnly = true;
      continue;
    }
    if (key === '--synthetic-render-smoke') {
      args.syntheticRenderSmoke = true;
      continue;
    }
    if (key === '--headed') {
      args.browserMode = 'headed';
      continue;
    }
    if (key === '--headless') {
      args.browserMode = 'headless';
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
    const value = argv[i + 1];
    i += 1;
    if (key === '--manifest') args.manifest = path.resolve(value);
    else if (key === '--output-dir') args.outputDir = path.resolve(value);
    else if (key === '--chrome-exe') args.chromeExe = path.resolve(value);
    else if (key === '--port') args.port = Number(value);
    else if (key === '--timeout-ms') args.timeoutMs = Number(value);
    else if (key === '--browser-mode') args.browserMode = String(value);
    else if (key === '--display') args.display = String(value);
    else if (key === '--chrome-arg') args.chromeArgs.push(String(value));
  }
  if (!args.manifest) throw new Error('--manifest is required');
  if (!args.outputDir) throw new Error('--output-dir is required');
  if (!['headless', 'headed'].includes(args.browserMode)) {
    throw new Error(`--browser-mode must be headless or headed, got ${args.browserMode}`);
  }
  return args;
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

function validateInstanceRenderManifest(manifest) {
  if (!manifest || manifest.schema !== INSTANCE_RENDER_MANIFEST_SCHEMA) {
    throw new Error(
      `refusing non-instance render manifest; expected ${INSTANCE_RENDER_MANIFEST_SCHEMA}`,
    );
  }
  if (manifest.idEncoding !== INSTANCE_ID_ENCODING) {
    throw new Error('manifest must encode componentGlobalId, not globalGlbId');
  }
  if (Number(manifest.renderFovYDeg) !== RENDER_FOV_Y_DEG) {
    throw new Error(`Render FOV must be exactly ${RENDER_FOV_Y_DEG} degrees`);
  }
  if (manifest.formalImageEvaluationReady !== false) {
    throw new Error('component browser renderer is not marked formal-ready');
  }
  if (Number(manifest.modelInputFovYDeg) !== MODEL_INPUT_FOV_Y_DEG) {
    throw new Error(`Model-input FOV must be exactly ${MODEL_INPUT_FOV_Y_DEG} degrees`);
  }
  const bindings = manifest.instanceBindings;
  if (!bindings || bindings.schema !== INSTANCE_BINDING_SCHEMA) {
    throw new Error(`manifest.instanceBindings must use ${INSTANCE_BINDING_SCHEMA}`);
  }
  const componentToBinding = bindings.componentToBinding || {};
  if (!Array.isArray(manifest.selectedGlbs) || manifest.selectedGlbs.some((value) => !Number.isInteger(Number(value)) || Number(value) < 0)) {
    throw new Error('manifest.selectedGlbs must be a list of non-negative integers');
  }
  const selected = new Set(manifest.selectedGlbs.map((value) => Number(value)));
  const byGlb = bindings.byGlobalGlbId || {};
  const bindingGlbs = new Set(Object.keys(byGlb).map((value) => Number(value)));
  if (selected.size !== bindingGlbs.size || [...bindingGlbs].some((value) => !selected.has(value))) {
    throw new Error('selectedGlbs must be the complete GLB inventory, not a predicted subset');
  }
  for (const [componentId, binding] of Object.entries(componentToBinding)) {
    const numericComponentId = Number(componentId);
    const globalGlbId = Number(binding.globalGlbId);
    const instanceIndex = Number(binding.instanceIndex);
    if (!Number.isInteger(numericComponentId) || numericComponentId < 0 ||
        !Number.isInteger(globalGlbId) || globalGlbId < 0 ||
        !Number.isInteger(instanceIndex) || instanceIndex < 0) {
      throw new Error(`invalid component binding for ${componentId}`);
    }
    if (!byGlb[String(globalGlbId)]) {
      throw new Error(`component ${componentId} references missing GLB ${globalGlbId}`);
    }
    if (!selected.has(globalGlbId)) {
      throw new Error(`component ${componentId} references an unselected GLB ${globalGlbId}`);
    }
    const glbBinding = byGlb[String(globalGlbId)];
    const componentIds = glbBinding.componentGlobalIds || [];
    if (instanceIndex >= componentIds.length || Number(componentIds[instanceIndex]) !== numericComponentId) {
      throw new Error(`component ${componentId} does not match GLB ${globalGlbId} instance slot ${instanceIndex}`);
    }
  }
  if (!manifest.reference || manifest.reference.mode !== 'full_scene_renderable_instances' ||
      manifest.reference.idSource !== 'componentGlobalId') {
    throw new Error('manifest reference must be a full component-ID scene render');
  }
  if (!manifest.prediction ||
      manifest.prediction.field !== PREDICTION_COMPONENT_IDS_BY_KEY_FIELD ||
      manifest.prediction.keyField !== PREDICTION_KEY_FIELD) {
    throw new Error('manifest prediction must use predictionComponentIdsByKey and predictionKey');
  }
  const predictionComponentIdsByKey = manifest[PREDICTION_COMPONENT_IDS_BY_KEY_FIELD];
  if (!predictionComponentIdsByKey || typeof predictionComponentIdsByKey !== 'object' ||
      Array.isArray(predictionComponentIdsByKey)) {
    throw new Error(`manifest.${PREDICTION_COMPONENT_IDS_BY_KEY_FIELD} is missing`);
  }
  for (const [predictionKey, componentIds] of Object.entries(predictionComponentIdsByKey)) {
    if (!predictionKey) throw new Error('manifest prediction table contains an empty key');
    if (!Array.isArray(componentIds)) {
      throw new Error(`prediction key ${predictionKey} must map to a component ID list`);
    }
    for (const componentId of componentIds) {
      if (!Number.isInteger(Number(componentId)) || Number(componentId) < 0 ||
          !componentToBinding[String(Number(componentId))]) {
        throw new Error(`prediction key ${predictionKey} contains unknown componentGlobalId ${componentId}`);
      }
    }
  }
  if (manifest.syntheticComponentIdSmoke) {
    const smoke = manifest.syntheticComponentIdSmoke;
    for (const field of ['referenceComponentIds', 'predictionComponentIds']) {
      if (!Array.isArray(smoke[field])) throw new Error(`synthetic smoke is missing ${field}`);
      for (const componentId of smoke[field]) {
        if (!Number.isInteger(Number(componentId)) || Number(componentId) < 0 ||
            !componentToBinding[String(Number(componentId))]) {
          throw new Error(`synthetic smoke contains unknown componentGlobalId ${componentId}`);
        }
      }
    }
  }
  const samples = manifest.samples || [];
  const predictionKeyByViewcell = new Map();
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    if (Object.prototype.hasOwnProperty.call(sample, 'referenceGlbs') ||
        Object.prototype.hasOwnProperty.call(sample, 'testGlbs') ||
        Object.prototype.hasOwnProperty.call(sample, 'predictionGlbIds')) {
      throw new Error(`sample ${index} still contains GLB-level image IDs`);
    }
    if (Object.prototype.hasOwnProperty.call(sample, 'predictionComponentIds')) {
      throw new Error(`sample ${index} stores predictionComponentIds; use predictionKey`);
    }
    const predictionKey = sample[PREDICTION_KEY_FIELD];
    if (typeof predictionKey !== 'string' || !predictionKey) {
      throw new Error(`sample ${index} is missing ${PREDICTION_KEY_FIELD}`);
    }
    if (!Object.prototype.hasOwnProperty.call(predictionComponentIdsByKey, predictionKey)) {
      throw new Error(`sample ${index} references unknown predictionKey ${predictionKey}`);
    }
    if (Object.prototype.hasOwnProperty.call(sample, 'viewcellRow')) {
      const viewcellRow = sample.viewcellRow;
      if (!Number.isInteger(Number(viewcellRow)) || Number(viewcellRow) < 0) {
        throw new Error(`sample ${index} has an invalid viewcellRow: ${viewcellRow}`);
      }
      const previousKey = predictionKeyByViewcell.get(Number(viewcellRow));
      if (previousKey !== undefined && previousKey !== predictionKey) {
        throw new Error(`view-cell row ${viewcellRow} uses multiple prediction keys`);
      }
      predictionKeyByViewcell.set(Number(viewcellRow), predictionKey);
    }
    if (Number(sample.renderFovYDeg) !== RENDER_FOV_Y_DEG) {
      throw new Error(`sample ${index} does not use the 60 degree render camera`);
    }
    if (Number(sample.modelInputFovYDeg) !== MODEL_INPUT_FOV_Y_DEG) {
      throw new Error(`sample ${index} does not use the 66 degree model-input camera`);
    }
  }
  return {
    schema: 'component-id-render-schema-validation-v1',
    manifestSchema: manifest.schema,
    idEncoding: manifest.idEncoding,
    renderFovYDeg: RENDER_FOV_Y_DEG,
    selectedGlbCount: selected.size,
    componentBindingCount: Object.keys(componentToBinding).length,
    sampleCount: samples.length,
    predictionKeyCount: Object.keys(predictionComponentIdsByKey).length,
    predictionKeyReuseCount: samples.length - new Set(samples.map((sample) => sample[PREDICTION_KEY_FIELD])).size,
    formalImageEvaluationReady: false,
    browserInstanceReorderImplemented: false,
  };
}

function validateFormalInstanceRenderManifest(manifest) {
  if (!manifest || manifest.schema !== FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA) {
    throw new Error(`refusing non-formal image manifest; expected ${FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA}`);
  }
  if (manifest.formalImageEvaluationReady !== true) {
    throw new Error('formal image manifest must set formalImageEvaluationReady=true');
  }
  if (manifest.syntheticComponentIdSmoke) throw new Error('synthetic component-ID smoke cannot be formal');
  const common = { ...manifest, schema: INSTANCE_RENDER_MANIFEST_SCHEMA, formalImageEvaluationReady: false };
  const base = validateInstanceRenderManifest(common);
  const requirements = manifest.formalRequirements || {};
  if (requirements.requiresHardwareWebGL !== true || requirements.syntheticSmokeAllowed !== false || requirements.completeGlbInventory !== true) {
    throw new Error('formal image requirements must require hardware WebGL, disallow synthetic smoke, and retain complete inventory');
  }
  if (manifest.reference?.geometrySource !== 'original_local_glb_meshes' || manifest.reference?.completeInventory !== true) {
    throw new Error('formal image reference must use the complete original local GLB mesh inventory');
  }
  if (manifest.prediction?.postFilter !== 'component_visibility_mask_after_conservative_render_submission') {
    throw new Error('formal image prediction must use an instance-level visibility mask');
  }
  if (manifest.spatialCulling?.completeInventoryRetained !== true || manifest.spatialCulling?.componentLevelMask !== true) {
    throw new Error('formal spatial submission must retain complete inventory and component-level masks');
  }
  if (!Array.isArray(manifest.samples) || manifest.samples.length === 0) {
    throw new Error('formal image evaluation requires real samples');
  }
  for (let index = 0; index < manifest.samples.length; index += 1) {
    const sample = manifest.samples[index];
    if (sample.referenceMode !== 'full_scene_renderable_instances') {
      throw new Error(`formal sample ${index} is not a full-scene reference`);
    }
  }
  return { ...base, schema: 'formal-component-id-render-schema-validation-v1', formalImageEvaluationReady: true };
}

function validateInstanceRenderBatchManifest(manifest) {
  if (!manifest || manifest.schema !== INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA) {
    throw new Error(
      `refusing non-batch instance render manifest; expected ${INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA}`,
    );
  }
  if (!Array.isArray(manifest.batches) || manifest.batches.length === 0) {
    throw new Error('batch manifest must contain a non-empty batches list');
  }
  const batchIds = new Set();
  const sampleIds = new Set();
  const flattened = [];
  for (let batchIndex = 0; batchIndex < manifest.batches.length; batchIndex += 1) {
    const batch = manifest.batches[batchIndex];
    const batchId = String(batch && batch.batchId ? batch.batchId : '');
    if (!batchId) throw new Error(`batch ${batchIndex} is missing batchId`);
    if (batchIds.has(batchId)) throw new Error(`duplicate batchId: ${batchId}`);
    batchIds.add(batchId);
    if (!Array.isArray(batch.samples) || batch.samples.length === 0) {
      throw new Error(`batch ${batchId} must contain a non-empty samples list`);
    }
    for (let sampleIndex = 0; sampleIndex < batch.samples.length; sampleIndex += 1) {
      const sample = batch.samples[sampleIndex];
      const sampleId = String(sample && sample.sampleId ? sample.sampleId : '');
      if (!sampleId) throw new Error(`batch ${batchId} sample ${sampleIndex} is missing sampleId`);
      if (sampleIds.has(sampleId)) throw new Error(`duplicate sampleId across batches: ${sampleId}`);
      sampleIds.add(sampleId);
      flattened.push(sample);
    }
  }
  const common = { ...manifest, schema: INSTANCE_RENDER_MANIFEST_SCHEMA, samples: flattened };
  const validation = validateInstanceRenderManifest(common);
  return {
    ...validation,
    schema: 'component-id-render-batch-schema-validation-v1',
    manifestSchema: manifest.schema,
    batchCount: manifest.batches.length,
    sampleCount: flattened.length,
  };
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
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error('Chrome/Edge executable not found. Pass --chrome-exe or set CHROME_EXE.');
}

function assertRendererDependencies() {
  const required = [
    'three/build/three.module.js',
    'three/examples/jsm/loaders/GLTFLoader.js',
    'three/examples/jsm/loaders/DRACOLoader.js',
    'three/examples/jsm/libs/meshopt_decoder.module.js',
    'three/examples/jsm/libs/draco/gltf/draco_decoder.wasm',
  ];
  const missing = required
    .map((rel) => path.join(SLM2_ROOT, 'node_modules', rel))
    .filter((file) => !fs.existsSync(file));
  if (missing.length > 0) {
    throw new Error(
      'Missing true GLB renderer dependencies under slm2viewer/node_modules. ' +
      'Run `npm ci` in slm2viewer before using --image-renderer true_glb. Missing: ' +
      missing.map((file) => path.relative(REPO_ROOT, file)).join(', ')
    );
  }
}

function contentType(file) {
  const ext = path.extname(file).toLowerCase();
  if (ext === '.js' || ext === '.mjs') return 'text/javascript; charset=utf-8';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.wasm') return 'application/wasm';
  if (ext === '.glb') return 'model/gltf-binary';
  if (ext === '.png') return 'image/png';
  if (ext === '.html') return 'text/html; charset=utf-8';
  return 'application/octet-stream';
}

function safeJoin(root, rel) {
  const resolved = path.resolve(root, rel.replace(/^[/\\]+/, ''));
  const rootResolved = path.resolve(root);
  if (resolved !== rootResolved && !resolved.startsWith(rootResolved + path.sep)) {
    throw new Error(`Path escapes root: ${rel}`);
  }
  return resolved;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function writeJson(res, value, status = 200) {
  const text = JSON.stringify(value);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(text);
}

function rendererHtml() {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Local GLB Color-ID Renderer</title>
  <style>html,body{margin:0;background:#111;color:#ddd;font:12px Consolas,monospace}#status{padding:8px;white-space:pre-wrap}</style>
</head>
<body>
  <div id="status">booting</div>
  <script type="importmap">
    {"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}
  </script>
  <script type="module" src="/renderer.js"></script>
</body>
</html>`;
}

function syntheticRendererJs() {
  return `
import * as THREE from 'three';

const FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA = 'local-true-component-id-formal-render-manifest-v2';
const statusEl = document.getElementById('status');
function status(message) {
  statusEl.textContent = message;
  console.log('[component-id-smoke]', message);
  fetch('/log', { method: 'POST', body: message }).catch(() => {});
}

function makeComponentIdMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {},
    vertexShader: [
      'attribute float componentId;',
      'varying float vComponentId;',
      'void main() {',
      '  vComponentId = componentId;',
      '  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);',
      '}',
    ].join('\\n'),
    fragmentShader: [
      'uniform float maskEnabled;',
      'varying float vComponentId;',
      'void main() {',
      '  float encoded = vComponentId + 1.0;',
      '  float red = mod(encoded, 256.0);',
      '  float green = mod(floor(encoded / 256.0), 256.0);',
      '  float blue = mod(floor(encoded / 65536.0), 256.0);',
      '  gl_FragColor = vec4(vec3(red, green, blue) / 255.0, 1.0);',
      '}',
    ].join('\\n'),
    depthTest: true,
    depthWrite: true,
    toneMapped: false,
  });
}

function makeInstancedComponentMesh(componentIds, positions) {
  if (componentIds.length !== positions.length) throw new Error('synthetic component/position count mismatch');
  const geometry = new THREE.BoxGeometry(1.2, 1.2, 1.2);
  geometry.setAttribute(
    'componentId',
    new THREE.InstancedBufferAttribute(new Float32Array(componentIds), 1),
  );
  const mesh = new THREE.InstancedMesh(geometry, makeComponentIdMaterial(), positions.length);
  const matrix = new THREE.Matrix4();
  for (let index = 0; index < positions.length; index += 1) {
    matrix.makeTranslation(positions[index][0], positions[index][1], positions[index][2]);
    mesh.setMatrixAt(index, matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
  return mesh;
}

function decodePixels(rgba) {
  const ids = new Uint32Array(rgba.length / 4);
  for (let index = 0, pixel = 0; index < rgba.length; index += 4, pixel += 1) {
    ids[pixel] = rgba[index + 3] === 0
      ? 0
      : (rgba[index] | (rgba[index + 1] << 8) | (rgba[index + 2] << 16));
  }
  return ids;
}

function compareIdBuffers(reference, prediction) {
  let validReferencePixels = 0;
  let errorPixels = 0;
  let missPixels = 0;
  let wrongInstancePixels = 0;
  let extraPixels = 0;
  const diff = new Uint8Array(reference.length);
  for (let index = 0; index < reference.length; index += 1) {
    const referenceId = reference[index];
    const predictionId = prediction[index];
    if (referenceId !== 0) {
      validReferencePixels += 1;
      if (referenceId !== predictionId) {
        errorPixels += 1;
        if (predictionId === 0) {
          missPixels += 1;
          diff[index] = 1;
        } else {
          wrongInstancePixels += 1;
          diff[index] = 2;
        }
      }
    } else if (predictionId !== 0) {
      extraPixels += 1;
      diff[index] = 3;
    }
  }
  return {
    metrics: {
      totalPixels: reference.length,
      validReferencePixels,
      errorPixels,
      missPixels,
      wrongInstancePixels,
      extraPixels,
      PER: errorPixels / Math.max(1, validReferencePixels),
      missPixelRate: missPixels / Math.max(1, validReferencePixels),
      wrongInstancePixelRate: wrongInstancePixels / Math.max(1, validReferencePixels),
      extraPixelRateOverImage: extraPixels / Math.max(1, reference.length),
    },
    diff,
  };
}

async function postBuffer(name, typedArray) {
  await fetch('/buffer?name=' + encodeURIComponent(name), { method: 'POST', body: typedArray.buffer });
}

async function main() {
  const manifest = await (await fetch('/manifest', { cache: 'no-store' })).json();
  const formalManifest = manifest.schema === FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA;
  const smoke = manifest.syntheticComponentIdSmoke;
  if (!smoke) throw new Error('manifest.syntheticComponentIdSmoke is required');
  const width = 160;
  const height = 90;
  const referencePositions = [[-1.25, 0, 0], [1.25, 0, 0]];
  const predictionPositions = [[-1.25, 0, 0]];
  if (smoke.referenceComponentIds.length !== 2 || smoke.predictionComponentIds.length !== 1) {
    throw new Error('synthetic smoke expects two reference and one prediction component');
  }
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.NoToneMapping;
  document.body.appendChild(renderer.domElement);
  const gl = renderer.getContext();
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuBackend = {
    api: 'WebGL',
    vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
  };
  status('WebGL backend vendor="' + gpuBackend.vendor + '" renderer="' + gpuBackend.renderer + '"');
  const target = new THREE.WebGLRenderTarget(width, height, { depthBuffer: true, stencilBuffer: false });
  if (target.texture && 'colorSpace' in target.texture && THREE.NoColorSpace !== undefined) target.texture.colorSpace = THREE.NoColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100.0);
  camera.position.set(0, 0, 7);
  camera.lookAt(0, 0, 0);
  const referenceMesh = makeInstancedComponentMesh(smoke.referenceComponentIds, referencePositions);
  const predictionMesh = makeInstancedComponentMesh(smoke.predictionComponentIds, predictionPositions);
  scene.add(referenceMesh);
  scene.add(predictionMesh);
  const pixels = new Uint8Array(width * height * 4);
  const renderMesh = async (activeMesh) => {
    referenceMesh.visible = activeMesh === referenceMesh;
    predictionMesh.visible = activeMesh === predictionMesh;
    renderer.setRenderTarget(target);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
    return decodePixels(pixels);
  };
  status('rendering component-ID reference');
  const reference = await renderMesh(referenceMesh);
  status('rendering component-ID prediction');
  const prediction = await renderMesh(predictionMesh);
  const { metrics, diff } = compareIdBuffers(reference, prediction);
  const referenceValues = new Set(reference);
  const predictionValues = new Set(prediction);
  if (!referenceValues.has(Number(smoke.referenceComponentIds[0]) + 1) ||
      !referenceValues.has(Number(smoke.referenceComponentIds[1]) + 1)) {
    throw new Error('reference ID buffer did not contain both component IDs');
  }
  if (predictionValues.has(Number(smoke.referenceComponentIds[1]) + 1) || metrics.missPixels <= 0) {
    throw new Error('prediction ID buffer did not produce the expected component miss');
  }
  await postBuffer('synthetic_reference_component_id_u32.bin', reference);
  await postBuffer('synthetic_prediction_component_id_u32.bin', prediction);
  await postBuffer('synthetic_diff_mask_u8.bin', diff);
  await fetch('/done', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      schema: 'component-id-browser-render-smoke-v1',
      status: 'rendered',
      formalImageEvaluationReady: false,
      browserInstanceReorderImplemented: false,
      synthetic: true,
      gpuBackend,
      renderFovYDeg: 60,
      width,
      height,
      referenceComponentIds: smoke.referenceComponentIds,
      predictionComponentIds: smoke.predictionComponentIds,
      imageMetrics: metrics,
      maskSemantics: { miss: 1, wrong: 2, extra: 3 },
    }),
  });
}

main().catch(async (error) => {
  await fetch('/done', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ schema: 'component-id-browser-render-smoke-v1', status: 'failed', formalImageEvaluationReady: false, error: String(error && error.stack ? error.stack : error) }),
  }).catch(() => {});
});
`;
}

function rendererJs() {
  return `
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const RENDER_FOV_Y_DEG = 60;
const FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA = 'local-true-component-id-formal-render-manifest-v2';
const PREDICTION_COMPONENT_IDS_BY_KEY_FIELD = 'predictionComponentIdsByKey';
const PREDICTION_KEY_FIELD = 'predictionKey';
const statusEl = document.getElementById('status');
function status(message) {
  statusEl.textContent = message;
  console.log('[true-glb-render]', message);
  fetch('/log', { method: 'POST', body: message }).catch(() => {});
}

function makeInstancedIdMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { maskEnabled: { value: 0.0 } },
    vertexShader: [
      'attribute float componentId;',
      'attribute float componentVisible;',
      'attribute float componentSpatialVisible;',
      'varying float vComponentId;',
      'varying float vComponentVisible;',
      'varying float vComponentSpatialVisible;',
      'void main() {',
      '  vComponentId = componentId;',
      '  vComponentVisible = componentVisible;',
      '  vComponentSpatialVisible = componentSpatialVisible;',
      '  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);',
      '}',
    ].join('\\n'),
    fragmentShader: [
      'uniform float maskEnabled;',
      'varying float vComponentId;',
      'varying float vComponentVisible;',
      'varying float vComponentSpatialVisible;',
      'void main() {',
      '  if (vComponentSpatialVisible < 0.5) discard;',
      '  if (maskEnabled > 0.5 && vComponentVisible < 0.5) discard;',
      '  float encoded = vComponentId + 1.0;',
      '  float red = mod(encoded, 256.0);',
      '  float green = mod(floor(encoded / 256.0), 256.0);',
      '  float blue = mod(floor(encoded / 65536.0), 256.0);',
      '  gl_FragColor = vec4(vec3(red, green, blue) / 255.0, 1.0);',
      '}',
    ].join('\\n'),
    depthTest: true,
    depthWrite: true,
    toneMapped: false,
  });
}

function makeStaticIdMaterial(componentId) {
  return new THREE.ShaderMaterial({
    uniforms: { componentId: { value: Number(componentId) }, componentVisible: { value: 0.0 }, maskEnabled: { value: 0.0 } },
    vertexShader: [
      'uniform float componentId;',
      'uniform float componentVisible;',
      'varying float vComponentId;',
      'varying float vComponentVisible;',
      'void main() {',
      '  vComponentId = componentId;',
      '  vComponentVisible = componentVisible;',
      '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
      '}',
    ].join('\\n'),
    fragmentShader: [
      'uniform float maskEnabled;',
      'varying float vComponentId;',
      'varying float vComponentVisible;',
      'void main() {',
      '  if (maskEnabled > 0.5 && vComponentVisible < 0.5) discard;',
      '  float encoded = vComponentId + 1.0;',
      '  float red = mod(encoded, 256.0);',
      '  float green = mod(floor(encoded / 256.0), 256.0);',
      '  float blue = mod(floor(encoded / 65536.0), 256.0);',
      '  gl_FragColor = vec4(vec3(red, green, blue) / 255.0, 1.0);',
      '}',
    ].join('\\n'),
    depthTest: true,
    depthWrite: true,
    toneMapped: false,
  });
}

function decodePixels(rgba) {
  const out = new Uint32Array(rgba.length / 4);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 1) {
    out[j] = rgba[i + 3] === 0 ? 0 : (rgba[i] | (rgba[i + 1] << 8) | (rgba[i + 2] << 16));
  }
  return out;
}

function computeMetrics(reference, test) {
  let valid = 0, background = 0, error = 0, miss = 0, wrong = 0, extra = 0;
  const contributors = new Map();
  for (let i = 0; i < reference.length; i += 1) {
    const r = reference[i];
    const t = test[i];
    if (r !== 0) {
      valid += 1;
      if (t !== r) {
        error += 1;
        contributors.set(r - 1, (contributors.get(r - 1) || 0) + 1);
        if (t === 0) miss += 1;
        else wrong += 1;
      }
    } else {
      background += 1;
      if (t !== 0) extra += 1;
    }
  }
  const total = reference.length;
  return {
    metrics: {
      totalPixels: total,
      validReferencePixels: valid,
      backgroundReferencePixels: background,
      errorPixels: error,
      missPixels: miss,
      wrongInstancePixels: wrong,
      extraPixels: extra,
      PER: error / Math.max(1, valid),
      missPixelRate: miss / Math.max(1, valid),
      wrongInstancePixelRate: wrong / Math.max(1, valid),
      extraPixelRateOverImage: extra / Math.max(1, total),
    },
    contributors,
  };
}

function diffMask(reference, test) {
  const out = new Uint8Array(reference.length);
  for (let i = 0; i < reference.length; i += 1) {
    const r = reference[i], t = test[i];
    if (r !== 0 && t === 0) out[i] = 1;
    else if (r !== 0 && t !== 0 && r !== t) out[i] = 2;
    else if (r === 0 && t !== 0) out[i] = 3;
  }
  return out;
}

function hashRgb(ids) {
  const out = new Uint8ClampedArray(ids.length * 4);
  for (let i = 0; i < ids.length; i += 1) {
    const id = ids[i];
    out[i * 4 + 0] = id === 0 ? 0 : ((id * 37 + 17) & 255);
    out[i * 4 + 1] = id === 0 ? 0 : ((id * 67 + 29) & 255);
    out[i * 4 + 2] = id === 0 ? 0 : ((id * 97 + 53) & 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

function diffRgb(diff) {
  const out = new Uint8ClampedArray(diff.length * 4);
  for (let i = 0; i < diff.length; i += 1) {
    const v = diff[i];
    if (v === 1) { out[i*4] = 220; out[i*4+1] = 40; out[i*4+2] = 40; }
    else if (v === 2) { out[i*4] = 240; out[i*4+1] = 180; out[i*4+2] = 30; }
    else if (v === 3) { out[i*4] = 40; out[i*4+1] = 110; out[i*4+2] = 230; }
    out[i * 4 + 3] = 255;
  }
  return out;
}

async function postBuffer(url, typedArray) {
  await fetch(url, { method: 'POST', body: typedArray.buffer });
}

async function postPreview(name, reference, test, diff, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width * 3;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(hashRgb(reference), width, height), 0, 0);
  ctx.putImageData(new ImageData(hashRgb(test), width, height), width, 0);
  ctx.putImageData(new ImageData(diffRgb(diff), width, height), width * 2, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  await fetch('/preview?name=' + encodeURIComponent(name), { method: 'POST', body: blob });
}

function setCamera(camera, sample) {
  if (Number(sample.renderFovYDeg) !== RENDER_FOV_Y_DEG) {
    throw new Error('render camera FOV must be exactly 60 degrees');
  }
  camera.fov = RENDER_FOV_Y_DEG;
  camera.aspect = Number(sample.aspect || 16 / 9);
  camera.near = 0.05;
  camera.far = 20000;
  camera.updateProjectionMatrix();
  const p = sample.cameraPosition;
  const f = sample.cameraForward;
  camera.position.set(p[0], p[1], p[2]);
  camera.up.set(0, 1, 0);
  camera.lookAt(p[0] + f[0], p[1] + f[1], p[2] + f[2]);
}

function sampleCameraKey(sample) {
  return JSON.stringify([
    ...(sample.cameraPosition || []).map(Number),
    ...(sample.cameraForward || []).map(Number),
    Number(sample.aspect || 16 / 9),
    Number(sample.renderFovYDeg),
  ]);
}

function makeRenderFrustum(camera) {
  camera.updateMatrixWorld(true);
  const viewProjection = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  return new THREE.Frustum().setFromProjectionMatrix(viewProjection);
}

function parseSpatialAabb(value) {
  if (!value || !Array.isArray(value.min) || !Array.isArray(value.max) ||
      value.min.length !== 3 || value.max.length !== 3) {
    return null;
  }
  const minimum = value.min.map(Number);
  const maximum = value.max.map(Number);
  if ([...minimum, ...maximum].some((coordinate) => !Number.isFinite(coordinate))) {
    return null;
  }
  if (minimum.some((coordinate, index) => coordinate > maximum[index])) {
    return null;
  }
  return new THREE.Box3(
    new THREE.Vector3(minimum[0], minimum[1], minimum[2]),
    new THREE.Vector3(maximum[0], maximum[1], maximum[2]),
  );
}

function spatialGlbIds(manifest, selected, frustum) {
  const bounds = manifest.glbAabbs;
  const componentBounds = manifest.componentAabbs;
  if (!manifest.spatialCulling || (!bounds && !componentBounds)) {
    return { glbIds: selected.slice(), componentIds: null };
  }
  const result = [];
  const visibleComponents = new Set();
  const byGlb = (manifest.instanceBindings && manifest.instanceBindings.byGlobalGlbId) || {};
  for (const rawGid of selected) {
    const gid = Number(rawGid);
    const binding = byGlb[String(gid)] || {};
    const componentIds = (binding.componentGlobalIds || []).map(Number);
    // Per-instance spatial filtering is valid only when every bound in the
    // GLB has been audited. A partial table is not evidence that the missing
    // instances are outside the frustum: fail open for the whole GLB so the
    // reference image remains a complete-scene render.
    const componentBoxes = componentIds.map((componentId) => parseSpatialAabb(
      componentBounds && componentBounds[String(componentId)],
    ));
    const completeComponentBounds = componentIds.length > 0 && componentBoxes.every(Boolean);
    if (completeComponentBounds) {
      let hasVisibleComponent = false;
      for (let index = 0; index < componentIds.length; index += 1) {
        if (frustum.intersectsBox(componentBoxes[index])) {
          visibleComponents.add(componentIds[index]);
          hasVisibleComponent = true;
        }
      }
      if (hasVisibleComponent) result.push(gid);
      continue;
    }

    const aabb = bounds[String(gid)];
    // Missing or partial instance bounds must fail open. Even a valid GLB
    // bound is not used here because it cannot certify the unknown instances.
    if (componentIds.length > 0 && !completeComponentBounds) {
      result.push(gid);
      for (const componentId of componentIds) visibleComponents.add(componentId);
      continue;
    }
    const box = parseSpatialAabb(aabb);
    if (!box) {
      result.push(gid);
      for (const componentId of componentIds) visibleComponents.add(componentId);
      continue;
    }
    if (frustum.intersectsBox(box)) {
      result.push(gid);
      for (const componentId of componentIds) visibleComponents.add(componentId);
    }
  }
  return { glbIds: result, componentIds: visibleComponents };
}

function setSpatialVisibility(groups, visibleIds) {
  const visible = new Set(visibleIds.map(Number));
  for (const [gid, group] of groups.entries()) group.scene.visible = visible.has(Number(gid));
}

function predictionGlbIds(manifest, requiredGlbIds, predictionComponentIds, spatialComponentIds) {
  const prediction = new Set((predictionComponentIds || []).map(Number));
  const spatial = spatialComponentIds ? new Set(spatialComponentIds) : null;
  const byGlb = (manifest.instanceBindings && manifest.instanceBindings.byGlobalGlbId) || {};
  const result = [];
  for (const rawGid of requiredGlbIds) {
    const binding = byGlb[String(Number(rawGid))] || {};
    const hasPrediction = (binding.componentGlobalIds || []).some((rawComponentId) => {
      const componentId = Number(rawComponentId);
      return prediction.has(componentId) && (spatial === null || spatial.has(componentId));
    });
    if (hasPrediction) result.push(Number(rawGid));
  }
  return result;
}

function predictionComponentsForSample(manifest, sample) {
  const table = manifest[PREDICTION_COMPONENT_IDS_BY_KEY_FIELD] || {};
  const predictionKey = sample[PREDICTION_KEY_FIELD];
  if (!Object.prototype.hasOwnProperty.call(table, predictionKey)) {
    throw new Error('sample ' + (sample.sampleId || '<unknown>') +
      ' references unknown predictionKey ' + predictionKey);
  }
  return table[predictionKey];
}

function setSpatialInstanceVisibility(groups, visibleComponentIds) {
  const visible = visibleComponentIds ? new Set(visibleComponentIds) : null;
  const dirtyAttributes = new Set();
  for (const group of groups.values()) {
    for (const entry of group.instanced) {
      for (let index = 0; index < entry.componentIds.length; index += 1) {
        const componentId = entry.componentIds[index];
        const value = visible === null || visible.has(componentId) ? 1.0 : 0.0;
        if (entry.spatialVisibleAttribute.array[index] !== value) {
          entry.spatialVisibleAttribute.array[index] = value;
          dirtyAttributes.add(entry.spatialVisibleAttribute);
        }
      }
    }
  }
  for (const attribute of dirtyAttributes) attribute.needsUpdate = true;
}

function setVisibilityMaskEnabled(groups, enabled) {
  const value = enabled ? 1.0 : 0.0;
  for (const group of groups.values()) {
    for (const entry of group.instanced) {
      if (entry.material.uniforms.maskEnabled.value !== value) {
        entry.material.uniforms.maskEnabled.value = value;
      }
    }
    for (const entry of group.staticMeshes) {
      if (entry.material.uniforms.maskEnabled.value !== value) {
        entry.material.uniforms.maskEnabled.value = value;
      }
    }
  }
}

function setComponentVisibility(slot, value, dirtyAttributes) {
  if (slot.kind === 'instanced') {
    if (slot.entry.visibleAttribute.array[slot.instanceIndex] !== value) {
      slot.entry.visibleAttribute.array[slot.instanceIndex] = value;
      dirtyAttributes.add(slot.entry.visibleAttribute);
    }
  } else if (slot.entry.material.uniforms.componentVisible.value !== value) {
    slot.entry.material.uniforms.componentVisible.value = value;
  }
}

function setPredictionComponents(componentSlots, previousVisible, visibleIds) {
  const nextVisible = new Set((visibleIds || []).map((value) => Number(value)));
  const dirtyAttributes = new Set();
  for (const componentId of previousVisible) {
    if (!nextVisible.has(componentId)) {
      const slot = componentSlots.get(componentId);
      if (slot) setComponentVisibility(slot, 0.0, dirtyAttributes);
    }
  }
  for (const componentId of nextVisible) {
    if (!previousVisible.has(componentId)) {
      const slot = componentSlots.get(componentId);
      if (slot) setComponentVisibility(slot, 1.0, dirtyAttributes);
    }
  }
  for (const attribute of dirtyAttributes) attribute.needsUpdate = true;
  return nextVisible;
}

function renderIds(renderer, scene, camera, target, pixels, width, height) {
  renderer.setRenderTarget(target);
  renderer.clear(true, true, true);
  renderer.render(scene, camera);
  renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
  return decodePixels(pixels);
}

async function main() {
  const pageStarted = performance.now();
  const manifest = await (await fetch('/manifest', { cache: 'no-store' })).json();
  const formalManifest = manifest.schema === FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA;
  const glbIndex = await (await fetch('/glb-index', { cache: 'no-store' })).json();
  const entriesById = new Map((glbIndex.entries || []).map((entry) => [Number(entry.globalId), entry]));
  const width = Number(manifest.width);
  const height = Number(manifest.height);
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.NoToneMapping;
  document.body.appendChild(renderer.domElement);
  const gl = renderer.getContext();
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuBackend = {
    api: 'WebGL',
    vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
  };
  status('WebGL backend vendor="' + gpuBackend.vendor + '" renderer="' + gpuBackend.renderer + '"');
  const target = new THREE.WebGLRenderTarget(width, height, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: true,
    stencilBuffer: false,
  });
  if (target.texture && 'colorSpace' in target.texture && THREE.NoColorSpace !== undefined) {
    target.texture.colorSpace = THREE.NoColorSpace;
  }
  const pixels = new Uint8Array(width * height * 4);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.05, 20000);
  const dracoLoader = new DRACOLoader().setDecoderPath('/node_modules/three/examples/jsm/libs/draco/gltf/');
  const loader = new GLTFLoader().setDRACOLoader(dracoLoader).setMeshoptDecoder(MeshoptDecoder);
  const groups = new Map();
  const componentSlots = new Map();
  const selected = manifest.selectedGlbs || [];
  const spatialMode = Boolean(
    manifest.spatialCulling && (manifest.glbAabbs || manifest.componentAabbs),
  );
  const loadStarted = performance.now();
  const loadedIds = new Set();
  const loadingPromises = new Map();
  let loadCallCount = 0;
  async function loadGlb(gid) {
    const numericGid = Number(gid);
    if (groups.has(numericGid)) return groups.get(numericGid);
    if (loadingPromises.has(numericGid)) return loadingPromises.get(numericGid);
    const entry = entriesById.get(numericGid);
    if (!entry) throw new Error('glbIndex is missing selected GLB ' + numericGid);
    const binding = (manifest.instanceBindings.byGlobalGlbId || {})[String(numericGid)];
    if (!binding) throw new Error('manifest is missing instance binding for GLB ' + numericGid);
    const url = '/assets/' + entry.path.replace(/\\\\/g, '/');
    const promise = (async () => {
      try {
        const gltf = await loader.loadAsync(url);
        const group = { scene: gltf.scene, instanced: [], staticMeshes: [] };
        let meshCount = 0;
        gltf.scene.traverse((object) => {
          // Spatial visibility is applied at the GLB group boundary using
          // audited world-space AABBs. Keep per-object culling disabled so
          // the existing instancing transform contract is unchanged.
          object.frustumCulled = false;
          if (object.isInstancedMesh) {
            meshCount += 1;
            if (meshCount > 1) throw new Error('GLB contains more than one mesh node: ' + url);
            if (object.count !== binding.componentGlobalIds.length) {
              throw new Error('GLB instance count does not match manifest: ' + url);
            }
            const componentIds = binding.componentGlobalIds.map((value) => Number(value));
            object.geometry.setAttribute(
              'componentId',
              new THREE.InstancedBufferAttribute(new Float32Array(componentIds), 1),
            );
            const visibleAttribute = new THREE.InstancedBufferAttribute(
              new Float32Array(componentIds.length).fill(0),
              1,
            );
            visibleAttribute.setUsage(THREE.DynamicDrawUsage);
            object.geometry.setAttribute('componentVisible', visibleAttribute);
            const spatialVisibleAttribute = new THREE.InstancedBufferAttribute(
              new Float32Array(componentIds.length).fill(0),
              1,
            );
            spatialVisibleAttribute.setUsage(THREE.DynamicDrawUsage);
            object.geometry.setAttribute('componentSpatialVisible', spatialVisibleAttribute);
            const material = makeInstancedIdMaterial();
            object.material = material;
            const meshEntry = {
              object,
              componentIds,
              visibleAttribute,
              spatialVisibleAttribute,
              material,
            };
            group.instanced.push(meshEntry);
            for (let instanceIndex = 0; instanceIndex < componentIds.length; instanceIndex += 1) {
              componentSlots.set(componentIds[instanceIndex], {
                kind: 'instanced', entry: meshEntry, instanceIndex,
              });
            }
          } else if (object.isMesh) {
            meshCount += 1;
            if (meshCount > 1) throw new Error('GLB contains more than one mesh node: ' + url);
            if (binding.componentGlobalIds.length !== 1) {
              throw new Error('non-instanced GLB must bind exactly one component: ' + url);
            }
            const componentId = Number(binding.componentGlobalIds[0]);
            const material = makeStaticIdMaterial(componentId);
            object.material = material;
            const meshEntry = { object, componentId, material };
            group.staticMeshes.push(meshEntry);
            componentSlots.set(componentId, { kind: 'static', entry: meshEntry, instanceIndex: 0 });
          }
        });
        if (meshCount !== 1 && binding.renderable) throw new Error('renderable GLB has no supported mesh: ' + url);
        gltf.scene.visible = false;
        scene.add(gltf.scene);
        groups.set(numericGid, group);
        loadedIds.add(numericGid);
        loadCallCount += 1;
        if (loadCallCount % 25 === 0 || loadCallCount === selected.length) {
          status('loaded ' + loadCallCount + '/' + selected.length + ' GLBs (spatial lazy load)');
        }
        return group;
      } catch (error) {
        throw new Error('failed to load/bind ' + url + ': ' + String(error && error.message ? error.message : error));
      } finally {
        loadingPromises.delete(numericGid);
      }
    })();
    loadingPromises.set(numericGid, promise);
    return promise;
  }
  async function ensureGlbs(requiredIds) {
    const missing = requiredIds.filter((gid) => !groups.has(Number(gid)));
    // Loading in small concurrent waves bounds decoder memory while avoiding
    // one network/decode round trip per visible GLB.
    const waveSize = spatialMode ? 8 : 4;
    for (let offset = 0; offset < missing.length; offset += waveSize) {
      await Promise.all(missing.slice(offset, offset + waveSize).map((gid) => loadGlb(gid)));
    }
  }

  const batches = Array.isArray(manifest.batches)
    ? manifest.batches
    : [{ batchId: 'default', samples: manifest.samples || [] }];
  const samples = [];
  for (const batch of batches) {
    for (const sample of batch.samples || []) samples.push({ ...sample, batchId: String(batch.batchId || 'default') });
  }
  const sampleGroupsByCamera = new Map();
  for (const sample of samples) {
    const key = sampleCameraKey(sample);
    if (!sampleGroupsByCamera.has(key)) {
      sampleGroupsByCamera.set(key, { key, samples: [] });
    }
    sampleGroupsByCamera.get(key).samples.push(sample);
  }
  const sampleGroups = Array.from(sampleGroupsByCamera.values())
    .sort((left, right) => left.key.localeCompare(right.key));

  const totals = { totalPixels: 0, validReferencePixels: 0, backgroundReferencePixels: 0, errorPixels: 0, missPixels: 0, wrongInstancePixels: 0, extraPixels: 0 };
  const top = new Map();
  const imageRateSamples = [];
  let selfConsistencyPER = null;
  const sampleRows = [];
  const batchTotals = new Map(batches.map((batch) => [String(batch.batchId || 'default'), {
    totalPixels: 0, validReferencePixels: 0, backgroundReferencePixels: 0,
    errorPixels: 0, missPixels: 0, wrongInstancePixels: 0, extraPixels: 0,
  }]));
  const started = performance.now();
  let activePrediction = new Set();
  let sampleOrdinal = 0;
  const spatialStats = {
    enabled: spatialMode,
    sampleGroupCount: sampleGroups.length,
    referenceRenderCount: 0,
    referenceReuseCount: 0,
    requiredGlbCountSum: 0,
    requiredGlbCountMin: null,
    requiredGlbCountMax: 0,
    predictionGlbCountSum: 0,
    predictionGlbCountMin: null,
    predictionGlbCountMax: 0,
  };
  for (const sampleGroup of sampleGroups) {
    const representative = sampleGroup.samples[0];
    setCamera(camera, representative);
    const frustum = makeRenderFrustum(camera);
    const spatialSelection = spatialGlbIds(manifest, selected, frustum);
    const requiredGlbs = spatialSelection.glbIds;
    spatialStats.requiredGlbCountSum += requiredGlbs.length;
    spatialStats.requiredGlbCountMin = spatialStats.requiredGlbCountMin === null
      ? requiredGlbs.length
      : Math.min(spatialStats.requiredGlbCountMin, requiredGlbs.length);
    spatialStats.requiredGlbCountMax = Math.max(spatialStats.requiredGlbCountMax, requiredGlbs.length);
    await ensureGlbs(requiredGlbs);
    setSpatialVisibility(groups, requiredGlbs);
    setSpatialInstanceVisibility(groups, spatialSelection.componentIds);
    const referenceMaskStarted = performance.now();
    setVisibilityMaskEnabled(groups, false);
    const referenceMaskMs = performance.now() - referenceMaskStarted;
    const referenceStarted = performance.now();
    const reference = renderIds(renderer, scene, camera, target, pixels, width, height);
    const referenceRenderMs = performance.now() - referenceStarted;
    spatialStats.referenceRenderCount += 1;
    spatialStats.referenceReuseCount += Math.max(0, sampleGroup.samples.length - 1);
    for (const sample of sampleGroup.samples) {
      const batchId = String(sample.batchId || 'default');
      const predictionComponentIds = predictionComponentsForSample(manifest, sample);
      const visibilityStarted = performance.now();
      activePrediction = setPredictionComponents(componentSlots, activePrediction, predictionComponentIds);
      const visibilityUpdateMs = performance.now() - visibilityStarted;
      const predictionMaskStarted = performance.now();
      const predictionGlbs = predictionGlbIds(
        manifest,
        requiredGlbs,
        predictionComponentIds,
        spatialSelection.componentIds,
      );
      spatialStats.predictionGlbCountSum += predictionGlbs.length;
      spatialStats.predictionGlbCountMin = spatialStats.predictionGlbCountMin === null
        ? predictionGlbs.length
        : Math.min(spatialStats.predictionGlbCountMin, predictionGlbs.length);
      spatialStats.predictionGlbCountMax = Math.max(spatialStats.predictionGlbCountMax, predictionGlbs.length);
      // A prediction render only needs GLBs containing predicted instances.
      // The reference scene remains the complete conservative frustum scene;
      // this switch changes draw submission, never prediction membership.
      setSpatialVisibility(groups, predictionGlbs);
      setVisibilityMaskEnabled(groups, true);
      const predictionMaskMs = performance.now() - predictionMaskStarted;
      const predictionStarted = performance.now();
      const test = renderIds(renderer, scene, camera, target, pixels, width, height);
      const predictionRenderMs = performance.now() - predictionStarted;
      const { metrics, contributors } = computeMetrics(reference, test);
      imageRateSamples.push({
        PER: metrics.PER,
        missPixelRate: metrics.missPixelRate,
        wrongInstancePixelRate: metrics.wrongInstancePixelRate,
        extraPixelRateOverImage: metrics.extraPixelRateOverImage,
      });
      for (const key of Object.keys(totals)) totals[key] += metrics[key] || 0;
      const batchAggregate = batchTotals.get(batchId);
      if (!batchAggregate) throw new Error('missing batch accumulator for ' + batchId);
      for (const key of Object.keys(batchAggregate)) batchAggregate[key] += metrics[key] || 0;
      for (const [gid, pixelCount] of contributors.entries()) {
        top.set(gid, (top.get(gid) || 0) + pixelCount);
      }
      const diff = diffMask(reference, test);
      if (manifest.saveIdBuffers || sampleOrdinal < manifest.previewSamples) {
        await postBuffer('/buffer?name=' + encodeURIComponent(sample.sampleId + '_reference_u32.bin'), reference);
        await postBuffer('/buffer?name=' + encodeURIComponent(sample.sampleId + '_test_u32.bin'), test);
        await postBuffer('/buffer?name=' + encodeURIComponent(sample.sampleId + '_diff_u8.bin'), diff);
      }
      if (sampleOrdinal < manifest.previewSamples) {
        await postPreview(sample.sampleId + '_preview.png', reference, test, diff, width, height);
      }
      if (sampleOrdinal === 0) {
        const self = computeMetrics(reference, reference).metrics;
        selfConsistencyPER = self.PER;
      }
      sampleRows.push({
        ...sample,
        batchId,
        imageMetrics: metrics,
        timing: {
          referenceMaskMs,
          referenceRenderMs: sample === representative ? referenceRenderMs : 0,
          visibilityUpdateMs,
          predictionMaskMs,
          predictionRenderMs,
        },
      });
      sampleOrdinal += 1;
      if (sampleOrdinal % 16 === 0 || sampleOrdinal === samples.length) {
        status('rendered ' + sampleOrdinal + '/' + samples.length + ' samples in ' + ((performance.now() - started) / 1000).toFixed(1) + 's');
      }
    }
  }
  const loadFinished = performance.now();
  const valid = Math.max(1, totals.validReferencePixels);
  const total = Math.max(1, totals.totalPixels);
  const topMissedComponents = Array.from(top.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 50)
    .map(([componentGlobalId, missPixels]) => {
      const binding = (manifest.instanceBindings.componentToBinding || {})[String(componentGlobalId)] || {};
      const glbEntry = entriesById.get(Number(binding.globalGlbId)) || {};
      return { componentGlobalId, globalGlbId: Number(binding.globalGlbId), missPixels, path: glbEntry.path || '' };
    });
  const batchSummaries = batches.map((batch) => {
    const batchId = String(batch.batchId || 'default');
    const aggregate = batchTotals.get(batchId);
    const validPixels = Math.max(1, aggregate.validReferencePixels);
    const totalPixels = Math.max(1, aggregate.totalPixels);
    return {
      batchId,
      sampleCount: (batch.samples || []).length,
      imageMetrics: {
        schema: 'color-id-per-v1-aggregate',
        ...aggregate,
        PER: aggregate.errorPixels / validPixels,
        missPixelRate: aggregate.missPixels / validPixels,
        wrongInstancePixelRate: aggregate.wrongInstancePixels / validPixels,
        extraPixelRateOverImage: aggregate.extraPixels / totalPixels,
        evaluatedSubposeCount: (batch.samples || []).length,
        renderFailedSubposeCount: 0,
        missingGlbSubposeCount: 0,
      },
    };
  });
  const renderElapsedMs = performance.now() - started;
  const percentile = (key, value) => {
    const values = imageRateSamples.map((row) => Number(row[key])).filter(Number.isFinite).sort((a, b) => a - b);
    if (values.length === 0) return null;
    const position = (values.length - 1) * (Number(value) / 100);
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return values[lower];
    return values[lower] + (values[upper] - values[lower]) * (position - lower);
  };
  const meanRate = (key) => imageRateSamples.length === 0
    ? null
    : imageRateSamples.reduce((sum, row) => sum + Number(row[key]), 0) / imageRateSamples.length;
  spatialStats.meanRequiredGlbCount = samples.length > 0
    ? spatialStats.requiredGlbCountSum / sampleGroups.length
    : 0;
  spatialStats.meanPredictionGlbCount = samples.length > 0
    ? spatialStats.predictionGlbCountSum / samples.length
    : 0;
  spatialStats.loadedGlbCount = groups.size;
  spatialStats.loaderCalls = loadCallCount;
  const assetReuse = {
    schema: 'm5-browser-asset-reuse-v1',
    browserPageCount: 1,
    glbLoadPasses: 1,
    selectedGlbCount: selected.length,
    loadedGlbCount: groups.size,
    glbLoaderCalls: loadCallCount,
    sampleCount: samples.length,
    batchCount: batches.length,
    loadElapsedMs: loadFinished - loadStarted,
    renderElapsedMs,
    totalPageElapsedMs: performance.now() - pageStarted,
    reloadsPerPoseWithinPage: 0,
    reuseScope: 'all batches and samples in this manifest share one browser page; same-camera samples reuse one reference render',
    spatialCulling: spatialStats,
    note: 'AABB frustum filtering is render-submission-only; selectedGlbs and component predictions remain complete and unchanged.',
  };
  const summary = {
    schema: 'local-true-component-id-browser-summary-v3',
    renderer: navigator.userAgent,
    renderStatus: 'rendered_component_id_buffers',
    componentIdShaderImplemented: true,
    browserInstanceReorderImplemented: false,
    formalImageEvaluationReady: Boolean(formalManifest && sampleOrdinal === samples.length),
    gpuBackend,
    renderFovYDeg: 60,
    selectedGlbCount: selected.length,
    sampleCount: samples.length,
    batchCount: batches.length,
    loadedGlbCount: groups.size,
    spatialCulling: spatialStats,
    width,
    height,
    elapsedMs: renderElapsedMs,
    loadElapsedMs: loadFinished - loadStarted,
    totalPageElapsedMs: performance.now() - pageStarted,
    assetReuse,
    batchSummaries,
    selfConsistencyPER,
    imageMetrics: {
      schema: 'color-id-per-v1-aggregate',
      ...totals,
      PER: totals.errorPixels / valid,
      missPixelRate: totals.missPixels / valid,
      wrongInstancePixelRate: totals.wrongInstancePixels / valid,
      extraPixelRateOverImage: totals.extraPixels / total,
      meanPER: meanRate('PER'),
      medianPER: percentile('PER', 50),
      p95PER: percentile('PER', 95),
      meanMissPixelRate: meanRate('missPixelRate'),
      medianMissPixelRate: percentile('missPixelRate', 50),
      p95MissPixelRate: percentile('missPixelRate', 95),
      meanWrongInstancePixelRate: meanRate('wrongInstancePixelRate'),
      medianWrongInstancePixelRate: percentile('wrongInstancePixelRate', 50),
      p95WrongInstancePixelRate: percentile('wrongInstancePixelRate', 95),
      meanExtraPixelRateOverImage: meanRate('extraPixelRateOverImage'),
      medianExtraPixelRateOverImage: percentile('extraPixelRateOverImage', 50),
      p95ExtraPixelRateOverImage: percentile('extraPixelRateOverImage', 95),
      evaluatedSubposeCount: samples.length,
      renderFailedSubposeCount: 0,
      missingGlbSubposeCount: 0,
    },
    topMissedComponents,
  };
  await fetch('/sample-results', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(sampleRows) });
  await fetch('/done', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(summary) });
}

main().catch(async (error) => {
  await fetch('/done', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ schema: 'local-true-component-id-browser-summary-v2', renderStatus: 'failed', formalImageEvaluationReady: false, error: String(error && error.stack ? error.stack : error) }),
  }).catch(() => {});
});
`;
}

async function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(args.outputDir, { recursive: true });
  fs.mkdirSync(path.join(args.outputDir, 'samples'), { recursive: true });
  fs.mkdirSync(path.join(args.outputDir, 'previews'), { recursive: true });
  for (const dir of ['samples', 'previews']) {
    for (const file of fs.readdirSync(path.join(args.outputDir, dir))) {
      fs.rmSync(path.join(args.outputDir, dir, file), { force: true });
    }
  }
  const manifest = JSON.parse(fs.readFileSync(args.manifest, 'utf8'));
  const isBatchManifest = manifest.schema === INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA;
  const isFormalManifest = manifest.schema === FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA;
  if (isFormalManifest && args.syntheticRenderSmoke) {
    throw new Error('formal image manifests cannot be used with synthetic render smoke');
  }
  const validation = isBatchManifest
    ? validateInstanceRenderBatchManifest(manifest)
    : (isFormalManifest
      ? validateFormalInstanceRenderManifest(manifest)
      : validateInstanceRenderManifest(manifest));
  if (args.validateOnly) {
    const summary = {
      schema: isBatchManifest
        ? 'component-id-render-batch-schema-validation-summary-v1'
        : 'component-id-render-schema-validation-summary-v1',
      status: 'schema_validated_not_rendered',
      formalImageEvaluationReady: false,
      browserInstanceReorderImplemented: false,
      imageMetrics: null,
      validation,
      manifest: args.manifest,
    };
    fs.writeFileSync(path.join(args.outputDir, 'render_summary.json'), JSON.stringify(summary, null, 2), 'utf8');
    console.log(JSON.stringify(summary, null, 2));
    return;
  }
  if (args.syntheticRenderSmoke && !manifest.syntheticComponentIdSmoke) {
    throw new Error('--synthetic-render-smoke requires manifest.syntheticComponentIdSmoke');
  }
  const glbRoot = path.resolve(manifest.glbRoot);
  const glbIndexPath = path.resolve(manifest.glbIndex);
  const chromeExe = findChrome(args.chromeExe);
  assertRendererDependencies();
  let doneResolve;
  let doneReject;
  const donePromise = new Promise((resolve, reject) => {
    doneResolve = resolve;
    doneReject = reject;
  });
  const logs = [];
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      if (req.method === 'GET' && url.pathname === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
        res.end(rendererHtml());
        return;
      }
      if (req.method === 'GET' && url.pathname === '/renderer.js') {
        res.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store' });
        res.end(args.syntheticRenderSmoke ? syntheticRendererJs() : rendererJs());
        return;
      }
      if (req.method === 'GET' && url.pathname === '/manifest') return writeJson(res, manifest);
      if (req.method === 'GET' && url.pathname === '/glb-index') {
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
        fs.createReadStream(glbIndexPath).pipe(res);
        return;
      }
      if (req.method === 'GET' && url.pathname.startsWith('/node_modules/')) {
        const file = safeJoin(path.join(SLM2_ROOT, 'node_modules'), url.pathname.slice('/node_modules/'.length));
        if (!fs.existsSync(file)) throw new Error(`missing node module file ${file}`);
        res.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'public, max-age=3600' });
        fs.createReadStream(file).pipe(res);
        return;
      }
      if (req.method === 'GET' && url.pathname.startsWith('/assets/')) {
        const file = safeJoin(glbRoot, url.pathname.slice('/assets/'.length));
        if (!fs.existsSync(file)) throw new Error(`missing GLB asset ${file}`);
        res.writeHead(200, { 'Content-Type': contentType(file), 'Cache-Control': 'public, max-age=3600' });
        fs.createReadStream(file).pipe(res);
        return;
      }
      if (req.method === 'POST' && url.pathname === '/buffer') {
        const name = path.basename(url.searchParams.get('name') || 'buffer.bin');
        const body = await readBody(req);
        fs.writeFileSync(path.join(args.outputDir, 'samples', name), body);
        return writeJson(res, { ok: true });
      }
      if (req.method === 'POST' && url.pathname === '/preview') {
        const name = path.basename(url.searchParams.get('name') || 'preview.png');
        const body = await readBody(req);
        fs.writeFileSync(path.join(args.outputDir, 'previews', name), body);
        return writeJson(res, { ok: true });
      }
      if (req.method === 'POST' && url.pathname === '/sample-results') {
        const body = await readBody(req);
        fs.writeFileSync(path.join(args.outputDir, 'sample_image_metrics.json'), body);
        return writeJson(res, { ok: true });
      }
      if (req.method === 'POST' && url.pathname === '/log') {
        const body = (await readBody(req)).toString('utf8');
        logs.push({ time: new Date().toISOString(), message: body });
        console.log('[browser]', body);
        return writeJson(res, { ok: true });
      }
      if (req.method === 'POST' && url.pathname === '/done') {
        const body = JSON.parse((await readBody(req)).toString('utf8'));
        body.chromeLaunch = {
          executable: chromeExe,
          args: chromeArgs,
        };
        if (args.requireHardwareGpu) {
          const gpuGate = classifyGpuBackend(body.gpuBackend);
          body.gpuGate = {
            required: true,
            ...gpuGate,
          };
          if (!gpuGate.hardware && !body.error) {
            body.error = `hardware GPU required, browser reported ${gpuGate.renderer || 'no WebGL renderer'}`;
            body.renderStatus = 'failed_hardware_gpu_gate';
          }
        }
        if (isFormalManifest && !body.error) {
          if (body.formalImageEvaluationReady !== true || body.componentIdShaderImplemented !== true ||
              body.renderStatus !== 'rendered_component_id_buffers' ||
              Number(body.sampleCount) !== Number((manifest.samples || []).length)) {
            body.error = 'formal image gate failed: complete real component-ID render was not reported';
            body.renderStatus = 'failed_formal_image_gate';
            body.formalImageEvaluationReady = false;
          }
        }
        fs.writeFileSync(path.join(args.outputDir, 'render_summary.json'), JSON.stringify(body, null, 2), 'utf8');
        fs.writeFileSync(path.join(args.outputDir, 'browser_logs.json'), JSON.stringify(logs, null, 2), 'utf8');
        if (body.error) doneReject(new Error(body.error));
        else doneResolve(body);
        return writeJson(res, { ok: !body.error, error: body.error || null }, body.error ? 409 : 200);
      }
      writeJson(res, { error: 'not found' }, 404);
    } catch (error) {
      writeJson(res, { error: String(error && error.message ? error.message : error) }, 500);
    }
  });
  await new Promise((resolve) => server.listen(args.port, '127.0.0.1', resolve));
  const port = server.address().port;
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pvs-color-id-chrome-'));
  if (args.requireHardwareGpu && args.chromeArgs.some((value) => /swiftshader|disable-gpu/i.test(value))) {
    throw new Error('--require-hardware-gpu cannot be combined with SwiftShader or --disable-gpu');
  }
  const chromeArgs = [
    ...(args.browserMode === 'headless' ? ['--headless=new'] : []),
    '--no-first-run',
    '--disable-dev-shm-usage',
    '--disable-background-networking',
    '--disable-extensions',
    '--hide-scrollbars',
    '--mute-audio',
    '--enable-gpu',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    '--use-angle=vulkan',
    '--enable-accelerated-2d-canvas',
    '--enable-zero-copy',
    ...(args.requireHardwareGpu ? ['--disable-software-rasterizer'] : []),
    ...args.chromeArgs,
    `--user-data-dir=${userDataDir}`,
    `http://127.0.0.1:${port}/`,
  ];
  console.log(`[true-glb-render] launching ${chromeExe}`);
  const chromeEnv = { ...process.env };
  if (args.display) chromeEnv.DISPLAY = args.display;
  const chrome = spawn(chromeExe, chromeArgs, { env: chromeEnv, stdio: ['ignore', 'pipe', 'pipe'] });
  chrome.stdout.on('data', (chunk) => process.stdout.write(chunk));
  chrome.stderr.on('data', (chunk) => process.stderr.write(chunk));
  const timer = setTimeout(() => doneReject(new Error(`renderer timeout after ${args.timeoutMs} ms`)), args.timeoutMs);
  try {
    await donePromise;
    clearTimeout(timer);
  } finally {
    chrome.kill();
    server.close();
    try {
      fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 });
    } catch (error) {
      console.warn(`[true-glb-render] could not remove temporary Chrome profile ${userDataDir}: ${error && error.message ? error.message : error}`);
    }
  }
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
