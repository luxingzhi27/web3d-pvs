#!/usr/bin/env node
/* Hardware-WebGPU parity probe for the ray-context/survival OWRB bundle. */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..');

function parseArgs(argv) {
  const options = {
    bundle: null,
    cases: null,
    out: null,
    chromeExe: process.env.CHROME_PATH || process.env.CHROMIUM_PATH || null,
    port: 0,
    timeoutMs: 180000,
    requireHardwareGpu: true,
  };
  const valueOptions = new Set(['bundle', 'cases', 'out', 'chrome-exe', 'port', 'timeout-ms']);
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--allow-software-gpu') { options.requireHardwareGpu = false; continue; }
    if (token === '--require-hardware-gpu') { options.requireHardwareGpu = true; continue; }
    const equal = token.indexOf('=');
    const key = token.slice(0, equal >= 0 ? equal : undefined).replace(/^--/, '');
    if (!valueOptions.has(key)) throw new Error(`unknown argument ${token}`);
    const value = equal >= 0 ? token.slice(equal + 1) : argv[++i];
    if (key === 'bundle') options.bundle = path.resolve(value);
    else if (key === 'cases') options.cases = path.resolve(value);
    else if (key === 'out') options.out = path.resolve(value);
    else if (key === 'chrome-exe') options.chromeExe = path.resolve(value);
    else if (key === 'port') options.port = Number(value);
    else if (key === 'timeout-ms') options.timeoutMs = Number(value);
  }
  if (!options.bundle || !options.cases || !options.out) throw new Error('--bundle, --cases and --out are required');
  return options;
}

function findChrome(explicit) {
  const candidates = [explicit, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'].filter(Boolean);
  for (const candidate of candidates) if (fs.existsSync(candidate)) return candidate;
  throw new Error('Chrome executable not found');
}

async function loadPlaywright() {
  try { return await import('playwright'); } catch {
    const local = path.join(REPO_ROOT, 'slm2viewer', 'node_modules', 'playwright', 'index.mjs');
    return import(`file://${local}`);
  }
}

function classifyGpu(adapterInfo) {
  const text = [adapterInfo?.vendor, adapterInfo?.architecture, adapterInfo?.device, adapterInfo?.description].filter(Boolean).join(' ');
  const software = /swiftshader|llvmpipe|softpipe|swrast|software|no-webgpu/i.test(text);
  return {
    vendor: String(adapterInfo?.vendor || ''),
    architecture: String(adapterInfo?.architecture || ''),
    device: String(adapterInfo?.device || ''),
    description: String(adapterInfo?.description || ''),
    hardware: Boolean(text) && !software,
    softwareMarkers: software ? text : null,
  };
}

function hostGpuEvidence() {
  const run = (args) => {
    try {
      return { available: true, text: execFileSync('nvidia-smi', args, { encoding: 'utf8', timeout: 10000 }) };
    } catch (error) {
      return { available: false, error: String(error?.message || error) };
    }
  };
  return {
    nvidiaSmi: run(['--query-gpu=index,name,driver_version,utilization.gpu,memory.used', '--format=csv,noheader,nounits']),
    nvidiaSmiPmon: run(['pmon', '-c', '1']),
  };
}

function startGpuPmonSampler() {
  let child;
  let stdout = '';
  let stderr = '';
  let spawnError = null;
  let closed = false;
  const closeWaiters = [];
  let stopped = false;
  let stopPromise;
  try {
    // Keep pmon alive for the whole browser probe.  A single sample taken
    // after WebGPU completes cannot establish that the computation window
    // had a GPU process attached to it.
    child = spawn('nvidia-smi', ['pmon', '-s', 'um', '-d', '1'], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    child.on('error', (error) => { spawnError = String(error?.message || error); });
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('close', () => {
      closed = true;
      while (closeWaiters.length) closeWaiters.shift()(finish());
    });
  } catch (error) {
    return {
      stop: async () => ({ available: false, error: String(error?.message || error), sampleCount: 0 }),
    };
  }
  const finish = () => {
    const lines = stdout.split(/\r?\n/);
    const sampleCount = lines.filter((line) => /^\s*\d+\s+\S+/.test(line)).length;
    return {
      available: !spawnError,
      error: spawnError,
      exitText: stderr.trim() || null,
      sampleCount,
      text: stdout,
    };
  };
  return {
    stop: async () => {
      if (stopped) return stopPromise;
      stopped = true;
      if (closed) return finish();
      stopPromise = new Promise((resolve) => {
        closeWaiters.push(resolve);
        child.kill('SIGTERM');
        setTimeout(() => {
          if (!closed) child.kill('SIGKILL');
        }, 2000);
      });
      return stopPromise;
    },
  };
}

function denseFunction(name, inputSize, outputSize, weightOffset, biasOffset, activation) {
  const lines = [];
  lines.push(`fn ${name}(x: array<f32,${inputSize}>) -> array<f32,${outputSize}> {`);
  lines.push(`  var y: array<f32,${outputSize}>;`);
  lines.push(`  for (var i: u32 = 0u; i < ${outputSize}u; i = i + 1u) {`);
  lines.push(`    var value: f32 = weightBuffer[${biasOffset}u + i];`);
  lines.push(`    for (var j: u32 = 0u; j < ${inputSize}u; j = j + 1u) {`);
  lines.push(`      value = value + weightBuffer[${weightOffset}u + i * ${inputSize}u + j] * x[j];`);
  lines.push('    }');
  lines.push(`    y[i] = ${activation === 'relu' ? 'max(value, 0.0)' : 'value'};`);
  lines.push('  }');
  lines.push('  return y;');
  lines.push('}');
  return lines.join('\n');
}

function weightLayout(weightsPayload) {
  const names = [
    'survival_direction_basis.0.weight','survival_direction_basis.0.bias','survival_direction_basis.2.weight','survival_direction_basis.2.bias',
    'context_direction_basis.0.weight','context_direction_basis.0.bias','context_direction_basis.2.weight','context_direction_basis.2.bias',
    'base_trunk.0.weight','base_trunk.0.bias','base_trunk.2.weight','base_trunk.2.bias','base_visibility_head.weight','base_visibility_head.bias',
    'utility_head.0.weight','utility_head.0.bias','utility_head.2.weight','utility_head.2.bias',
    'download_head.0.weight','download_head.0.bias','download_head.2.weight','download_head.2.bias',
  ];
  const values = [];
  const offsets = {};
  for (const name of names) {
    const record = weightsPayload.weights[name];
    if (!record) throw new Error(`missing query weight ${name}`);
    offsets[name] = values.length;
    values.push(...record.values);
  }
  return { values: new Float32Array(values), offsets };
}

function rendererSource() {
  return String.raw`
const meta = await (await fetch('/meta')).json();
const weightsPayload = await (await fetch('/weights')).json();
const casesPayload = await (await fetch('/cases')).json();
const runtimeBytes = await (await fetch('/runtime')).arrayBuffer();
const aabbBytes = await (await fetch('/aabb')).arrayBuffer();

function classifyAdapter(adapter) {
  const info = adapter.info || {};
  return { vendor: info.vendor || '', architecture: info.architecture || '', device: info.device || '', description: info.description || '' };
}

function makeShader(layout, modelConfig = {}) {
  const o = layout.offsets;
  const survivalEnabled = modelConfig.survivalEnabled !== false;
  const parts = [];
  parts.push('const survivalEnabled: bool = ' + (survivalEnabled ? 'true' : 'false') + ';');
  parts.push('struct Params { cameraPos: vec3<f32>, tanX: f32, forward: vec3<f32>, tanY: f32, candidateOffset: u32, candidateCount: u32, threshold: f32, pad: f32 };');
  parts.push('@group(0) @binding(0) var<storage, read> runtimeBuffer: array<u32>;');
  parts.push('@group(0) @binding(1) var<storage, read> aabbBuffer: array<u32>;');
  parts.push('@group(0) @binding(2) var<storage, read> candidateBuffer: array<u32>;');
  parts.push('@group(0) @binding(3) var<storage, read> weightBuffer: array<f32>;');
  parts.push('@group(0) @binding(4) var<storage, read_write> outputBuffer: array<u32>;');
  parts.push('@group(0) @binding(5) var<uniform> params: Params;');
  parts.push('fn feature(id: u32, index: u32) -> f32 { if (!survivalEnabled && index >= 128u) { return 0.0; } let packed = runtimeBuffer[id * 78u + index / 2u]; let pair = unpack2x16float(packed); return select(pair.y, pair.x, (index & 1u) == 0u); }');
  parts.push('fn sigmoid(value: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(value, -80.0, 80.0))); }');
  parts.push('fn softplus(value: f32) -> f32 { return max(value, 0.0) + log(1.0 + exp(-abs(value))); }');
  parts.push(denseFunction('surv0', 3, 16, o['survival_direction_basis.0.weight'], o['survival_direction_basis.0.bias'], 'relu'));
  parts.push(denseFunction('surv2', 16, 4, o['survival_direction_basis.2.weight'], o['survival_direction_basis.2.bias'], 'none'));
  parts.push(denseFunction('ctx0', 3, 16, o['context_direction_basis.0.weight'], o['context_direction_basis.0.bias'], 'relu'));
  parts.push(denseFunction('ctx2', 16, 4, o['context_direction_basis.2.weight'], o['context_direction_basis.2.bias'], 'none'));
  parts.push(denseFunction('base0', 177, 64, o['base_trunk.0.weight'], o['base_trunk.0.bias'], 'relu'));
  parts.push(denseFunction('base2', 64, 32, o['base_trunk.2.weight'], o['base_trunk.2.bias'], 'relu'));
  parts.push(denseFunction('baseVis', 32, 1, o['base_visibility_head.weight'], o['base_visibility_head.bias'], 'none'));
  parts.push(denseFunction('util0', 33, 32, o['utility_head.0.weight'], o['utility_head.0.bias'], 'relu'));
  parts.push(denseFunction('util2', 32, 1, o['utility_head.2.weight'], o['utility_head.2.bias'], 'none'));
  parts.push(denseFunction('down0', 33, 32, o['download_head.0.weight'], o['download_head.0.bias'], 'relu'));
  parts.push(denseFunction('down2', 32, 1, o['download_head.2.weight'], o['download_head.2.bias'], 'none'));
  parts.push([
    '@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {',
    '  let localIndex = gid.x;',
    '  if (localIndex >= params.candidateCount) { return; }',
    '  let instanceId = candidateBuffer[params.candidateOffset + localIndex];',
    '  let aabbBase = instanceId * 3u;',
    '  let a0 = unpack2x16float(aabbBuffer[aabbBase]);',
    '  let a1 = unpack2x16float(aabbBuffer[aabbBase + 1u]);',
    '  let a2 = unpack2x16float(aabbBuffer[aabbBase + 2u]);',
    '  let minimum = vec3<f32>(a0.x, a0.y, a1.x);',
    '  let maximum = vec3<f32>(a1.y, a2.x, a2.y);',
    '  let size = max(maximum - minimum, vec3<f32>(0.0001));',
    '  let center = (minimum + maximum) * 0.5;',
    '  let delta = center - params.cameraPos;',
    '  let distance = max(length(delta), 0.0001);',
    '  let rayDir = delta / distance;',
    '  let forward = normalize(params.forward);',
    '  var upSeed = vec3<f32>(0.0, 1.0, 0.0);',
    '  if (abs(dot(forward, upSeed)) > 0.98) { upSeed = vec3<f32>(0.0, 0.0, 1.0); }',
    '  let right = normalize(cross(forward, upSeed));',
    '  let up = normalize(cross(right, forward));',
    '  let dotForward = dot(rayDir, forward);',
    '  let u = dot(rayDir, right) / max(abs(dotForward) * params.tanX, 0.0001);',
    '  let v = dot(rayDir, up) / max(abs(dotForward) * params.tanY, 0.0001);',
    '  let radius = length(size) * 0.5;',
    '  let angular = radius / max(distance, 1.0);',
    '  var ray9: array<f32,9>;',
    '  ray9[0] = rayDir.x; ray9[1] = rayDir.y; ray9[2] = rayDir.z;',
    '  ray9[3] = clamp(log(1.0 + distance / 100.0) / 4.0, 0.0, 2.0) - 1.0;',
    '  ray9[4] = clamp(dotForward, -1.0, 1.0);',
    '  ray9[5] = clamp(u, -4.0, 4.0) / 4.0;',
    '  ray9[6] = clamp(v, -4.0, 4.0) / 4.0;',
    '  ray9[7] = clamp(angular / params.tanX, 0.0, 4.0) / 2.0 - 1.0;',
    '  ray9[8] = clamp(angular / params.tanY, 0.0, 4.0) / 2.0 - 1.0;',
    '  var d: array<f32,3>; d[0] = rayDir.x; d[1] = rayDir.y; d[2] = rayDir.z;',
    '  let survivalPhiHidden = surv0(d); let survivalPhiLinear = surv2(survivalPhiHidden);',
    '  let contextPhiHidden = ctx0(d); let contextPhiLinear = ctx2(contextPhiHidden);',
    '  var survivalPhi: array<f32,4>; var contextPhi: array<f32,4>;',
    '  let survivalGate = select(0.0, 1.0, survivalEnabled);',
    '  for (var i: u32 = 0u; i < 4u; i = i + 1u) { survivalPhi[i] = survivalGate * tanh(survivalPhiLinear[i]); contextPhi[i] = tanh(contextPhiLinear[i]); }',
    '  var survivalParams: array<f32,7>;',
    '  for (var k: u32 = 0u; k < 7u; k = k + 1u) { survivalParams[k] = survivalPhi[0] * feature(instanceId, 128u + k) + survivalPhi[1] * feature(instanceId, 135u + k) + survivalPhi[2] * feature(instanceId, 142u + k) + survivalPhi[3] * feature(instanceId, 149u + k); }',
    '  var semantic: array<f32,8>;',
    '  if (survivalEnabled) {',
    '    let noBlock = sigmoid(survivalParams[0]);',
    '    let expMax = max(survivalParams[1], survivalParams[2]); let exp0 = exp(survivalParams[1] - expMax); let exp1 = exp(survivalParams[2] - expMax); let expDen = max(exp0 + exp1, 0.000001);',
    '    let rawMix0 = exp0 / expDen; let rawMix1 = exp1 / expDen;',
    '    let rawDepth0 = 0.05 + 0.90 * sigmoid(survivalParams[3]); let rawDepth1 = 0.05 + 0.90 * sigmoid(survivalParams[4]);',
    '    let depth0 = min(rawDepth0, rawDepth1); let depth1 = max(rawDepth0, rawDepth1);',
    '    let mix0 = select(rawMix1, rawMix0, rawDepth0 <= rawDepth1); let mix1 = select(rawMix0, rawMix1, rawDepth0 <= rawDepth1);',
    '    let rawScale0 = 0.03 + softplus(survivalParams[5]); let rawScale1 = 0.03 + softplus(survivalParams[6]);',
    '    let scale0 = select(rawScale1, rawScale0, rawDepth0 <= rawDepth1); let scale1 = select(rawScale0, rawScale1, rawDepth0 <= rawDepth1);',
    '    let rho = clamp(distance / max(distance + radius, 0.0001), 0.0, 1.0);',
    '    let cdf = (1.0 - noBlock) * (mix0 * sigmoid((rho - depth0) / scale0) + mix1 * sigmoid((rho - depth1) / scale1));',
    '    let survival = clamp(1.0 - cdf, 0.00001, 1.0);',
    '    let expectedDepth = mix0 * depth0 + mix1 * depth1;',
    '    let variance = mix0 * (depth0 - expectedDepth) * (depth0 - expectedDepth) + mix1 * (depth1 - expectedDepth) * (depth1 - expectedDepth);',
    '    let uncertainty = sqrt(max(variance, 0.00000001));',
    '    let s0 = sigmoid((rho - depth0) / scale0); let s1 = sigmoid((rho - depth1) / scale1);',
    '    let localSlope = (1.0 - noBlock) * (mix0 / scale0 * s0 * (1.0 - s0) + mix1 / scale1 * s1 * (1.0 - s1));',
    '    semantic[0] = survival; semantic[1] = 1.0 - survival; semantic[2] = noBlock; semantic[3] = expectedDepth; semantic[4] = uncertainty; semantic[5] = mix0; semantic[6] = mix1; semantic[7] = localSlope;',
    '  } else {',
    '    for (var semanticIndex: u32 = 0u; semanticIndex < 8u; semanticIndex = semanticIndex + 1u) { semantic[semanticIndex] = 0.0; }',
    '  }',
    '  var contextQuery: array<f32,8>;',
    '  for (var k: u32 = 0u; k < 8u; k = k + 1u) { contextQuery[k] = contextPhi[0] * feature(instanceId, 96u + k) + contextPhi[1] * feature(instanceId, 104u + k) + contextPhi[2] * feature(instanceId, 112u + k) + contextPhi[3] * feature(instanceId, 120u + k); }',
    '  var baseInput: array<f32,177>; for (var i: u32 = 0u; i < 156u; i = i + 1u) { baseInput[i] = feature(instanceId, i); }',
    '  for (var i: u32 = 0u; i < 9u; i = i + 1u) { baseInput[156u + i] = ray9[i]; }',
    '  for (var i: u32 = 0u; i < 8u; i = i + 1u) { baseInput[165u + i] = contextQuery[i]; }',
    '  baseInput[173u] = semantic[0]; baseInput[174u] = semantic[2]; baseInput[175u] = semantic[3]; baseInput[176u] = semantic[4];',
    '  let baseHidden0 = base0(baseInput); let baseHidden = base2(baseHidden0); let baseLogit = baseVis(baseHidden)[0];',
    '  let visibilityLogit = baseLogit;',
    '  var taskInput: array<f32,33>; for (var i: u32 = 0u; i < 32u; i = i + 1u) { taskInput[i] = baseHidden[i]; } taskInput[32] = sigmoid(visibilityLogit);',
    '  let utilityLogit = util2(util0(taskInput))[0]; let downloadLogit = down2(down0(taskInput))[0];',
    '  let outputBase = localIndex * 6u;',
    '  outputBuffer[outputBase] = bitcast<u32>(visibilityLogit); outputBuffer[outputBase + 1u] = bitcast<u32>(sigmoid(visibilityLogit));',
    '  outputBuffer[outputBase + 2u] = bitcast<u32>(utilityLogit); outputBuffer[outputBase + 3u] = bitcast<u32>(sigmoid(utilityLogit));',
    '  outputBuffer[outputBase + 4u] = bitcast<u32>(downloadLogit); outputBuffer[outputBase + 5u] = bitcast<u32>(sigmoid(downloadLogit));',
    '}',
  ].join('\n'));
  return parts.join('\n');
}

async function runBrowser(casesPayload, meta, weights, runtimeBytes, aabbBytes) {
  if (!navigator.gpu) throw new Error('navigator.gpu is unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter returned null');
  const adapterInfo = classifyAdapter(adapter);
  const device = await adapter.requestDevice();
  const layout = weightLayout(weights);
  const shader = makeShader(layout, meta.modelConfig || {});
  const module = device.createShaderModule({ code: shader });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length) throw new Error('WGSL compilation failed: ' + JSON.stringify(errors));
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const runtimeWords = new Uint32Array(runtimeBytes);
  const aabbWords = new Uint32Array(aabbBytes);
  const runtimeBuffer = device.createBuffer({ size: runtimeWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const aabbBuffer = device.createBuffer({ size: aabbWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const weightBuffer = device.createBuffer({ size: layout.values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(runtimeBuffer, 0, runtimeWords);
  device.queue.writeBuffer(aabbBuffer, 0, aabbWords);
  device.queue.writeBuffer(weightBuffer, 0, layout.values);
  const results = [];
  for (const testCase of casesPayload.cases) {
    const ids = new Uint32Array(testCase.candidateIds);
    const candidateBuffer = device.createBuffer({ size: Math.max(4, ids.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(candidateBuffer, 0, ids);
    const outputSize = Math.max(24, ids.length * 6 * 4);
    const outputBuffer = device.createBuffer({ size: outputSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: outputSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const uniform = new ArrayBuffer(48);
    const floatView = new Float32Array(uniform);
    const uintView = new Uint32Array(uniform);
    floatView.set(testCase.camera.position, 0);
    floatView[3] = Number(testCase.camera.cameraView[3]);
    floatView.set(testCase.camera.cameraForward, 4);
    floatView[7] = Number(testCase.camera.cameraView[4]);
    uintView[8] = 0; uintView[9] = ids.length; floatView[10] = Number(meta.threshold);
    const uniformBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniformBuffer, 0, uniform);
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: runtimeBuffer } },
      { binding: 1, resource: { buffer: aabbBuffer } },
      { binding: 2, resource: { buffer: candidateBuffer } },
      { binding: 3, resource: { buffer: weightBuffer } },
      { binding: 4, resource: { buffer: outputBuffer } },
      { binding: 5, resource: { buffer: uniformBuffer } },
    ] });
    const started = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(Math.ceil(ids.length / 64)); pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, ids.length * 6 * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const words = Array.from(new Uint32Array(readback.getMappedRange().slice(0)));
    readback.unmap();
    results.push({ caseId: Number(testCase.caseId), candidateIds: Array.from(ids), rawOutput: words, forwardMs: performance.now() - started });
    candidateBuffer.destroy(); outputBuffer.destroy(); readback.destroy(); uniformBuffer.destroy();
  }
  runtimeBuffer.destroy(); aabbBuffer.destroy(); weightBuffer.destroy();
  return { schema: 'ray-context-survival-owrb-webgpu-parity-capture-v1', adapterInfo, results, backend: 'webgpu' };
}

runBrowser(casesPayload, meta, weightsPayload, runtimeBytes, aabbBytes)
  .then((result) => fetch('/done', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(result) }))
  .catch(async (error) => {
    await fetch('/done', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ schema: 'ray-context-survival-owrb-webgpu-parity-capture-v1', error: String(error?.stack || error), backend: 'webgpu' }) }).catch(() => {});
  });
`;
}

function rendererSourceWithPayload() {
  return rendererSource();
}

async function main() {
  const args = parseArgs(process.argv);
  const cases = JSON.parse(fs.readFileSync(args.cases, 'utf8'));
  const meta = JSON.parse(fs.readFileSync(path.join(args.bundle, 'model_meta.json'), 'utf8'));
  const weights = JSON.parse(fs.readFileSync(path.join(args.bundle, meta.files.queryWeights), 'utf8'));
  const outputDir = path.dirname(args.out);
  fs.mkdirSync(outputDir, { recursive: true });
  const hostGpuBefore = hostGpuEvidence();
  const gpuPmonSampler = startGpuPmonSampler();
  let doneResolve; let doneReject;
  const done = new Promise((resolve, reject) => { doneResolve = resolve; doneReject = reject; });
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      if (req.method === 'GET' && url.pathname === '/') { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end('<!doctype html><script type="module" src="/renderer.js"></script>'); return; }
      if (req.method === 'GET' && url.pathname === '/renderer.js') {
        const helperSource = [
          `const denseFunction = ${denseFunction.toString()};`,
          `const weightLayout = ${weightLayout.toString()};`,
        ].join('\n');
        const source = helperSource + `\nconst casesPayload = ${JSON.stringify(cases)}; const meta = ${JSON.stringify(meta)}; const weightsPayload = ${JSON.stringify(weights)};\n` +
          rendererSourceWithPayload().replace("const meta = await (await fetch('/meta')).json();\nconst weightsPayload = await (await fetch('/weights')).json();\nconst casesPayload = await (await fetch('/cases')).json();\nconst runtimeBytes = await (await fetch('/runtime')).arrayBuffer();\nconst aabbBytes = await (await fetch('/aabb')).arrayBuffer();", "const runtimeBytes = await (await fetch('/runtime')).arrayBuffer();\nconst aabbBytes = await (await fetch('/aabb')).arrayBuffer();");
        res.writeHead(200, { 'Content-Type': 'text/javascript' }); res.end(source); return;
      }
      if (req.method === 'GET' && url.pathname === '/meta') { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(meta)); return; }
      if (req.method === 'GET' && url.pathname === '/weights') { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(weights)); return; }
      if (req.method === 'GET' && url.pathname === '/cases') { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(cases)); return; }
      if (req.method === 'GET' && url.pathname === '/runtime') { fs.createReadStream(path.join(args.bundle, meta.files.runtimeFeatures)).pipe(res); return; }
      if (req.method === 'GET' && url.pathname === '/aabb') { fs.createReadStream(path.join(args.bundle, meta.files.instanceAabb)).pipe(res); return; }
      if (req.method === 'POST' && url.pathname === '/done') {
        const chunks = []; for await (const chunk of req) chunks.push(chunk);
        const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        body.gpuGate = { required: args.requireHardwareGpu, ...classifyGpu(body.adapterInfo) };
        body.hostGpuBefore = hostGpuBefore;
        const duringPmon = await gpuPmonSampler.stop();
        body.hostGpuDuring = {
          // The pmon stream spans the browser execution window.  The
          // nvidia-smi snapshot is captured at the same completion boundary
          // so existing GPU-evidence readers retain their schema.
          nvidiaSmi: hostGpuEvidence().nvidiaSmi,
          nvidiaSmiPmon: duringPmon,
        };
        body.hostGpuAfter = hostGpuEvidence();
        body.formalReady = Boolean(
          body.backend === 'webgpu'
          && body.gpuGate.hardware
          && body.hostGpuBefore.nvidiaSmi.available
          && body.hostGpuBefore.nvidiaSmiPmon.available
          && body.hostGpuDuring.nvidiaSmi.available
          && body.hostGpuDuring.nvidiaSmiPmon.available
          && Number(body.hostGpuDuring.nvidiaSmiPmon.sampleCount || 0) > 0
          && body.hostGpuAfter.nvidiaSmi.available
          && body.hostGpuAfter.nvidiaSmiPmon.available
        );
        // Always resolve with the complete body.  The caller writes it to the
        // requested output before turning a failed hardware gate into a
        // process error.  This preserves adapter/compiler diagnostics instead
        // of losing them behind a rejected promise.
        doneResolve(body);
        res.writeHead(body.error ? 409 : 200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: !body.error })); return;
      }
      res.writeHead(404); res.end();
    } catch (error) {
      console.error('[owrb-parity-server]', error?.stack || error);
      res.writeHead(500); res.end(String(error?.stack || error));
    }
  });
  await new Promise((resolve) => server.listen(args.port, '127.0.0.1', resolve));
  const port = server.address().port;
  const chrome = findChrome(args.chromeExe);
  const { chromium } = await loadPlaywright();
  const launchArgs = [
    '--headless=new', '--no-sandbox', '--no-first-run', '--disable-dev-shm-usage',
    '--disable-background-networking', '--disable-extensions',
    '--enable-gpu', '--enable-unsafe-webgpu', '--enable-webgpu', '--enable-webgl',
    '--enable-features=Vulkan', '--use-vulkan', '--use-angle=vulkan', '--enable-accelerated-2d-canvas', '--enable-zero-copy',
    '--ignore-gpu-blocklist', '--disable-gpu-sandbox',
    '--ozone-platform=headless', '--ozone-override-screen-size=1280,720',
    ...(args.requireHardwareGpu ? ['--disable-software-rasterizer'] : []),
  ];
  let browser;
  let result = null;
  let chromeGpuInfo = null;
  const timer = setTimeout(() => doneReject(new Error(`WebGPU parity timed out after ${args.timeoutMs}ms`)), args.timeoutMs);
  const writeCapture = (payload) => {
    fs.writeFileSync(args.out, JSON.stringify({
      schema: 'ray-context-survival-owrb-webgpu-parity-capture-v1',
      cases: args.cases,
      bundle: args.bundle,
      capturedAt: new Date().toISOString(),
      ...payload,
    }, null, 2) + '\n');
  };
  try {
    browser = await chromium.launch({ executablePath: chrome, headless: true, args: launchArgs });
    try {
      const browserCdp = await browser.newBrowserCDPSession();
      chromeGpuInfo = await browserCdp.send('SystemInfo.getInfo');
      await browserCdp.detach().catch(() => {});
    } catch (error) {
      chromeGpuInfo = { error: String(error?.stack || error) };
    }
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', (message) => console.log('[owrb-parity-browser]', message.text()));
    page.on('pageerror', (error) => console.error('[owrb-parity-pageerror]', error));
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded' });
    result = await done;
    clearTimeout(timer);
    result.browserLaunch = {
      executablePath: chrome,
      headless: true,
      args: launchArgs,
      vulkanIcd: process.env.VK_ICD_FILENAMES || null,
      chromeGpuInfo,
    };
    const failure = result.error
      || (result.backend !== 'webgpu' ? 'parity backend was not WebGPU' : null)
      || (args.requireHardwareGpu && result.formalReady !== true ? 'WebGPU hardware gate failed' : null);
    if (failure) throw new Error(String(failure));
    writeCapture(result);
    console.log(JSON.stringify({ output: args.out, caseCount: result.results?.length || 0, gpuGate: result.gpuGate }, null, 2));
  } catch (error) {
    const failure = String(error?.stack || error);
    if (result) {
      if (!result.error) result.error = failure;
      result.browserLaunch = {
        executablePath: chrome,
        headless: true,
        args: launchArgs,
        vulkanIcd: process.env.VK_ICD_FILENAMES || null,
        chromeGpuInfo,
      };
      writeCapture(result);
    } else {
      writeCapture({
        backend: 'webgpu',
        adapterInfo: null,
        results: [],
        formalReady: false,
        gpuGate: { required: args.requireHardwareGpu, hardware: false, software: false, reason: 'probe_failed' },
        hostGpuBefore,
        error: failure,
      });
    }
    throw error;
  } finally {
    clearTimeout(timer);
    await gpuPmonSampler.stop().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    server.close();
  }
}

main().catch((error) => { console.error(error?.stack || error); process.exitCode = 1; });
