#!/usr/bin/env node
/* Run the geometry-shell HZB browser MVP with an explicit shell and dataset. */

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  classifyGpuText,
  headlessWebGpuArgs,
  resolveVulkanEnvironment,
} from './chrome_gpu_flags.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = path.resolve(SCRIPT_DIR, '..');
const REPO_ROOT = path.resolve(VIEWER_ROOT, '..');
const POSE_STRIDE_BYTES = 64;
const SOFTWARE_PATTERN = /swiftshader|llvmpipe|softpipe|swrast|software/i;

function parseArgs(argv) {
  const options = {
    shellDir: '',
    datasetDir: '',
    regionDatasetDir: '',
    output: '',
    mode: 'Point60',
    split: 'test',
    limit: 0,
    warmupCount: 0,
    width: 512,
    height: 288,
    aspect: 16 / 9,
    fovYDeg: 60,
    regionFovYDeg: 66,
    near: 0.05,
    far: 20000,
    depthBiasM: 0.001,
    chromeExe: process.env.CHROME_PATH || process.env.CHROMIUM_PATH || '',
    port: 0,
    timeoutMs: 30 * 60 * 1000,
    requireHardwareGpu: true,
  };
  const valueOptions = new Set([
    'shell-dir', 'dataset-dir', 'region-dataset-dir', 'output', 'mode', 'split', 'limit',
    'warmup-count', 'width', 'height', 'aspect', 'fov-y-deg', 'region-fov-y-deg', 'near', 'far',
    'depth-bias-m', 'chrome-exe', 'port', 'timeout-ms',
  ]);
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--allow-software-gpu') {
      options.requireHardwareGpu = false;
      continue;
    }
    if (token === '--require-hardware-gpu') {
      options.requireHardwareGpu = true;
      continue;
    }
    const equal = token.indexOf('=');
    const key = token.slice(0, equal >= 0 ? equal : undefined).replace(/^--/, '');
    if (!valueOptions.has(key)) throw new Error(`Unknown argument: ${token}`);
    const value = equal >= 0 ? token.slice(equal + 1) : argv[++index];
    if (value == null) throw new Error(`Missing value for --${key}`);
    if (key === 'shell-dir') options.shellDir = path.resolve(value);
    else if (key === 'dataset-dir') options.datasetDir = path.resolve(value);
    else if (key === 'region-dataset-dir') options.regionDatasetDir = path.resolve(value);
    else if (key === 'output') options.output = path.resolve(value);
    else if (key === 'mode') options.mode = String(value);
    else if (key === 'split') options.split = String(value);
    else if (key === 'limit') options.limit = Number(value);
    else if (key === 'warmup-count') options.warmupCount = Number(value);
    else if (key === 'width') options.width = Number(value);
    else if (key === 'height') options.height = Number(value);
    else if (key === 'aspect') options.aspect = Number(value);
    else if (key === 'fov-y-deg') options.fovYDeg = Number(value);
    else if (key === 'region-fov-y-deg') options.regionFovYDeg = Number(value);
    else if (key === 'near') options.near = Number(value);
    else if (key === 'far') options.far = Number(value);
    else if (key === 'depth-bias-m') options.depthBiasM = Number(value);
    else if (key === 'chrome-exe') options.chromeExe = path.resolve(value);
    else if (key === 'port') options.port = Number(value);
    else if (key === 'timeout-ms') options.timeoutMs = Number(value);
  }
  if (!options.shellDir || !options.datasetDir || !options.output) {
    throw new Error('--shell-dir, --dataset-dir and --output are required.');
  }
  if (!['Point60', 'Region66'].includes(options.mode)) throw new Error('--mode must be Point60 or Region66.');
  if (options.mode === 'Region66' && !options.regionDatasetDir) {
    throw new Error('Region66 requires --region-dataset-dir with subpose camera arrays.');
  }
  for (const [name, value] of [
    ['limit', options.limit], ['warmup-count', options.warmupCount], ['width', options.width], ['height', options.height],
  ]) {
    if (!Number.isInteger(value) || value < 0 || (name === 'width' && value < 8) || (name === 'height' && value < 8)) {
      throw new Error(`--${name} is invalid.`);
    }
  }
  if (!(options.aspect > 0) || !(options.near >= 0) || !(options.far > options.near)
      || !(options.depthBiasM >= 0) || !(options.timeoutMs > 0)) throw new Error('camera/timing options are invalid.');
  if (!Number.isInteger(options.port) || options.port < 0 || options.port > 65535) throw new Error('--port is invalid.');
  return options;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function readPoseRecord(buffer, index) {
  const offset = index * POSE_STRIDE_BYTES;
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return {
    position: [view.getFloat32(offset + 12, true), view.getFloat32(offset + 16, true), view.getFloat32(offset + 20, true)],
    forward: [view.getFloat32(offset + 24, true), view.getFloat32(offset + 28, true), view.getFloat32(offset + 32, true)],
    split: view.getUint8(offset + 44),
    category: view.getUint8(offset + 45),
  };
}

function readVec3File(filePath, count) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.byteLength !== count * 3 * 4) throw new Error(`${filePath} has an invalid vec3 length.`);
  const values = new Float32Array(buffer.buffer, buffer.byteOffset, count * 3);
  return values;
}

function readU64File(filePath, count) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.byteLength !== count * 8) throw new Error(`${filePath} has an invalid uint64 length.`);
  return new BigUint64Array(buffer.buffer, buffer.byteOffset, count);
}

function readU32File(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.byteLength % 4 !== 0) throw new Error(`${filePath} has an invalid uint32 length.`);
  return new Uint32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

function requireFile(directory, name) {
  const file = path.join(directory, name);
  if (!fs.existsSync(file)) throw new Error(`missing dataset file: ${file}`);
  return file;
}

function buildPointRows(options, meta, poseBuffer, candidateOffsets, candidateIds, queryCenters) {
  const splitId = Number(meta.splitIds?.[options.split]);
  if (!Number.isInteger(splitId)) throw new Error(`dataset_meta.json has no split ${options.split}.`);
  const rows = [];
  for (let poseId = 0; poseId < Number(meta.poseCount); poseId += 1) {
    const pose = readPoseRecord(poseBuffer, poseId);
    if (pose.split !== splitId) continue;
    const sourceStart = Number(candidateOffsets[poseId]);
    const sourceEnd = Number(candidateOffsets[poseId + 1]);
    const source = candidateIds.subarray(sourceStart, sourceEnd);
    const position = queryCenters
      ? Array.from(queryCenters.subarray(poseId * 3, poseId * 3 + 3))
      : pose.position;
    rows.push({
      poseId,
      position,
      forward: pose.forward,
      fovYDeg: options.fovYDeg,
      aspect: options.aspect,
      near: options.near,
      far: options.far,
      sourceCandidateIds: source,
      category: pose.category,
    });
    if (options.limit > 0 && rows.length >= options.limit) break;
  }
  return rows;
}

function buildRegionRows(options, meta, poseBuffer, candidateOffsets, candidateIds, queryCenters, regionDir) {
  const pointRows = buildPointRows({ ...options, limit: 0 }, meta, poseBuffer, candidateOffsets, candidateIds, queryCenters);
  const regionMeta = readJson(requireFile(regionDir, 'dataset_meta.json'));
  const regionPoseCount = Number(regionMeta.viewcellCount ?? regionMeta.poseCount);
  const offsetsPath = path.join(regionDir, 'subpose_offsets.bin');
  const positionsPath = path.join(regionDir, 'subpose_camera_pos.bin');
  const forwardsPath = path.join(regionDir, 'subpose_camera_forward.bin');
  if (!fs.existsSync(offsetsPath) || !fs.existsSync(positionsPath) || !fs.existsSync(forwardsPath)) {
    throw new Error('Region66 requires subpose_offsets.bin, subpose_camera_pos.bin and subpose_camera_forward.bin.');
  }
  const offsets = readU64File(offsetsPath, regionPoseCount + 1);
  const positions = readVec3File(positionsPath, Number(offsets[regionPoseCount]));
  const forwards = readVec3File(forwardsPath, Number(offsets[regionPoseCount]));
  const rows = [];
  for (const row of pointRows) {
    if (row.poseId >= regionPoseCount) throw new Error(`Region66 pose ${row.poseId} is outside region dataset.`);
    const start = Number(offsets[row.poseId]);
    const end = Number(offsets[row.poseId + 1]);
    const subposes = [];
    for (let index = start; index < end; index += 1) {
      subposes.push({
        position: Array.from(positions.subarray(index * 3, index * 3 + 3)),
        forward: Array.from(forwards.subarray(index * 3, index * 3 + 3)),
        fovYDeg: options.regionFovYDeg,
        aspect: options.aspect,
        near: options.near,
        far: options.far,
      });
    }
    rows.push({ ...row, subposes, fovYDeg: options.regionFovYDeg });
    if (options.limit > 0 && rows.length >= options.limit) break;
  }
  return rows;
}

function buildWorkload(options) {
  const datasetMeta = readJson(requireFile(options.datasetDir, 'dataset_meta.json'));
  const shellMeta = readJson(requireFile(options.shellDir, 'shell_meta.json'));
  if (shellMeta.schema !== 'geometry-shell-hzb-v1') throw new Error('shell directory has an unsupported schema.');
  if (Number(datasetMeta.numInstances) !== Number(shellMeta.instanceCount)) {
    throw new Error('dataset and geometry shell instance counts disagree.');
  }
  const poseCount = Number(datasetMeta.poseCount);
  const poseBytes = fs.readFileSync(requireFile(options.datasetDir, 'poses.bin'));
  if (poseBytes.byteLength !== poseCount * POSE_STRIDE_BYTES) throw new Error('poses.bin length does not match dataset_meta.json.');
  const poseBuffer = new Uint8Array(poseBytes.buffer, poseBytes.byteOffset, poseBytes.byteLength);
  const candidateOffsets = readU64File(requireFile(options.datasetDir, 'candidate_offsets.bin'), poseCount + 1);
  const candidateIds = readU32File(requireFile(options.datasetDir, 'candidate_ids.bin'));
  if (Number(candidateOffsets[0]) !== 0 || Number(candidateOffsets.at(-1)) !== candidateIds.length) {
    throw new Error('candidate CSR offsets are invalid.');
  }
  const queryCentersPath = path.join(options.datasetDir, 'query_center_world.bin');
  const queryCenters = fs.existsSync(queryCentersPath) ? readVec3File(queryCentersPath, poseCount) : null;
  const rows = options.mode === 'Region66'
    ? buildRegionRows(options, datasetMeta, poseBuffer, candidateOffsets, candidateIds, queryCenters, options.regionDatasetDir)
    : buildPointRows(options, datasetMeta, poseBuffer, candidateOffsets, candidateIds, queryCenters);
  if (rows.length === 0) throw new Error('selected split has no poses.');
  const outputDir = path.dirname(options.output);
  fs.mkdirSync(outputDir, { recursive: true });
  const workloadCandidates = [];
  let cursor = 0;
  const workloadRows = rows.map((row, ordinal) => {
    const ids = row.sourceCandidateIds;
    workloadCandidates.push(ids);
    const output = {
      ordinal,
      poseId: row.poseId,
      position: row.position,
      forward: row.forward,
      fovYDeg: row.fovYDeg,
      aspect: row.aspect,
      near: row.near,
      far: row.far,
      candidateOffset: cursor,
      candidateCount: ids.length,
      category: row.category,
    };
    if (row.subposes) output.subposes = row.subposes;
    cursor += ids.length;
    return output;
  });
  const flat = new Uint32Array(cursor);
  let flatOffset = 0;
  for (const ids of workloadCandidates) {
    flat.set(ids, flatOffset);
    flatOffset += ids.length;
  }
  const candidateFile = 'geometry_shell_hzb_candidates_uint32.bin';
  fs.writeFileSync(path.join(outputDir, candidateFile), Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
  const workload = {
    schema: 'geometry-shell-hzb-browser-workload-v1',
    scene: shellMeta.sceneName,
    split: options.split,
    mode: options.mode,
    poseCount: workloadRows.length,
    candidateCount: flat.length,
    candidateFile,
    candidateDtype: 'uint32-little-endian',
    width: options.width,
    height: options.height,
    aspect: options.aspect,
    fovYDeg: options.mode === 'Region66' ? options.regionFovYDeg : options.fovYDeg,
    near: options.near,
    far: options.far,
    source: {
      datasetDirName: path.basename(options.datasetDir),
      regionDatasetDirName: options.regionDatasetDir ? path.basename(options.regionDatasetDir) : null,
      candidateSemantics: datasetMeta.candidateSemantics || null,
    },
    poses: workloadRows,
  };
  const workloadFile = path.join(outputDir, 'geometry_shell_hzb_workload.json');
  fs.writeFileSync(workloadFile, `${JSON.stringify(workload, null, 2)}\n`, 'utf8');
  return { workload, workloadFile, outputDir, shellMeta };
}

function findChrome(explicit) {
  const candidates = [
    explicit,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const candidate of candidates) if (fs.existsSync(candidate)) return candidate;
  throw new Error('Chrome/Chromium executable not found.');
}

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    return import(`file://${path.join(VIEWER_ROOT, 'node_modules', 'playwright', 'index.mjs')}`);
  }
}

function runNvidiaSmi(args) {
  try {
    return { available: true, text: execFileSync('nvidia-smi', args, { encoding: 'utf8', timeout: 10000 }) };
  } catch (error) {
    return { available: false, error: String(error?.message || error) };
  }
}

function hostGpuEvidence() {
  return {
    nvidiaSmi: runNvidiaSmi([
      '--query-gpu=index,name,driver_version,utilization.gpu,memory.used',
      '--format=csv,noheader,nounits',
    ]),
    nvidiaSmiPmon: runNvidiaSmi(['pmon', '-c', '1']),
  };
}

function competingComputeProcesses(...evidence) {
  const processes = [];
  const seen = new Set();
  for (const sample of evidence) {
    const text = sample?.nvidiaSmiPmon?.text || '';
    for (const line of text.split(/\r?\n/)) {
      const fields = line.trim().split(/\s+/);
      if (fields.length < 8 || !/^\d+$/.test(fields[0]) || !/^\d+$/.test(fields[1])) continue;
      const type = fields[2];
      if (!type.includes('C')) continue;
      const command = fields.slice(fields.length >= 10 ? 9 : 7).join(' ');
      if (/chrome|chromium|xorg|nvidia-smi/i.test(command)) continue;
      const key = `${fields[0]}:${fields[1]}:${command}`;
      if (seen.has(key)) continue;
      seen.add(key);
      processes.push({ gpu: Number(fields[0]), pid: Number(fields[1]), type, command });
    }
  }
  return processes;
}

function startPmonSampler() {
  let output = '';
  let errorOutput = '';
  let spawnError = null;
  let closed = false;
  const child = spawn('nvidia-smi', ['pmon', '-s', 'um', '-d', '1'], { stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.on('data', (chunk) => { output += chunk.toString(); });
  child.stderr.on('data', (chunk) => { errorOutput += chunk.toString(); });
  child.on('error', (error) => { spawnError = String(error?.message || error); });
  child.on('close', () => { closed = true; });
  return {
    stop: async () => new Promise((resolve) => {
      if (closed) {
        resolve({ available: !spawnError, error: spawnError, text: output, errorText: errorOutput, sampleCount: output.split(/\r?\n/).filter((line) => /^\s*\d+\s+\S+/.test(line)).length });
        return;
      }
      child.once('close', () => resolve({
        available: !spawnError,
        error: spawnError,
        text: output,
        errorText: errorOutput,
        sampleCount: output.split(/\r?\n/).filter((line) => /^\s*\d+\s+\S+/.test(line)).length,
      }));
      child.kill('SIGTERM');
      setTimeout(() => { if (!closed) child.kill('SIGKILL'); }, 2000);
    }),
  };
}

function captureChromeProcess(browserServer) {
  const child = browserServer?.process?.() || null;
  let stdout = '';
  let stderr = '';
  child?.stdout?.on('data', (chunk) => { stdout += chunk.toString(); });
  child?.stderr?.on('data', (chunk) => { stderr += chunk.toString(); });
  return {
    available: Boolean(child?.stdout && child?.stderr),
    pid: child?.pid || null,
    snapshot: () => ({
      available: Boolean(child?.stdout && child?.stderr),
      pid: child?.pid || null,
      exitCode: child?.exitCode ?? null,
      signalCode: child?.signalCode ?? null,
      stdout,
      stderr,
    }),
  };
}

function mimeType(file) {
  return {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.bin': 'application/octet-stream',
  }[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

function startStaticServer(bundleDir, mappings, requestedPort) {
  const resolvedBundle = path.resolve(bundleDir);
  const routes = mappings.map(([prefix, root]) => [prefix, path.resolve(root)]);
  const server = http.createServer((request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (url.pathname === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
      }
      let root = resolvedBundle;
      let relative = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'geometry_shell_hzb_benchmark.html';
      for (const [prefix, mappedRoot] of routes) {
        if (url.pathname === prefix.slice(0, -1) || url.pathname.startsWith(prefix)) {
          root = mappedRoot;
          relative = decodeURIComponent(url.pathname.slice(prefix.length)).replace(/^\/+/, '');
          break;
        }
      }
      const file = path.resolve(root, relative);
      if (file !== root && !file.startsWith(`${root}${path.sep}`)) throw new Error('path traversal');
      const stat = fs.statSync(file);
      const served = stat.isDirectory() ? path.join(file, 'index.html') : file;
      response.writeHead(200, { 'Content-Type': mimeType(served), 'Cache-Control': 'no-store' });
      fs.createReadStream(served).pipe(response);
    } catch {
      response.writeHead(404);
      response.end('Not found');
    }
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(requestedPort, '127.0.0.1', () => resolve(server));
  });
}

function buildBundle(bundleDir) {
  const parcel = path.join(VIEWER_ROOT, 'node_modules', '.bin', 'parcel');
  if (!fs.existsSync(parcel)) throw new Error(`${parcel} is missing; run npm ci in slm2viewer first.`);
  const result = spawnSync(parcel, [
    'build', 'geometry_shell_hzb_benchmark.html', '--no-cache', '--no-source-maps', '--no-minify', '--out-dir', bundleDir,
  ], { cwd: VIEWER_ROOT, encoding: 'utf8', timeout: 10 * 60 * 1000 });
  if (result.status !== 0) throw new Error(`Parcel HZB benchmark build failed:\n${result.stdout}\n${result.stderr}`);
}

function classifyBackend(info) {
  const values = [info?.vendor, info?.architecture, info?.device, info?.description, info?.renderer, info?.version];
  const gate = classifyGpuText(values);
  return { ...info, hardware: Boolean(values.join(' ').trim()) && !SOFTWARE_PATTERN.test(values.join(' ')), ...gate };
}

async function main() {
  const options = parseArgs(process.argv);
  const workloadInfo = buildWorkload(options);
  const bundleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'geometry-shell-hzb-bundle-'));
  buildBundle(bundleDir);
  const server = await startStaticServer(bundleDir, [
    ['/shell/', options.shellDir],
    ['/workload/', workloadInfo.outputDir],
  ], options.port);
  const port = server.address().port;
  const chrome = findChrome(options.chromeExe);
  const { chromium } = await loadPlaywright();
  const launchArgs = headlessWebGpuArgs({ screenSize: '1280,720', quietBrowser: true });
  const launchEnvironment = resolveVulkanEnvironment();
  const hostGpuBefore = hostGpuEvidence();
  const pmon = startPmonSampler();
  let browser = null;
  let browserServer = null;
  let chromeProcess = null;
  let pageConsole = [];
  let pageErrors = [];
  let result = null;
  let failure = null;
  try {
    browserServer = await chromium.launchServer({
      executablePath: chrome,
      headless: true,
      args: launchArgs,
      env: launchEnvironment,
      dumpio: false,
    });
    chromeProcess = captureChromeProcess(browserServer);
    browser = await chromium.connect(browserServer.wsEndpoint());
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', (message) => {
      const entry = { type: message.type(), text: message.text() };
      pageConsole.push(entry);
      console.log(`[geometry-shell-hzb-page:${entry.type}] ${entry.text}`);
    });
    page.on('pageerror', (error) => {
      const entry = String(error.stack || error);
      pageErrors.push(entry);
      console.error(`[geometry-shell-hzb-pageerror] ${entry}`);
    });
    await page.goto(`http://127.0.0.1:${port}/geometry_shell_hzb_benchmark.html`, {
      waitUntil: 'domcontentloaded',
      timeout: options.timeoutMs,
    });
    await page.waitForFunction(() => window.__geometryShellHZBReady === true, null, { timeout: options.timeoutMs });
    result = await page.evaluate(async (config) => window.runGeometryShellHZBBenchmark(config), {
      assetBaseUrl: `http://127.0.0.1:${port}/shell/`,
      workloadUrl: `http://127.0.0.1:${port}/workload/geometry_shell_hzb_workload.json`,
      mode: options.mode,
      width: options.width,
      height: options.height,
      fovYDeg: options.fovYDeg,
      near: options.near,
      far: options.far,
      depthBiasM: options.depthBiasM,
      warmupCount: options.warmupCount,
    });
  } catch (error) {
    failure = String(error?.stack || error);
  } finally {
    const hostGpuDuring = {
      nvidiaSmi: hostGpuEvidence().nvidiaSmi,
      nvidiaSmiPmon: await pmon.stop(),
    };
    await browser?.close().catch(() => {});
    await browserServer?.close().catch(() => {});
    const hostGpuAfter = hostGpuEvidence();
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(bundleDir, { recursive: true, force: true });
    const chromeProcessEvidence = chromeProcess?.snapshot() || {
      available: false,
      pid: null,
      exitCode: null,
      signalCode: null,
      stdout: '',
      stderr: '',
    };
    const competingProcesses = competingComputeProcesses(hostGpuBefore, hostGpuDuring);
    const concurrentGpuWork = competingProcesses.length > 0;
    const adapterGate = classifyBackend(result?.gpuBackend || result?.adapterInfo || {});
    const webglGate = classifyBackend(result?.webglInfo || {});
    const gpuGate = {
      required: options.requireHardwareGpu,
      hardware: adapterGate.hardware && webglGate.hardware,
      adapter: adapterGate,
      webgl: webglGate,
    };
    const runtimeHealthy = Boolean(
      result && pageErrors.length === 0 && (result.gpuValidationErrors || []).length === 0,
    );
    const evidenceReady = hostGpuBefore.nvidiaSmi.available
      && hostGpuBefore.nvidiaSmiPmon.available
      && hostGpuDuring.nvidiaSmi.available
      && hostGpuDuring.nvidiaSmiPmon.available
      && hostGpuDuring.nvidiaSmiPmon.sampleCount > 0
      && hostGpuAfter.nvidiaSmi.available
      && hostGpuAfter.nvidiaSmiPmon.available
      && chromeProcessEvidence.available;
    const formalReady = Boolean(
      options.requireHardwareGpu && !failure && runtimeHealthy && gpuGate.hardware && evidenceReady && !concurrentGpuWork,
    );
    if (!failure && options.requireHardwareGpu && !gpuGate.hardware) {
      failure = 'Geometry-shell HZB WebGPU/WebGL hardware GPU gate failed.';
    } else if (!failure && options.requireHardwareGpu && !runtimeHealthy) {
      failure = 'Geometry-shell HZB browser runtime reported a page or WebGPU validation error.';
    } else if (!failure && options.requireHardwareGpu && !formalReady && !concurrentGpuWork) {
      failure = 'Geometry-shell HZB hardware GPU gate or execution evidence failed.';
    }
    const executionClass = !options.requireHardwareGpu
      ? 'software-debug-only'
      : !gpuGate.hardware
        ? 'hardware-gate-failed'
        : failure
          ? 'hardware-gate-failed'
          : concurrentGpuWork
            ? 'hardware-smoke-concurrent'
            : formalReady
              ? 'formal-hardware-gpu'
              : 'hardware-gate-failed';
    const capture = {
      ...(result || { schema: 'geometry-shell-hzb-browser-result-v1', mode: options.mode }),
      capturedAt: new Date().toISOString(),
      shellDir: options.shellDir,
      datasetDir: options.datasetDir,
      regionDatasetDir: options.regionDatasetDir || null,
      output: options.output,
      gpuGate,
      formalReady,
      executionClass,
      gpuConcurrency: {
        concurrentComputeDetected: concurrentGpuWork,
        competingProcesses,
      },
      browserPageConsole: pageConsole,
      browserPageErrors: pageErrors,
      chromeProcess: chromeProcessEvidence,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter,
      browserLaunch: {
        executablePath: chrome,
        headless: true,
        dumpio: false,
        transport: 'launchServer+connect',
        args: launchArgs,
        vulkanIcd: launchEnvironment.VK_ICD_FILENAMES || null,
      },
      error: failure,
    };
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, `${JSON.stringify(capture, null, 2)}\n`, 'utf8');
    fs.writeFileSync(path.join(path.dirname(options.output), 'geometry_shell_hzb_gpu_evidence.json'), `${JSON.stringify({
      schema: 'geometry-shell-hzb-gpu-evidence-v1',
      gpuBackend: result?.gpuBackend || null,
      formalReady,
      gpuGate,
      browserLaunch: capture.browserLaunch,
      chromeProcess: chromeProcessEvidence,
      gpuConcurrency: capture.gpuConcurrency,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter,
      error: failure,
    }, null, 2)}\n`, 'utf8');
  }
  if (failure) throw new Error(failure);
  console.log(JSON.stringify({
    output: options.output,
    mode: result.mode,
    poseCount: result.samples.length,
    candidateCount: result.workload.candidateCount,
    summary: result.summary,
    gpuBackend: result.gpuBackend,
    adapterInfo: result.adapterInfo,
    webglInfo: result.webglInfo,
  }, null, 2));
}

main().catch((error) => {
  console.error(error?.stack || error);
  process.exitCode = 1;
});
