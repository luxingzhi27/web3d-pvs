#!/usr/bin/env node
/* Capture one prediction from the deployed V4 frontend for PyTorch parity checking. */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { headlessWebGpuArgs, resolveVulkanEnvironment } from './chrome_gpu_flags.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = path.resolve(SCRIPT_DIR, '..');
const SOFTWARE_PATTERN = /swiftshader|llvmpipe|softpipe|swrast|software/i;

function parseArgs(argv) {
  const options = {
    viewerDir: path.join(VIEWER_ROOT, 'public'),
    out: null,
    scene: 'hkust-v3',
    chromeExe: process.env.CHROME_PATH || process.env.CHROMIUM_PATH || null,
    port: 0,
    timeoutMs: 180000,
    requireHardwareGpu: true,
  };
  const valueOptions = new Set([
    'viewer-dir', 'out', 'scene', 'chrome-exe', 'port', 'timeout-ms',
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
    if (key === 'viewer-dir') options.viewerDir = path.resolve(value);
    else if (key === 'out') options.out = path.resolve(value);
    else if (key === 'scene') options.scene = String(value);
    else if (key === 'chrome-exe') options.chromeExe = path.resolve(value);
    else if (key === 'port') options.port = Number(value);
    else if (key === 'timeout-ms') options.timeoutMs = Number(value);
  }
  if (!options.out) throw new Error('--out is required');
  if (!Number.isInteger(options.port) || options.port < 0 || options.port > 65535) {
    throw new Error('--port must be an integer between 0 and 65535');
  }
  if (!(options.timeoutMs > 0)) throw new Error('--timeout-ms must be positive');
  if (!fs.existsSync(path.join(options.viewerDir, 'index.html'))) {
    throw new Error(`Built viewer not found at ${options.viewerDir}; run npm run build first`);
  }
  return options;
}

function findChrome(explicit) {
  const candidates = [
    explicit,
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error('Chrome executable not found');
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
    return {
      available: true,
      text: execFileSync('nvidia-smi', args, { encoding: 'utf8', timeout: 10000 }),
    };
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

function startPmonSampler() {
  let output = '';
  let errorOutput = '';
  let spawnError = null;
  let closed = false;
  let stopPromise = null;
  const child = spawn('nvidia-smi', ['pmon', '-s', 'um', '-d', '1'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (chunk) => { output += chunk.toString(); });
  child.stderr.on('data', (chunk) => { errorOutput += chunk.toString(); });
  child.on('error', (error) => { spawnError = String(error?.message || error); });
  child.on('close', () => { closed = true; });

  const finish = () => ({
    available: !spawnError,
    error: spawnError,
    exitText: errorOutput.trim() || null,
    sampleCount: output.split(/\r?\n/).filter((line) => /^\s*\d+\s+\S+/.test(line)).length,
    text: output,
  });

  return {
    stop: async () => {
      if (stopPromise) return stopPromise;
      if (closed) return finish();
      stopPromise = new Promise((resolve) => {
        child.once('close', () => resolve(finish()));
        child.kill('SIGTERM');
        setTimeout(() => {
          if (!closed) child.kill('SIGKILL');
        }, 2000);
      });
      return stopPromise;
    },
  };
}

function classifyBackend(info) {
  const text = [info?.vendor, info?.architecture, info?.device, info?.description]
    .filter(Boolean)
    .join(' ');
  return {
    ...info,
    hardware: Boolean(text) && !SOFTWARE_PATTERN.test(text),
    softwareMarkers: SOFTWARE_PATTERN.test(text) ? text : null,
  };
}

function mimeType(file) {
  const extension = path.extname(file).toLowerCase();
  return {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.bin': 'application/octet-stream',
    '.glb': 'model/gltf-binary',
    '.wasm': 'application/wasm',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
  }[extension] || 'application/octet-stream';
}

function startStaticServer(root, requestedPort) {
  const resolvedRoot = path.resolve(root);
  const server = http.createServer((request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      const relative = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'index.html';
      const file = path.resolve(resolvedRoot, relative);
      if (file !== resolvedRoot && !file.startsWith(`${resolvedRoot}${path.sep}`)) {
        response.writeHead(403);
        response.end();
        return;
      }
      const stat = fs.statSync(file);
      const servedFile = stat.isDirectory() ? path.join(file, 'index.html') : file;
      response.writeHead(200, {
        'Content-Type': mimeType(servedFile),
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(servedFile).pipe(response);
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

async function capturePage(page, timeoutMs) {
  await page.waitForFunction(
    () => window.__slmApp?.viewer?.slm2Loader?.neuralPVS?.isReady === true,
    null,
    { timeout: timeoutMs },
  );
  await page.evaluate(() => {
    const loader = window.__slmApp.viewer.slm2Loader;
    loader.lastNeuralPrediction = null;
    loader.forceSceneCullingNow({ forceNeural: true });
  });
  await page.waitForFunction(
    () => {
      const prediction = window.__slmApp?.viewer?.slm2Loader?.lastNeuralPrediction;
      return prediction?.candidateInstanceIds?.length > 0
        && prediction.candidateScores?.length === prediction.candidateInstanceIds.length;
    },
    null,
    { timeout: timeoutMs },
  );

  return page.evaluate(async () => {
    const viewer = window.__slmApp.viewer;
    const loader = viewer.slm2Loader;
    const prediction = loader.lastNeuralPrediction;
    const camera = loader.activeCamera || viewer.activeCamera;
    camera.updateMatrixWorld(true);
    const world = camera.matrixWorld.elements;
    const forward = [-world[8], -world[9], -world[10]];
    const singleton = globalThis.__SLM_INSTANCE_PVS_WEBGPU_SINGLETON__ || {};
    const adapter = singleton.adapter
      || await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const adapterInfo = adapter?.info || {};
    const canvas = document.querySelector('canvas');
    const gl = canvas?.getContext('webgl2') || canvas?.getContext('webgl');
    const rendererInfo = gl?.getExtension('WEBGL_debug_renderer_info');

    return {
      schema: 'pvs-v4-frontend-parity-capture-v1',
      backend: prediction.backend,
      camera: {
        position: camera.position.toArray(),
        quaternion: camera.quaternion.toArray(),
        forward,
        aspect: camera.aspect,
        fov: camera.fov,
        near: camera.near,
        far: camera.far,
      },
      candidateInstanceIds: Array.from(prediction.candidateInstanceIds),
      candidateScores: Array.from(prediction.candidateScores),
      candidateDiagnostics: Array.from(prediction.candidateDiagnostics || []),
      diagnosticOutputFloats: Number(prediction.diagnosticOutputFloats || 0),
      visibleInstanceIds: Array.from(prediction.componentModelList || []),
      renderVisibleInstanceIds: Array.from(prediction.renderComponentModelList || []),
      adapterInfo: {
        vendor: String(adapterInfo.vendor || ''),
        architecture: String(adapterInfo.architecture || ''),
        device: String(adapterInfo.device || ''),
        description: String(adapterInfo.description || ''),
      },
      webglInfo: {
        vendor: String(gl && rendererInfo ? gl.getParameter(rendererInfo.UNMASKED_VENDOR_WEBGL) : ''),
        description: String(gl && rendererInfo ? gl.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL) : ''),
        version: String(gl ? gl.getParameter(gl.VERSION) : ''),
      },
      modelInfo: loader.neuralPVS?.modelInfo || null,
      predictTimings: loader.neuralPVS?.lastPredictTimings || null,
    };
  });
}

async function main() {
  const options = parseArgs(process.argv);
  fs.mkdirSync(path.dirname(options.out), { recursive: true });
  const chrome = findChrome(options.chromeExe);
  const { chromium } = await loadPlaywright();
  const launchArgs = headlessWebGpuArgs({
    screenSize: '1280,720',
    quietBrowser: true,
  });
  const launchEnvironment = resolveVulkanEnvironment();
  const hostGpuBefore = hostGpuEvidence();
  const pmon = startPmonSampler();
  const server = await startStaticServer(options.viewerDir, options.port);
  const port = server.address().port;
  let browser = null;
  let result = null;
  let chromeGpuInfo = null;
  let failure = null;
  try {
    browser = await chromium.launch({
      executablePath: chrome,
      headless: true,
      args: launchArgs,
      env: launchEnvironment,
    });
    try {
      const cdp = await browser.newBrowserCDPSession();
      chromeGpuInfo = await cdp.send('SystemInfo.getInfo');
      await cdp.detach();
    } catch (error) {
      chromeGpuInfo = { error: String(error?.stack || error) };
    }
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('pageerror', (error) => console.error('[v4-parity-pageerror]', error));
    await page.goto(
      `http://127.0.0.1:${port}/?scene=${encodeURIComponent(options.scene)}&neuralDebugLogs=true&showGUI=false`,
      { waitUntil: 'domcontentloaded', timeout: options.timeoutMs },
    );
    result = await capturePage(page, options.timeoutMs);
  } catch (error) {
    failure = String(error?.stack || error);
  } finally {
    const hostGpuDuring = {
      nvidiaSmi: hostGpuEvidence().nvidiaSmi,
      nvidiaSmiPmon: await pmon.stop(),
    };
    const hostGpuAfter = hostGpuEvidence();
    if (browser) await browser.close().catch(() => {});
    await new Promise((resolve) => server.close(resolve));

    const adapterGate = classifyBackend(result?.adapterInfo || {});
    const webglGate = classifyBackend({
      vendor: result?.webglInfo?.vendor || '',
      architecture: '',
      device: '',
      description: result?.webglInfo?.description || '',
    });
    const gpuGate = {
      required: options.requireHardwareGpu,
      hardware: adapterGate.hardware && webglGate.hardware,
      adapter: adapterGate,
      webgl: webglGate,
    };
    const formalReady = Boolean(
      !failure
      && result?.backend === 'worker-webgpu-v4'
      && gpuGate.hardware
      && hostGpuBefore.nvidiaSmi.available
      && hostGpuBefore.nvidiaSmiPmon.available
      && hostGpuDuring.nvidiaSmi.available
      && hostGpuDuring.nvidiaSmiPmon.available
      && hostGpuDuring.nvidiaSmiPmon.sampleCount > 0
      && hostGpuAfter.nvidiaSmi.available
      && hostGpuAfter.nvidiaSmiPmon.available
    );
    const capture = {
      ...(result || { schema: 'pvs-v4-frontend-parity-capture-v1', backend: null }),
      capturedAt: new Date().toISOString(),
      viewerDir: options.viewerDir,
      scene: options.scene,
      gpuGate,
      formalReady,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter,
      browserLaunch: {
        executablePath: chrome,
        headless: true,
        args: launchArgs,
        vulkanIcd: launchEnvironment.VK_ICD_FILENAMES || null,
        chromeGpuInfo,
      },
      error: failure,
    };
    fs.writeFileSync(options.out, `${JSON.stringify(capture, null, 2)}\n`);
    if (!failure && options.requireHardwareGpu && !formalReady) {
      failure = 'WebGPU hardware gate failed';
    }
  }

  if (failure) throw new Error(failure);
  console.log(JSON.stringify({
    output: options.out,
    candidateCount: result.candidateInstanceIds.length,
    visibleCount: result.visibleInstanceIds.length,
    renderVisibleCount: result.renderVisibleInstanceIds.length,
    gpuGate: classifyBackend(result.adapterInfo),
  }, null, 2));
}

main().catch((error) => {
  console.error(error?.stack || error);
  process.exitCode = 1;
});
