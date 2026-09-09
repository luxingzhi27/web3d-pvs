#!/usr/bin/env node
/* 用途：浏览器/Three.js GPU Color-ID 备用采样器。
 *
 * 当前正式采样优先使用 neural_instance_culling/rvc_sampler。
 * 本脚本会加载场景 GLB，用颜色 ID 离屏渲染统计每个相机位置下可见的实例集合。
 *
 * 典型运行：
 *   node neural_instance_culling/sampler/run_sampler.mjs \
 *     --assets-dir hkust-v3/assets \
 *     --output neural_instance_culling/sampler/out/instance_vis_samples.jsonl
 *
 * 重要参数：
 *   --grid-step 控制 XZ 采样密度；--width/--height 控制离屏 tile 分辨率；
 *   --yaws/--pitches 控制每个位置采样多少方向；--smoke 用于小规模测试。
 *   默认模型采样严格使用 66 度；--point60-gt 只用于 canonical subpose=0 的
 *   真实 60 度 Color-ID GT，并强制使用硬件 GPU 证据。
 *   默认要求浏览器回报硬件 WebGL 后端；只有显式传入 --allow-software-gpu
 *   才允许软件后端进行非正式语义调试。
 *   完整参数表见 INSTANCE_PVS_SCENE_MIGRATION_GUIDE.md 的“14.2 GPU Color-ID 备用采样器参数”。
 */
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import readline from 'node:readline';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { MODEL_FOV_Y_DEG } from './neuralpvs_fov_protocol.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const POINT60_GT_FOV_Y_DEG = 60;
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const WEB_ROOT = path.join(__dirname, 'web');
const DEFAULT_ASSETS_DIR = path.join(REPO_ROOT, 'hkust-v3', 'assets');
const NODE_MODULES_DIR = path.join(REPO_ROOT, 'slm2viewer', 'node_modules');
const SYSTEM_CHROME_CANDIDATES = [
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
];

function parseArgs(argv) {
  const args = {
    assetsDir: DEFAULT_ASSETS_DIR,
    output: path.join(__dirname, 'out', 'instance_vis_samples.jsonl'),
    posePlan: '',
    glbIdList: '',
    port: 0,
    width: 256,
    height: 144,
    fovYDeg: MODEL_FOV_Y_DEG,
    fovYExplicit: false,
    point60Gt: false,
    poseStart: 0,
    poseCount: 0,
    gridStep: 25,
    roadHeights: [2, 5, 10, 15, 25],
    aerialHeights: [40, 60, 90, 120, 150],
    farHeights: [30, 80, 130],
    farDistances: [250, 500, 1000, 2000],
    yaws: Array.from({ length: 12 }, (_v, i) => i * 30),
    pitches: [-25, 0, 20],
    maxPositions: null,
    maxGlbs: null,
    loadConcurrency: 64,
    occupiedOnly: true,
    occupiedExpandCells: 1,
    coarseGridStep: null,
    includeCoarseGlobal: true,
    heightMode: 'world',
    atlasColumns: 12,
    atlasRows: 8,
    directionTilesPerAtlas: 96,
    cameraBoundsExpandRatio: 0.1,
    cameraMinY: -20,
    cameraMaxY: 180,
    roadSearchCells: 6,
    roadMinNeighbors: 2,
    headless: true,
    smoke: false,
    requireHardwareGpu: true,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    if (key === '--headed') {
      args.headless = false;
      continue;
    }
    if (key === '--full-bounds') {
      args.occupiedOnly = false;
      continue;
    }
    if (key === '--no-coarse-global') {
      args.includeCoarseGlobal = false;
      continue;
    }
    if (key === '--smoke') {
      args.smoke = true;
      args.maxPositions = 4;
      args.maxGlbs = 8;
      args.gridStep = 500;
      args.directionTilesPerAtlas = 36;
      continue;
    }
    if (key === '--point60-gt') {
      if (args.fovYExplicit && Math.abs(Number(args.fovYDeg) - POINT60_GT_FOV_Y_DEG) > 1e-6) {
        throw new Error(`--point60-gt only supports a ${POINT60_GT_FOV_Y_DEG} degree FOV.`);
      }
      args.point60Gt = true;
      args.fovYDeg = POINT60_GT_FOV_Y_DEG;
      continue;
    }
    if (key === '--require-hardware-gpu' || key === '--no-allow-software-gpu') {
      args.requireHardwareGpu = true;
      continue;
    }
    if (key === '--allow-software-gpu' || key === '--no-require-hardware-gpu') {
      args.requireHardwareGpu = false;
      continue;
    }
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'assets-dir') args.assetsDir = path.resolve(value);
    else if (name === 'output') args.output = path.resolve(value);
    else if (name === 'pose-plan') args.posePlan = path.resolve(value);
    else if (name === 'glb-id-list') args.glbIdList = path.resolve(value);
    else if (name === 'port') args.port = Number(value);
    else if (name === 'width') args.width = Number(value);
    else if (name === 'height') args.height = Number(value);
    else if (name === 'fov-y-deg') {
      args.fovYExplicit = true;
      args.fovYDeg = Number(value);
    }
    else if (name === 'pose-start') args.poseStart = Number(value);
    else if (name === 'pose-count') args.poseCount = Number(value);
    else if (name === 'grid-step') args.gridStep = Number(value);
    else if (name === 'road-heights') args.roadHeights = value.split(',').map(Number).filter(Number.isFinite);
    else if (name === 'aerial-heights') args.aerialHeights = value.split(',').map(Number).filter(Number.isFinite);
    else if (name === 'far-heights') args.farHeights = value.split(',').map(Number).filter(Number.isFinite);
    else if (name === 'far-distances') args.farDistances = value.split(',').map(Number).filter(Number.isFinite);
    else if (name === 'yaws') args.yaws = value.split(',').map(Number).filter(Number.isFinite);
    else if (name === 'pitches') args.pitches = value.split(',').map(Number).filter(Number.isFinite);
    else if (name === 'max-positions') args.maxPositions = Number(value);
    else if (name === 'max-glbs') args.maxGlbs = Number(value);
    else if (name === 'load-concurrency') args.loadConcurrency = Number(value);
    else if (name === 'occupied-expand-cells') args.occupiedExpandCells = Number(value);
    else if (name === 'coarse-grid-step') args.coarseGridStep = Number(value);
    else if (name === 'height-mode') args.heightMode = value;
    else if (name === 'atlas-columns') args.atlasColumns = Number(value);
    else if (name === 'atlas-rows') args.atlasRows = Number(value);
    else if (name === 'direction-tiles-per-atlas') args.directionTilesPerAtlas = Number(value);
    else if (name === 'camera-bounds-expand-ratio') args.cameraBoundsExpandRatio = Number(value);
    else if (name === 'camera-min-y') args.cameraMinY = Number(value);
    else if (name === 'camera-max-y') args.cameraMaxY = Number(value);
    else if (name === 'road-search-cells') args.roadSearchCells = Number(value);
    else if (name === 'road-min-neighbors') args.roadMinNeighbors = Number(value);
  }
  const expectedFovY = args.point60Gt ? POINT60_GT_FOV_Y_DEG : MODEL_FOV_Y_DEG;
  if (Math.abs(Number(args.fovYDeg) - expectedFovY) > 1e-6) {
    if (args.point60Gt) {
      throw new Error(`--point60-gt only supports a ${POINT60_GT_FOV_Y_DEG} degree FOV.`);
    }
    throw new Error(`The current sampler only supports the model FOV of ${MODEL_FOV_Y_DEG} degrees.`);
  }
  if (args.point60Gt && !args.requireHardwareGpu) {
    throw new Error('--point60-gt requires the hardware GPU gate; software GPU mode is not allowed.');
  }
  return args;
}

function finitePlanVector(value, field, index, requireNonZero = false) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((item) => !Number.isFinite(Number(item)))) {
    throw new Error(`Point60 pose ${index} has an invalid ${field}.`);
  }
  if (requireNonZero && Math.hypot(...value.map(Number)) <= 1e-8) {
    throw new Error(`Point60 pose ${index} has a zero ${field}.`);
  }
}

function validatePoint60Plan(posePlan) {
  if (!Array.isArray(posePlan) || posePlan.length === 0) {
    throw new Error('--point60-gt requires a non-empty canonical --pose-plan.');
  }
  const seenViewcellIds = new Set();
  let viewport = null;
  for (let index = 0; index < posePlan.length; index += 1) {
    const pose = posePlan[index];
    const viewcellId = Number(pose?.viewcell_id);
    const subposeId = Number(pose?.subpose_id);
    if (!Number.isInteger(viewcellId) || viewcellId < 0) {
      throw new Error(`Point60 pose ${index} must have a non-negative integer viewcell_id.`);
    }
    if (pose?.pose_index !== undefined) {
      const poseIndex = Number(pose.pose_index);
      if (!Number.isInteger(poseIndex) || poseIndex < 0) {
        throw new Error(`Point60 pose ${index} has an invalid pose_index.`);
      }
    }
    if (seenViewcellIds.has(viewcellId)) {
      throw new Error(`Point60 pose plan contains duplicate viewcell_id ${viewcellId}.`);
    }
    seenViewcellIds.add(viewcellId);
    if (subposeId !== 0) {
      throw new Error(`Point60 pose ${index} must use canonical subpose_id=0.`);
    }
    finitePlanVector(pose.camera_pos, 'camera_pos');
    finitePlanVector(pose.camera_forward, 'camera_forward', true);

    const fovY = Number(pose.fov_y);
    if (!Number.isFinite(fovY) || Math.abs(fovY - POINT60_GT_FOV_Y_DEG) > 1e-6) {
      throw new Error(`Point60 pose ${index} must have fov_y=${POINT60_GT_FOV_Y_DEG}.`);
    }
    if (pose.render_fov_y !== undefined
      && Math.abs(Number(pose.render_fov_y) - POINT60_GT_FOV_Y_DEG) > 1e-6) {
      throw new Error(`Point60 pose ${index} must have render_fov_y=${POINT60_GT_FOV_Y_DEG}.`);
    }
    const aspect = Number(pose.aspect);
    const width = Number(pose.width);
    const height = Number(pose.height);
    if (!Number.isFinite(aspect) || aspect <= 0
      || !Number.isInteger(width) || width <= 0
      || !Number.isInteger(height) || height <= 0) {
      throw new Error(`Point60 pose ${index} must have a positive aspect, width, and height.`);
    }
    if (Math.max(1, Math.round(height * aspect)) !== width) {
      throw new Error(`Point60 pose ${index} width/height/aspect cannot reproduce one viewport.`);
    }
    if (viewport === null) {
      viewport = { width, height, aspect };
    } else if (viewport.width !== width || viewport.height !== height || Math.abs(viewport.aspect - aspect) > 1e-9) {
      throw new Error('--point60-gt requires one plan per exact width/height/aspect viewport group.');
    }
  }
  return viewport;
}

function validateModel66Plan(posePlan) {
  if (!Array.isArray(posePlan)) return;
  for (let index = 0; index < posePlan.length; index += 1) {
    const pose = posePlan[index];
    if (pose?.fov_y === undefined) continue;
    const fovY = Number(pose.fov_y);
    if (!Number.isFinite(fovY) || Math.abs(fovY - MODEL_FOV_Y_DEG) > 1e-6) {
      throw new Error(`The default sampler requires pose ${index} to use fov_y=${MODEL_FOV_Y_DEG}.`);
    }
  }
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.html') return 'text/html; charset=utf-8';
  if (ext === '.js' || ext === '.mjs') return 'text/javascript; charset=utf-8';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.glb') return 'model/gltf-binary';
  if (ext === '.wasm') return 'application/wasm';
  if (ext === '.bin') return 'application/octet-stream';
  return 'application/octet-stream';
}

function resolveServedPath(urlPath, assetsDir) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  if (decoded === '/' || decoded === '/index.html') return path.join(WEB_ROOT, 'index.html');
  if (decoded.startsWith('/sampler/')) return path.join(WEB_ROOT, decoded.slice('/sampler/'.length));
  if (decoded.startsWith('/assets/')) return path.join(assetsDir, decoded.slice('/assets/'.length));
  if (decoded.startsWith('/node_modules/')) return path.join(NODE_MODULES_DIR, decoded.slice('/node_modules/'.length));
  return null;
}

function isWithin(child, parent) {
  const rel = path.relative(parent, child);
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel);
}

function createServer(args) {
  const server = http.createServer((req, res) => {
    const filePath = resolveServedPath(req.url || '/', args.assetsDir);
    const roots = [WEB_ROOT, args.assetsDir, NODE_MODULES_DIR];
    if (!filePath || !roots.some((root) => isWithin(filePath, root) || filePath === root)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      res.writeHead(200, {
        'content-type': contentType(filePath),
        'cache-control': 'no-store',
        'access-control-allow-origin': '*',
      });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(args.port, '127.0.0.1', () => {
      const address = server.address();
      resolve({ server, url: `http://127.0.0.1:${address.port}` });
    });
  });
}

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch (err) {
    const localPlaywright = path.join(NODE_MODULES_DIR, 'playwright', 'index.mjs');
    if (fs.existsSync(localPlaywright)) return import(pathToFileURL(localPlaywright).href);
    throw new Error('Playwright is required for the sampler. Install it in this workspace or run through an environment that already provides playwright.');
  }
}

async function readJsonl(filePath, start = 0, count = 0) {
  if (!filePath) return null;
  const rows = [];
  const begin = Math.max(0, Math.floor(Number(start) || 0));
  const limit = Number.isFinite(Number(count)) && Number(count) > 0 ? Math.floor(Number(count)) : Infinity;
  const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
  let lineIndex = 0;
  for await (const rawLine of rl) {
    const line = rawLine.trim();
    if (!line) continue;
    if (lineIndex < begin) {
      lineIndex += 1;
      continue;
    }
    if (rows.length >= limit) break;
    try {
      rows.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`Failed to parse pose plan ${filePath}:${lineIndex + 1}: ${error.message}`);
    }
    lineIndex += 1;
  }
  rl.close();
  stream.destroy();
  return rows;
}

function readNumericIdList(filePath) {
  if (!filePath) return null;
  const values = fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map((line) => Number(line))
    .filter((value) => Number.isInteger(value) && value >= 0);
  return Array.from(new Set(values));
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
    software: softwarePattern.test(text),
  };
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

function gpuEvidencePath(outputPath) {
  return `${outputPath}.gpu_evidence.json`;
}

function writeGpuEvidence(outputPath, evidence) {
  fs.writeFileSync(gpuEvidencePath(outputPath), `${JSON.stringify(evidence, null, 2)}\n`, 'utf8');
}

async function main() {
  const args = parseArgs(process.argv);
  const posePlan = await readJsonl(args.posePlan, args.poseStart, args.poseCount);
  if (args.point60Gt && !args.posePlan) {
    throw new Error('--point60-gt requires --pose-plan; fallback poses are not formal Point60 GT.');
  }
  if (!args.point60Gt) validateModel66Plan(posePlan);
  const point60Viewport = args.point60Gt ? validatePoint60Plan(posePlan) : null;
  if (point60Viewport) {
    // The browser camera currently derives the render width from height*aspect.
    // Use the plan group dimensions so the evidence records the actual viewport.
    args.width = point60Viewport.width;
    args.height = point60Viewport.height;
  }
  const glbIdList = readNumericIdList(args.glbIdList);
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  const output = fs.createWriteStream(args.output, { encoding: 'utf8' });
  const { server, url } = await createServer(args);
  const { chromium } = await loadPlaywright();
  const executablePath = SYSTEM_CHROME_CANDIDATES.find((candidate) => fs.existsSync(candidate));
  const chromeArgs = [
    '--disable-dev-shm-usage',
    '--ignore-gpu-blocklist',
    '--enable-gpu',
    '--enable-webgl',
    '--use-angle=vulkan',
    '--enable-accelerated-2d-canvas',
    '--enable-zero-copy',
    ...(args.requireHardwareGpu ? ['--disable-software-rasterizer'] : []),
  ];
  const hostGpuBefore = captureHostGpuEvidence();
  let browser = null;
  let result = null;
  let hostGpuDuring = null;
  let hostGpuAfter = null;
  let evidence = null;
  let failure = null;
  try {
    if (args.requireHardwareGpu && !executablePath) {
      throw new Error('hardware GPU required, but no system Chrome executable was found');
    }
    browser = await chromium.launch({
    headless: args.headless,
    executablePath,
    args: chromeArgs,
    });
    const page = await browser.newPage({
    viewport: { width: Math.max(640, args.width), height: Math.max(360, args.height) },
    });
    page.on('console', (msg) => {
    const type = msg.type();
    console.log(`[browser:${type}] ${msg.text()}`);
    });
    page.on('pageerror', (err) => {
    console.error('[browser:pageerror]', err);
    });
    page.on('requestfailed', (request) => {
    const urlText = request.url();
    const errorText = request.failure()?.errorText || '';
    if (/\/assets\/.*\.glb(?:$|\?)/i.test(urlText) && /ERR_ABORTED/i.test(errorText)) return;
    console.error('[browser:requestfailed]', urlText, errorText);
    });

    let lastLog = 0;
    await page.exposeFunction('emitInstanceSample', (record) => {
    output.write(`${JSON.stringify(record)}\n`);
    });
    await page.exposeFunction('emitInstanceProgress', (progress) => {
    const now = Date.now();
    if (now - lastLog > 1000 || progress.poseIndex === progress.poseCount) {
      lastLog = now;
      const elapsed = progress.elapsedMs / 1000;
      const rate = progress.poseIndex / Math.max(1e-6, elapsed);
      const eta = (progress.poseCount - progress.poseIndex) / Math.max(1e-6, rate);
      console.log(
        `[sampler] ${progress.poseIndex}/${progress.poseCount} positions | visible=${progress.visibleCount} | ${rate.toFixed(2)} pos/s | eta=${eta.toFixed(1)}s`,
      );
    }
    });

    page.setDefaultTimeout(0);
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    try {
      await page.waitForFunction(() => typeof window.runInstanceSampler === 'function', null, { timeout: 30000 });
    } catch (error) {
      const bodyText = await page.locator('body').innerText().catch(() => '');
      throw new Error(`Sampler page did not expose runInstanceSampler within 30s. Body: ${bodyText}`);
    }
    result = await page.evaluate((options) => window.runInstanceSampler(options), {
      width: args.width,
      height: args.height,
      point60Gt: args.point60Gt,
      fovYDeg: args.fovYDeg,
      posePlan,
      glbIdList,
      poseStart: 0,
      poseCount: 0,
      gridStep: args.gridStep,
      roadHeights: args.roadHeights,
      aerialHeights: args.aerialHeights,
      farHeights: args.farHeights,
      farDistances: args.farDistances,
      yaws: args.yaws,
      pitches: args.pitches,
      maxPositions: args.maxPositions,
      maxGlbs: args.maxGlbs,
      loadConcurrency: args.loadConcurrency,
      occupiedOnly: args.occupiedOnly,
      occupiedExpandCells: args.occupiedExpandCells,
      coarseGridStep: args.coarseGridStep,
      includeCoarseGlobal: args.includeCoarseGlobal,
      heightMode: args.heightMode,
      atlasColumns: args.atlasColumns,
      atlasRows: args.atlasRows,
      directionTilesPerAtlas: args.directionTilesPerAtlas,
      cameraBoundsExpandRatio: args.cameraBoundsExpandRatio,
      cameraMinY: args.cameraMinY,
      cameraMaxY: args.cameraMaxY,
      roadSearchCells: args.roadSearchCells,
      roadMinNeighbors: args.roadMinNeighbors,
    });
    const gpuGate = classifyGpuBackend(result.gpuBackend);
    result.gpuGate = {
      required: args.requireHardwareGpu,
      ...gpuGate,
    };
    if (args.requireHardwareGpu && !gpuGate.hardware) {
      throw new Error(`hardware GPU required, sampler reported ${gpuGate.renderer || 'no WebGL renderer'}`);
    }
    hostGpuDuring = captureHostGpuEvidence();
    if (args.requireHardwareGpu && (!hostGpuDuring.nvidiaSmi.available || !hostGpuDuring.nvidiaSmiPmon.available)) {
      throw new Error('hardware GPU required, but nvidia-smi/pmon evidence was unavailable');
    }
    evidence = {
      schema: 'color-id-sampler-gpu-evidence-v1',
      formalReady: false,
      output: args.output,
      posePlan: args.posePlan || null,
      poseStart: args.poseStart,
      poseCount: args.poseCount,
      fovYDeg: args.fovYDeg,
      mode: args.point60Gt ? 'Point60' : 'Model66',
      point60Gt: args.point60Gt,
      width: args.width,
      height: args.height,
      chrome: {
        executablePath: executablePath || null,
        args: chromeArgs,
      },
      gpuBackend: result.gpuBackend || null,
      gpuGate,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter: null,
    };
    console.log(JSON.stringify({ output: args.output, ...result }, null, 2));
  } catch (error) {
    failure = error;
    if (!hostGpuDuring) hostGpuDuring = captureHostGpuEvidence();
    evidence = {
      schema: 'color-id-sampler-gpu-evidence-v1',
      formalReady: false,
      output: args.output,
      posePlan: args.posePlan || null,
      poseStart: args.poseStart,
      poseCount: args.poseCount,
      fovYDeg: args.fovYDeg,
      mode: args.point60Gt ? 'Point60' : 'Model66',
      point60Gt: args.point60Gt,
      width: args.width,
      height: args.height,
      chrome: {
        executablePath: executablePath || null,
        args: chromeArgs,
      },
      gpuBackend: result?.gpuBackend || null,
      gpuGate: result?.gpuGate || null,
      hostGpuBefore,
      hostGpuDuring,
      hostGpuAfter: null,
      error: String(error?.message || error),
    };
  } finally {
    await new Promise((resolve) => output.end(resolve));
    if (browser) await browser.close().catch(() => {});
    await new Promise((resolve) => server.close(resolve));
    hostGpuAfter = captureHostGpuEvidence();
    if (!evidence) {
      evidence = {
        schema: 'color-id-sampler-gpu-evidence-v1',
        formalReady: false,
        output: args.output,
        posePlan: args.posePlan || null,
        poseStart: args.poseStart,
        poseCount: args.poseCount,
        fovYDeg: args.fovYDeg,
        mode: args.point60Gt ? 'Point60' : 'Model66',
        point60Gt: args.point60Gt,
        width: args.width,
        height: args.height,
        chrome: {
          executablePath: executablePath || null,
          args: chromeArgs,
        },
        gpuBackend: result?.gpuBackend || null,
        gpuGate: result?.gpuGate || null,
        hostGpuBefore,
        hostGpuDuring: hostGpuDuring || captureHostGpuEvidence(),
        hostGpuAfter: null,
      };
    }
    evidence.hostGpuAfter = hostGpuAfter;
    const hostEvidenceReady = (hostEvidence) => Boolean(
      hostEvidence?.nvidiaSmi?.available && hostEvidence?.nvidiaSmiPmon?.available,
    );
    evidence.formalReady = Boolean(
      !failure
      && args.requireHardwareGpu
      && evidence.gpuGate?.hardware
      && hostEvidenceReady(hostGpuBefore)
      && hostEvidenceReady(evidence.hostGpuDuring)
      && hostEvidenceReady(hostGpuAfter),
    );
    evidence.capturedAt = new Date().toISOString();
    writeGpuEvidence(args.output, evidence);
  }
  if (failure) throw failure;
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

export {
  POINT60_GT_FOV_Y_DEG,
  classifyGpuBackend,
  parseArgs,
  validateModel66Plan,
  validatePoint60Plan,
};
