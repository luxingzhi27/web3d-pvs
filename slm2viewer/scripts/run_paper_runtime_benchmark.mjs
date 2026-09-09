#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { headlessWebGpuArgs, resolveVulkanEnvironment } from './chrome_gpu_flags.mjs';

function argument(name, fallback = null) {
  const prefix = `--${name}=`;
  const value = process.argv.find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
}

function booleanArgument(name, fallback) {
  const value = argument(name, fallback ? 'true' : 'false');
  if (!['true', 'false'].includes(value)) throw new Error(`--${name} must be true or false.`);
  return value === 'true';
}

function runEvidenceCommand(command, args) {
  try {
    return { ok: true, output: execFileSync(command, args, { encoding: 'utf8', timeout: 10000 }) };
  } catch (error) {
    return { ok: false, output: `ERROR: ${error.message}` };
  }
}

function gpuEvidence() {
  return {
    nvidiaSmi: runEvidenceCommand('nvidia-smi', []),
    nvidiaPmon: runEvidenceCommand('nvidia-smi', ['pmon', '-c', '1', '-s', 'um']),
  };
}

function evidenceReady(record) {
  return Boolean(record?.nvidiaSmi?.ok && record?.nvidiaPmon?.ok);
}

function competingGpuProcesses(evidence) {
  const lines = [evidence?.before, evidence?.during, evidence?.after]
    .flatMap((record) => String(record?.nvidiaPmon?.output || '').split('\n'))
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'));
  return [...new Set(lines.filter((line) => /\bpython(?:3|\d[\d.]*)?\b/i.test(line)))];
}

async function webGlInfo(page) {
  return page.evaluate(() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return { vendor: '', renderer: '', version: '' };
    const extension = gl.getExtension('WEBGL_debug_renderer_info');
    return {
      vendor: extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      version: gl.getParameter(gl.VERSION),
    };
  });
}

function formalGate(result, evidence, browserErrors, webgl, requireHardwareGpu) {
  const competingProcesses = competingGpuProcesses(evidence);
  const adapterGate = result?.environment?.hardwareGate;
  const evidenceComplete = ['before', 'during', 'after'].every((key) => evidenceReady(evidence[key]));
  const webglText = `${webgl?.vendor || ''} ${webgl?.renderer || ''}`;
  const webglHardware = Boolean(webglText.trim())
    && !/swiftshader|llvmpipe|softpipe|swrast|software/i.test(webglText);
  const backend = result?.environment?.backend;
  const hardware = backend === 'wasm-simd-v4'
    ? result?.environment?.wasmSimd === true && webglHardware
    : adapterGate?.hardware === true && webglHardware;
  const formalReady = Boolean(
    requireHardwareGpu
    && hardware
    && evidenceComplete
    && competingProcesses.length === 0
    && browserErrors.length === 0
  );
  return {
    required: requireHardwareGpu,
    formalReady,
    hardware,
    evidenceComplete,
    webglHardware,
    concurrentComputeDetected: competingProcesses.length > 0,
    competingProcesses,
  };
}

async function run() {
  const url = argument('url', 'http://127.0.0.1:8321/runtime-benchmark.html');
  const scene = argument('scene', 'hkust');
  const backend = argument('backend', 'webgpu');
  const sessions = Number(argument('sessions', '5'));
  const output = resolve(argument('output', `paper_runtime_${scene}_${backend}.json`));
  const requireHardwareGpu = booleanArgument('require-hardware-gpu', true);
  if (!['webgpu', 'wasm'].includes(backend)) throw new Error(`Unsupported backend ${backend}.`);
  if (!Number.isInteger(sessions) || sessions !== 5) {
    throw new Error('Formal paper runtime requires --sessions=5.');
  }

  const chromeArgs = headlessWebGpuArgs({ screenSize: '1280,720', quietBrowser: true });
  const launchEnvironment = resolveVulkanEnvironment();
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
    headless: true,
    args: chromeArgs,
    env: launchEnvironment,
  });
  const evidence = { before: gpuEvidence() };
  let finalPayload = null;
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const browserErrors = [];
    page.on('pageerror', (error) => browserErrors.push(error.message));
    page.on('console', (message) => {
      if (message.type() === 'error') browserErrors.push(message.text());
    });
    await page.goto(url, { waitUntil: 'networkidle' });
    const webgl = await webGlInfo(page);
    await page.fill('#device-label', 'RTX A6000 automated paper benchmark');
    await page.fill('#device-model', 'NVIDIA RTX A6000');
    await page.selectOption('#scene-id', scene);
    await page.selectOption('#runtime-backend', backend);
    await page.selectOption('#session-count', String(sessions));
    await page.click('#start-button');
    await page.waitForTimeout(1000);
    evidence.during = gpuEvidence();
    await page.waitForFunction(() => {
      const upload = document.querySelector('#upload-value')?.textContent || '';
      const button = document.querySelector('#start-button');
      return upload.startsWith('成功：') || (button && !button.disabled && upload === '未上传');
    }, null, { timeout: 15 * 60 * 1000 });
    const state = await page.evaluate(() => ({
      message: document.querySelector('#message')?.textContent || '',
      result: window.__pvsRuntimeBenchmark?.result || null,
    }));
    if (!state.result) throw new Error(state.message || 'Benchmark did not produce a result.');
    evidence.after = gpuEvidence();
    const gate = formalGate(state.result, evidence, browserErrors, webgl, requireHardwareGpu);
    finalPayload = {
      ...state.result,
      automationEvidence: {
        schema: 'pvs-paper-runtime-hardware-evidence-v1',
        formalReady: gate.formalReady,
        gate,
        browserErrors,
        webgl,
        chromeArgs,
        vulkanIcd: launchEnvironment.VK_ICD_FILENAMES || null,
        hostGpu: evidence,
      },
    };
    fs.mkdirSync(dirname(output), { recursive: true });
    fs.writeFileSync(output, `${JSON.stringify(finalPayload, null, 2)}\n`);
    if (!gate.formalReady) {
      throw new Error(`Formal A6000 runtime hardware gate failed; inspect ${output}.`);
    }
    console.log(JSON.stringify({
      output,
      scene,
      backend: state.result.environment.backend,
      summary: state.result.summary,
      formalReady: true,
    }, null, 2));
  } finally {
    await browser.close();
  }
  return finalPayload;
}

export { competingGpuProcesses, evidenceReady, formalGate, run };

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
