#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';
import { headlessWebGpuArgs, resolveVulkanEnvironment } from './chrome_gpu_flags.mjs';

function argument(name, fallback = null) {
  const prefix = `--${name}=`;
  const value = process.argv.find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
}

function gpuEvidence() {
  const run = (command, args) => {
    try {
      return execFileSync(command, args, { encoding: 'utf8', timeout: 10000 });
    } catch (error) {
      return `ERROR: ${error.message}`;
    }
  };
  return {
    nvidiaSmi: run('nvidia-smi', []),
    nvidiaPmon: run('nvidia-smi', ['pmon', '-c', '1', '-s', 'um']),
  };
}

const url = argument('url', 'http://127.0.0.1:8321/runtime-benchmark.html');
const scene = argument('scene', 'hkust');
const backend = argument('backend', 'webgpu');
const sessions = Number(argument('sessions', '1'));
const output = resolve(argument('output', `paper_runtime_${scene}_${backend}.json`));
if (!['webgpu', 'wasm'].includes(backend)) throw new Error(`Unsupported backend ${backend}.`);
if (!Number.isInteger(sessions) || sessions < 1 || sessions > 5) {
  throw new Error('--sessions must be between 1 and 5.');
}

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
  headless: true,
  args: headlessWebGpuArgs(),
  env: resolveVulkanEnvironment(),
});
const evidence = { before: gpuEvidence() };
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const browserErrors = [];
  page.on('pageerror', (error) => browserErrors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') browserErrors.push(message.text());
  });
  await page.goto(url, { waitUntil: 'networkidle' });
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
  const capture = {
    schema: 'pvs-paper-runtime-automated-capture-v1',
    result: state.result,
    browserErrors,
    chromeArgs: headlessWebGpuArgs(),
    vulkanIcd: resolveVulkanEnvironment().VK_ICD_FILENAMES || null,
    gpuEvidence: evidence,
  };
  fs.mkdirSync(dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(capture, null, 2)}\n`);
  console.log(JSON.stringify({
    output,
    scene,
    backend: state.result.environment.backend,
    summary: state.result.summary,
    browserErrors: browserErrors.length,
  }, null, 2));
} finally {
  await browser.close();
}
