#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import {
  classifyGpuText,
  headlessWebGpuArgs,
  resolveVulkanEnvironment,
} from './chrome_gpu_flags.mjs';

const allowSoftware = process.argv.includes('--allow-software-gpu');
const chrome = [
  process.env.CHROME_PATH,
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
].find((candidate) => candidate && fs.existsSync(candidate));
if (!chrome) throw new Error('Chrome/Chromium was not found.');

const server = http.createServer((_request, response) => {
  response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  response.end('<!doctype html><title>WebGPU hardware probe</title>');
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});

const args = headlessWebGpuArgs();
const environment = resolveVulkanEnvironment();
let browser;
try {
  browser = await chromium.launch({
    executablePath: chrome,
    headless: true,
    args,
    env: environment,
  });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: 'domcontentloaded' });
  const api = await page.evaluate(async () => {
    let adapter = null;
    for (let attempt = 0; attempt < 10 && !adapter; attempt += 1) {
      adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) await new Promise((resolve) => setTimeout(resolve, 200));
    }
    const info = adapter?.info || {};
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl');
    const extension = gl?.getExtension('WEBGL_debug_renderer_info');
    return {
      adapterAvailable: Boolean(adapter),
      adapter: {
        vendor: String(info.vendor || ''),
        architecture: String(info.architecture || ''),
        device: String(info.device || ''),
        description: String(info.description || ''),
      },
      webgl: {
        vendor: String(gl && extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : ''),
        renderer: String(gl && extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : ''),
      },
    };
  });
  const adapterGate = classifyGpuText(Object.values(api.adapter));
  const webglGate = classifyGpuText(Object.values(api.webgl));
  let nvidiaSmi = false;
  try {
    execFileSync('nvidia-smi', ['-L'], { stdio: 'ignore', timeout: 10000 });
    nvidiaSmi = true;
  } catch {}
  const hardware = api.adapterAvailable && adapterGate.hardware && webglGate.hardware && nvidiaSmi;
  console.log(JSON.stringify({
    hardware,
    api,
    adapterGate,
    webglGate,
    nvidiaSmi,
    chrome,
    vulkanIcd: environment.VK_ICD_FILENAMES || null,
    args,
  }, null, 2));
  if (!allowSoftware && !hardware) process.exitCode = 1;
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}
