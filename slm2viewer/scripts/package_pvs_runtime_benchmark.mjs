#!/usr/bin/env node
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
} from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const viewerDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const publicDir = resolve(viewerDir, 'public');
const outputDir = resolve(viewerDir, 'public_runtime_benchmark');
const htmlName = 'runtime-benchmark.html';
const modelAssetDir = 'assets/neural_instance_culling/pvs_mainline_v4';
const workloadAssetDir = 'assets/benchmark/pvs_v4_frontend_inference_latency_v1';

if (!existsSync(resolve(publicDir, htmlName))) {
  throw new Error('Build the frontend before packaging the runtime benchmark.');
}
if (!existsSync(resolve(publicDir, workloadAssetDir, 'workload.json'))) {
  throw new Error('Build the frozen runtime workload before packaging.');
}

rmSync(outputDir, { recursive: true, force: true });
mkdirSync(outputDir, { recursive: true });
copyFileSync(resolve(publicDir, htmlName), resolve(outputDir, htmlName));

const html = readFileSync(resolve(publicDir, htmlName), 'utf8');
const references = [...html.matchAll(/(?:src|href)="([^"?#]+)(?:[?#][^"]*)?"/g)]
  .map((match) => match[1])
  .filter((path) => !path.includes('://') && !path.startsWith('data:'));
for (const relative of references) {
  const source = resolve(publicDir, relative);
  if (!existsSync(source)) throw new Error(`Benchmark HTML references missing file ${relative}.`);
  const destination = resolve(outputDir, relative);
  mkdirSync(dirname(destination), { recursive: true });
  copyFileSync(source, destination);
}

for (const relative of [modelAssetDir, workloadAssetDir]) {
  cpSync(resolve(publicDir, relative), resolve(outputDir, relative), { recursive: true });
}

console.log(`Packaged PVS runtime benchmark: ${outputDir}`);
