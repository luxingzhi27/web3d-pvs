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
const workloadAssetDir = 'assets/benchmark/pvs_paper_runtime';

if (!existsSync(resolve(publicDir, htmlName))) {
  throw new Error('Build the frontend before packaging the runtime benchmark.');
}
if (!existsSync(resolve(publicDir, workloadAssetDir, 'scenes.json'))) {
  throw new Error('Build the paper runtime workloads before packaging.');
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

const manifest = JSON.parse(readFileSync(resolve(publicDir, workloadAssetDir, 'scenes.json'), 'utf8'));
const runtimeAssets = new Set([workloadAssetDir, 'assets/wasm/instance_pvs_v4.wasm']);
for (const scene of manifest.scenes || []) {
  const modelPath = String(scene.modelAssetPath || '').replace(/^\.\//, '');
  if (!modelPath) throw new Error(`Scene ${scene.id || '?'} has no modelAssetPath.`);
  runtimeAssets.add(modelPath);
}
for (const relative of runtimeAssets) {
  if (!existsSync(resolve(publicDir, relative))) {
    throw new Error(`Runtime benchmark asset is missing: ${relative}`);
  }
  const destination = resolve(outputDir, relative);
  mkdirSync(dirname(destination), { recursive: true });
  cpSync(resolve(publicDir, relative), destination, { recursive: true });
}

console.log(`Packaged PVS runtime benchmark: ${outputDir}`);
