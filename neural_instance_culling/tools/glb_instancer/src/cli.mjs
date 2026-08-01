#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildInstancedScene } from './pipeline.mjs';
import { buildSubGlbInstancedScene } from './sub_pipeline.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..');

function parseArgs(argv) {
  const options = {
    input: '',
    outputAssets: '',
    sceneName: '',
    report: '',
    fingerprintTolerance: 0.001,
    rmsTolerance: 0.001,
    maxTolerance: 0.005,
    triangleProgressEvery: 5_000_000,
    regionProgressEvery: 10_000,
    analyzeOnly: false,
    overwrite: false,
    sourceAssets: '',
    lod: 'LOD0',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const name = key.slice(2);
    if (name === 'analyze-only') { options.analyzeOnly = true; continue; }
    if (name === 'overwrite') { options.overwrite = true; continue; }
    const value = argv[i + 1];
    i += 1;
    if (name === 'input') options.input = path.resolve(value);
    else if (name === 'source-assets') options.sourceAssets = path.resolve(value);
    else if (name === 'output-assets') options.outputAssets = path.resolve(value);
    else if (name === 'scene-name') options.sceneName = value;
    else if (name === 'lod') options.lod = value;
    else if (name === 'report') options.report = path.resolve(value);
    else if (name === 'fingerprint-tolerance') options.fingerprintTolerance = Number(value);
    else if (name === 'rms-tolerance') options.rmsTolerance = Number(value);
    else if (name === 'max-tolerance') options.maxTolerance = Number(value);
    else if (name === 'triangle-progress-every') options.triangleProgressEvery = Number(value);
    else if (name === 'region-progress-every') options.regionProgressEvery = Number(value);
  }
  if (options.input && options.sourceAssets) {
    throw new Error('Use exactly one of --input or --source-assets.');
  }
  if (!options.input && !options.sourceAssets) {
    throw new Error('An explicit --input or --source-assets path is required.');
  }
  if (!options.analyzeOnly && !options.outputAssets) {
    throw new Error('--output-assets is required when building an instanced scene.');
  }
  if (!options.sceneName && options.outputAssets) {
    options.sceneName = path.basename(path.dirname(options.outputAssets));
  }
  return options;
}

const options = parseArgs(process.argv);
const result = options.sourceAssets
  ? await buildSubGlbInstancedScene(options)
  : await buildInstancedScene(options);
console.log(JSON.stringify({
  input: result.input,
  sourceAssets: result.sourceAssets || null,
  outputAssets: result.outputAssets || null,
  componentCount: result.componentCount ?? result.connectedRegionCount,
  connectedRegionCount: result.connectedRegionCount ?? null,
  prototypeCount: result.prototypeCount,
  reusedComponentCount: result.reusedComponentCount ?? null,
  reusedRegionCount: result.reusedRegionCount ?? null,
  estimatedByteReductionRatio: result.estimatedByteReductionRatio,
  actualByteReductionRatio: result.actualByteReductionRatio ?? null,
  elapsedSec: result.elapsedSec,
}, null, 2));
