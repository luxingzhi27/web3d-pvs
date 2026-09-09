#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  competingGpuProcesses,
  evidenceReady,
  formalGate,
} from './run_paper_runtime_benchmark.mjs';

const host = {
  nvidiaSmi: { ok: true, output: 'NVIDIA RTX A6000' },
  nvidiaPmon: { ok: true, output: '# header\n0 10 C chrome 1 2' },
};
assert.equal(evidenceReady(host), true);
assert.deepEqual(competingGpuProcesses({ before: host, during: host, after: host }), []);

const result = { environment: { hardwareGate: { hardware: true } } };
const webgl = { vendor: 'Google Inc. (NVIDIA)', renderer: 'ANGLE (NVIDIA RTX A6000)' };
assert.equal(
  formalGate(result, { before: host, during: host, after: host }, [], webgl, true).formalReady,
  true,
);

const busy = { ...host, nvidiaPmon: { ok: true, output: '0 22 C python 90 0' } };
const failed = formalGate(result, { before: host, during: busy, after: host }, [], webgl, true);
assert.equal(failed.formalReady, false);
assert.equal(failed.concurrentComputeDetected, true);
assert.equal(failed.competingProcesses.length, 1);

assert.equal(
  formalGate(
    result,
    { before: host, during: host, after: host },
    [],
    { vendor: 'Google', renderer: 'SwiftShader' },
    true,
  ).formalReady,
  false,
);

const wasm = {
  environment: {
    backend: 'wasm-simd-v4',
    wasmSimd: true,
    hardwareGate: null,
  },
};
assert.equal(
  formalGate(wasm, { before: host, during: host, after: host }, [], webgl, true).formalReady,
  true,
);

console.log('Paper runtime benchmark runner tests passed.');
