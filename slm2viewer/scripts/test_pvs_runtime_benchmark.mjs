#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  adapterHardwareGate,
  percentile,
  shuffledOrdinals,
  summarizeSamples,
} from '../src/PVSRuntimeBenchmark.js';

assert.equal(percentile([1, 2, 3, 4], 0.5), 2.5);
assert.equal(percentile([1, 2, 3, 4], 0.95), 3.8499999999999996);
assert.equal(percentile([], 0.5), null);

const firstOrder = shuffledOrdinals(20, 1234);
const secondOrder = shuffledOrdinals(20, 1234);
assert.deepEqual(firstOrder, secondOrder);
assert.deepEqual([...firstOrder].sort((a, b) => a - b), Array.from({ length: 20 }, (_, index) => index));

const summary = summarizeSamples([
  { candidateCount: 100, modelInferenceMs: 1, submitCompletionMs: 2 },
  { candidateCount: 300, modelInferenceMs: 3, submitCompletionMs: 4 },
]);
assert.equal(summary.sampleCount, 2);
assert.equal(summary.candidateMean, 200);
assert.equal(summary.modelInferenceP50Ms, 2);

assert.equal(adapterHardwareGate({ vendor: 'nvidia', architecture: 'ampere' }).hardware, true);
assert.equal(adapterHardwareGate({ vendor: 'Google', device: 'SwiftShader' }).hardware, false);
assert.equal(adapterHardwareGate({}).hardware, false);

console.log('PVS runtime benchmark unit tests passed.');
