#!/usr/bin/env node
import assert from 'node:assert/strict';
import {
  canAttemptWebGPU,
  describeBackendFallback,
  isWebGPUBackendFailure,
  normalizeInstancePVSBackend,
} from '../src/InstancePVSBackendPolicy.js';

assert.equal(normalizeInstancePVSBackend(), 'auto');
assert.equal(normalizeInstancePVSBackend('WEBGPU'), 'webgpu');
assert.equal(normalizeInstancePVSBackend('cpu'), 'cpu');
assert.equal(normalizeInstancePVSBackend('unknown'), 'auto');
assert.equal(canAttemptWebGPU({ navigator: { gpu: {} } }), true);
assert.equal(canAttemptWebGPU({ navigator: {} }), false);
assert.equal(isWebGPUBackendFailure(
  new Error('requestAdapter returned null'),
  { scope: { navigator: { gpu: {} } } },
), true);
assert.equal(isWebGPUBackendFailure(
  new Error('model_meta.json is missing'),
  { scope: { navigator: { gpu: {} } } },
), false);
assert.equal(describeBackendFallback(new Error('  device   was lost  ')), 'device was lost');

console.log('Instance PVS backend policy tests passed.');
