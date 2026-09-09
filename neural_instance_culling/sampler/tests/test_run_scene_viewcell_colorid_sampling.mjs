#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  HOST_GPU_EVIDENCE_FIELDS,
  gpuEvidencePath,
  writeGpuExecutionSummary,
} from '../run_scene_viewcell_colorid_sampling.mjs';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'viewcell-sampler-wrapper-test-'));
try {
  const output = path.join(root, 'shard.jsonl');
  const job = {
    output,
    start: 0,
    end: 128,
    stdout: path.join(root, 'stdout.log'),
    stderr: path.join(root, 'stderr.log'),
  };
  const args = {
    scene: 'synthetic',
    posePlan: path.join(root, 'plan.jsonl'),
    outputDir: root,
  };
  const phases = {
    hostGpuBefore: { phase: 'before' },
    hostGpuDuring: { phase: 'during' },
    hostGpuAfter: { phase: 'after' },
  };
  fs.writeFileSync(gpuEvidencePath(output), JSON.stringify({
    formalReady: true,
    gpuBackend: { renderer: 'NVIDIA' },
    gpuGate: { hardware: true },
    ...phases,
  }));

  const summary = writeGpuExecutionSummary(args, [job]);
  assert.equal(summary.schema, 'viewcell-color-id-sampler-gpu-execution-v2');
  assert.deepEqual(summary.hostGpuEvidenceFields, HOST_GPU_EVIDENCE_FIELDS);
  assert.deepEqual(summary.jobs[0].hostGpuBefore, phases.hostGpuBefore);
  assert.deepEqual(summary.jobs[0].hostGpuDuring, phases.hostGpuDuring);
  assert.deepEqual(summary.jobs[0].hostGpuAfter, phases.hostGpuAfter);
  assert.equal(summary.jobs[0].hostGpuEvidenceComplete, true);

  fs.writeFileSync(gpuEvidencePath(output), JSON.stringify({
    formalReady: true,
    ...phases,
    hostGpuAfter: undefined,
  }));
  assert.throws(
    () => writeGpuExecutionSummary(args, [job]),
    /missing host GPU phase evidence/,
  );
  const failed = JSON.parse(fs.readFileSync(path.join(root, 'gpu_execution_summary.json'), 'utf8'));
  assert.equal(failed.formalReady, false);
  assert.deepEqual(failed.jobs[0].missingHostGpuEvidence, ['hostGpuAfter']);
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}

console.log('View-cell sampler wrapper tests passed.');
