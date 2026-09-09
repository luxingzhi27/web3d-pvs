#!/usr/bin/env node
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

function makeGlb() {
  const json = Buffer.from(JSON.stringify({ asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [] }], nodes: [] }));
  const paddedLength = Math.ceil(json.length / 4) * 4;
  const buffer = Buffer.alloc(12 + 8 + paddedLength, 0x20);
  buffer.writeUInt32LE(0x46546c67, 0);
  buffer.writeUInt32LE(2, 4);
  buffer.writeUInt32LE(buffer.length, 8);
  buffer.writeUInt32LE(paddedLength, 12);
  buffer.writeUInt32LE(0x4e4f534a, 16);
  json.copy(buffer, 20);
  return buffer;
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'slm-real-streaming-'));
try {
  const assetRoot = path.join(root, 'assets');
  const outputDir = path.join(root, 'out');
  fs.mkdirSync(assetRoot);
  const entries = [];
  for (let id = 0; id < 3; id += 1) {
    const relative = `asset-${id}.glb`;
    fs.writeFileSync(path.join(assetRoot, relative), makeGlb());
    entries.push({ globalId: id, path: relative });
  }
  const runtimePath = path.join(root, 'runtime.json');
  const indexPath = path.join(root, 'glbIndex.json');
  const planPath = path.join(root, 'plan.json');
  fs.writeFileSync(runtimePath, JSON.stringify({ globalGlbCount: 3 }));
  fs.writeFileSync(indexPath, JSON.stringify({ entries }));
  const poses = Array.from({ length: 12 }, (_, ordinal) => ({
    poseId: ordinal,
    ordinal,
    candidateGlbIds: [0, 1, 2],
    gtGlbIds: [1],
    orderedGlbIds: [1, 2, 0],
    tiers: { urgent: [1], warm: [2], speculative: [0] },
  }));
  fs.writeFileSync(planPath, JSON.stringify({
    schema: 'pvs-real-scheduler-plan-v1',
    split: 'test',
    testRead: true,
    poseCount: 12,
    poseIds: poses.map((pose) => pose.poseId),
    tiering: { schedulerTiers: ['urgent', 'warm', 'speculative'] },
    startup100: { enabled: false, startupTierUsed: false },
    methods: { full: { status: 'available', poseCount: 12, poses } },
  }));

  execFileSync(process.execPath, [
    path.resolve('scripts/run_real_scheduler_streaming.mjs'),
    '--plan', planPath,
    '--runtime-meta', runtimePath,
    '--glb-index', indexPath,
    '--glb-root', assetRoot,
    '--output-dir', outputDir,
    '--methods', 'full',
    '--bandwidths', '100',
    '--repeats', '1',
    '--concurrency', '1',
  ], { encoding: 'utf8' });

  const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'real_scheduler_summary.json'), 'utf8'));
  const runs = fs.readFileSync(path.join(outputDir, 'real_scheduler_runs.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(summary.scheduler, 'GlbResourceScheduler');
  assert.deepEqual(summary.schedulerTiers, ['urgent', 'warm', 'speculative']);
  assert.equal(summary.startup100Enabled, false);
  assert.equal(runs.length, 12);
  assert.ok(runs.every((run) => run.firstFrame.reached));
  assert.ok(runs.every((run) => run.firstFrame.downloadedBytes === runs[0].firstFrame.downloadedBytes));
  assert.ok(runs.every((run) => run.schedulerSnapshot.counts['urgent:resident'] === 1));
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}

console.log('real scheduler streaming fixture: ok');
