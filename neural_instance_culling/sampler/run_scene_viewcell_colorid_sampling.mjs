#!/usr/bin/env node
/**
 * Generic NeuralPVS-style viewcell Color-ID sampler for one local scene.
 *
 * This is the single-scene counterpart of run_three_scene_viewcell_colorid_sampling.mjs.
 * It assumes the viewcell subpose plan has already been generated and launches
 * multiple run_sampler.mjs shards in parallel.
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { MODEL_FOV_Y_DEG } from './neuralpvs_fov_protocol.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const HOST_GPU_EVIDENCE_FIELDS = ['hostGpuBefore', 'hostGpuDuring', 'hostGpuAfter'];

function parseArgs(argv) {
  const args = {
    scene: '',
    assetsDir: '',
    posePlan: '',
    outputDir: '',
    outputPrefix: '',
    glbIdList: '',
    subposesPerViewcell: 8,
    height: 288,
    width: 512,
    // Training observations and model candidate inference use the same 66°
    // camera. The browser's final display camera is separately fixed at 60°.
    fovY: MODEL_FOV_Y_DEG,
    parallel: 4,
    shards: 16,
    force: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--force') {
      args.force = true;
      continue;
    }
    if (!key.startsWith('--')) continue;
    const value = argv[i + 1];
    i += 1;
    const name = key.slice(2);
    if (name === 'scene') args.scene = value;
    else if (name === 'assets-dir') args.assetsDir = path.resolve(value);
    else if (name === 'pose-plan') args.posePlan = path.resolve(value);
    else if (name === 'output-dir') args.outputDir = path.resolve(value);
    else if (name === 'output-prefix') args.outputPrefix = value;
    else if (name === 'glb-id-list') args.glbIdList = path.resolve(value);
    else if (name === 'subposes-per-viewcell') args.subposesPerViewcell = Number(value);
    else if (name === 'height') args.height = Number(value);
    else if (name === 'width') args.width = Number(value);
    else if (name === 'fov-y') args.fovY = Number(value);
    else if (name === 'parallel') args.parallel = Number(value);
    else if (name === 'shards') args.shards = Number(value);
  }
  if (!args.scene) args.scene = path.basename(path.dirname(args.assetsDir || 'scene'));
  if (!args.assetsDir) throw new Error('Expected --assets-dir.');
  if (!args.posePlan) throw new Error('Expected --pose-plan.');
  if (!args.outputDir) throw new Error('Expected --output-dir.');
  if (Math.abs(Number(args.fovY) - MODEL_FOV_Y_DEG) > 1e-6) {
    throw new Error(`The current sampler only supports the model FOV of ${MODEL_FOV_Y_DEG} degrees.`);
  }
  if (!args.outputPrefix) args.outputPrefix = `samples_viewcell_renderfov${args.fovY}_h${args.height}_k${args.subposesPerViewcell}`;
  return args;
}

function lineCount(file) {
  const text = fs.readFileSync(file, 'utf8');
  if (!text) return 0;
  return text.endsWith('\n') ? text.split('\n').length - 1 : text.split('\n').length;
}

function gpuEvidencePath(outputPath) {
  return `${outputPath}.gpu_evidence.json`;
}

function hasHostGpuEvidence(value) {
  return value !== null
    && typeof value === 'object'
    && !Array.isArray(value)
    && Object.keys(value).length > 0;
}

function makeJobs(args) {
  const outputDir = args.outputDir;
  const logDir = path.join(outputDir, 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  const count = lineCount(args.posePlan);
  const shardCount = Math.max(1, Math.floor(args.shards));
  const shardSize = Math.ceil(count / shardCount);
  const jobs = [];
  for (let shard = 0; shard < shardCount; shard += 1) {
    const start = shard * shardSize;
    if (start >= count) break;
    const end = Math.min(count, start + shardSize);
    const output = path.join(outputDir, `${args.outputPrefix}_${String(start).padStart(6, '0')}_${String(end).padStart(6, '0')}.jsonl`);
    const stdout = path.join(logDir, path.basename(output).replace(/\.jsonl$/i, '_stdout.log'));
    const stderr = path.join(logDir, path.basename(output).replace(/\.jsonl$/i, '_stderr.log'));
    const skip = !args.force && fs.existsSync(output) && lineCount(output) === end - start;
    jobs.push({
      start,
      end,
      output,
      gpuEvidence: gpuEvidencePath(output),
      skip,
      stdout,
      stderr,
      command: 'node',
      args: [
        'neural_instance_culling/sampler/run_sampler.mjs',
        '--assets-dir', args.assetsDir,
        '--pose-plan', args.posePlan,
        '--output', output,
        '--pose-start', String(start),
        '--pose-count', String(end - start),
        '--height', String(args.height),
        '--width', String(args.width),
        '--fov-y-deg', String(args.fovY),
        '--require-hardware-gpu',
      ],
    });
    if (args.glbIdList) {
      jobs[jobs.length - 1].args.push('--glb-id-list', args.glbIdList);
    }
  }
  return jobs;
}

function writeGpuExecutionSummary(args, jobs) {
  const evidence = jobs.map((job) => {
    const evidencePath = gpuEvidencePath(job.output);
    if (!fs.existsSync(evidencePath)) {
      throw new Error(`Missing sampler GPU evidence: ${evidencePath}`);
    }
    const record = JSON.parse(fs.readFileSync(evidencePath, 'utf8'));
    const missingHostGpuEvidence = HOST_GPU_EVIDENCE_FIELDS.filter(
      (field) => !hasHostGpuEvidence(record[field]),
    );
    return {
      output: job.output,
      evidencePath,
      stdout: job.stdout,
      stderr: job.stderr,
      poseStart: job.start,
      poseEndExclusive: job.end,
      formalReady: Boolean(record.formalReady),
      gpuBackend: record.gpuBackend || null,
      gpuGate: record.gpuGate || null,
      hostGpuBefore: record.hostGpuBefore || null,
      hostGpuDuring: record.hostGpuDuring || null,
      hostGpuAfter: record.hostGpuAfter || null,
      hostGpuEvidenceComplete: missingHostGpuEvidence.length === 0,
      missingHostGpuEvidence,
      error: record.error || null,
    };
  });
  const failed = evidence.filter(
    (record) => !record.formalReady || record.error || !record.hostGpuEvidenceComplete,
  );
  const summary = {
    schema: 'viewcell-color-id-sampler-gpu-execution-v2',
    formalReady: failed.length === 0,
    scene: args.scene,
    posePlan: args.posePlan,
    outputDir: args.outputDir,
    requireHardwareGpu: true,
    hostGpuEvidenceFields: HOST_GPU_EVIDENCE_FIELDS,
    jobs: evidence,
    failureCount: failed.length,
    capturedAt: new Date().toISOString(),
  };
  const output = path.join(args.outputDir, 'gpu_execution_summary.json');
  fs.writeFileSync(output, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  if (!summary.formalReady) {
    const missing = failed.filter((record) => !record.hostGpuEvidenceComplete).length;
    const missingMessage = missing ? `; ${missing} shard(s) are missing host GPU phase evidence` : '';
    throw new Error(`Formal sampler GPU evidence failed for ${failed.length} shard(s)${missingMessage}; see ${output}`);
  }
  return summary;
}

async function runJobs(scene, jobs, parallel) {
  let cursor = 0;
  let active = 0;
  let finished = 0;
  await new Promise((resolve, reject) => {
    const launchNext = () => {
      while (active < parallel && cursor < jobs.length) {
        const job = jobs[cursor++];
        if (job.skip) {
          finished += 1;
          console.log(`[skip] ${scene} ${job.start}-${job.end} already has a complete JSONL; GPU evidence will still be audited`);
          if (finished >= jobs.length) resolve();
          continue;
        }
        active += 1;
        const out = fs.openSync(job.stdout, 'w');
        const err = fs.openSync(job.stderr, 'w');
        console.log(`[launch] ${scene} ${job.start}-${job.end} -> ${path.relative(REPO_ROOT, job.output)}`);
        const child = spawn(job.command, job.args, {
          cwd: REPO_ROOT,
          stdio: ['ignore', out, err],
        });
        child.on('exit', (code) => {
          fs.closeSync(out);
          fs.closeSync(err);
          active -= 1;
          finished += 1;
          if (code !== 0) {
            reject(new Error(`[failed] ${scene} ${job.start}-${job.end} exited with ${code}; see ${job.stderr}`));
            return;
          }
          console.log(`[done] ${finished}/${jobs.length} ${scene} ${job.start}-${job.end}`);
          if (finished >= jobs.length) resolve();
          else launchNext();
        });
      }
      if (jobs.length === 0) resolve();
    };
    launchNext();
  });
}

async function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(args.outputDir, { recursive: true });
  const jobs = makeJobs(args);
  const summary = {
    scene: args.scene,
    assetsDir: args.assetsDir,
    posePlan: args.posePlan,
    outputDir: args.outputDir,
    jobs: jobs.length,
    parallel: Math.max(1, Math.floor(args.parallel)),
    shards: Math.max(1, Math.floor(args.shards)),
    subposesPerViewcell: args.subposesPerViewcell,
    height: args.height,
    width: args.width,
    fovY: args.fovY,
    gpuPolicy: {
      requireHardwareGpu: true,
      softwareOverride: 'not exposed by the formal view-cell wrapper',
    },
    glbIdList: args.glbIdList || null,
  };
  fs.writeFileSync(path.join(args.outputDir, `${args.outputPrefix}_run_summary.json`), JSON.stringify(summary, null, 2), 'utf8');
  console.log(JSON.stringify(summary, null, 2));
  await runJobs(args.scene, jobs, summary.parallel);
  const gpuSummary = writeGpuExecutionSummary(args, jobs);
  console.log(JSON.stringify(gpuSummary, null, 2));
}

export {
  HOST_GPU_EVIDENCE_FIELDS,
  gpuEvidencePath,
  makeJobs,
  parseArgs,
  writeGpuExecutionSummary,
};

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
