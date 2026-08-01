#!/usr/bin/env node
/* Aggregate bounded M9 browser reports without hiding invalid runs. */
import fs from 'node:fs';
import path from 'node:path';

function parseArgs() {
  const args = { inputDir: null, files: [], out: null };
  for (let i = 2; i < process.argv.length; i += 1) {
    const arg = process.argv[i];
    if (arg === '--input-dir') args.inputDir = path.resolve(process.argv[++i]);
    else if (arg === '--files') args.files = String(process.argv[++i]).split(',').map((v) => v.trim()).filter(Boolean);
    else if (arg === '--out') args.out = path.resolve(process.argv[++i]);
    else if (arg === '--help') {
      console.log('Usage: node scripts/summarize_m9_runtime_runs.mjs --input-dir DIR --files run2.json,run3.json --out FILE');
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.inputDir || !args.files.length || !args.out) throw new Error('--input-dir, --files and --out are required');
  return args;
}

function percentile(values, probability) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function finiteValues(runs, selector) {
  return runs.map(selector).filter((value) => Number.isFinite(value));
}

function distribution(values) {
  return {
    count: values.length,
    p50: percentile(values, 0.50),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    min: values.length ? Math.min(...values) : null,
    max: values.length ? Math.max(...values) : null,
  };
}

function loadRun(inputDir, file) {
  const filePath = path.join(inputDir, file);
  const report = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const summary = report.summary || {};
  const prediction = summary.visibility?.finalRuntimePrediction || {};
  const errors = summary.runtimeAssetErrors || {};
  const constraints = summary.constraints || {};
  const probe = constraints.latestProbe?.probe || {};
  const componentSets = probe.componentSets || {};
  const frustum = probe.frustum || {};
  return {
    file,
    reportStatus: report.status || null,
    cold0Status: summary.cold0?.status || null,
    webgpuStatus: summary.webgpu?.status || null,
    backend: summary.visibility?.finalBackend || null,
    browserVersion: report.run?.browser?.version || null,
    firstPredictionMetricMs: summary.visibility?.firstPredictionMetricMs ?? null,
    totalMs: prediction.totalMs ?? null,
    inferenceMs: prediction.inferenceMs ?? null,
    postMs: prediction.postMs ?? null,
    candidateMs: prediction.candidateMs ?? null,
    candidateCount: prediction.candidateCount ?? null,
    rawInstanceCount: prediction.rawInstanceCount ?? null,
    renderInstanceCount: prediction.renderInstanceCount ?? null,
    rawGlbCount: prediction.rawGlbCount ?? null,
    renderGlbCount: prediction.renderGlbCount ?? null,
    targetGlbRequests: summary.targetGlb?.requestsObserved ?? null,
    targetGlbResponses: summary.targetGlb?.completedObserved ?? null,
    validatedGlbResponses: summary.targetGlb?.responseSummary?.validatedGlbResponses ?? null,
    htmlFallbackResponses: summary.targetGlb?.responseSummary?.htmlFallbackResponses ?? null,
    consoleErrorCount: errors.consoleErrorCount ?? null,
    loadErrorCount: errors.loadErrorCount ?? null,
    pageErrorCount: errors.pageErrorCount ?? null,
    requestFailureCount: errors.requestFailureCount ?? null,
    instanceLevelEvidence: constraints.instanceLevelEvidence || null,
    aabbFrustumQueryEvidence: constraints.aabbFrustumQueryEvidence || null,
    independentFrustumRefreshEvidence: constraints.independentFrustumRefreshEvidence || null,
    renderOutsideRaw: componentSets.renderOutsideRaw ?? null,
    unknownRender: componentSets.unknownRender ?? null,
    realFov: frustum.activeCamera?.fov ?? null,
    backFov: frustum.backCamera?.fov ?? null,
  };
}

function main() {
  const args = parseArgs();
  const runs = args.files.map((file) => loadRun(args.inputDir, file));
  const validRuns = runs.filter((run) =>
    run.reportStatus === 'completed-observation' &&
    run.cold0Status === 'pass-no-target-glb-request-before-first-prediction' &&
    run.webgpuStatus === 'observed-success' &&
    run.backend === 'worker-webgpu' &&
    run.consoleErrorCount === 0 && run.loadErrorCount === 0 &&
    run.pageErrorCount === 0 && run.requestFailureCount === 0 &&
    run.instanceLevelEvidence === 'observed' &&
    run.aabbFrustumQueryEvidence === 'observed' &&
    run.independentFrustumRefreshEvidence === 'observed' &&
    run.renderOutsideRaw === 0 && run.unknownRender === 0
  );
  const metrics = {
    firstPredictionMetricMs: distribution(finiteValues(validRuns, (r) => Number(r.firstPredictionMetricMs))),
    totalMs: distribution(finiteValues(validRuns, (r) => Number(r.totalMs))),
    inferenceMs: distribution(finiteValues(validRuns, (r) => Number(r.inferenceMs))),
    postMs: distribution(finiteValues(validRuns, (r) => Number(r.postMs))),
    candidateMs: distribution(finiteValues(validRuns, (r) => Number(r.candidateMs))),
    candidateCount: distribution(finiteValues(validRuns, (r) => Number(r.candidateCount))),
    rawInstanceCount: distribution(finiteValues(validRuns, (r) => Number(r.rawInstanceCount))),
    renderInstanceCount: distribution(finiteValues(validRuns, (r) => Number(r.renderInstanceCount))),
  };
  const payload = {
    schema: 'neuralstreamweb3d-m9-runtime-runs-summary-v1',
    scene: 'hkust-v3',
    source: 'local frontend with remote HKUST GLB base',
    inputFiles: args.files,
    validRunCount: validRuns.length,
    excludedRunCount: runs.length - validRuns.length,
    runs,
    validRuns: validRuns.map((run) => run.file),
    metrics,
    fov: { modelAndBackCameraDegrees: 66, realRenderCameraDegrees: 60 },
    qualityGates: {
      cold0AllValid: validRuns.every((run) => run.cold0Status === 'pass-no-target-glb-request-before-first-prediction'),
      noRuntimeErrorsAllValid: validRuns.every((run) => run.consoleErrorCount === 0 && run.loadErrorCount === 0 && run.pageErrorCount === 0),
      instanceLevelEvidenceAllValid: validRuns.every((run) => run.instanceLevelEvidence === 'observed'),
      fovObservedAllValid: validRuns.every((run) => run.realFov === 60 && run.backFov === 66),
      mobileEvidence: 'missing; this is desktop headless Chromium only',
    },
    limitations: [
      'Headless Chrome reports are not mobile or hardware WebGPU evidence.',
      'Only the bounded GLB response sample is magic-validated by the runtime auditor.',
      'The three runs use one fixed initial pose and do not constitute a navigation or p95 user-trace benchmark.',
    ],
  };
  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  fs.writeFileSync(args.out, `${JSON.stringify(payload, null, 2)}\n`);
  console.log(JSON.stringify({ output: args.out, validRunCount: validRuns.length, excludedRunCount: runs.length - validRuns.length }, null, 2));
}

main();
