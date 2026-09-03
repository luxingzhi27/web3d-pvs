import { PVSQuerySession } from './PVSQuerySession.js';

let session = null;

function postResult(payload) {
  const transfers = [];
  for (const key of [
    'componentModelList', 'modelList', 'weightList',
    'prefetchGlbIds', 'prefetchWeights', 'renderComponentModelList', 'renderModelList',
    'renderComponentAddedIds', 'renderComponentRemovedIds',
    'renderGlbAddedIds', 'renderGlbRemovedIds',
    'candidateInstanceIds', 'candidateScores', 'candidateDiagnostics',
  ]) {
    if (payload[key]?.buffer) transfers.push(payload[key].buffer);
  }
  self.postMessage(payload, transfers);
}

self.onmessage = (event) => {
  const message = event.data || {};
  Promise.resolve().then(async () => {
    if (message.type === 'init') {
      session?.dispose();
      session = new PVSQuerySession(message.assetBaseUrl, {
        assetVersion: message.assetVersion,
        debugLogging: message.debugLogging,
        maxPrefetch: message.maxPrefetch,
        prefetchThreshold: message.prefetchThreshold,
        downloadPlanMode: message.downloadPlanMode,
        backendPreference: 'wasm',
        executionLocation: 'worker',
      });
      await session.init();
      self.postMessage({
        type: 'ready',
        backend: session.backend,
        timings: session.initTimings,
        fallbackReason: message.fallbackReason || session.pvs?.fallbackReason || null,
        modelInfo: session.getModelInfo(),
      });
      return;
    }
    if (!session?.ready) throw new Error('PVS worker is not ready.');
    if (message.type === 'setDownloadPlanMode') {
      session.setDownloadPlanMode(message.downloadPlanMode);
    } else if (message.type === 'predict') {
      postResult(await session.predict(message.snapshot || {}, message.serial));
    } else if (message.type === 'filter') {
      postResult(await session.refilter(message.snapshot || {}, message.serial));
    }
  }).catch((error) => {
    self.postMessage({
      type: 'error',
      serial: message.serial,
      message: error?.message || String(error),
      stack: error?.stack || null,
    });
  });
};
