export function warmNeuralAsset(url) {
  if (!url) return Promise.resolve(null);
  return fetch(url, { cache: 'force-cache' })
    .then((response) => {
      if (!response.ok) throw new Error(`Failed to warm ${url}: ${response.status}`);
      return response.arrayBuffer();
    })
    .catch((error) => {
      console.warn('[NeuralAssetWorker] warm failed:', error);
      return null;
    });
}
