function finitePositive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

export function resolveRenderSurface(options = {}) {
  const cssWidth = Math.max(1, Math.round(finitePositive(options.width, 1)));
  const cssHeight = Math.max(1, Math.round(finitePositive(options.height, 1)));
  const pixelRatio = finitePositive(options.devicePixelRatio, 1);
  return {
    cssWidth,
    cssHeight,
    pixelRatio,
    drawingBufferWidth: Math.floor(cssWidth * pixelRatio),
    drawingBufferHeight: Math.floor(cssHeight * pixelRatio),
  };
}

export function sameRenderSurface(a, b) {
  return Boolean(a && b)
    && a.cssWidth === b.cssWidth
    && a.cssHeight === b.cssHeight
    && a.pixelRatio === b.pixelRatio;
}
