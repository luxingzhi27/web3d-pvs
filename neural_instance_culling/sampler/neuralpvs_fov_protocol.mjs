import fs from 'node:fs';
import path from 'node:path';

const CONFIG_PATH = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'config', 'neuralpvs_viewcell_protocol.json');
const CONFIG = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));

export const NEURALPVS_FOV_PROTOCOL = Object.freeze(CONFIG);
export const RENDER_FOV_Y_DEG = Number(CONFIG.renderFovYDeg);
export const MODEL_FOV_Y_DEG = Number(CONFIG.modelFovYDeg);

export function backOffsetForRadius(radius, baseFovYDeg = RENDER_FOV_Y_DEG) {
  const safeRadius = Math.max(0, Number(radius) || 0);
  const halfFovRad = (Number(baseFovYDeg) * Math.PI) / 360;
  return safeRadius / Math.max(1e-6, Math.tan(halfFovRad));
}

export function horizontalFovFromVertical(verticalFovDeg, aspect) {
  const tanHalfY = Math.tan((Number(verticalFovDeg) * Math.PI) / 360);
  return (2 * Math.atan(tanHalfY * Math.max(1e-6, Number(aspect) || 1)) * 180) / Math.PI;
}
