import { Vector3, Quaternion } from 'three';
import {
  FRONTEND_RENDER_FOV_Y_DEG,
  MODEL_INPUT_FOV_Y_DEG,
} from './neuralPvsFovProtocol.js';

const _pos = new Vector3();
const _quat = new Quaternion();
const _forward = new Vector3();
const RAD_TO_DEG = 180 / Math.PI;
const HKUST_VIEWCELL_RADIUS_M = 2;
const VIEWCELL_VERTICAL_EPSILON_M = 1e-4;
const VIEWCELL_ORIENTATION_EPSILON_DEG = 1e-4;
const VIEWCELL_FOV_EPSILON_DEG = 1e-4;
const RENDER_POSITION_EPSILON_M = 1e-4;

function finiteNumber(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function wrappedAngleDeltaDeg(a, b) {
  let delta = Math.abs(a - b) % 360;
  if (delta > 180) delta = 360 - delta;
  return delta;
}

export class CameraPredictionGate {
  constructor(options = {}) {
    this.configure(options);
    this.lastPosition = null;
    this.lastQuaternion = null;
    this.lastYawDeg = null;
    this.lastPitchDeg = null;
    this.lastFov = null;
    this.lastAspect = null;
    this.lastRenderPosition = null;
    this.lastRenderQuaternion = null;
    this.lastRenderFov = null;
    this.lastRenderAspect = null;
    this.lastPredictionAt = 0;
  }

  configure(options = {}) {
    this.mode = String(options.mode || this.mode || 'delta').toLowerCase();
    const requestedPositionDelta = finiteNumber(
      options.positionDelta !== undefined ? options.positionDelta : this.positionDelta,
      8,
    );
    // HKUST uses a horizontal view-cell disk with a fixed 2 m radius.
    this.positionDelta = this.mode === 'viewcell'
      ? HKUST_VIEWCELL_RADIUS_M
      : requestedPositionDelta;
    this.angleDeltaDeg = finiteNumber(options.angleDeltaDeg !== undefined ? options.angleDeltaDeg : this.angleDeltaDeg, 5);
    this.yawDeltaDeg = finiteNumber(options.yawDeltaDeg !== undefined ? options.yawDeltaDeg : this.yawDeltaDeg, this.angleDeltaDeg);
    this.pitchDeltaDeg = finiteNumber(options.pitchDeltaDeg !== undefined ? options.pitchDeltaDeg : this.pitchDeltaDeg, this.angleDeltaDeg);
    this.fovDeltaDeg = finiteNumber(options.fovDeltaDeg !== undefined ? options.fovDeltaDeg : this.fovDeltaDeg, 2);
    this.aspectDelta = finiteNumber(options.aspectDelta !== undefined ? options.aspectDelta : this.aspectDelta, 0.08);
    this.minIntervalMs = finiteNumber(options.minIntervalMs !== undefined ? options.minIntervalMs : this.minIntervalMs, 100);
    return this;
  }

  reset() {
    this.lastPosition = null;
    this.lastQuaternion = null;
    this.lastYawDeg = null;
    this.lastPitchDeg = null;
    this.lastFov = null;
    this.lastAspect = null;
    this.lastRenderPosition = null;
    this.lastRenderQuaternion = null;
    this.lastRenderFov = null;
    this.lastRenderAspect = null;
    this.lastPredictionAt = 0;
  }

  _readCamera(camera) {
    camera.getWorldPosition(_pos);
    camera.getWorldQuaternion(_quat);
    _forward.set(0, 0, -1).applyQuaternion(_quat).normalize();
    const yawDeg = Math.atan2(-_forward.x, -_forward.z) * RAD_TO_DEG;
    const pitchDeg = Math.asin(Math.max(-1, Math.min(1, _forward.y))) * RAD_TO_DEG;
    return {
      position: _pos.clone(),
      quaternion: _quat.clone(),
      yawDeg,
      pitchDeg,
      fov: finiteNumber(camera.fov, FRONTEND_RENDER_FOV_Y_DEG),
      aspect: finiteNumber(camera.aspect, 16 / 9),
    };
  }

  shouldPredict(camera, nowMs = performance.now()) {
    if (!camera) return true;
    const current = this._readCamera(camera);

    if (!this.lastPosition || !this.lastQuaternion) {
      return true;
    }

    const moved = this.lastPosition.distanceTo(current.position);
    const angleRad = 2 * Math.acos(Math.min(1, Math.abs(this.lastQuaternion.dot(current.quaternion))));
    const angleDeg = angleRad * 180 / Math.PI;
    const fovDelta = Math.abs(current.fov - finiteNumber(this.lastFov, current.fov));
    const aspectDelta = Math.abs(current.aspect - finiteNumber(this.lastAspect, current.aspect));
    if (this.mode === 'viewcell') {
      const horizontalMoved = Math.hypot(
        current.position.x - this.lastPosition.x,
        current.position.z - this.lastPosition.z,
      );
      const verticalMoved = Math.abs(current.position.y - this.lastPosition.y);
      // The offline HKUST subposes keep one fixed orientation. Use the full
      // quaternion so roll cannot silently widen the registered cell either.
      const orientationChanged = angleDeg > VIEWCELL_ORIENTATION_EPSILON_DEG;
      const fovChanged = fovDelta > VIEWCELL_FOV_EPSILON_DEG;
      // A cell boundary must trigger immediately; minIntervalMs cannot make
      // an out-of-cell prediction look valid.
      return horizontalMoved >= this.positionDelta ||
        verticalMoved > VIEWCELL_VERTICAL_EPSILON_M ||
        orientationChanged ||
        fovChanged ||
        aspectDelta >= this.aspectDelta;
    }

    if (nowMs - this.lastPredictionAt < this.minIntervalMs) {
      return false;
    }

    return moved >= this.positionDelta ||
      angleDeg >= this.angleDeltaDeg ||
      fovDelta >= this.fovDeltaDeg ||
      aspectDelta >= this.aspectDelta;
  }

  shouldRefilter(camera) {
    if (!camera || !this.lastPosition || this.shouldPredict(camera)) return false;
    const current = this._readCamera(camera);
    if (!this.lastRenderPosition || !this.lastRenderQuaternion) return true;
    const positionDelta = this.lastRenderPosition.distanceTo(current.position);
    const quaternionDelta = 2 * Math.acos(
      Math.min(1, Math.abs(this.lastRenderQuaternion.dot(current.quaternion))),
    ) * RAD_TO_DEG;
    return positionDelta > RENDER_POSITION_EPSILON_M
      || quaternionDelta > VIEWCELL_ORIENTATION_EPSILON_DEG
      || Math.abs(current.fov - finiteNumber(this.lastRenderFov, current.fov)) > VIEWCELL_FOV_EPSILON_DEG
      || Math.abs(current.aspect - finiteNumber(this.lastRenderAspect, current.aspect)) > 1e-6;
  }

  commitRender(camera) {
    if (!camera) return;
    const current = this._readCamera(camera);
    this.lastRenderPosition = current.position;
    this.lastRenderQuaternion = current.quaternion;
    this.lastRenderFov = current.fov;
    this.lastRenderAspect = current.aspect;
  }

  commit(camera, nowMs = performance.now()) {
    if (!camera) return;
    const current = this._readCamera(camera);
    this.lastPosition = current.position;
    this.lastQuaternion = current.quaternion;
    this.lastYawDeg = current.yawDeg;
    this.lastPitchDeg = current.pitchDeg;
    this.lastFov = current.fov;
    this.lastAspect = current.aspect;
    this.lastPredictionAt = nowMs;
    this.lastRenderPosition = current.position.clone();
    this.lastRenderQuaternion = current.quaternion.clone();
    this.lastRenderFov = current.fov;
    this.lastRenderAspect = current.aspect;
  }

  getState() {
    return {
      mode: this.mode,
      positionDelta: this.positionDelta,
      angleDeltaDeg: this.angleDeltaDeg,
      yawDeltaDeg: this.yawDeltaDeg,
      pitchDeltaDeg: this.pitchDeltaDeg,
      fovDeltaDeg: this.fovDeltaDeg,
      aspectDelta: this.aspectDelta,
      minIntervalMs: this.minIntervalMs,
      positionShape: this.mode === 'viewcell' ? 'horizontal_disk' : 'euclidean',
      verticalPositionEpsilonM: this.mode === 'viewcell' ? VIEWCELL_VERTICAL_EPSILON_M : null,
      orientationMode: this.mode === 'viewcell' ? 'fixed_quaternion' : 'delta',
      orientationEpsilonDeg: this.mode === 'viewcell' ? VIEWCELL_ORIENTATION_EPSILON_DEG : null,
      renderFovYDeg: FRONTEND_RENDER_FOV_Y_DEG,
      modelInputFovYDeg: MODEL_INPUT_FOV_Y_DEG,
      hasCommittedPrediction: Boolean(this.lastPosition && this.lastQuaternion),
      hasCommittedRenderFilter: Boolean(this.lastRenderPosition && this.lastRenderQuaternion),
    };
  }
}
