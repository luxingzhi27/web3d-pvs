import { Vector3, Quaternion } from 'three';
import { FRONTEND_RENDER_FOV_Y_DEG } from './neuralPvsFovProtocol.js';

const _pos = new Vector3();
const _quat = new Quaternion();
const _forward = new Vector3();
const RAD_TO_DEG = 180 / Math.PI;

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
    this.lastPredictionAt = 0;
  }

  configure(options = {}) {
    this.mode = String(options.mode || this.mode || 'delta').toLowerCase();
    this.positionDelta = finiteNumber(options.positionDelta !== undefined ? options.positionDelta : this.positionDelta, 8);
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

    if (nowMs - this.lastPredictionAt < this.minIntervalMs) {
      return false;
    }

    const moved = this.lastPosition.distanceTo(current.position);
    const angleRad = 2 * Math.acos(Math.min(1, Math.abs(this.lastQuaternion.dot(current.quaternion))));
    const angleDeg = angleRad * 180 / Math.PI;
    const fovDelta = Math.abs(current.fov - finiteNumber(this.lastFov, current.fov));
    const aspectDelta = Math.abs(current.aspect - finiteNumber(this.lastAspect, current.aspect));
    if (this.mode === 'viewcell') {
      const yawDelta = wrappedAngleDeltaDeg(current.yawDeg, finiteNumber(this.lastYawDeg, current.yawDeg));
      const pitchDelta = Math.abs(current.pitchDeg - finiteNumber(this.lastPitchDeg, current.pitchDeg));
      return moved >= this.positionDelta ||
        yawDelta >= this.yawDeltaDeg ||
        pitchDelta >= this.pitchDeltaDeg ||
        fovDelta >= this.fovDeltaDeg ||
        aspectDelta >= this.aspectDelta;
    }
    return moved >= this.positionDelta ||
      angleDeg >= this.angleDeltaDeg ||
      fovDelta >= this.fovDeltaDeg ||
      aspectDelta >= this.aspectDelta;
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
      hasCommittedPrediction: Boolean(this.lastPosition && this.lastQuaternion),
    };
  }
}
