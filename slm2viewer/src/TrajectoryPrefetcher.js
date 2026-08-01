import { Vector3, Quaternion, Euler } from 'three';

const _pos = new Vector3();
const _quat = new Quaternion();
const _euler = new Euler(0, 0, 0, 'YXZ');

export class TrajectoryPrefetcher {
  constructor(options = {}) {
    this.historyWindowMs = Number(options.historyWindowMs !== undefined ? options.historyWindowMs : 500);
    this.horizonsMs = Array.isArray(options.horizonsMs) ? options.horizonsMs.slice() : [800, 1600];
    this.history = [];
  }

  reset() {
    this.history = [];
  }

  record(camera, nowMs = performance.now()) {
    if (!camera) return;
    camera.getWorldPosition(_pos);
    camera.getWorldQuaternion(_quat);
    this.history.push({
      time: nowMs,
      position: _pos.clone(),
      quaternion: _quat.clone(),
    });
    const cutoff = nowMs - this.historyWindowMs;
    while (this.history.length > 2 && this.history[0].time < cutoff) {
      this.history.shift();
    }
  }

  predictFuturePoses(nowMs = performance.now()) {
    if (this.history.length < 2) return [];
    const first = this.history[0];
    const last = this.history[this.history.length - 1];
    const dt = Math.max((last.time - first.time) / 1000, 1e-3);
    const velocity = last.position.clone().sub(first.position).multiplyScalar(1 / dt);

    const poses = [];
    for (const horizonMs of this.horizonsMs) {
      const seconds = horizonMs / 1000;
      const position = last.position.clone().addScaledVector(velocity, seconds);
      // V1 dead reckoning keeps orientation stable. It is safer than over-rotating from noisy mouse input.
      _euler.setFromQuaternion(last.quaternion, 'YXZ');
      poses.push({
        horizonMs,
        position,
        rotation: _euler.clone(),
      });
    }
    return poses;
  }
}
