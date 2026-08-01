import { rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const require = createRequire(import.meta.url);
const WorkerFarm = require('@parcel/workers');

// Parcel v1 eagerly forks worker processes during build. That fails with EPERM in
// restricted Windows environments, while the local worker path is sufficient here.
WorkerFarm.prototype.shouldUseRemoteWorkers = function shouldUseRemoteWorkers() {
  return false;
};
WorkerFarm.prototype.shouldStartRemoteWorkers = function shouldStartRemoteWorkers() {
  return false;
};
WorkerFarm.prototype.startMaxWorkers = function startMaxWorkers() {};

const Bundler = require('parcel-bundler');
const entry = resolve('index.html');
const outDir = resolve('public');

// Parcel v1 can leave same-named hashed assets behind when the output folder is
// reused. Clean it so deploy packages never serve stale bundle contents.
rmSync(outDir, { recursive: true, force: true });

const bundler = new Bundler(entry, {
  outDir,
  publicUrl: '.',
  watch: false,
  hmr: false,
  sourceMaps: false,
});

try {
  const bundle = await bundler.bundle();
  if (!bundle) {
    throw new Error('Parcel returned an empty bundle.');
  }
} catch (error) {
  console.error(error);
  process.exit(1);
}
