import { readdirSync, readFileSync, rmSync } from 'node:fs';
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
const buildCacheDir = resolve('.parcel-build-cache');

// Parcel v1 can leave same-named hashed assets behind when the output folder is
// reused. Clean it so deploy packages never serve stale bundle contents.
rmSync(outDir, { recursive: true, force: true });
rmSync(buildCacheDir, { recursive: true, force: true });

const bundler = new Bundler(entry, {
  outDir,
  publicUrl: '.',
  production: true,
  watch: false,
  hmr: false,
  cache: false,
  cacheDir: buildCacheDir,
  minify: false,
  sourceMaps: false,
});

try {
  const bundle = await bundler.bundle();
  if (!bundle) {
    throw new Error('Parcel returned an empty bundle.');
  }

  const javascriptFiles = readdirSync(outDir).filter((name) => name.endsWith('.js'));
  for (const name of javascriptFiles) {
    const source = readFileSync(resolve(outDir, name), 'utf8');
    if (source.includes('__parcel__error__overlay__') ||
        source.includes('parcel-bundler/src/builtins/hmr-runtime.js')) {
      throw new Error(`Production bundle contains Parcel HMR runtime: ${name}`);
    }
  }
} catch (error) {
  console.error(error);
  process.exit(1);
}
