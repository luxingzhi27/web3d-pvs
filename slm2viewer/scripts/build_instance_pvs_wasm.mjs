#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const viewerRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const crateRoot = resolve(viewerRoot, 'wasm/instance_pvs_v4');
const manifest = resolve(crateRoot, 'Cargo.toml');
const compiled = resolve(crateRoot, 'target/wasm32-unknown-unknown/release/instance_pvs_v4.wasm');
const destination = resolve(viewerRoot, 'assets/wasm/instance_pvs_v4.wasm');
const cargoCandidates = [
  process.env.CARGO,
  resolve(homedir(), '.cargo/bin/cargo'),
  'cargo',
].filter(Boolean);
const cargo = cargoCandidates.find((candidate) => candidate === 'cargo' || existsSync(candidate));

if (!cargo) throw new Error('Rust cargo was not found. Install the user-level Rust toolchain first.');

const result = spawnSync(cargo, [
  'build',
  '--release',
  '--target',
  'wasm32-unknown-unknown',
  '--manifest-path',
  manifest,
], {
  cwd: viewerRoot,
  env: {
    ...process.env,
    RUSTFLAGS: `${process.env.RUSTFLAGS || ''} -C target-feature=+simd128`.trim(),
  },
  encoding: 'utf8',
});

if (result.status !== 0) {
  process.stderr.write(result.stdout || '');
  process.stderr.write(result.stderr || '');
  throw new Error(`WASM SIMD build failed with exit code ${result.status}.`);
}
if (!existsSync(compiled) || statSync(compiled).size === 0) {
  throw new Error('Cargo did not produce instance_pvs_v4.wasm.');
}

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(compiled, destination);
console.log(`Built ${destination} (${statSync(destination).size} bytes).`);
