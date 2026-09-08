import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const home = process.env.HOME || '/mnt/sda/rhyang';
const drawio = process.env.DRAWIO_BIN || path.join(home, '.local/share/drawio-31.3.2/squashfs-root/AppRun');
const xvfbBin = process.env.XVFB_BIN || path.join(home, '.local/share/xvfb/usr/bin/Xvfb');

if (!fs.existsSync(drawio)) {
  throw new Error(`draw.io exporter not found: ${drawio}`);
}
if (!fs.existsSync(xvfbBin)) {
  throw new Error(`Xvfb not found: ${xvfbBin}`);
}

const requested = new Set(process.argv.slice(2).map((name) => path.basename(name).replace(/\.drawio$/, '')));
const drawioFiles = fs.readdirSync(dir)
  .filter((name) => /^figure_\d{2}_.+\.drawio$/.test(name))
  .filter((name) => requested.size === 0 || requested.has(name.replace(/\.drawio$/, '')))
  .sort();

if (drawioFiles.length === 0) {
  throw new Error('No matching draw.io figures found.');
}

const displayNumber = 100 + (process.pid % 400);
const display = `:${displayNumber}`;
const displaySocket = `/tmp/.X11-unix/X${displayNumber}`;
const xvfb = spawn(xvfbBin, [display, '-screen', '0', '1920x1200x24', '-nolisten', 'tcp', '-ac'], {
  stdio: ['ignore', 'ignore', 'inherit'],
});

for (let attempt = 0; attempt < 50 && !fs.existsSync(displaySocket); attempt += 1) {
  await new Promise((resolve) => setTimeout(resolve, 100));
}
if (!fs.existsSync(displaySocket)) {
  xvfb.kill('SIGTERM');
  throw new Error(`Xvfb did not start on ${display}`);
}

try {
  const env = { ...process.env, DISPLAY: display };
  for (const drawioFile of drawioFiles) {
    const source = path.join(dir, drawioFile);
    const svg = source.replace(/\.drawio$/, '.svg');
    const png = source.replace(/\.drawio$/, '.png');

    execFileSync(drawio, ['-x', '-f', 'svg', '-e', '-o', svg, source, '--disable-gpu', '--no-sandbox'], {
      env,
      stdio: 'inherit',
    });
    execFileSync(drawio, ['-x', '-f', 'png', '-s', '2', '-o', png, source, '--disable-gpu', '--no-sandbox'], {
      env,
      stdio: 'inherit',
    });
  }
} finally {
  xvfb.kill('SIGTERM');
}

console.log(`Exported ${drawioFiles.length} draw.io figures to SVG and PNG.`);
