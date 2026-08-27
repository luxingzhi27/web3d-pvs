#!/usr/bin/env node
/*
 * Build a standalone HKUST deployment package.
 *
 * The package contains the current obfuscated frontend, HKUST runtime metadata,
 * the HKUST neural assets, and small proxy geometry. Scene GLB files are not
 * copied: the browser loads them from the configured liteweb3d URL.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const viewerDir = path.resolve(scriptDir, '..');
const defaultSourceDir = path.join(viewerDir, 'public_deploy');
const defaultOutputDir = path.join(viewerDir, 'public_deploy_hkust_liteweb3d');
const defaultArchive = path.join(viewerDir, 'hkust_v3_public_deploy_liteweb3d.tar.gz');
const hkustModelDir = 'pvs_mainline_v4';
const hkustModelFiles = [
  'model_meta.json',
  'instance_runtime_features_fp16.bin',
  'instance_aabb_fp32.bin',
  'instance_to_glb_uint32.bin',
  'query_weights_fp16.bin',
  'frequency_cycles_fp32.bin',
  'chi_table_fp32.bin',
];
const defaultAssetBaseUrl = 'https://www.liteweb3d.com/data/hkust-v3/';

function parseArgs() {
  const args = {
    sourceDir: defaultSourceDir,
    outputDir: defaultOutputDir,
    archive: defaultArchive,
    assetBaseUrl: defaultAssetBaseUrl,
  };

  for (let i = 2; i < process.argv.length; i += 1) {
    const arg = process.argv[i];
    if (arg === '--source-dir') args.sourceDir = path.resolve(process.argv[++i]);
    else if (arg === '--output-dir') args.outputDir = path.resolve(process.argv[++i]);
    else if (arg === '--archive') args.archive = path.resolve(process.argv[++i]);
    else if (arg === '--asset-base-url') args.assetBaseUrl = process.argv[++i];
    else if (arg === '--help') {
      console.log([
        'Usage: node scripts/package_hkust_liteweb3d_deploy.mjs [options]',
        `  --source-dir <dir>       Source deploy directory (default: ${defaultSourceDir})`,
        `  --output-dir <dir>       Standalone directory (default: ${defaultOutputDir})`,
        `  --archive <file>         Archive path (default: ${defaultArchive})`,
        `  --asset-base-url <url>   Remote GLB base URL (default: ${defaultAssetBaseUrl})`,
      ].join('\n'));
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  args.assetBaseUrl = `${String(args.assetBaseUrl).replace(/\/+$/, '')}/`;
  return args;
}

function toPosix(value) {
  return value.split(path.sep).join('/');
}

function walkFiles(dir, output = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const absolute = path.join(dir, entry.name);
    if (entry.isDirectory()) walkFiles(absolute, output);
    else if (entry.isFile()) output.push(absolute);
  }
  return output;
}

function ensureParent(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function copyFile(source, target) {
  ensureParent(target);
  fs.copyFileSync(source, target);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function shouldCopy(relativePath) {
  if (relativePath === 'deploy_manifest.json' ||
      relativePath === 'README_DEPLOY.md' ||
      relativePath === 'Caddyfile.slm2viewer' ||
      relativePath === 'nginx-neural-culling.conf') {
    return false;
  }

  if (relativePath.startsWith('assets/scenes/')) {
    if (!relativePath.startsWith('assets/scenes/hkust-v3/')) return false;
    if (relativePath.includes('/glb/') ||
        (relativePath.endsWith('.glb') && !relativePath.endsWith('/proxy/proxy.glb'))) {
      return false;
    }
    return true;
  }

  if (relativePath.startsWith('assets/neural_instance_culling/')) {
    return relativePath.startsWith(`assets/neural_instance_culling/${hkustModelDir}/`);
  }

  if (relativePath.startsWith('assets/neural_culling/') ||
      relativePath.startsWith('assets/models/') ||
      relativePath.startsWith('assets/ort/')) {
    return false;
  }

  return true;
}

function rewriteConfig(outputDir, assetBaseUrl) {
  const configPath = path.join(outputDir, 'assets', 'config.json');
  if (!fs.existsSync(configPath)) throw new Error(`Missing ${configPath}`);

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const sourceScene = config.scenes && config.scenes['hkust-v3'];
  if (!sourceScene || !sourceScene.loaderConfig) {
    throw new Error('Source config has no hkust-v3 scene');
  }

  const hkustScene = clone(sourceScene);
  const loader = hkustScene.loaderConfig;
  loader.resourcesBaseUrl = './assets/scenes/hkust-v3';
  loader.glbResourcesBaseUrl = assetBaseUrl;
  delete loader.resourcesWS;
  delete loader.rcServerAddress;
  delete loader.remoteResourcesBaseUrl;
  delete loader.remoteResourcesWS;
  delete loader.remoteRvcServerAddress;
  loader.schedulingStrategy = loader.schedulingStrategy || 'auto';

  // Keep both names so the normal URL and the no-query fallback open HKUST.
  config.scenes = {
    default_config: clone(hkustScene),
    'hkust-v3': hkustScene,
  };
  delete config.lbs;
  fs.writeFileSync(configPath, `${JSON.stringify(config, null, 2)}\n`);

  const raw = fs.readFileSync(configPath);
  fs.writeFileSync(`${configPath}.br`, brotliCompressSync(raw, {
    params: { [zlibConstants.BROTLI_PARAM_QUALITY]: 11 },
  }));
  fs.writeFileSync(`${configPath}.gz`, gzipSync(raw, { level: 9 }));
  return loader;
}

function writeDeploymentFiles(outputDir, assetBaseUrl) {
  const readme = `# Standalone HKUST deployment\n\n` +
    `This package uses the current obfuscated SLM2Viewer frontend and the\n` +
    `HKUST neural visibility runtime assets. Draco and KTX2 decoders are local.\n` +
    `Scene component GLB files are intentionally\n` +
    `excluded from the package and are fetched from:\n\n` +
    `    ${assetBaseUrl}\n\n` +
    `## Deploy\n\n` +
    `Extract the archive into the nginx document root:\n\n` +
    `    mkdir -p /var/www/slm2viewer/public_deploy\n` +
    `    tar -xzf hkust_v3_public_deploy_liteweb3d.tar.gz \\\n     -C /var/www/slm2viewer/public_deploy --strip-components=1\n\n` +
    `The default page and ?scene=hkust-v3 both open HKUST. The server only\n` +
    `needs to serve this directory; no local scene_glbs directory is required.\n\n` +
    `The remote origin must allow browser GET/Range requests and CORS.\n`;
  fs.writeFileSync(path.join(outputDir, 'README_DEPLOY.md'), readme);

  const nginx = `# Standalone HKUST package. Remote GLB base: ${assetBaseUrl}\n` +
    `server {\n` +
    `    listen 80;\n` +
    `    server_name _;\n` +
    `    root /var/www/slm2viewer/public_deploy;\n` +
    `    index index.html;\n\n` +
    `    brotli_static on;\n` +
    `    gzip_static on;\n` +
    `    gzip_vary on;\n\n` +
    `    location = /assets/config.json {\n` +
    `        add_header Cache-Control "no-cache";\n` +
    `        try_files $uri =404;\n` +
    `    }\n\n` +
    `    location ^~ /assets/neural_instance_culling/ {\n` +
    `        add_header Cache-Control "no-cache";\n` +
    `        try_files $uri =404;\n` +
    `    }\n\n` +
    `    location ~* \\.(js|css|wasm|mjs|jpg|jpeg|png|ico|hdr)$ {\n` +
    `        add_header Cache-Control "public, max-age=31536000, immutable";\n` +
    `        try_files $uri =404;\n` +
    `    }\n\n` +
    `    location / {\n` +
    `        try_files $uri $uri/ /index.html;\n` +
    `    }\n` +
    `}\n`;
  fs.writeFileSync(path.join(outputDir, 'nginx-neural-culling.conf'), nginx);

  const caddy = `# Standalone HKUST package. Remote GLB base: ${assetBaseUrl}\n` +
    `:8080 {\n` +
    `    root * /var/www/slm2viewer/public_deploy\n` +
    `    encode zstd gzip\n` +
    `    header /assets/config.json Cache-Control "no-cache"\n` +
    `    header /assets/neural_instance_culling/* Cache-Control "no-cache"\n` +
    `    try_files {path} /index.html\n` +
    `    file_server {\n` +
    `        precompressed br gzip\n` +
    `    }\n` +
    `}\n`;
  fs.writeFileSync(path.join(outputDir, 'Caddyfile.slm2viewer'), caddy);
}

function sha256(filePath) {
  return createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function writeManifest(outputDir, sourceDir, assetBaseUrl, copiedFiles) {
  const files = walkFiles(outputDir)
    .filter((filePath) => !filePath.endsWith('.br') && !filePath.endsWith('.gz'))
    .map((filePath) => ({
      path: toPosix(path.relative(outputDir, filePath)),
      bytes: fs.statSync(filePath).size,
      sha256: sha256(filePath),
    }))
    .sort((a, b) => a.path.localeCompare(b.path));
  const rawBytes = files.reduce((sum, item) => sum + item.bytes, 0);
  fs.writeFileSync(path.join(outputDir, 'deploy_manifest.json'), `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    package: 'standalone-hkust-v3',
    sourceDeployDir: path.basename(sourceDir),
    scene: 'hkust-v3',
    localRuntimeAssets: true,
    remoteGlbBaseUrl: assetBaseUrl,
    glbIncluded: false,
    copiedFiles: copiedFiles.length,
    rawBytes,
    files,
  }, null, 2)}\n`);
}

function main() {
  const args = parseArgs();
  if (!fs.existsSync(args.sourceDir)) {
    throw new Error(`Source deploy directory not found: ${args.sourceDir}`);
  }

  const sourceModelDir = path.join(
    args.sourceDir,
    'assets',
    'neural_instance_culling',
    hkustModelDir,
  );
  if (hkustModelFiles.some((file) => !fs.existsSync(path.join(sourceModelDir, file)))) {
    throw new Error(`HKUST runtime model is incomplete: ${sourceModelDir}`);
  }

  fs.rmSync(args.outputDir, { recursive: true, force: true });
  fs.mkdirSync(args.outputDir, { recursive: true });

  const copiedFiles = [];
  for (const sourceFile of walkFiles(args.sourceDir)) {
    const relativePath = toPosix(path.relative(args.sourceDir, sourceFile));
    if (!shouldCopy(relativePath) || relativePath === 'assets/config.json.br' || relativePath === 'assets/config.json.gz') {
      continue;
    }
    const targetFile = path.join(args.outputDir, relativePath);
    copyFile(sourceFile, targetFile);
    copiedFiles.push(relativePath);
  }

  const loader = rewriteConfig(args.outputDir, args.assetBaseUrl);
  writeDeploymentFiles(args.outputDir, args.assetBaseUrl);
  writeManifest(args.outputDir, args.sourceDir, args.assetBaseUrl, copiedFiles);

  fs.mkdirSync(path.dirname(args.archive), { recursive: true });
  fs.rmSync(args.archive, { force: true });
  execFileSync('tar', [
    '-czf', args.archive,
    '-C', path.dirname(args.outputDir),
    path.basename(args.outputDir),
  ], { stdio: 'inherit' });

  const manifest = JSON.parse(fs.readFileSync(path.join(args.outputDir, 'deploy_manifest.json'), 'utf8'));
  console.log(JSON.stringify({
    outputDir: args.outputDir,
    archive: args.archive,
    archiveBytes: fs.statSync(args.archive).size,
    copiedFiles: manifest.copiedFiles,
    rawBytes: manifest.rawBytes,
    scene: 'hkust-v3',
    glbBaseUrl: loader.glbResourcesBaseUrl,
    glbIncluded: manifest.glbIncluded,
    model: hkustModelDir,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exit(1);
}
