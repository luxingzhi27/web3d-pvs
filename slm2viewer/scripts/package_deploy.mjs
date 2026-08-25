import {
  constants as fsConstants,
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, relative, resolve, sep } from 'node:path';
import { brotliCompressSync, constants as zlibConstants, gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import JavaScriptObfuscator from 'javascript-obfuscator';

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourceDir = resolve(rootDir, 'public');
const outputDirEqualsArg = process.argv.find((arg) => arg.startsWith('--output-dir='));
const outputDirFlagIndex = process.argv.indexOf('--output-dir');
const outputDirValue = outputDirEqualsArg
  ? outputDirEqualsArg.slice('--output-dir='.length).trim()
  : (outputDirFlagIndex >= 0 ? String(process.argv[outputDirFlagIndex + 1] || '').trim() : 'public_deploy');
const deployDir = resolve(rootDir, outputDirValue || 'public_deploy');
const deployAssetProxyPath = '/hkust-v3-assets';
const remoteAssetBaseUrl = 'https://smart3d.hkust-gz.edu.cn/proxy/hkust-v3/assets';
const sceneModelDirByName = Object.freeze({
  'hkust-v3': 'pvs_mainline_v4',
});
const v4RuntimeFiles = Object.freeze([
  'model_meta.json',
  'instance_runtime_features_fp16.bin',
  'instance_aabb_fp32.bin',
  'instance_to_glb_uint32.bin',
  'query_weights_fp16.bin',
  'frequency_cycles_fp32.bin',
  'chi_table_fp32.bin',
]);
const sceneEqualsArg = process.argv.find((arg) => arg.startsWith('--scene='));
const sceneFlagIndex = process.argv.indexOf('--scene');
const selectedSceneName = sceneEqualsArg
  ? sceneEqualsArg.slice('--scene='.length).trim()
  : (sceneFlagIndex >= 0 ? String(process.argv[sceneFlagIndex + 1] || '').trim() : '');
const assetModeArg = process.argv.find((arg) => arg.startsWith('--asset-mode='));
const assetMode = (
  assetModeArg ? assetModeArg.split('=')[1] : (process.env.SLM_DEPLOY_ASSET_MODE || 'direct')
).toLowerCase();
const obfuscateArg = process.argv.find((arg) => arg.startsWith('--obfuscate-js='));
const obfuscateJs = (
  obfuscateArg ? obfuscateArg.split('=')[1] : (process.env.SLM_DEPLOY_OBFUSCATE_JS || 'true')
).toLowerCase() !== 'false';
const skipLargeBinaryCompression = (
  process.env.SLM_DEPLOY_SKIP_LARGE_BINARY_COMPRESSION || 'true'
).toLowerCase() === 'true';
const skipRuntimeAssetCompression = (
  process.env.SLM_DEPLOY_SKIP_RUNTIME_ASSET_COMPRESSION || 'true'
).toLowerCase() === 'true';
const largeBinaryCompressionThreshold = Math.max(
  1,
  Number(process.env.SLM_DEPLOY_LARGE_BINARY_COMPRESSION_THRESHOLD || 8 * 1024 * 1024),
);

if (!['direct', 'proxy'].includes(assetMode)) {
  console.error(`Unsupported asset mode: ${assetMode}`);
  console.error('Use --asset-mode=direct or --asset-mode=proxy.');
  process.exit(1);
}

const sourceConfigPath = resolve(rootDir, 'assets', 'config.json');
const sourceConfig = existsSync(sourceConfigPath)
  ? JSON.parse(readFileSync(sourceConfigPath, 'utf8'))
  : null;
if (selectedSceneName) {
  if (!sourceConfig?.scenes?.[selectedSceneName]) {
    throw new Error(`Unknown --scene ${selectedSceneName}. Check assets/config.json.`);
  }
}

// 未指定场景时保留当前源目录中的全部运行模型；指定场景时只保留该场景对应的模型。
const localNeuralInstanceCullingRoot = resolve(rootDir, 'assets', 'neural_instance_culling');
const selectedModelDirs = selectedSceneName
  ? new Set(sceneModelDirByName[selectedSceneName] ? [sceneModelDirByName[selectedSceneName]] : [])
  : null;
const runtimeNeuralFiles = new Set();
if (existsSync(localNeuralInstanceCullingRoot)) {
  for (const entry of readdirSync(localNeuralInstanceCullingRoot)) {
    if (selectedModelDirs && !selectedModelDirs.has(entry)) continue;
    const modelDir = resolve(localNeuralInstanceCullingRoot, entry);
    if (!existsSync(modelDir)) continue;
    const toRel = (p) => toPosix(relative(rootDir, p));
    for (const file of v4RuntimeFiles) {
      const source = resolve(modelDir, file);
      if (existsSync(source)) runtimeNeuralFiles.add(toRel(source));
    }
  }
}
if (selectedSceneName && sceneModelDirByName[selectedSceneName]) {
  const modelDir = sceneModelDirByName[selectedSceneName];
  const requiredModelFiles = v4RuntimeFiles.map(
    (file) => `assets/neural_instance_culling/${modelDir}/${file}`,
  );
  const missing = requiredModelFiles.filter((rel) => !runtimeNeuralFiles.has(rel));
  if (missing.length > 0) {
    throw new Error(`Runtime assets for --scene ${selectedSceneName} are missing: ${missing.join(', ')}`);
  }
}

const alwaysInclude = new Set([
  'assets/config.json',
  'assets/custom_starts.json',
  'assets/environment/cloudy_puresky.jpg',
]);
// 当前运行时元数据始终按场景放在 assets/scenes/<scene>/ 下。
// 不再把历史根目录元数据打入包内，避免某个场景请求失败时误用 HKUST 数据。

const precompressExts = new Set([
  '.bin',
  '.css',
  '.html',
  '.js',
  '.json',
  '.mjs',
  '.svg',
  '.txt',
  '.wasm',
]);

function toPosix(path) {
  return path.split(sep).join('/');
}

function relativePosix(path) {
  return toPosix(relative(sourceDir, path));
}

function hasExtension(path, extensions) {
  const lower = path.toLowerCase();
  for (const ext of extensions) {
    if (lower.endsWith(ext)) return true;
  }
  return false;
}

function shouldInclude(rel) {
  if (rel.startsWith('assets/neural_culling/')) {
    return runtimeNeuralFiles.has(rel);
  }

  if (rel.startsWith('assets/neural_instance_culling/')) {
    return runtimeNeuralFiles.has(rel);
  }

  if (rel.startsWith('assets/scenes/')) {
    if (selectedSceneName && !rel.startsWith(`assets/scenes/${selectedSceneName}/`)) {
      return false;
    }
    // conversionManifest 只供离线场景元数据重建使用，前端不会请求它；源场景中继续保留。
    if (rel.endsWith('/conversionManifest.json')) return false;
    // public_deploy 本体只携带场景元数据（sceneWeb/glbIndex/runtimeVisibilityMeta/
    // conversionManifest 及其 .br/.gz 预压缩）+ proxy.glb(SLM2Loader 用 resourcesBaseUrl 加载,
    // 体量小,几 MB)。场景 GLB 本体（task-*/glb/LOD0/sub_*.glb）文件量大,用
    // `npm run package:scene-glb -- --scene <name>` 单独打 GLB 包,不进 public_deploy。
    if (rel.endsWith('/proxy/proxy.glb')) return true;
    if (rel.endsWith('.glb')) return false;
    return !(
      rel.includes('/glb/') ||
      rel.includes('/foi/') ||
      rel.includes('/distWebViewer/')
    );
  }

  if (rel.startsWith('assets/models/') || rel.startsWith('assets/ort/')) {
    return false;
  }

  if (rel.startsWith('assets/environment/')) {
    return rel === 'assets/environment/cloudy_puresky.jpg';
  }

  if (alwaysInclude.has(rel)) {
    return true;
  }

  if (rel.startsWith('assets/icons/')) {
    return true;
  }

  if (rel === 'assets/favicon.ico') {
    return true;
  }

  if (!rel.startsWith('assets/')) {
    return true;
  }

  return false;
}

function walkFiles(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(path, out);
    } else if (entry.isFile()) {
      out.push(path);
    }
  }
  return out;
}

function ensureParent(path) {
  mkdirSync(dirname(path), { recursive: true });
}

function copySelectedFiles() {
  const copied = [];
  const skipped = [];

  for (const src of walkFiles(sourceDir)) {
    const rel = relativePosix(src);
    if (!shouldInclude(rel)) {
      skipped.push(rel);
      continue;
    }

    const dst = resolve(deployDir, rel);
    ensureParent(dst);
    copyFileSync(src, dst, fsConstants.COPYFILE_FICLONE);
    copied.push(rel);
  }

  return { copied, skipped };
}

function precompressFiles(files) {
  const compressed = [];
  for (const rel of files) {
    if (!hasExtension(rel, precompressExts)) continue;
    const lowerRel = rel.toLowerCase();
    const isJsonMetadata = lowerRel.endsWith('.json');
    const isRuntimeAsset = (
      rel.startsWith('assets/neural_instance_culling/') ||
      rel.startsWith('assets/neural_culling/') ||
      rel.startsWith('assets/scenes/')
    );

    // JSON 是可直接由 nginx/caddy 旁路发送的元数据；模型二进制和 GLB
    // 不在这里生成预压缩副本，避免部署包膨胀和构建阶段长时间占用 CPU。
    if (skipRuntimeAssetCompression && isRuntimeAsset && !isJsonMetadata) {
      continue;
    }
    const path = resolve(deployDir, rel);
    if (skipLargeBinaryCompression && !isJsonMetadata &&
        statSync(path).size >= largeBinaryCompressionThreshold) {
      continue;
    }
    const buffer = readFileSync(path);
    if (buffer.length < 1024) continue;

    const br = brotliCompressSync(buffer, {
      params: {
        [zlibConstants.BROTLI_PARAM_QUALITY]: 11,
      },
    });
    const gz = gzipSync(buffer, { level: 9 });
    writeFileSync(`${path}.br`, br);
    writeFileSync(`${path}.gz`, gz);
    compressed.push({
      path: rel,
      rawBytes: buffer.length,
      brBytes: br.length,
      gzBytes: gz.length,
    });
  }
  return compressed;
}

function shouldObfuscate(rel) {
  return /^app\.[^/]+\.js$/.test(rel) || /^LightweightPVSWorker\.[^/]+\.js$/.test(rel);
}

function obfuscateJavaScript(files) {
  if (!obfuscateJs) return [];

  const obfuscated = [];
  for (const rel of files) {
    if (!shouldObfuscate(rel)) continue;
    const path = resolve(deployDir, rel);
    const before = readFileSync(path, 'utf8');
    const startedAt = Date.now();
    const result = JavaScriptObfuscator.obfuscate(before, {
      target: 'browser',
      compact: true,
      simplify: true,
      sourceMap: false,
      renameGlobals: false,
      selfDefending: false,
      debugProtection: false,
      debugProtectionInterval: 0,
      disableConsoleOutput: false,
      controlFlowFlattening: false,
      deadCodeInjection: false,
      transformObjectKeys: false,
      numbersToExpressions: false,
      splitStrings: false,
      stringArray: true,
      stringArrayCallsTransform: false,
      stringArrayEncoding: [],
      stringArrayIndexesType: ['hexadecimal-number'],
      stringArrayRotate: true,
      stringArrayShuffle: true,
      stringArrayThreshold: 0.45,
      identifierNamesGenerator: 'hexadecimal',
      unicodeEscapeSequence: false,
    });
    const after = result.getObfuscatedCode();
    writeFileSync(path, after);
    obfuscated.push({
      path: rel,
      rawBytesBefore: Buffer.byteLength(before),
      rawBytesAfter: Buffer.byteLength(after),
      ms: Date.now() - startedAt,
      note: 'Obfuscation raises reverse-engineering cost, but browser code can still be inspected at runtime.',
    });
  }
  return obfuscated;
}

function hashFile(rel) {
  return createHash('sha256')
    .update(readFileSync(resolve(deployDir, rel)))
    .digest('hex')
    .slice(0, 12);
}

function addIndexAssetVersions() {
  const indexPath = resolve(deployDir, 'index.html');
  if (!existsSync(indexPath)) return null;

  const files = walkFiles(deployDir)
    .map((path) => toPosix(relative(deployDir, path)))
    .filter((rel) => !rel.endsWith('.br') && !rel.endsWith('.gz'));
  const appJs = files.find((rel) => /^app\.[^/]+\.js$/.test(rel));
  const styleCss = files.find((rel) => /^style\.[^/]+\.css$/.test(rel));
  const favicon = files.find((rel) => /^favicon\.[^/]+\.ico$/.test(rel));
  const versions = {};
  let html = readFileSync(indexPath, 'utf8');

  const rewriteRef = (rel) => {
    if (!rel) return;
    const version = hashFile(rel);
    versions[rel] = version;
    const escaped = rel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    html = html.replace(new RegExp(`${escaped}(?:\\?v=[a-f0-9]+)?`, 'g'), `${rel}?v=${version}`);
  };

  rewriteRef(appJs);
  rewriteRef(styleCss);
  rewriteRef(favicon);
  writeFileSync(indexPath, html);
  return versions;
}

function rewriteDeployConfig() {
  const configPath = resolve(deployDir, 'assets/config.json');
  if (!existsSync(configPath)) return null;

  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  if (!config.scenes) return null;

  if (selectedSceneName) {
    const selectedScene = config.scenes[selectedSceneName];
    if (!selectedScene) {
      throw new Error(`Selected scene ${selectedSceneName} is missing from the copied config.`);
    }
    config.scenes = {
      default_config: JSON.parse(JSON.stringify(selectedScene)),
      [selectedSceneName]: JSON.parse(JSON.stringify(selectedScene)),
    };
  }

  // 场景清单完全由 assets/config.json 控制——打包脚本不写死场景名。
  // 只做清理:删除 remote fallback 残留、清掉无用字段。
  for (const [name, scene] of Object.entries(config.scenes)) {
    const lc = scene && scene.loaderConfig;
    if (!lc) continue;
    // 删除 remote fallback 字段(防止走回 smart3d 等远程地址)
    delete lc.remoteResourcesBaseUrl;
    delete lc.remoteResourcesWS;
    delete lc.remoteRvcServerAddress;
    lc.schedulingStrategy = lc.schedulingStrategy || 'auto';
  }

  // default_config 如果没有显式设置,用第一个有 resourcesBaseUrl 的场景兜底
  const dc = config.scenes['default_config'];
  if (dc && dc.loaderConfig) {
    const fallback = Object.values(config.scenes).find(
      (s) => s !== dc && s && s.loaderConfig && s.loaderConfig.resourcesBaseUrl,
    );
    if (!dc.loaderConfig.resourcesBaseUrl && fallback) {
      dc.loaderConfig.resourcesBaseUrl = fallback.loaderConfig.resourcesBaseUrl;
      dc.loaderConfig.glbResourcesBaseUrl = fallback.loaderConfig.glbResourcesBaseUrl || fallback.loaderConfig.resourcesBaseUrl;
    }
    dc.loaderConfig.resourcesWS = '';
    dc.loaderConfig.rcServerAddress = '';
  }
  config.lbs = null;

  writeFileSync(configPath, `${JSON.stringify(config, null, 2)}\n`);

  const main = (selectedSceneName && config.scenes[selectedSceneName])
    || config.scenes['hkust-v3']
    || Object.values(config.scenes).find((s) => s && s.loaderConfig);
  const loaderConfig = main ? main.loaderConfig : {};
  return {
    resourcesBaseUrl: loaderConfig.resourcesBaseUrl || '',
    resourcesWS: loaderConfig.resourcesWS || '',
    rcServerAddress: loaderConfig.rcServerAddress || '',
    lbs: config.lbs,
  };
}

function writeCaddyTemplate() {
  const assetProxyBlock = assetMode === 'proxy'
    ? `
    handle_path ${deployAssetProxyPath}/* {
        rewrite * /proxy/hkust-v3/assets{uri}
        reverse_proxy https://smart3d.hkust-gz.edu.cn {
            header_up Host smart3d.hkust-gz.edu.cn
        }
    }
`
    : `
    # Direct asset mode: scene metadata and proxy geometry are served from this
    # deployment. Each scene's loaderConfig controls its GLB base URL.
    # The proxy mode is retained only for the legacy HKUST upstream template.
`;
  const template = `# Generated by npm run package:deploy
# This Caddyfile serves the viewer by IP on port 8080.
# Asset mode: ${assetMode}
#
# Usage:
#   sudo cp Caddyfile.slm2viewer /etc/caddy/conf.d/slm2viewer.caddy
#   echo 'import /etc/caddy/conf.d/*.caddy' | sudo tee -a /etc/caddy/Caddyfile
#   sudo caddy validate --config /etc/caddy/Caddyfile
#   sudo systemctl reload caddy

:8080 {
    root * /var/www/slm2viewer/public_deploy
    encode zstd gzip
${assetProxyBlock}

    @html path / /index.html
    header @html Cache-Control "no-cache"

    @runtime path /assets/config.json /assets/neural_culling/* /assets/neural_instance_culling/* /assets/scenes/*
    header @runtime Cache-Control "no-cache"

    @static path *.js *.css *.wasm *.mjs *.jpg *.png *.ico
    header @static Cache-Control "public, max-age=31536000, immutable"

    handle /assets/neural_culling/* {
        file_server {
            precompressed br gzip
        }
    }

    handle /assets/neural_instance_culling/* {
        file_server {
            precompressed br gzip
        }
    }

    handle /assets/* {
        file_server {
            precompressed br gzip
        }
    }

    handle /favicon* {
        file_server {
            precompressed br gzip
        }
    }

    handle {
        try_files {path} /index.html
        file_server {
            precompressed br gzip
        }
    }
}
`;
  writeFileSync(resolve(deployDir, 'Caddyfile.slm2viewer'), template);
}

function writeNginxTemplate() {
  const template = `# Generated by npm run package:deploy
# Copy this server block into your nginx config and set root to public_deploy.
server {
    listen 80;
    server_name _;
    root /var/www/slm2viewer/public_deploy;
    index index.html;

    brotli_static on;
    gzip_static on;
    gzip_vary on;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location ^~ /assets/neural_culling/ {
        add_header Cache-Control "no-cache";
        try_files $uri =404;
    }

    location ^~ /assets/neural_instance_culling/ {
        add_header Cache-Control "no-cache";
        try_files $uri =404;
    }

    location = /assets/config.json {
        add_header Cache-Control "no-cache";
        try_files $uri =404;
    }

    location ~* \\.(js|css|html|wasm|mjs)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }

    location = /index.html {
        add_header Cache-Control "no-cache";
    }

    location ~* \\.(jpg|jpeg|png|ico|hdr)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }
}
`;
  writeFileSync(resolve(deployDir, 'nginx-neural-culling.conf'), template);
}

function writeManifest(copied, skipped, compressed, indexAssetVersions, obfuscated) {
  const files = walkFiles(deployDir)
    .filter((path) => !path.endsWith('.br') && !path.endsWith('.gz'))
    .map((path) => {
      const rel = toPosix(relative(deployDir, path));
      return {
        path: rel,
        bytes: statSync(path).size,
      };
    })
    .sort((a, b) => a.path.localeCompare(b.path));

  const totalRawBytes = files.reduce((sum, item) => sum + item.bytes, 0);
  const totalBrBytes = compressed.reduce((sum, item) => sum + item.brBytes, 0);
  const totalGzBytes = compressed.reduce((sum, item) => sum + item.gzBytes, 0);
  const compressedByPath = new Map(compressed.map((item) => [item.path, item]));
  const brotliTransferBytes = files.reduce((sum, item) => {
    const compressedItem = compressedByPath.get(item.path);
    return sum + (compressedItem ? compressedItem.brBytes : item.bytes);
  }, 0);
  const gzipTransferBytes = files.reduce((sum, item) => {
    const compressedItem = compressedByPath.get(item.path);
    return sum + (compressedItem ? compressedItem.gzBytes : item.bytes);
  }, 0);

  writeFileSync(resolve(deployDir, 'deploy_manifest.json'), JSON.stringify({
    generatedAt: new Date().toISOString(),
    sourceDir,
    deployDir,
    selectedScene: selectedSceneName || null,
    selectedModelDirs: selectedModelDirs ? Array.from(selectedModelDirs) : null,
    totals: {
      copiedFiles: copied.length,
      skippedFiles: skipped.length,
      rawBytes: totalRawBytes,
      brotliBytes: totalBrBytes,
      gzipBytes: totalGzBytes,
      brotliTransferBytes,
      gzipTransferBytes,
    },
    runtimeNeuralFiles: Array.from(runtimeNeuralFiles),
    deploymentAssets: {
      mode: assetMode,
      localPath: deployAssetProxyPath,
      remoteAssetBaseUrl,
      skipLargeBinaryCompression,
      largeBinaryCompressionThreshold,
      skipRuntimeAssetCompression,
      compressionPolicy: 'JSON metadata receives .br/.gz sidecars; runtime model binaries and large binary assets do not.',
      note: assetMode === 'proxy'
        ? 'Legacy HKUST proxy template is generated; current scene metadata remains local and current GLB paths come from each scene loaderConfig.'
        : 'Scene metadata, proxy geometry, neural assets and GLB paths are served from the local deployment configuration. resourcesWS and rcServerAddress are disabled.',
    },
    indexAssetVersions,
    obfuscation: {
      enabled: obfuscateJs,
      note: obfuscateJs
        ? 'Deploy app.*.js and LightweightPVSWorker.*.js are obfuscated. This is not DRM; it increases reverse-engineering cost but cannot fully hide browser code.'
        : 'Disabled by --obfuscate-js=false or SLM_DEPLOY_OBFUSCATE_JS=false.',
      files: obfuscated,
    },
    files,
    compressed,
  }, null, 2));
}

if (!existsSync(sourceDir)) {
  console.error(`Build output not found: ${sourceDir}`);
  console.error('Run npm run build first.');
  process.exit(1);
}

rmSync(deployDir, { recursive: true, force: true });
mkdirSync(deployDir, { recursive: true });

const { copied, skipped } = copySelectedFiles();

// 部署说明随包走
const readmePath = resolve(rootDir, 'README_DEPLOY.md');
if (existsSync(readmePath)) {
  copyFileSync(readmePath, resolve(deployDir, 'README_DEPLOY.md'));
  copied.push('README_DEPLOY.md');
}

const configRewrite = rewriteDeployConfig();
const obfuscated = obfuscateJavaScript(copied);
const indexAssetVersions = addIndexAssetVersions();
const compressed = precompressFiles(copied);
writeCaddyTemplate();
writeNginxTemplate();
writeManifest(copied, skipped, compressed, indexAssetVersions, obfuscated);

const copiedRuntimeNeuralFiles = Array.from(runtimeNeuralFiles)
  .filter((rel) => existsSync(resolve(deployDir, rel)));
const rawRuntimeSize = copiedRuntimeNeuralFiles
  .reduce((sum, rel) => sum + statSync(resolve(deployDir, rel)).size, 0);
const brRuntimeSize = copiedRuntimeNeuralFiles
  .filter((rel) => existsSync(resolve(deployDir, `${rel}.br`)))
  .reduce((sum, rel) => sum + statSync(resolve(deployDir, `${rel}.br`)).size, 0);
const manifest = JSON.parse(readFileSync(resolve(deployDir, 'deploy_manifest.json'), 'utf8'));

console.log(JSON.stringify({
  deployDir,
  copiedFiles: copied.length,
  skippedFiles: skipped.length,
  compressedFiles: compressed.length,
  obfuscatedJs: obfuscated,
  rawDeployBytes: manifest.totals.rawBytes,
  brotliTransferBytes: manifest.totals.brotliTransferBytes,
  neuralRuntimeRawBytes: rawRuntimeSize,
  neuralRuntimeBrotliBytes: brRuntimeSize,
  assetMode,
  selectedScene: selectedSceneName || null,
  selectedModelDirs: selectedModelDirs ? Array.from(selectedModelDirs) : null,
  deploymentProxyPath: deployAssetProxyPath,
  remoteAssetBaseUrl,
  indexAssetVersions,
  configRewrite,
  caddyTemplate: resolve(deployDir, 'Caddyfile.slm2viewer'),
  nginxTemplate: resolve(deployDir, 'nginx-neural-culling.conf'),
  manifest: resolve(deployDir, 'deploy_manifest.json'),
}, null, 2));
