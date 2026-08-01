#!/usr/bin/env node
/* 单独打包某场景的 GLB 本体（不进 public_deploy）。
 *
 * 用途：场景 GLB 体量大(如 hkust-v3 和 IFCBench Metropolis)，远端只需上传一次，
 *   不随每次 public_deploy 发布重传。public_deploy 本体只带 JS/CSS/PVS 模型/场景元数据。
 *
 * 前后端约定：
 *   resourcesBaseUrl   → assets/scenes/<scene>  (sceneWeb/glbIndex/runtimeVisibilityMeta/proxy)
 *   glbResourcesBaseUrl → scene_glbs/<scene>     (task-N/glb/LOD0/sub_*.glb)
 *
 * 本脚本从场景资源目录提取 task-N/glb/LOD0/ 下的 .glb 文件，按 scene_glbs/<scene>/ 结构打包。
 * 上传到远端后解到 /var/www/slm2viewer/scene_glbs/<scene>/ 即可，再由 nginx alias serve。
 *
 * 用法：
 *   npm run package:scene-glb -- --scene ifcbench_fantasy_metropolis_instanced_v2
 *   npm run package:scene-glb -- --scene hkust-v3
 *   npm run package:scene-glb -- --scene hkust-v3 --source /path/to/hkust-v3/assets
 *   npm run package:scene-glb -- --scene hkust-v3 --source /path/to/hkust-v3/assets
 *
 * 产出：dist_scene_glb/<scene>_glb.tar.gz
 * 上传：scp dist_scene_glb/<scene>_glb.tar.gz root@<ip>:/tmp/
 *        ssh root@<ip> "tar -xzf /tmp/<scene>_glb.tar.gz -C /var/www/slm2viewer/scene_glbs"
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const argv = (() => {
  const out = { scene: '', source: '', outDir: '', gzip: true };
  for (let i = 2; i < process.argv.length; i += 1) {
    const k = process.argv[i];
    if (k === '--scene') { out.scene = process.argv[++i]; continue; }
    if (k === '--source') { out.source = path.resolve(process.argv[++i]); continue; }
    if (k === '--out-dir') { out.outDir = path.resolve(process.argv[++i]); continue; }
    if (k === '--no-gzip') { out.gzip = false; continue; }
  }
  if (!out.scene) throw new Error('Expected --scene <name>');
  if (!out.source) out.source = path.join(REPO_ROOT, out.scene, 'assets');
  if (!fs.existsSync(out.source)) throw new Error(`Scene assets not found: ${out.source}`);
  if (!fs.existsSync(path.join(out.source, 'sceneWeb.json'))) {
    throw new Error(`${out.source} has no sceneWeb.json — is it a scene assets dir?`);
  }
  if (!out.outDir) out.outDir = path.join(REPO_ROOT, 'dist_scene_glb');
  return out;
})();

// 建临时目录，只放入 task-N/glb/LOD0/*.glb (不含 proxy、不含 metadata)
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), `scene-glb-${argv.scene}-`));
const glbRoot = path.join(tmp, argv.scene);
fs.mkdirSync(glbRoot, { recursive: true });

const taskDirs = fs.readdirSync(argv.source).filter((n) => /^task-\d+$/.test(n));
let glbCount = 0;
for (const td of taskDirs) {
  const srcGlb = path.join(argv.source, td, 'glb', 'LOD0');
  if (!fs.existsSync(srcGlb)) continue;
  const dstGlb = path.join(glbRoot, td, 'glb', 'LOD0');
  fs.mkdirSync(dstGlb, { recursive: true });
  for (const f of fs.readdirSync(srcGlb)) {
    if (f.endsWith('.glb')) {
      fs.copyFileSync(path.join(srcGlb, f), path.join(dstGlb, f));
      glbCount += 1;
    }
  }
}

if (glbCount === 0) {
  fs.rmSync(tmp, { recursive: true });
  throw new Error(`No .glb files found under ${argv.source}/task-*/glb/LOD0/`);
}

fs.mkdirSync(argv.outDir, { recursive: true });
const outName = argv.gzip ? `${argv.scene}_glb.tar.gz` : `${argv.scene}_glb.tar`;
const outFile = path.join(argv.outDir, outName);

execSync(
  `tar ${argv.gzip ? '-zcf' : '-cf'} '${outFile}' -C '${tmp}' '${argv.scene}'`,
  { stdio: 'inherit' },
);

fs.rmSync(tmp, { recursive: true });

const sizeMB = (fs.statSync(outFile).size / (1024 * 1024)).toFixed(1);
console.log(`\n✓ 场景 GLB 包: ${outFile}`);
console.log(`  大小: ${sizeMB} MB | GLB 数: ${glbCount}`);
console.log(`  结构: ${argv.scene}/task-N/glb/LOD0/sub_*.glb`);
console.log(`  上传: scp ${outFile} root@<ip>:/tmp/`);
console.log(`  远端解: tar -xzf /tmp/${outName} -C /var/www/slm2viewer/scene_glbs`);
