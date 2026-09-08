import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.dirname(fileURLToPath(import.meta.url));
const FONT = 'Noto Sans CJK SC';
const DEFAULT_WIDTH = 1800;
const DEFAULT_HEIGHT = 1100;

const PALETTE = Object.freeze({
  ink: '#243244',
  edge: '#506176',
  offline: { fill: '#E9F2FB', stroke: '#4E79A7' },
  data: { fill: '#E8F6EF', stroke: '#2E7D62' },
  compute: { fill: '#F0ECF8', stroke: '#7563A8' },
  schedule: { fill: '#FFF3DF', stroke: '#B97824' },
  render: { fill: '#E8F3F7', stroke: '#3E7B91' },
  neutral: { fill: '#F4F6F8', stroke: '#6B7585' },
  danger: { fill: '#FBEAE8', stroke: '#B94C48' },
});

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function xmlLabel(value) {
  return String(value).split('\n').map(escapeXml).join('&#xa;');
}

function roleColors(role) {
  return PALETTE[role] || PALETTE.neutral;
}

class Diagram {
  constructor(number, title, width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT) {
    this.number = number;
    this.title = title;
    this.width = width;
    this.height = height;
    this.cells = [];
  }

  vertex(id, value, style, x, y, width, height, parent = '1') {
    this.cells.push(
      `        <mxCell id="${escapeXml(id)}" value="${xmlLabel(value)}" style="${style}" vertex="1" parent="${escapeXml(parent)}">\n` +
      `          <mxGeometry x="${x}" y="${y}" width="${width}" height="${height}" as="geometry"/>\n` +
      '        </mxCell>',
    );
    return id;
  }

  container(id, label, x, y, width, height, role = 'neutral') {
    const colors = roleColors(role);
    return this.vertex(id, label, [
      'swimlane',
      'startSize=48',
      'horizontal=1',
      'rounded=1',
      'arcSize=8',
      'collapsible=0',
      'container=0',
      'whiteSpace=wrap',
      'html=1',
      `fillColor=${colors.fill}`,
      'swimlaneFillColor=#FFFFFF',
      `strokeColor=${colors.stroke}`,
      `separatorColor=${colors.stroke}`,
      `fontColor=${PALETTE.ink}`,
      `fontFamily=${FONT}`,
      'fontSize=23',
      'fontStyle=1',
      'strokeWidth=2',
      'spacingLeft=14',
      'align=left',
    ].join(';') + ';', x, y, width, height);
  }

  node(id, label, x, y, width, height, role = 'neutral', options = {}) {
    const colors = roleColors(role);
    return this.vertex(id, label, [
      options.shape || 'rounded=1;arcSize=8',
      'whiteSpace=wrap',
      'html=1',
      `fillColor=${colors.fill}`,
      `strokeColor=${colors.stroke}`,
      `fontColor=${PALETTE.ink}`,
      `fontFamily=${FONT}`,
      `fontSize=${options.fontSize ?? 19}`,
      `fontStyle=${options.bold === false ? 0 : 1}`,
      `strokeWidth=${options.strokeWidth ?? 2}`,
      `dashed=${options.dashed ? 1 : 0}`,
      `align=${options.align || 'center'}`,
      'verticalAlign=middle',
      'spacing=9',
    ].join(';') + ';', x, y, width, height, options.parent || '1');
  }

  store(id, label, x, y, width, height, role = 'data', options = {}) {
    return this.node(id, label, x, y, width, height, role, {
      ...options,
      shape: 'shape=cylinder3;size=14;boundedLbl=1',
    });
  }

  cube(id, label, x, y, width, height, role = 'data', options = {}) {
    return this.node(id, label, x, y, width, height, role, {
      ...options,
      shape: 'shape=cube;size=12;direction=south',
    });
  }

  document(id, label, x, y, width, height, role = 'data', options = {}) {
    return this.node(id, label, x, y, width, height, role, {
      ...options,
      shape: 'shape=document;size=0.16;boundedLbl=1',
    });
  }

  circle(id, label, x, y, diameter, role = 'compute', options = {}) {
    return this.node(id, label, x, y, diameter, diameter, role, {
      ...options,
      shape: 'ellipse;aspect=fixed',
      fontSize: options.fontSize ?? 15,
      strokeWidth: options.strokeWidth ?? 1.6,
    });
  }

  chip(id, label, x, y, width, height, role = 'compute', options = {}) {
    const colors = roleColors(role);
    const body = this.node(id, label, x, y, width, height, role, {
      ...options,
      shape: 'shape=process;size=0.12',
      fontSize: options.fontSize ?? 20,
    });
    const pinColor = colors.stroke;
    const pinCount = 5;
    for (let i = 0; i < pinCount; i += 1) {
      const px = x + 18 + i * ((width - 36) / (pinCount - 1));
      const top = this.point(`${id}_pin_t_${i}`, px, y - 12);
      const topBody = this.point(`${id}_pin_tb_${i}`, px, y);
      const bottomBody = this.point(`${id}_pin_bb_${i}`, px, y + height);
      const bottom = this.point(`${id}_pin_b_${i}`, px, y + height + 12);
      this.edge(`${id}_pin_te_${i}`, top, topBody, { straight: true, arrow: false, color: pinColor, strokeWidth: 1.4 });
      this.edge(`${id}_pin_be_${i}`, bottomBody, bottom, { straight: true, arrow: false, color: pinColor, strokeWidth: 1.4 });
    }
    return body;
  }

  decision(id, label, x, y, width, height, role = 'schedule', options = {}) {
    return this.node(id, label, x, y, width, height, role, {
      ...options,
      shape: 'rhombus',
      fontSize: options.fontSize ?? 18,
    });
  }

  text(id, label, x, y, width, height, options = {}) {
    return this.vertex(id, label, [
      'text',
      'html=1',
      'strokeColor=none',
      'fillColor=none',
      'whiteSpace=wrap',
      `fontColor=${options.color || PALETTE.ink}`,
      `fontFamily=${FONT}`,
      `fontSize=${options.fontSize ?? 18}`,
      `fontStyle=${options.bold ? 1 : 0}`,
      `align=${options.align || 'center'}`,
      'verticalAlign=middle',
    ].join(';') + ';', x, y, width, height, options.parent || '1');
  }

  point(id, x, y, parent = '1') {
    return this.vertex(id, '', 'ellipse;html=1;fillColor=none;strokeColor=none;opacity=0;', x, y, 2, 2, parent);
  }

  edge(id, source, target, options = {}) {
    const ports = [
      options.exitX == null ? '' : `exitX=${options.exitX}`,
      options.exitY == null ? '' : `exitY=${options.exitY}`,
      options.entryX == null ? '' : `entryX=${options.entryX}`,
      options.entryY == null ? '' : `entryY=${options.entryY}`,
      options.exitX == null ? '' : 'exitDx=0',
      options.exitY == null ? '' : 'exitDy=0',
      options.entryX == null ? '' : 'entryDx=0',
      options.entryY == null ? '' : 'entryDy=0',
    ].filter(Boolean);
    const style = [
      options.straight ? 'edgeStyle=none' : 'edgeStyle=orthogonalEdgeStyle',
      'orthogonalLoop=1',
      'jettySize=18',
      'html=1',
      'rounded=1',
      `strokeColor=${options.color || PALETTE.edge}`,
      `fontColor=${PALETTE.ink}`,
      `fontFamily=${FONT}`,
      `fontSize=${options.fontSize ?? 16}`,
      `strokeWidth=${options.strokeWidth ?? 2}`,
      `dashed=${options.dashed ? 1 : 0}`,
      `endArrow=${options.arrow === false ? 'none' : 'blockThin'}`,
      `endFill=${options.arrow === false ? 0 : 1}`,
      'endSize=9',
      'labelBackgroundColor=#FFFFFF',
      ...ports,
    ].join(';') + ';';
    const points = options.points?.length
      ? `\n            <Array as="points">${options.points.map(([x, y]) => `<mxPoint x="${x}" y="${y}"/>`).join('')}</Array>`
      : '';
    this.cells.push(
      `        <mxCell id="${escapeXml(id)}" value="${xmlLabel(options.label || '')}" style="${style}" edge="1" parent="${escapeXml(options.parent || '1')}" source="${escapeXml(source)}" target="${escapeXml(target)}">\n` +
      `          <mxGeometry relative="1" as="geometry">${points}\n          </mxGeometry>\n` +
      '        </mxCell>',
    );
  }

  toXml() {
    this.text(`figure_${this.number}_number`, `图 ${this.number}`, this.width / 2 - 80, this.height - 52, 160, 34, {
      fontSize: 24,
      bold: true,
    });
    return `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" modified="2026-09-04T00:00:00.000Z" agent="drawio-skill" version="31.3.2" type="device">
  <diagram id="patent-figure-${this.number}" name="图 ${this.number} ${escapeXml(this.title)}">
    <mxGraphModel dx="${this.width}" dy="${this.height}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="${this.width}" pageHeight="${this.height}" background="#FFFFFF" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
${this.cells.join('\n')}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
`;
  }
}

function drawMatrix(f, id, label, x, y, columns, rows, role = 'data', options = {}) {
  const colors = roleColors(role);
  const width = options.width ?? 210;
  const height = options.height ?? 150;
  const labelHeight = options.labelHeight ?? 42;
  const gap = options.gap ?? 4;
  f.text(`${id}_label`, label, x, y, width, labelHeight, { fontSize: options.fontSize ?? 17, bold: true });
  const gridY = y + labelHeight + 4;
  const cellWidth = (width - gap * (columns - 1)) / columns;
  const cellHeight = (height - labelHeight - 4 - gap * (rows - 1)) / rows;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const emphasis = options.emphasis?.includes(row * columns + column);
      f.vertex(`${id}_cell_${row}_${column}`, '', [
        'rounded=0',
        `fillColor=${emphasis ? colors.stroke : colors.fill}`,
        `strokeColor=${colors.stroke}`,
        `opacity=${emphasis ? 86 : 100}`,
        'strokeWidth=1.2',
      ].join(';') + ';', x + column * (cellWidth + gap), gridY + row * (cellHeight + gap), cellWidth, cellHeight);
    }
  }
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawNeuralNetwork(f, id, label, x, y, width, height, role = 'compute', options = {}) {
  const layers = options.layers ?? [3, 5, 3, 1];
  const colors = roleColors(role);
  const labelHeight = 44;
  f.text(`${id}_label`, label, x, y, width, labelHeight, { fontSize: options.fontSize ?? 18, bold: true });
  const usableHeight = height - labelHeight - 12;
  const nodeSize = options.nodeSize ?? 22;
  const layerXs = layers.map((_, index) => x + 20 + index * ((width - 40 - nodeSize) / Math.max(1, layers.length - 1)));
  const layerNodes = layers.map((count, layerIndex) => {
    const spacing = count === 1 ? 0 : (usableHeight - nodeSize) / (count - 1);
    return Array.from({ length: count }, (_, nodeIndex) => {
      const nodeY = y + labelHeight + 6 + (count === 1 ? (usableHeight - nodeSize) / 2 : nodeIndex * spacing);
      return f.circle(`${id}_l${layerIndex}_n${nodeIndex}`, '', layerXs[layerIndex], nodeY, nodeSize, role, {
        strokeWidth: 1.3,
      });
    });
  });
  for (let layerIndex = 0; layerIndex < layerNodes.length - 1; layerIndex += 1) {
    for (const source of layerNodes[layerIndex]) {
      for (const target of layerNodes[layerIndex + 1]) {
        f.edge(`${id}_e_${source}_${target}`, source, target, {
          straight: true,
          arrow: false,
          color: colors.stroke,
          strokeWidth: 0.8,
        });
      }
    }
  }
  return { input: layerNodes[0], output: layerNodes.at(-1)[0] };
}

function drawSceneGlyph(f, id, label, x, y, width, height, role = 'data') {
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const baseY = y + height - 28;
  const buildings = [
    [x + 25, baseY - 55, 58, 55],
    [x + 92, baseY - 92, 66, 92],
    [x + 167, baseY - 68, 54, 68],
    [x + 230, baseY - 112, 72, 112],
  ];
  buildings.forEach(([bx, by, bw, bh], index) => f.cube(`${id}_building_${index}`, '', bx, by, bw, bh, role, { strokeWidth: 1.4 }));
  const left = f.point(`${id}_ground_l`, x + 10, baseY + 10);
  const right = f.point(`${id}_ground_r`, x + width - 10, baseY + 10);
  f.edge(`${id}_ground`, left, right, { straight: true, arrow: false, color: roleColors(role).stroke, strokeWidth: 1.5 });
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawCameraFrustum(f, id, label, x, y, width, height, role = 'render', options = {}) {
  const colors = roleColors(role);
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: options.fontSize ?? 17, bold: true });
  const camera = f.node(`${id}_camera`, '', x + 12, y + height / 2 - 18, 42, 36, role, {
    shape: 'shape=trapezoid;direction=east',
    strokeWidth: 1.6,
  });
  const top = f.point(`${id}_top`, x + width - 18, y + 48);
  const bottom = f.point(`${id}_bottom`, x + width - 18, y + height - 15);
  f.edge(`${id}_ray_top`, camera, top, { straight: true, arrow: false, dashed: options.dashed, color: colors.stroke, strokeWidth: 1.4 });
  f.edge(`${id}_ray_bottom`, camera, bottom, { straight: true, arrow: false, dashed: options.dashed, color: colors.stroke, strokeWidth: 1.4 });
  for (let i = 0; i < 3; i += 1) {
    f.cube(`${id}_object_${i}`, '', x + width - 80 + i * 20, y + 68 + i * 30, 28, 28, i === 1 ? 'schedule' : 'data', { strokeWidth: 1.1 });
  }
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawRelationGraph(f, id, label, x, y, width, height, role = 'compute') {
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const coords = [[0.12, 0.58], [0.33, 0.25], [0.50, 0.68], [0.69, 0.30], [0.86, 0.62]];
  const nodes = coords.map(([px, py], index) => f.circle(
    `${id}_node_${index}`,
    index === 2 ? '目标' : '',
    x + px * width - 18,
    y + 36 + py * (height - 56) - 18,
    36,
    index === 2 ? 'schedule' : role,
    { fontSize: 12 },
  ));
  [[0, 2], [1, 2], [3, 2], [4, 2], [1, 3]].forEach(([a, b], index) => f.edge(
    `${id}_edge_${index}`,
    nodes[a],
    nodes[b],
    { straight: true, color: roleColors(role).stroke, strokeWidth: index === 4 ? 1 : 1.5 },
  ));
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawHierarchy(f, id, label, x, y, width, height, role = 'compute') {
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const bottom = [0.12, 0.32, 0.58, 0.82].map((px, index) => f.circle(
    `${id}_instance_${index}`, '', x + px * width - 14, y + height - 38, 28, 'data', { strokeWidth: 1.2 },
  ));
  const groups = [0.28, 0.70].map((px, index) => f.circle(
    `${id}_group_${index}`, '', x + px * width - 18, y + height * 0.54, 36, role, { strokeWidth: 1.3 },
  ));
  const root = f.circle(`${id}_root`, '', x + width / 2 - 22, y + 42, 44, 'schedule', { strokeWidth: 1.4 });
  bottom.forEach((node, index) => f.edge(`${id}_lower_${index}`, node, groups[index < 2 ? 0 : 1], {
    straight: true, arrow: false, strokeWidth: 1.1,
  }));
  groups.forEach((node, index) => f.edge(`${id}_upper_${index}`, node, root, {
    straight: true, arrow: false, strokeWidth: 1.3,
  }));
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawPointCloud(f, id, label, x, y, width, height, role = 'data') {
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const points = [[.14,.70],[.20,.45],[.28,.64],[.34,.32],[.42,.52],[.48,.74],[.56,.39],[.64,.57],[.70,.26],[.78,.48],[.84,.68],[.57,.80],[.37,.82]];
  points.forEach(([px, py], index) => f.circle(
    `${id}_point_${index}`, '', x + px * width - 5, y + 24 + py * (height - 38) - 5, 10,
    index % 4 === 0 ? 'schedule' : role, { strokeWidth: 0.8 },
  ));
  const tl = f.point(`${id}_tl`, x + 18, y + 45);
  const tr = f.point(`${id}_tr`, x + width - 18, y + 45);
  const bl = f.point(`${id}_bl`, x + 18, y + height - 10);
  const br = f.point(`${id}_br`, x + width - 18, y + height - 10);
  f.edge(`${id}_box_t`, tl, tr, { straight: true, dashed: true, arrow: false, strokeWidth: 1 });
  f.edge(`${id}_box_l`, tl, bl, { straight: true, dashed: true, arrow: false, strokeWidth: 1 });
  f.edge(`${id}_box_b`, bl, br, { straight: true, dashed: true, arrow: false, strokeWidth: 1 });
  f.edge(`${id}_box_r`, tr, br, { straight: true, dashed: true, arrow: false, strokeWidth: 1 });
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawSpectrum(f, id, label, x, y, width, height, role = 'compute') {
  const colors = roleColors(role);
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const origin = f.point(`${id}_origin`, x + 24, y + height - 18);
  const xEnd = f.point(`${id}_x_end`, x + width - 12, y + height - 18);
  const yEnd = f.point(`${id}_y_end`, x + 24, y + 42);
  f.edge(`${id}_axis_x`, origin, xEnd, { straight: true, color: colors.stroke, strokeWidth: 1.2 });
  f.edge(`${id}_axis_y`, origin, yEnd, { straight: true, color: colors.stroke, strokeWidth: 1.2 });
  const curve = [];
  const samples = 14;
  for (let index = 0; index < samples; index += 1) {
    const px = x + 30 + index * ((width - 50) / (samples - 1));
    const phase = index / (samples - 1) * Math.PI * 3.5;
    const py = y + 82 + Math.sin(phase) * 28 + Math.sin(phase * 0.43) * 11;
    curve.push(f.point(`${id}_curve_${index}`, px, py));
    if (index > 0) {
      f.edge(`${id}_curve_edge_${index}`, curve[index - 1], curve[index], {
        straight: true, arrow: false, color: colors.stroke, strokeWidth: 2,
      });
    }
  }
  [1, 5, 9, 13].forEach((index) => f.circle(`${id}_sample_${index}`, '', x + 30 + index * ((width - 50) / (samples - 1)) - 5,
    y + 82 + Math.sin(index / (samples - 1) * Math.PI * 3.5) * 28 + Math.sin(index / (samples - 1) * Math.PI * 3.5 * 0.43) * 11 - 5,
    10, 'schedule', { strokeWidth: 0.8 }));
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawSurvivalCurves(f, id, label, x, y, width, height, role = 'compute') {
  const colors = [roleColors(role).stroke, PALETTE.schedule.stroke, PALETTE.render.stroke];
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const origin = f.point(`${id}_origin`, x + 24, y + height - 18);
  const xEnd = f.point(`${id}_x_end`, x + width - 12, y + height - 18);
  const yEnd = f.point(`${id}_y_end`, x + 24, y + 42);
  f.edge(`${id}_axis_x`, origin, xEnd, { straight: true, color: PALETTE.edge, strokeWidth: 1.1 });
  f.edge(`${id}_axis_y`, origin, yEnd, { straight: true, color: PALETTE.edge, strokeWidth: 1.1 });
  colors.forEach((color, curveIndex) => {
    let previous = null;
    for (let index = 0; index < 12; index += 1) {
      const t = index / 11;
      const px = x + 30 + t * (width - 50);
      const response = Math.exp(-(1.0 + curveIndex * 0.65) * t) * (0.80 - curveIndex * 0.11) + 0.08;
      const py = y + height - 24 - response * (height - 74);
      const point = f.point(`${id}_${curveIndex}_${index}`, px, py);
      if (previous) {
        f.edge(`${id}_edge_${curveIndex}_${index}`, previous, point, {
          straight: true, arrow: false, color, strokeWidth: 2,
        });
      }
      previous = point;
    }
  });
  f.text(`${id}_x_label`, '归一化距离', x + width - 112, y + height - 16, 100, 22, { fontSize: 13, align: 'right' });
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawVector(f, id, label, x, y, length, role = 'data', options = {}) {
  const colors = roleColors(role);
  const cellWidth = options.cellWidth ?? 25;
  const cellHeight = options.cellHeight ?? 32;
  f.text(`${id}_label`, label, x, y, Math.max(120, length * cellWidth), 30, { fontSize: options.fontSize ?? 16, bold: true });
  for (let index = 0; index < length; index += 1) {
    f.vertex(`${id}_cell_${index}`, '', [
      'rounded=0',
      `fillColor=${index % 3 === 0 ? colors.stroke : colors.fill}`,
      `strokeColor=${colors.stroke}`,
      `opacity=${index % 3 === 0 ? 78 : 100}`,
      'strokeWidth=1.1',
    ].join(';') + ';', x + index * cellWidth, y + 34, cellWidth - 3, cellHeight);
  }
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, length * cellWidth, cellHeight + 34);
}

function drawPriorityLanes(f, id, x, y, width, height) {
  const lanes = [
    ['当前显示', 'render'],
    ['首屏准备', 'data'],
    ['区域候选', 'compute'],
    ['低阈值预取', 'schedule'],
  ];
  const laneHeight = height / lanes.length;
  lanes.forEach(([label, role], index) => {
    const colors = roleColors(role);
    f.node(`${id}_lane_${index}`, label, x, y + index * laneHeight, width, laneHeight - 5, role, {
      shape: 'shape=parallelogram;perimeter=parallelogramPerimeter;fixedSize=1',
      align: 'left', fontSize: 15, strokeWidth: 1.3,
    });
    f.document(`${id}_file_${index}`, '', x + width - 45, y + index * laneHeight + 8, 26, 30, role, {
      strokeWidth: 1,
    });
    f.circle(`${id}_priority_${index}`, String(index + 1), x + width - 78, y + index * laneHeight + 11, 23, role, {
      fontSize: 12, strokeWidth: 1,
    });
    void colors;
  });
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawStateTable(f, id, label, x, y, width, height) {
  f.text(`${id}_label`, label, x, y, width, 36, { fontSize: 17, bold: true });
  const rows = 5;
  const columns = 4;
  const headers = ['资源', '等级', '得分', '状态'];
  const cellWidth = width / columns;
  const cellHeight = (height - 38) / rows;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const header = row === 0;
      f.node(`${id}_${row}_${column}`, header ? headers[column] : (column === 0 ? `G${row}` : ''),
        x + column * cellWidth, y + 38 + row * cellHeight, cellWidth, cellHeight,
        header ? 'schedule' : 'neutral', {
          shape: 'rounded=0', fontSize: header ? 13 : 12, strokeWidth: 1, bold: header,
        });
      if (!header && column > 0) {
        f.vertex(`${id}_mark_${row}_${column}`, '', [
          'ellipse',
          `fillColor=${[PALETTE.render.stroke, PALETTE.data.stroke, PALETTE.compute.stroke][column - 1]}`,
          'strokeColor=none',
        ].join(';') + ';', x + column * cellWidth + cellWidth / 2 - 5, y + 38 + row * cellHeight + cellHeight / 2 - 5, 10, 10);
      }
    }
  }
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawSlotArray(f, id, label, x, y, slots, role = 'render', options = {}) {
  const cellWidth = options.cellWidth ?? 52;
  const cellHeight = options.cellHeight ?? 58;
  f.text(`${id}_label`, label, x, y, slots * cellWidth, 34, { fontSize: 17, bold: true });
  for (let index = 0; index < slots; index += 1) {
    const active = index < (options.active ?? slots - 2);
    f.cube(`${id}_slot_${index}`, active ? `i${index}` : '', x + index * cellWidth, y + 40, cellWidth - 8, cellHeight, active ? role : 'neutral', {
      fontSize: 12, strokeWidth: 1.2, dashed: !active,
    });
  }
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, slots * cellWidth, cellHeight + 40);
}

function drawSceneTree(f, id, label, x, y, width, height) {
  f.text(`${id}_label`, label, x, y, width, 34, { fontSize: 17, bold: true });
  const root = f.circle(`${id}_root`, '', x + width / 2 - 17, y + 42, 34, 'neutral', { strokeWidth: 1.2 });
  const middle = [0.30, 0.70].map((px, index) => f.circle(`${id}_middle_${index}`, '', x + px * width - 15, y + 92, 30, 'compute', { strokeWidth: 1.1 }));
  const leaves = [0.13, 0.33, 0.57, 0.79].map((px, index) => f.cube(`${id}_leaf_${index}`, '', x + px * width - 14, y + height - 36, 28, 28, 'data', { strokeWidth: 1 }));
  middle.forEach((node, index) => f.edge(`${id}_root_e_${index}`, root, node, { straight: true, arrow: false, strokeWidth: 1.1 }));
  leaves.forEach((node, index) => f.edge(`${id}_leaf_e_${index}`, middle[index < 2 ? 0 : 1], node, { straight: true, arrow: false, strokeWidth: 1 }));
  return f.vertex(id, '', 'rounded=0;fillColor=none;strokeColor=none;opacity=0;', x, y, width, height);
}

function drawBitmap(f, id, label, x, y, width, height, role = 'render', emphasis = []) {
  return drawMatrix(f, id, label, x, y, 8, 5, role, { width, height, labelHeight: 36, gap: 2, emphasis, fontSize: 16 });
}

function drawFileStack(f, id, label, x, y, width, height, role = 'data') {
  for (let index = 2; index >= 0; index -= 1) {
    f.document(`${id}_doc_${index}`, index === 0 ? label : '', x + index * 12, y - index * 10, width, height, role, {
      fontSize: 16,
      strokeWidth: 1.4,
    });
  }
  return `${id}_doc_0`;
}

function buildFigure1() {
  const f = new Diagram(1, '系统架构与端到端数据流图');
  f.container('f1_offline', '100  离线处理端', 35, 25, 1730, 250, 'offline');
  f.container('f1_server', '200  资源服务端', 35, 300, 1730, 165, 'data');
  f.container('f1_browser', '300  浏览器端', 35, 495, 1730, 535, 'render');
  f.container('f1_worker', '330  后台可见性计算', 70, 555, 720, 410, 'compute');
  f.container('f1_main', '340/350/360  页面调度、加载与显示', 825, 555, 905, 410, 'schedule');

  drawSceneGlyph(f, 'f1_scene', '111  完整实例化场景', 70, 82, 285, 150, 'data');
  drawCameraFrustum(f, 'f1_sample', '112  视点区域采样', 405, 83, 230, 145, 'offline');
  drawRelationGraph(f, 'f1_encode', '113/114  几何与遮挡关系编码', 690, 82, 245, 150, 'compute');
  const offlineNetwork = drawNeuralNetwork(f, 'f1_train', '115/116  查询模型训练与校准', 990, 76, 250, 160, 'compute', {
    layers: [3, 4, 2, 1], nodeSize: 18, fontSize: 17,
  });
  const exportFile = drawFileStack(f, 'f1_export', '117  浏览器运行包\n特征、包围体、映射与权重', 1375, 110, 285, 92, 'data');

  f.store('f1_runtime_store', '210  实例运行数据\n小体量、启动常驻', 185, 345, 330, 80, 'data');
  f.store('f1_glb_store', '220  主体模型资源\n几何、材质与纹理', 1285, 345, 330, 80, 'data');

  drawMatrix(f, 'f1_runtime_input', '210  常驻实例特征表', 95, 625, 8, 4, 'data', {
    width: 245, height: 145, labelHeight: 40, gap: 3, emphasis: [2, 5, 11, 19, 23],
  });
  drawCameraFrustum(f, 'f1_camera', '320  当前相机与视点区域', 95, 790, 245, 135, 'render');
  f.chip('f1_compute', '330  GPU 批量可见性查询', 410, 670, 260, 115, 'compute', { fontSize: 18 });
  drawBitmap(f, 'f1_raw_outputs', '335/336  实例位图与资源得分', 505, 830, 235, 105, 'compute', [1, 2, 7, 9, 12, 18, 23, 30]);

  drawFileStack(f, 'f1_startup', '342  首屏准备', 865, 640, 175, 70, 'data');
  f.node('f1_classify', '341/343/344  四级资源需求', 865, 770, 210, 80, 'schedule');
  drawMatrix(f, 'f1_state', '345  统一资源状态表', 1140, 665, 5, 5, 'schedule', {
    width: 235, height: 155, labelHeight: 38, gap: 2, emphasis: [1, 5, 12, 18, 22],
  });
  drawFileStack(f, 'f1_pipeline', '350  下载 → 解析 → 挂载', 1450, 665, 220, 82, 'schedule');
  f.node('f1_delta', '363  实例显示差量', 1145, 865, 225, 65, 'render');
  drawSceneGlyph(f, 'f1_display', '360  增量显示结果', 1440, 790, 245, 145, 'render');

  f.edge('f1_e_1', 'f1_scene', 'f1_sample');
  f.edge('f1_e_2', 'f1_sample', 'f1_encode');
  f.edge('f1_e_3', 'f1_encode', offlineNetwork.input[1]);
  f.edge('f1_e_4', offlineNetwork.output, exportFile);
  f.edge('f1_e_5', exportFile, 'f1_runtime_store', {
    points: [[1715, 190], [1715, 320], [530, 320], [530, 385]],
    exitX: 1, exitY: 0.5, entryX: 1, entryY: 0.5,
  });
  f.edge('f1_e_6', 'f1_runtime_store', 'f1_runtime_input', {
    points: [[350, 475], [218, 475], [218, 610]], exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0,
    label: '启动时下载并常驻',
  });
  f.edge('f1_e_7', 'f1_runtime_input', 'f1_compute');
  f.edge('f1_e_8', 'f1_camera', 'f1_compute');
  f.edge('f1_e_9', 'f1_compute', 'f1_raw_outputs', { exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0 });
  f.edge('f1_e_10', 'f1_raw_outputs', 'f1_classify', {
    points: [[800, 882], [800, 810]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5,
    label: '资源结果',
  });
  f.edge('f1_e_11', 'f1_raw_outputs', 'f1_delta', {
    points: [[950, 905], [950, 898]], exitX: 1, exitY: 0.75, entryX: 0, entryY: 0.5,
    label: '实例差量',
  });
  f.edge('f1_e_12', 'f1_startup_doc_0', 'f1_state', { points: [[1095, 675], [1110, 675], [1110, 742]] });
  f.edge('f1_e_13', 'f1_classify', 'f1_state');
  f.edge('f1_e_14', 'f1_state', 'f1_pipeline_doc_0');
  f.edge('f1_e_15', 'f1_glb_store', 'f1_pipeline_doc_0', {
    points: [[1450, 475], [1560, 475], [1560, 635]], exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0,
    label: '按状态表请求',
  });
  f.edge('f1_e_16', 'f1_pipeline_doc_0', 'f1_display', { exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0 });
  f.edge('f1_e_17', 'f1_delta', 'f1_display');
  return f;
}

function buildFigure2() {
  const f = new Diagram(2, '离线模型构建与运行数据导出图');
  f.container('f2_evidence', '110  监督与场景证据', 40, 35, 390, 940, 'offline');
  f.container('f2_representation', '118/119  逐实例固定表示', 465, 35, 560, 940, 'compute');
  f.container('f2_training', '115/120/121/122  查询模型训练', 1060, 35, 420, 940, 'offline');
  f.container('f2_exporting', '116/117  校准与导出', 1515, 35, 245, 940, 'data');

  drawSceneGlyph(f, 'f2_scene', '111  实例化场景及资源映射', 75, 95, 320, 155, 'data');
  drawCameraFrustum(f, 'f2_samples', '112  视点区域采样与可见并集', 75, 285, 320, 145, 'offline');
  drawRelationGraph(f, 'f2_relations', '113  真实几何遮挡边', 75, 470, 320, 150, 'offline');
  drawHierarchy(f, 'f2_hierarchy', '实例 → 局部组 → 结构组', 75, 660, 320, 175, 'compute');

  drawPointCloud(f, 'f2_geometry', '114  单实例几何编码', 500, 100, 205, 150, 'data');
  drawMatrix(f, 'f2_geo_table', '96 维几何特征', 775, 105, 8, 5, 'data', {
    width: 205, height: 140, labelHeight: 38, gap: 2, emphasis: [1, 6, 13, 19, 27, 32],
  });
  drawRelationGraph(f, 'f2_relation', '118  分层遮挡关系聚合', 500, 335, 235, 165, 'compute');
  const relationNetwork = drawNeuralNetwork(f, 'f2_shared', '共享关系先验', 785, 330, 195, 170, 'compute', {
    layers: [3, 4, 2], nodeSize: 18, fontSize: 16,
  });
  f.circle('f2_calibration_residual', '+', 555, 590, 54, 'schedule', { fontSize: 24 });
  f.text('f2_calibration_label', '119  逐实例校准残差', 495, 650, 175, 35, { fontSize: 16, bold: true });
  drawMatrix(f, 'f2_survival', '逐实例遮挡生存场  4 × 7', 745, 555, 7, 4, 'compute', {
    width: 240, height: 150, labelHeight: 40, gap: 3, emphasis: [1, 4, 8, 13, 18, 23, 27],
  });
  drawMatrix(f, 'f2_fixed', '逐实例固定表  96 + 28 维', 620, 775, 10, 5, 'data', {
    width: 255, height: 150, labelHeight: 42, gap: 2, emphasis: [3, 8, 15, 24, 38, 46],
  });

  drawSpectrum(f, 'f2_spectral', '120  视点区域积分频谱', 1100, 125, 340, 190, 'compute');
  const queryNetwork = drawNeuralNetwork(f, 'f2_query_model', '121  共享轻量可见性查询网络', 1105, 380, 330, 210, 'compute', {
    layers: [4, 6, 4, 1], nodeSize: 20,
  });
  f.node('f2_loss', '122  训练目标\n逐姿态平衡分类\n加权召回保护与边界间隔', 1110, 655, 320, 115, 'offline');

  f.decision('f2_threshold', '116  校准集\n冻结显示阈值', 1540, 215, 195, 130, 'offline');
  drawFileStack(f, 'f2_runtime', '117  浏览器运行包\n固定表、网络权重\n包围体、映射与阈值', 1540, 510, 175, 135, 'data');
  f.text('f2_note', '训练样本和关系图只参与离线计算；浏览器下载固定实例表与轻量查询参数。', 1010, 895, 680, 35, {
    fontSize: 18,
    bold: true,
  });

  f.edge('f2_e_1', 'f2_scene', 'f2_geometry');
  f.edge('f2_e_2', 'f2_geometry', 'f2_geo_table');
  f.edge('f2_e_3', 'f2_relations', 'f2_relation');
  f.edge('f2_e_4', 'f2_hierarchy', 'f2_relation');
  f.edge('f2_e_5', 'f2_relation', relationNetwork.input[1]);
  f.edge('f2_e_6', relationNetwork.output, 'f2_calibration_residual', {
    points: [[1000, 420], [1000, 530], [582, 530]], exitX: 1, exitY: 0.5, entryX: 0.5, entryY: 0,
  });
  f.edge('f2_e_7', 'f2_calibration_residual', 'f2_survival');
  f.edge('f2_e_8', 'f2_geo_table', 'f2_fixed', {
    points: [[1000, 170], [1000, 735], [750, 735]], exitX: 1, exitY: 0.5, entryX: 0.5, entryY: 0,
  });
  f.edge('f2_e_9', 'f2_survival', 'f2_fixed', { exitX: 0.5, exitY: 1, entryX: 1, entryY: 0.5 });
  f.edge('f2_e_10', 'f2_samples', 'f2_spectral', {
    points: [[450, 332], [450, 90], [1085, 90], [1085, 218]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5,
  });
  f.edge('f2_e_11', 'f2_spectral', queryNetwork.input[1]);
  f.edge('f2_e_12', 'f2_fixed', queryNetwork.input[2], {
    points: [[1035, 822], [1035, 450]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5,
  });
  f.edge('f2_e_14', queryNetwork.output, 'f2_loss');
  f.edge('f2_e_15', queryNetwork.output, 'f2_threshold');
  f.edge('f2_e_16', 'f2_threshold', 'f2_runtime_doc_0');
  f.edge('f2_e_17', 'f2_fixed', 'f2_runtime_doc_0', {
    points: [[1015, 822], [1495, 822], [1495, 580]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.65,
  });
  f.edge('f2_e_18', queryNetwork.output, 'f2_runtime_doc_0', { exitX: 1, exitY: 0.65, entryX: 0, entryY: 0.3 });
  return f;
}

function buildFigure3() {
  const f = new Diagram(3, '视角条件化可见性查询模型图');
  f.container('f3_inputs', '210/320  每个候选实例的输入', 40, 35, 350, 950, 'data');
  f.container('f3_query', '120  视点区域条件化查询', 425, 35, 830, 950, 'compute');
  f.container('f3_head', '121  浏览器轻量推理', 1290, 35, 470, 950, 'render');

  drawPointCloud(f, 'f3_geometry', '固定几何特征  96 维', 75, 100, 280, 165, 'data');
  drawMatrix(f, 'f3_survival_coeff', '逐实例遮挡生存场  4 × 7', 80, 320, 7, 4, 'compute', {
    width: 270, height: 160, labelHeight: 42, gap: 3, emphasis: [1, 4, 9, 13, 18, 24, 27],
  });
  drawCameraFrustum(f, 'f3_center', '中心视角：方向、距离与画面位置', 75, 535, 280, 145, 'render');
  drawMatrix(f, 'f3_region', '视点区域的空间与方向范围', 100, 755, 5, 3, 'render', {
    width: 230, height: 125, labelHeight: 42, gap: 4, emphasis: [1, 2, 6, 7, 8, 12],
  });

  const basisNetwork = drawNeuralNetwork(f, 'f3_basis', '共享方向基  视线 → 4 个方向响应', 470, 105, 300, 185, 'compute', {
    layers: [3, 5, 4], nodeSize: 19, fontSize: 17,
  });
  drawVector(f, 'f3_depth', '归一化观察深度', 500, 345, 7, 'render', { cellWidth: 34, cellHeight: 30 });
  drawSurvivalCurves(f, 'f3_survival_query', '方向条件化的遮挡生存响应', 840, 130, 350, 230, 'compute');
  f.text('f3_survival_note', '矩阵系数与方向响应共同确定不同观察方向下随距离变化的遮挡趋势', 790, 395, 425, 48, { fontSize: 16 });

  drawCameraFrustum(f, 'f3_variation', '中心视角与区域变化轴', 475, 515, 290, 155, 'render');
  drawSpectrum(f, 'f3_integral', '视点区域积分频谱', 840, 500, 350, 220, 'compute');
  drawVector(f, 'f3_moment', '区域均值、边界与高频摘要', 865, 765, 10, 'compute', { cellWidth: 30, cellHeight: 34 });
  f.text('f3_integral_note', '解析积分一次概括整个视点区域', 845, 860, 335, 35, { fontSize: 16 });

  drawVector(f, 'f3_assemble', '联合查询向量', 1350, 165, 11, 'render', { cellWidth: 29, cellHeight: 38, fontSize: 18 });
  const queryNetwork = drawNeuralNetwork(f, 'f3_mlp', '共享轻量查询网络', 1360, 370, 300, 260, 'compute', {
    layers: [4, 6, 4, 1], nodeSize: 22, fontSize: 18,
  });
  f.circle('f3_score', 'p', 1460, 730, 100, 'render', { fontSize: 32, strokeWidth: 2.4 });
  f.text('f3_score_label', '连续可见性分数  0 至 1', 1385, 845, 250, 38, { fontSize: 18, bold: true });
  f.text('f3_runtime_note', '固定实例表常驻内存，浏览器按候选实例批量计算。', 1330, 920, 390, 35, { fontSize: 16, bold: true });

  f.edge('f3_e_1', 'f3_center', basisNetwork.input[1]);
  f.edge('f3_e_2', 'f3_center', 'f3_depth');
  f.edge('f3_e_3', 'f3_survival_coeff', 'f3_survival_query', {
    points: [[405, 382], [405, 475], [790, 475], [790, 278]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5,
  });
  f.edge('f3_e_4', basisNetwork.output, 'f3_survival_query');
  f.edge('f3_e_5', 'f3_depth', 'f3_survival_query');
  f.edge('f3_e_6', 'f3_center', 'f3_variation');
  f.edge('f3_e_7', 'f3_region', 'f3_variation');
  f.edge('f3_e_8', 'f3_variation', 'f3_integral');
  f.edge('f3_e_8b', 'f3_integral', 'f3_moment');
  f.edge('f3_e_9', 'f3_geometry', 'f3_assemble', {
    points: [[405, 188], [405, 85], [1508, 85], [1508, 145]], exitX: 1, exitY: 0.5, entryX: 0.5, entryY: 0,
  });
  f.edge('f3_e_10', 'f3_center', 'f3_assemble', {
    points: [[405, 582], [405, 850], [1735, 850], [1735, 330]], exitX: 1, exitY: 0.5, entryX: 1, entryY: 0.55,
  });
  f.edge('f3_e_11', 'f3_survival_query', 'f3_assemble');
  f.edge('f3_e_12', 'f3_moment', 'f3_assemble', {
    points: [[1220, 820], [1320, 820], [1320, 225]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.8,
  });
  f.edge('f3_e_13', 'f3_assemble', queryNetwork.input[1]);
  f.edge('f3_e_14', queryNetwork.output, 'f3_score');
  return f;
}

function sequenceMessage(f, id, fromX, toX, y, label, options = {}) {
  const source = `${id}_source`;
  const target = `${id}_target`;
  f.point(source, fromX, y);
  f.point(target, toX, y);
  f.edge(id, source, target, {
    straight: true,
    dashed: options.dashed,
    label,
    fontSize: 16,
    color: options.color,
  });
}

function buildFigure4() {
  const f = new Diagram(4, '浏览器启动、可见性推理与资源加载时序图', 1800, 1300);
  f.container('f4_start_phase', '启动阶段', 45, 105, 1710, 300, 'data');
  f.container('f4_predict_phase', '首次区域查询', 45, 430, 1710, 315, 'compute');
  f.container('f4_stream_phase', '持续调度与区域复用', 45, 770, 1710, 445, 'schedule');

  const xs = { main: 195, worker: 635, scheduler: 1130, server: 1605 };
  f.node('f4_main', '300  页面主线程', 85, 25, 220, 60, 'render');
  f.node('f4_worker', '330  可见性工作线程', 515, 25, 240, 60, 'compute');
  f.node('f4_scheduler', '340/350  资源状态调度与流水线', 985, 25, 290, 60, 'schedule');
  f.node('f4_server', '200  资源服务端', 1495, 25, 220, 60, 'data');

  for (const [name, x] of Object.entries(xs)) {
    f.point(`f4_${name}_top`, x, 85);
    f.point(`f4_${name}_bottom`, x, 1215);
    f.edge(`f4_${name}_life`, `f4_${name}_top`, `f4_${name}_bottom`, {
      straight: true,
      dashed: true,
      arrow: false,
      strokeWidth: 1,
    });
  }

  sequenceMessage(f, 'f4_m1', xs.main, xs.scheduler, 165, '提交资源索引与首屏资源顺序');
  sequenceMessage(f, 'f4_m2', xs.scheduler, xs.server, 220, '请求首屏主体模型资源');
  sequenceMessage(f, 'f4_m3', xs.main, xs.worker, 275, '初始化并加载实例运行数据');
  sequenceMessage(f, 'f4_m4', xs.worker, xs.server, 330, '请求固定特征、包围体、映射与权重');
  sequenceMessage(f, 'f4_m5', xs.server, xs.worker, 375, '返回实例运行数据', { dashed: true });

  sequenceMessage(f, 'f4_m6', xs.server, xs.scheduler, 485, '首屏资源逐个返回', { dashed: true });
  sequenceMessage(f, 'f4_m7', xs.main, xs.worker, 540, '发送当前相机与视点区域');
  f.node('f4_worker_compute', '候选生成 → 可见性推理\n→ 当前视锥过滤 → GLB 汇总', 500, 575, 270, 80, 'compute', { fontSize: 17 });
  sequenceMessage(f, 'f4_m8', xs.worker, xs.main, 690, '返回实例编号、区域资源得分、当前显示资源及低阈值预取资源', { dashed: true });
  sequenceMessage(f, 'f4_m9', xs.main, xs.scheduler, 730, '完成需求分级并更新统一状态表');

  sequenceMessage(f, 'f4_m10', xs.scheduler, xs.server, 835, '按可执行时间、需求等级和得分请求资源');
  sequenceMessage(f, 'f4_m11', xs.server, xs.scheduler, 890, '资源数据逐个返回', { dashed: true });
  f.node('f4_pipeline', '排队 → 下载 → 解析 → 挂载 → 驻留\n失败任务按等待时间有限重试', 980, 925, 300, 85, 'schedule', { fontSize: 17 });
  sequenceMessage(f, 'f4_m12', xs.scheduler, xs.main, 1040, '通知驻留资源及实例映射');
  sequenceMessage(f, 'f4_m13', xs.main, xs.worker, 1095, '相机移动：提交新的当前显示视锥');
  sequenceMessage(f, 'f4_m14', xs.worker, xs.main, 1145, '复用区域结果并返回实例与资源差量', { dashed: true });
  sequenceMessage(f, 'f4_m15', xs.main, xs.scheduler, 1190, '提升新进入画面的资源，降级离开画面的区域资源');
  return f;
}

function buildFigure5() {
  const f = new Diagram(5, '一次运行时可见性批量查询数据流图');
  f.container('f5_input', '输入与候选生成', 35, 35, 430, 940, 'data');
  f.container('f5_compute', '330  后台批量计算', 500, 35, 805, 940, 'compute');
  f.container('f5_output', '压缩输出', 1340, 35, 425, 940, 'render');

  drawMatrix(f, 'f5_runtime', '210  实例运行数据', 90, 120, 8, 5, 'data', {
    width: 320, height: 160, labelHeight: 40, gap: 3, emphasis: [1, 7, 12, 18, 25, 31],
  });
  drawCameraFrustum(f, 'f5_camera', '320  当前相机与视点区域', 85, 330, 330, 155, 'render');
  drawCameraFrustum(f, 'f5_frusta', '323/324  候选视锥与显示视锥', 85, 540, 330, 165, 'compute', { dashed: true });
  f.circle('f5_anchor', '321', 205, 780, 92, 'schedule', { fontSize: 22 });
  f.text('f5_anchor_label', '预测锚点与复用边界', 120, 885, 260, 36, { fontSize: 17, bold: true });

  f.chip('f5_gpu', '', 530, 95, 745, 720, 'compute', { fontSize: 21 });
  f.text('f5_gpu_label', 'GPU 并行批量查询', 805, 105, 200, 34, { fontSize: 18, bold: true });
  drawBitmap(f, 'f5_candidate', '331  候选实例位图', 565, 160, 275, 125, 'compute', [0, 2, 4, 8, 11, 17, 21, 26, 31, 34]);
  drawMatrix(f, 'f5_model_input', '固定特征批次', 925, 155, 8, 4, 'data', {
    width: 275, height: 125, labelHeight: 36, gap: 2, emphasis: [2, 7, 11, 19, 25],
  });
  const batchNetwork = drawNeuralNetwork(f, 'f5_model', '332  并行可见性查询', 925, 320, 275, 170, 'compute', {
    layers: [3, 5, 3, 1], nodeSize: 17, fontSize: 16,
  });
  drawBitmap(f, 'f5_region', '333  阈值筛选后的区域位图', 565, 355, 275, 125, 'render', [0, 2, 8, 11, 17, 26, 34]);
  drawCameraFrustum(f, 'f5_display_filter', '334  真实 60° 视锥过滤', 565, 560, 275, 140, 'render');
  drawBitmap(f, 'f5_display', '当前显示实例位图', 925, 550, 275, 125, 'render', [0, 8, 11, 26]);
  drawFileStack(f, 'f5_glb', '336  按资源取最高分', 585, 735, 210, 70, 'schedule');
  drawVector(f, 'f5_compact', '335  压缩编号或差量', 935, 715, 8, 'compute', { cellWidth: 31, cellHeight: 30, fontSize: 16 });
  f.text('f5_note', '中间概率与特征留在计算后端，只回读压缩后的实例和资源结果。', 610, 865, 585, 45, {
    fontSize: 18,
    bold: true,
  });

  drawFileStack(f, 'f5_out_prefetch', '344  低阈值预取资源', 1410, 145, 280, 80, 'schedule');
  drawFileStack(f, 'f5_out_render', '341  当前显示资源', 1410, 355, 280, 80, 'render');
  drawFileStack(f, 'f5_out_region', '336  区域资源与得分', 1410, 565, 280, 80, 'data');
  drawVector(f, 'f5_out_instance', '363  当前实例编号或差量', 1415, 785, 9, 'render', { cellWidth: 30, cellHeight: 34, fontSize: 16 });

  f.edge('f5_e_1', 'f5_runtime', 'f5_candidate');
  f.edge('f5_e_2', 'f5_camera', 'f5_frusta', { exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0 });
  f.edge('f5_e_3', 'f5_frusta', 'f5_candidate', {
    points: [[480, 580], [520, 580], [520, 185]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5,
  });
  f.edge('f5_e_4', 'f5_anchor', 'f5_frusta', { exitX: 0.5, exitY: 0, entryX: 0.5, entryY: 1 });
  f.edge('f5_e_5', 'f5_candidate', 'f5_model_input');
  f.edge('f5_e_5b', 'f5_model_input', batchNetwork.input[1]);
  f.edge('f5_e_6', batchNetwork.output, 'f5_region');
  f.edge('f5_e_7', 'f5_region', 'f5_display_filter');
  f.edge('f5_e_7b', 'f5_display_filter', 'f5_display');
  f.edge('f5_e_8', batchNetwork.output, 'f5_glb_doc_0', {
    points: [[900, 405], [900, 700], [820, 700]], exitX: 0, exitY: 0.5, entryX: 1, entryY: 0.5,
  });
  f.edge('f5_e_9', 'f5_display', 'f5_compact');
  f.edge('f5_e_10', 'f5_compact', 'f5_out_instance', {
    points: [[1285, 765], [1285, 835]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5,
  });
  f.edge('f5_e_11', 'f5_glb_doc_0', 'f5_out_region_doc_0');
  f.edge('f5_e_12', 'f5_display', 'f5_out_render_doc_0');
  f.edge('f5_e_13', batchNetwork.output, 'f5_out_prefetch_doc_0');
  return f;
}

function buildFigure6() {
  const f = new Diagram(6, '视点区域、双视锥与结果复用图');
  f.container('f6_geometry', '视点区域与两个视锥', 35, 35, 820, 940, 'render');
  f.container('f6_reuse', '370  相机移动后的区域结果复用', 890, 35, 875, 940, 'compute');

  f.node('f6_back_camera', '候选相机', 95, 445, 150, 65, 'compute', { shape: 'shape=trapezoid;direction=east' });
  f.node('f6_viewcell', '322  视点区域', 280, 360, 220, 235, 'render');
  f.node('f6_current_camera', '320  当前相机', 315, 445, 150, 65, 'render', { shape: 'shape=trapezoid;direction=east' });
  f.point('f6_cand_top', 770, 165);
  f.point('f6_cand_bottom', 770, 790);
  f.point('f6_disp_top', 770, 315);
  f.point('f6_disp_bottom', 770, 650);
  f.edge('f6_cand_1', 'f6_back_camera', 'f6_cand_top', { straight: true, dashed: true, arrow: false });
  f.edge('f6_cand_2', 'f6_back_camera', 'f6_cand_bottom', { straight: true, dashed: true, arrow: false });
  f.edge('f6_disp_1', 'f6_current_camera', 'f6_disp_top', { straight: true, arrow: false, color: PALETTE.render.stroke });
  f.edge('f6_disp_2', 'f6_current_camera', 'f6_disp_bottom', { straight: true, arrow: false, color: PALETTE.render.stroke });
  f.text('f6_candidate_label', '323  覆盖视点区域的候选视锥', 485, 150, 275, 35, { fontSize: 18, bold: true });
  f.text('f6_display_label', '324  当前显示视锥', 520, 300, 225, 35, { fontSize: 18, bold: true, color: PALETTE.render.stroke });
  [[555, 410], [625, 515], [690, 355], [705, 665]].forEach(([x, y], index) => {
    f.cube(`f6_instance_${index}`, '', x, y, 42, 42, index === 3 ? 'neutral' : 'data', { strokeWidth: 1.2 });
  });
  f.text('f6_contract', '候选视锥覆盖视点区域内允许相机的显示范围并集。\n浏览器针对一个预测锚点执行一次区域查询。', 150, 845, 590, 65, { fontSize: 18 });

  drawBitmap(f, 'f6_cache', '321/362  锚点与缓存区域位图', 950, 120, 280, 145, 'data', [1, 2, 7, 8, 14, 22, 29]);
  f.node('f6_change', '320  相机位置或方向变化', 1370, 155, 300, 80, 'render');
  f.decision('f6_inside', '相机参数仍在\n视点区域内？', 1195, 350, 250, 155, 'schedule');
  f.node('f6_refilter', '334  使用新的当前显示视锥\n筛选缓存区域位图', 1445, 590, 265, 105, 'compute');
  f.node('f6_requery', '330  建立新预测锚点\n执行完整区域查询', 945, 590, 265, 105, 'compute');
  f.store('f6_delta', '363  输出实例差量\n以及资源提级和降级信号', 1195, 790, 280, 100, 'render');
  f.edge('f6_e_1', 'f6_cache', 'f6_inside');
  f.edge('f6_e_2', 'f6_change', 'f6_inside');
  f.edge('f6_e_3', 'f6_inside', 'f6_refilter', { label: '区域内' });
  f.edge('f6_e_4', 'f6_inside', 'f6_requery', { label: '超出区域' });
  f.edge('f6_e_5', 'f6_refilter', 'f6_delta');
  f.edge('f6_e_6', 'f6_requery', 'f6_delta');
  return f;
}

function buildFigure7() {
  const f = new Diagram(7, '实例显示与统一资源状态调度图');
  f.container('f7_signals', '335/336  同一次可见性查询的输出', 35, 35, 390, 530, 'compute');
  f.container('f7_scheduler', '340  统一资源状态调度', 460, 35, 1305, 530, 'schedule');
  f.container('f7_pipeline', '350  各资源独立推进的处理流水线', 35, 600, 1730, 395, 'render');

  drawVector(f, 'f7_instance_delta', '363  当前显示实例差量', 85, 105, 9, 'render', { cellWidth: 31, cellHeight: 32 });
  drawVector(f, 'f7_region_scores', '336  区域资源得分', 85, 225, 9, 'data', { cellWidth: 31, cellHeight: 32 });
  drawFileStack(f, 'f7_render_ids', '341  当前显示资源', 105, 370, 235, 65, 'render');
  drawFileStack(f, 'f7_prefetch_ids', '344  低阈值预取资源', 105, 490, 235, 65, 'schedule');

  drawFileStack(f, 'f7_startup', '342  首屏准备', 500, 110, 190, 62, 'data');
  drawPriorityLanes(f, 'f7_tiers', 500, 220, 225, 240);
  drawStateTable(f, 'f7_state_table', '345  统一资源状态表', 805, 125, 300, 255);
  f.node('f7_plan_update', '计划更新\n提级、降级或取消需求', 520, 470, 190, 65, 'neutral', { fontSize: 17 });
  f.node('f7_select', '346  任务选择\n就绪时间、需求等级、得分、入队次序', 1180, 145, 285, 115, 'schedule');
  drawFileStack(f, 'f7_selected_preview', '下一项资源任务', 1530, 155, 165, 70, 'schedule');
  f.node('f7_retry', '355  失败后有限重试\n递增等待时间', 1175, 385, 300, 90, 'danger');
  f.node('f7_cache_rule', '356  非当前结果\n按需求、内存和复用价值\n选择缓存或释放', 800, 425, 285, 95, 'neutral', { fontSize: 17 });

  drawFileStack(f, 'f7_selected_task', '346  已选资源', 70, 680, 180, 72, 'schedule');
  f.circle('f7_queued', '队列', 300, 690, 82, 'schedule', { fontSize: 17 });
  f.document('f7_fetch', '351\n下载', 455, 675, 130, 100, 'data', { fontSize: 16 });
  f.document('f7_parse', '352\n解析', 690, 675, 130, 100, 'compute', { fontSize: 16 });
  f.cube('f7_mount', '353\n挂载', 925, 675, 130, 100, 'render', { fontSize: 16 });
  f.store('f7_resident', '354\n驻留', 1160, 675, 145, 100, 'data', { fontSize: 16 });
  drawSceneGlyph(f, 'f7_display', '360  当前显示实例', 1430, 650, 270, 145, 'render');
  drawVector(f, 'f7_display_condition', '363  当前实例集合', 1450, 845, 8, 'render', { cellWidth: 29, cellHeight: 30, fontSize: 15 });
  f.text('f7_budget', '下载并发、解析并发和每帧挂载时间分别受控；只剩预取任务时降低下载并发。', 420, 900, 930, 40, {
    fontSize: 18,
    bold: true,
  });

  f.edge('f7_e_1', 'f7_region_scores', 'f7_tiers');
  f.edge('f7_e_2', 'f7_render_ids_doc_0', 'f7_tiers');
  f.edge('f7_e_3', 'f7_prefetch_ids_doc_0', 'f7_tiers');
  f.edge('f7_e_4', 'f7_startup_doc_0', 'f7_tiers');
  f.edge('f7_e_5', 'f7_tiers', 'f7_state_table');
  f.edge('f7_e_6', 'f7_plan_update', 'f7_state_table', {
    points: [[760, 502], [760, 350]], exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.8,
  });
  f.edge('f7_e_7', 'f7_state_table', 'f7_select');
  f.edge('f7_e_7b', 'f7_select', 'f7_selected_preview_doc_0');
  f.edge('f7_e_8', 'f7_retry', 'f7_state_table', {
    points: [[1745, 460], [1745, 90], [1042, 90]], exitX: 1, exitY: 0.5, entryX: 0.5, entryY: 0,
    label: '重新排队',
  });
  f.edge('f7_e_9', 'f7_state_table', 'f7_cache_rule');
  f.edge('f7_e_10', 'f7_selected_task_doc_0', 'f7_queued');
  f.edge('f7_e_11', 'f7_queued', 'f7_fetch');
  f.edge('f7_e_12', 'f7_fetch', 'f7_parse');
  f.edge('f7_e_13', 'f7_parse', 'f7_mount');
  f.edge('f7_e_14', 'f7_mount', 'f7_resident');
  f.edge('f7_e_15', 'f7_resident', 'f7_display');
  f.edge('f7_e_16', 'f7_display_condition', 'f7_display');
  return f;
}

function buildFigure8() {
  const f = new Diagram(8, '实例差量更新与按需渲染图');
  f.container('f8_delta', '360  实例显示差量更新', 35, 35, 1730, 455, 'render');
  f.container('f8_render', '381/382/383/384  静态场景整理与按需画面更新', 35, 525, 1730, 470, 'compute');

  drawVector(f, 'f8_delta_input', '363  新增与移除实例编号', 70, 115, 8, 'render', { cellWidth: 34, cellHeight: 34 });
  drawMatrix(f, 'f8_mapping', '372  实例 → 渲染对象映射', 95, 275, 5, 4, 'data', {
    width: 230, height: 130, labelHeight: 38, gap: 3, emphasis: [1, 4, 7, 13, 18],
  });
  f.node('f8_resolve', '根据实例编号定位\n几何原型或批对象', 405, 215, 250, 105, 'compute');
  drawSlotArray(f, 'f8_instanced', '371  稠密实例槽位：尾部写入 / 交换删除', 720, 105, 6, 'render', {
    cellWidth: 54, cellHeight: 62, active: 5,
  });
  drawBitmap(f, 'f8_batched', '382  批对象显示标记', 760, 285, 250, 125, 'render', [0, 2, 3, 8, 11, 18, 25, 28]);
  f.chip('f8_upload', '373  局部矩阵与标记上传', 1100, 210, 285, 115, 'compute', { fontSize: 17 });
  drawSceneGlyph(f, 'f8_scene', '稳定的渲染对象', 1450, 160, 255, 180, 'data');

  drawSceneTree(f, 'f8_flatten', '381  原始场景树', 75, 610, 300, 190);
  drawSlotArray(f, 'f8_ready', '382  压平并批量组织', 440, 650, 5, 'data', {
    cellWidth: 58, cellHeight: 62, active: 5,
  });
  f.node('f8_events', '383  收集变化事件\n相机、资源、实例集合\n材质或动画', 805, 650, 260, 135, 'render', { fontSize: 18 });
  f.decision('f8_changed', '存在需要绘制的变化？', 1140, 640, 245, 155, 'schedule');
  drawSceneGlyph(f, 'f8_request', '384  请求下一帧并绘制', 1460, 575, 255, 150, 'render');
  f.node('f8_idle', '暂停连续刷新\n等待变化事件', 1470, 775, 235, 90, 'neutral');
  f.text('f8_note', '主线程更新量随实际变化实例数量增长，而不随场景总实例数量增长。', 520, 890, 760, 40, {
    fontSize: 18,
    bold: true,
  });

  f.edge('f8_e_1', 'f8_delta_input', 'f8_resolve');
  f.edge('f8_e_2', 'f8_mapping', 'f8_resolve');
  f.edge('f8_e_3', 'f8_resolve', 'f8_instanced');
  f.edge('f8_e_4', 'f8_resolve', 'f8_batched');
  f.edge('f8_e_5', 'f8_instanced', 'f8_upload');
  f.edge('f8_e_6', 'f8_batched', 'f8_upload');
  f.edge('f8_e_7', 'f8_upload', 'f8_scene');
  f.edge('f8_e_8', 'f8_flatten', 'f8_ready');
  f.edge('f8_e_9', 'f8_ready', 'f8_events');
  f.edge('f8_e_10', 'f8_events', 'f8_changed');
  f.edge('f8_e_11', 'f8_changed', 'f8_request', { label: '有变化' });
  f.edge('f8_e_12', 'f8_changed', 'f8_idle', { label: '无变化' });
  return f;
}

const outputs = [
  ['figure_01_system_architecture.drawio', buildFigure1()],
  ['figure_02_offline_model_pipeline.drawio', buildFigure2()],
  ['figure_03_visibility_model_architecture.drawio', buildFigure3()],
  ['figure_04_browser_loading_sequence.drawio', buildFigure4()],
  ['figure_05_runtime_visibility_pipeline.drawio', buildFigure5()],
  ['figure_06_viewcell_frustum_reuse.drawio', buildFigure6()],
  ['figure_07_dual_granularity_streaming.drawio', buildFigure7()],
  ['figure_08_incremental_rendering.drawio', buildFigure8()],
];

for (const [filename, figure] of outputs) {
  fs.writeFileSync(path.join(DIR, filename), figure.toXml(), 'utf8');
}

console.log('Generated 8 editable draw.io patent figures in', DIR);
