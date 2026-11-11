/* 斑马鱼全脑钙成像神经计算分析系统 —— 前端核心：路由、接口、通用组件 */
'use strict';

const Pages = {};

const MENU = [
  {group: '概览'},
  {id: 'dashboard', name: '系统首页', ico: '◉'},
  {group: '数据管理'},
  {id: 'experiments', name: '实验数据管理', ico: '▤'},
  {id: 'templates', name: '标准脑模板库', ico: '◈'},
  {group: '图像处理'},
  {id: 'preprocess', name: '图像预处理', ico: '◧'},
  {id: 'register', name: '运动校正配准', ico: '⇄'},
  {id: 'atlas', name: '脑区模板映射', ico: '▦'},
  {group: '信号分析'},
  {id: 'signal', name: '钙信号提取', ico: '∿'},
  {id: 'feature', name: '神经活动特征', ico: '▥'},
  {group: '智能建模'},
  {id: 'library', name: '个体特征库', ico: '☰'},
  {id: 'kernel', name: '高斯核空间平滑', ico: '◎'},
  {id: 'dataset', name: '建模数据集', ico: '▣'},
  {id: 'train', name: '模型训练', ico: '⚙'},
  {id: 'evaluate', name: '模型评估', ico: '✓'},
  {id: 'predict', name: '智能预测', ico: '➤'},
  {group: '系统管理'},
  {id: 'tasks', name: '任务中心', ico: '☷'},
  {id: 'settings', name: '系统设置', ico: '⚒'},
];

const GROUP_NAME = {0: '对照组', 1: '药物处理组'};
const FEATURES = [
  ['num_peaks', '钙峰数量'], ['areas_sum', '峰面积总和'], ['areas_mean', '平均峰面积'], ['duration_mean', '平均持续时间'],
  ['dr', '占空比'], ['height_max', '最大峰高'], ['height_mean', '平均峰高'], ['ap_mean', '平均动作电位强度'],
  ['interval_mean', '平均峰间间隔'], ['separation_mean', '平均峰距'],
];
const PALETTE = ['#2f7cf6', '#12a594', '#f08c2b', '#7c5cdb', '#e5484d', '#0ea5e9', '#a16207', '#db2777'];

/* ---------------- API ---------------- */
const API = {
  async req(method, url, body) {
    const opt = {method, headers: {}};
    if (body !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
    const r = await fetch(url, opt);
    if (r.status === 401) { location.href = '/login'; throw new Error('未登录'); }
    const j = await r.json();
    if (!j.ok) { toast(j.error || '请求失败', 'error'); throw new Error(j.error); }
    return j.data;
  },
  get(u) { return this.req('GET', u); },
  post(u, b) { return this.req('POST', u, b || {}); },
  put(u, b) { return this.req('PUT', u, b || {}); },
  del(u) { return this.req('DELETE', u); },
};

/* ---------------- helpers ---------------- */
function $(sel, root) { return (root || document).querySelector(sel); }
function $$(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
function fmt(v, d = 3) { if (v === null || v === undefined || Number.isNaN(v)) return '-'; return Number(v).toFixed(d).replace(/\.?0+$/, m => m.includes('.') ? '' : m); }
function pct(v, d = 1) { return (v * 100).toFixed(d) + '%'; }
function groupTag(g) { return `<span class="tag ${g == 1 ? 'orange' : 'teal'}">${GROUP_NAME[g]}</span>`; }
function statusTag(s) {
  const m = {running: ['blue', '运行中'], success: ['green', '已完成'], failed: ['red', '失败'], pending: ['', '等待']};
  const [c, t] = m[s] || ['', s]; return `<span class="tag ${c}">${t}</span>`;
}
function sec(s) { s = Math.round(s); return s < 60 ? `${s} 秒` : `${Math.floor(s / 60)} 分 ${s % 60} 秒`; }

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = 'toast ' + type; el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

function modal({title, body, okText = '确定', cancelText = '取消', width = 640, onOk, noFoot}) {
  const mask = document.createElement('div');
  mask.className = 'mask';
  mask.innerHTML = `<div class="modal" style="width:${width}px"><div class="modal-head"><span>${title}</span><span class="x">✕</span></div>
    <div class="modal-body"></div>${noFoot ? '' : `<div class="modal-foot"><button class="btn c">${cancelText}</button><button class="btn primary o">${okText}</button></div>`}</div>`;
  const bodyEl = $('.modal-body', mask);
  if (typeof body === 'string') bodyEl.innerHTML = body; else bodyEl.appendChild(body);
  const onKey = e => { if (e.key === 'Escape') close(); };
  const close = () => { mask.remove(); document.removeEventListener('keydown', onKey); };
  document.addEventListener('keydown', onKey);
  $('.x', mask).onclick = close;
  if (!noFoot) {
    $('.c', mask).onclick = close;
    $('.o', mask).onclick = async () => { try { if (onOk && (await onOk(bodyEl)) === false) return; close(); } catch (e) { /* keep open */ } };
  }
  document.body.appendChild(mask);
  return {el: bodyEl, close};
}

function confirmBox(text, onOk) { modal({title: '操作确认', body: `<div style="padding:6px 0">${text}</div>`, width: 420, onOk}); }

/* ---------------- charts ---------------- */
const Charts = [];
function chart(el, option) {
  if (typeof el === 'string') el = $(el);
  const old = echarts.getInstanceByDom(el); if (old) old.dispose();
  const c = echarts.init(el, null, {renderer: 'canvas'});
  c.setOption(Object.assign({color: PALETTE, animation: false, textStyle: {fontFamily: 'PingFang SC, Microsoft YaHei, Noto Sans CJK SC'}}, option));
  Charts.push(c);
  return c;
}
window.addEventListener('resize', () => Charts.forEach(c => { try { c.resize(); } catch (e) {} }));
function axisBase(name) { return {name, nameTextStyle: {color: '#6b778c'}, axisLine: {lineStyle: {color: '#c5cdd8'}}, splitLine: {lineStyle: {color: '#eef1f5'}}, axisLabel: {color: '#6b778c'}}; }
function heatmapOption(mat, {title = '', min = -1, max = 1, colors} = {}) {
  const data = [];
  mat.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
  const n = mat.length;
  return {
    title: title ? {text: title, left: 'center', textStyle: {fontSize: 13}} : undefined,
    tooltip: {formatter: p => `脑区 ${p.value[1] + 1} ↔ 脑区 ${p.value[0] + 1}<br/>值：${fmt(p.value[2], 4)}`},
    grid: {left: 40, right: 70, top: title ? 30 : 10, bottom: 30},
    xAxis: {type: 'category', data: [...Array(n).keys()].map(i => i + 1), axisLabel: {interval: 9, fontSize: 10}},
    yAxis: {type: 'category', data: [...Array(n).keys()].map(i => i + 1), axisLabel: {interval: 9, fontSize: 10}, inverse: true},
    visualMap: {min, max, precision: 2, calculable: true, orient: 'vertical', right: 0, top: 'center', itemHeight: 160,
      inRange: {color: colors || ['#3b4cc0', '#8db0fe', '#f2f2f2', '#f49a7b', '#b40426']}},
    series: [{type: 'heatmap', data, progressive: 0}],
  };
}

/* ---------------- task watcher ---------------- */
function taskWatcher(container, taskId, {onDone, onTick, showLog = true} = {}) {
  container.innerHTML = `<div class="task-box"><div class="row"><b class="t-title">任务</b><span class="t-status right"></span></div>
    <div class="msg"><span class="t-msg"></span><span class="t-time"></span></div><div class="progress"><div style="width:0%"></div></div>
    ${showLog ? '<div class="log t-log" style="margin-top:10px"></div>' : ''}</div>`;
  let stopped = false, seenRunning = false;
  const tick = async () => {
    if (stopped || !document.body.contains(container)) return;
    let t;
    try { t = await API.get('/api/tasks/' + taskId); } catch (e) { return; }
    $('.t-title', container).textContent = `${t.type_name} · ${t.title}`;
    $('.t-status', container).innerHTML = statusTag(t.status);
    $('.t-msg', container).textContent = t.message || '';
    $('.t-time', container).textContent = `${(t.progress * 100).toFixed(1)}% · 用时 ${sec(t.elapsed)}`;
    $('.progress > div', container).style.width = (t.progress * 100) + '%';
    if (showLog) { const lg = $('.t-log', container); lg.textContent = t.logs.join('\n'); lg.scrollTop = lg.scrollHeight; }
    if (onTick) onTick(t);
    if (t.status === 'running') { seenRunning = true; setTimeout(tick, 1000); }
    else { stopped = true; if (seenRunning && t.status === 'success') toast('任务完成：' + t.title, 'success'); if (onDone) onDone(t); }
  };
  tick();
  return {stop() { stopped = true; }};
}

/* ---------------- experiment selector ---------------- */
const Store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem('bc_' + k)) ?? d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('bc_' + k, JSON.stringify(v)); } catch (e) {} },
};

async function expToolbar(container, {onChange, sides = true, requireStep} = {}) {
  const exps = await API.get('/api/experiments');
  const bar = document.createElement('div');
  bar.className = 'toolbar';
  if (!exps.length) {
    bar.innerHTML = `<span class="muted">暂无实验数据，请先在「实验数据管理」中导入实验。</span><a class="btn primary sm right" href="#experiments">前往导入</a>`;
    container.appendChild(bar); return null;
  }
  let cur = Store.get('exp');
  if (!exps.find(e => e.id === cur)) cur = exps[0].id;
  bar.innerHTML = `<span class="lbl">当前实验</span><select class="inline-select exp-sel">${exps.map(e => `<option value="${e.id}" ${e.id === cur ? 'selected' : ''}>${esc(e.name)}（${e.frames} 帧）</option>`).join('')}</select>
    ${sides ? '<span class="lbl" style="margin-left:10px">鱼体位置</span><select class="inline-select side-sel" style="min-width:150px"></select>' : ''}
    <span class="exp-info muted small"></span><span class="right exp-steps"></span>`;
  container.appendChild(bar);
  const state = {exp: null, side: Store.get('side', 'l')};
  const load = async () => {
    const id = $('.exp-sel', bar).value; Store.set('exp', id);
    state.exp = exps.find(e => e.id === id);
    if (sides) {
      const ss = $('.side-sel', bar);
      ss.innerHTML = state.exp.fish.map(f => `<option value="${f.side}" ${f.side === state.side ? 'selected' : ''}>${f.side === 'l' ? '左侧鱼' : '右侧鱼'} · ${GROUP_NAME[f.group]}</option>`).join('');
      if (!state.exp.fish.find(f => f.side === state.side)) state.side = state.exp.fish[0].side;
      ss.value = state.side;
    }
    $('.exp-info', bar).textContent = `${state.exp.layout === 'double' ? '双鱼并排' : '单鱼'} · ${state.exp.image_size.join('×')} · ${state.exp.bit_depth} bit`;
    const done = Object.values(state.exp.status).filter(Boolean).length;
    $('.exp-steps', bar).innerHTML = `<span class="tag ${done === 6 ? 'green' : 'blue'}">流程进度 ${done}/6</span>`;
    onChange && onChange(state);
  };
  $('.exp-sel', bar).onchange = load;
  if (sides) $('.side-sel', bar).onchange = e => { state.side = e.target.value; Store.set('side', state.side); onChange && onChange(state); };
  await load();
  return state;
}

function stepGuard(container, exp, step, prevName, href) {
  if (exp.status[step]) return false;
  container.innerHTML = `<div class="card"><div class="empty"><div class="big">⧗</div>当前实验尚未完成「${prevName}」步骤<br><a class="btn primary" style="margin-top:14px" href="#${href}">前往${prevName}</a></div></div>`;
  return true;
}

function imgPanel(url, label, h = 300, light = false) {
  return `<div class="img-panel ${light ? 'light' : ''}" style="height:${h}px">${label ? `<span class="img-label">${label}</span>` : ''}<img src="${url}"></div>`;
}

/* ---------------- router ---------------- */
const App = {
  me: null,
  async start() {
    this.me = await API.get('/api/me');
    $('#uname').textContent = `${this.me.name}（${this.me.username}）`;
    $('#avatar').textContent = this.me.name[0];
    $('#logout').onclick = async () => { await API.post('/api/logout'); location.href = '/login'; };
    $('#menu').innerHTML = MENU.map(m => m.group ? `<div class="menu-group">${m.group}</div>` :
      `<a href="#${m.id}" data-id="${m.id}"><span class="ico">${m.ico}</span>${m.name}</a>`).join('');
    window.addEventListener('hashchange', () => this.route());
    setInterval(() => this.clock(), 1000); this.clock();
    setInterval(() => this.taskChip(), 3000); this.taskChip();
    this.route();
  },
  clock() { const d = new Date(); $('#clock').textContent = d.toLocaleString('zh-CN', {hour12: false}); },
  async taskChip() {
    try {
      const ts = await API.get('/api/tasks');
      const run = ts.filter(t => t.status === 'running');
      $('#taskChip').innerHTML = run.length ? `<span class="dot busy"></span><a href="#tasks">${run.length} 个任务运行中</a>` : '<span class="dot"></span><span>无运行任务</span>';
    } catch (e) {}
  },
  async route() {
    const [id, q] = (location.hash.slice(1) || 'dashboard').split('?');
    const page = Pages[id] || Pages.dashboard;
    const m = MENU.find(x => x.id === id) || MENU[1];
    $$('#menu a').forEach(a => a.classList.toggle('active', a.dataset.id === m.id));
    const grp = (() => { let g = ''; for (const x of MENU) { if (x.group) g = x.group; if (x.id === m.id) return g; } return ''; })();
    $('#crumb').innerHTML = `${grp} / <b>${m.name}</b>`;
    $$('.mask').forEach(m => m.remove());
    Charts.splice(0).forEach(c => { try { c.dispose(); } catch (e) {} });
    const content = $('#content');
    content.innerHTML = '';
    content.scrollTop = 0;
    const view = document.createElement('div');   // 每次路由使用独立容器，避免上一页面的异步渲染覆盖当前页面
    content.appendChild(view);
    const params = Object.fromEntries(new URLSearchParams(q || ''));
    try { await page.render(view, params); } catch (e) { if (document.body.contains(view)) console.error(e); }
  },
};
