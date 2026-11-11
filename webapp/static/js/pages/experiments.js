/* 实验数据管理：实验列表、导入向导、元数据编辑、帧浏览与强度分析 */
Pages.experiments = {
  async render(root, params) {
    if (params.id) return this.detail(root, params.id);
    const exps = await API.get('/api/experiments');
    root.innerHTML = `
      <div class="page-head"><div><h2>实验数据管理</h2><p>管理钙荧光成像原始数据（TIFF 帧序列），登记实验信息与鱼体分组，跟踪各实验的分析流程进度。</p></div>
        <div class="row"><button class="btn primary" id="btnNew">＋ 导入新实验</button></div></div>
      <div class="card"><div class="card-head"><h3>实验列表</h3><div class="row"><input type="text" id="kw" placeholder="按实验编号 / 描述检索" style="width:240px"></div></div>
        <div class="card-body" style="padding:0"><table class="tbl"><thead><tr>
          <th>实验编号</th><th>成像布局</th><th>帧数</th><th>图像尺寸</th><th>鱼体分组</th><th>指示剂</th><th>流程进度</th><th>操作员</th><th>创建时间</th><th>操作</th></tr></thead>
          <tbody id="tb"></tbody></table></div></div>`;
    const draw = () => {
      const kw = $('#kw').value.trim();
      const list = exps.filter(e => !kw || e.name.includes(kw) || (e.description || '').includes(kw));
      $('#tb').innerHTML = list.map(e => `<tr>
        <td><a href="#experiments?id=${e.id}"><b>${esc(e.name)}</b></a><div class="small muted">${esc(e.description || '')}</div></td>
        <td>${e.layout === 'double' ? '<span class="tag blue">双鱼并排</span>' : '<span class="tag">单鱼</span>'}</td>
        <td>${e.frames}</td><td>${e.image_size.join(' × ')}</td>
        <td>${e.fish.map(f => `${f.side.toUpperCase()}：${groupTag(f.group)}`).join('<br>')}</td><td>${esc(e.indicator)}</td>
        <td style="width:150px"><div class="progress"><div style="width:${e.progress * 100}%"></div></div><div class="small muted">${Math.round(e.progress * 6)}/6 步</div></td>
        <td>${esc(e.operator)}</td><td class="small muted">${e.created}</td>
        <td><a class="btn sm" href="#experiments?id=${e.id}">详情</a> <button class="btn sm success" data-run="${e.id}">一键分析</button> <button class="btn sm danger" data-del="${e.id}">删除</button></td></tr>`).join('')
        || '<tr><td colspan="10"><div class="empty"><div class="big">▤</div>暂无实验，点击右上角「导入新实验」</div></td></tr>';
      $$('[data-del]').forEach(b => b.onclick = () => confirmBox(`确认删除实验 <b>${b.dataset.del}</b> 的全部分析结果？（原始数据不会被删除）`, async () => {
        await API.del('/api/experiments/' + b.dataset.del); toast('已删除', 'success'); App.route();
      }));
      $$('[data-run]').forEach(b => b.onclick = async () => {
        const r = await API.post(`/api/experiments/${b.dataset.run}/run_all`);
        toast('已提交一键全流程分析任务', 'success'); location.hash = 'tasks?id=' + r.task_id;
      });
    };
    $('#kw').oninput = draw; draw();
    $('#btnNew').onclick = () => this.wizard();
  },

  wizard() {
    const body = document.createElement('div');
    body.innerHTML = `
      <div class="form g2">
        <div class="field" style="grid-column:span 2"><label>原始数据目录（TIFF 帧序列，文件名格式 img_000000xxx_Default_000.tif）</label>
          <div class="row" style="flex-wrap:nowrap"><input type="text" id="wPath"><button class="btn" id="wBrowse">浏览</button><button class="btn primary" id="wScan">扫描</button></div></div>
        <div class="field" style="grid-column:span 2"><div id="wScanRes" class="small muted">请选择目录并点击「扫描」识别帧序列</div></div>
        <div class="field"><label>实验编号</label><input type="text" id="wName"></div>
        <div class="field"><label>成像布局</label><select id="wLayout"><option value="double">双鱼并排（左右拆分）</option><option value="single">单鱼</option></select></div>
        <div class="field"><label>左侧鱼分组</label><select id="wG0"><option value="0">对照组</option><option value="1">药物处理组</option></select></div>
        <div class="field"><label>右侧鱼分组</label><select id="wG1"><option value="0">对照组</option><option value="1" selected>药物处理组</option></select></div>
        <div class="field"><label>钙指示剂</label><input type="text" id="wInd" value="GCaMP6s"></div>
        <div class="field"><label>采样帧率 (Hz)</label><input type="number" id="wFps" value="2" step="0.5"></div>
        <div class="field" style="grid-column:span 2"><label>实验描述</label><input type="text" id="wDesc" placeholder="如：成瘾药物急性暴露 30 min 后全脑钙成像"></div>
      </div>`;
    let seq = null;
    const m = modal({title: '导入新实验', body, okText: '创建实验', width: 720, onOk: async () => {
      if (!seq) { toast('请先扫描并选择帧序列目录', 'error'); return false; }
      const meta = await API.post('/api/experiments', {name: $('#wName', body).value, seq_dir: seq, layout: $('#wLayout', body).value,
        groups: [+$('#wG0', body).value, +$('#wG1', body).value], indicator: $('#wInd', body).value, frame_rate: +$('#wFps', body).value,
        description: $('#wDesc', body).value});
      toast('实验创建成功：' + meta.id, 'success'); location.hash = 'experiments?id=' + meta.id;
    }});
    $('#wPath', body).value = '';
    API.get('/api/browse').then(r => { $('#wPath', body).value = r.path; });
    $('#wBrowse', body).onclick = () => dirPicker($('#wPath', body).value, p => { $('#wPath', body).value = p; });
    $('#wScan', body).onclick = async () => {
      const res = await API.post('/api/experiments/scan', {path: $('#wPath', body).value});
      if (!res.length) { $('#wScanRes', body).innerHTML = '<span style="color:#e5484d">未发现帧序列</span>'; return; }
      $('#wScanRes', body).innerHTML = `<div class="dir-list">${res.map((r, i) => `<div data-i="${i}"><span>${esc(r.path)}</span><span>${r.frames} 帧 ${r.complete ? '<span class="tag green">完整</span>' : '<span class="tag orange">帧数不足</span>'}</span></div>`).join('')}</div>`;
      $$('[data-i]', body).forEach(el => el.onclick = () => {
        $$('[data-i]', body).forEach(x => x.style.background = ''); el.style.background = '#e7f0ff';
        seq = res[+el.dataset.i].path;
        const parts = seq.split('/'); $('#wName', body).value = parts[parts.length - 1].startsWith('Pos') ? parts[parts.length - 2] : parts[parts.length - 1];
      });
      $$('[data-i]', body)[0].click();
    };
  },

  async detail(root, id) {
    const e = await API.get('/api/experiments/' + id);
    const steps = [['imported', '数据导入', 'experiments'], ['preprocess', '图像预处理', 'preprocess'], ['register', '运动校正配准', 'register'],
      ['atlas', '脑区模板映射', 'atlas'], ['signal', '钙信号提取', 'signal'], ['feature', '特征计算', 'feature']];
    Store.set('exp', id);
    root.innerHTML = `
      <div class="page-head"><div><h2>实验详情 · ${esc(e.name)}</h2><p>${esc(e.description || '无描述')}</p></div>
        <div class="row"><a class="btn" href="#experiments">‹ 返回列表</a><button class="btn" id="btnEdit">编辑信息</button><button class="btn success" id="btnRun">一键全流程分析</button></div></div>
      <div class="card"><div class="card-body"><div class="steps">${steps.map(([k, n, h], i) => `<a class="step ${e.status[k] ? 'done' : ''}" href="#${h}"><div class="c">${e.status[k] ? '✓' : i + 1}</div>${n}</a>`).join('')}</div></div></div>
      <div class="grid g3">
        <div class="card"><div class="card-head"><h3>基本信息</h3></div><div class="card-body"><dl class="desc-list">
          <dt>实验编号</dt><dd>${e.id}</dd><dt>数据目录</dt><dd class="mono small" style="word-break:break-all">${esc(e.seq_dir)}</dd>
          <dt>成像布局</dt><dd>${e.layout === 'double' ? '双鱼并排' : '单鱼'}</dd><dt>帧数</dt><dd>${e.frames}（#${e.first_frame} ~ #${e.last_frame}）</dd>
          <dt>图像尺寸</dt><dd>${e.image_size.join(' × ')} 像素，${e.bit_depth} bit</dd><dt>数据量</dt><dd>${e.results.imported?.size_mb ?? '-'} MB</dd>
          <dt>实验对象</dt><dd>${esc(e.species)}</dd><dt>钙指示剂</dt><dd>${esc(e.indicator)}</dd><dt>采样帧率</dt><dd>${e.frame_rate} Hz</dd>
          <dt>鱼体分组</dt><dd>${e.fish.map(f => `${f.fish_id} ${groupTag(f.group)}`).join('<br>')}</dd>
          <dt>操作员</dt><dd>${esc(e.operator)}</dd><dt>创建时间</dt><dd>${e.created}</dd></dl></div></div>
        <div class="card span2"><div class="card-head"><h3>帧序列浏览</h3><div class="row">
            <span class="small muted">伪彩色</span><select id="cmap" class="inline-select" style="min-width:110px"><option value="gray">灰度</option><option value="inferno">Inferno</option><option value="viridis">Viridis</option><option value="turbo">Turbo</option><option value="hot">Hot</option></select></div></div>
          <div class="card-body"><div class="grid" style="grid-template-columns: 1.6fr 1fr">
            <div><div class="img-panel" style="height:330px" id="frameBox"><img id="frameImg"></div>
              <div class="row" style="margin-top:10px"><span class="small muted">帧号</span><input type="range" id="fr" min="${e.first_frame}" max="${e.last_frame}" value="500" style="flex:1"><b id="frNo" class="mono">500</b></div></div>
            <div><div class="desc-list small" id="frInfo"></div><div class="chart" id="hist" style="height:200px;margin-top:8px"></div></div>
          </div></div></div>
      </div>
      <div class="card"><div class="card-head"><h3>全时程平均荧光强度</h3><span class="muted small">每 5 帧采样；骤降点通常对应鱼体甩动等异常帧</span></div>
        <div class="card-body"><div class="chart" id="prof" style="height:240px"></div></div></div>
      <div class="card"><div class="card-head"><h3>相关任务记录</h3></div><div class="card-body" style="padding:0"><table class="tbl"><thead><tr><th>任务编号</th><th>类型</th><th>名称</th><th>状态</th><th>开始时间</th><th>用时</th></tr></thead>
        <tbody>${e.tasks.map(t => `<tr><td class="mono">${t.id}</td><td>${t.type_name}</td><td>${esc(t.title)}</td><td>${statusTag(t.status)}</td><td>${t.created}</td><td>${sec(t.elapsed)}</td></tr>`).join('') || '<tr><td colspan="6" class="empty">暂无任务</td></tr>'}</tbody></table></div></div>`;
    const loadFrame = async () => {
      const f = await API.get(`/api/experiments/${id}/frame?idx=${$('#fr').value}&cmap=${$('#cmap').value}`);
      $('#frameImg').src = f.url; $('#frNo').textContent = f.frame;
      $('#frInfo').innerHTML = `<dt>文件名</dt><dd class="mono">${f.file}</dd><dt>尺寸/类型</dt><dd>${f.shape.join('×')} · ${f.dtype}</dd>
        <dt>灰度范围</dt><dd>${f.min} ~ ${f.max}</dd><dt>均值 ± 标准差</dt><dd>${f.mean} ± ${f.std}</dd><dt>显示窗宽</dt><dd>${Math.round(f.vmin)} ~ ${Math.round(f.vmax)}</dd>`;
      chart('#hist', {grid: {left: 45, right: 10, top: 10, bottom: 25}, tooltip: {trigger: 'axis'},
        xAxis: {type: 'category', data: f.hist.edges.slice(0, -1), axisLabel: {fontSize: 10}}, yAxis: {type: 'log', axisLabel: {fontSize: 10}},
        series: [{type: 'bar', data: f.hist.counts, itemStyle: {color: '#2f7cf6'}, barCategoryGap: 0}]});
    };
    let timer;
    $('#fr').oninput = () => { $('#frNo').textContent = $('#fr').value; clearTimeout(timer); timer = setTimeout(loadFrame, 150); };
    $('#cmap').onchange = loadFrame;
    loadFrame();
    API.get(`/api/experiments/${id}/profile`).then(p => chart('#prof', {
      tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 55, right: 20, top: 30, bottom: 30},
      xAxis: {type: 'category', data: p.frames, ...axisBase('帧号')}, yAxis: {type: 'value', scale: true, ...axisBase('平均灰度')},
      series: [{name: '左侧鱼区域', type: 'line', data: p.l, showSymbol: false}, ...(p.r.length ? [{name: '右侧鱼区域', type: 'line', data: p.r, showSymbol: false}] : [])]}));
    $('#btnRun').onclick = async () => { const r = await API.post(`/api/experiments/${id}/run_all`); location.hash = 'tasks?id=' + r.task_id; };
    $('#btnEdit').onclick = () => {
      const b = document.createElement('div');
      b.innerHTML = `<div class="form g2"><div class="field"><label>实验名称</label><input type="text" id="eN" value="${esc(e.name)}"></div>
        <div class="field"><label>钙指示剂</label><input type="text" id="eI" value="${esc(e.indicator)}"></div>
        ${e.fish.map((f, i) => `<div class="field"><label>${f.side === 'l' ? '左侧' : '右侧'}鱼分组</label><select id="eG${i}"><option value="0" ${f.group == 0 ? 'selected' : ''}>对照组</option><option value="1" ${f.group == 1 ? 'selected' : ''}>药物处理组</option></select></div>`).join('')}
        <div class="field" style="grid-column:span 2"><label>实验描述</label><textarea id="eD" rows="3">${esc(e.description || '')}</textarea></div></div>`;
      modal({title: '编辑实验信息', body: b, onOk: async () => {
        await API.put('/api/experiments/' + id, {name: $('#eN', b).value, indicator: $('#eI', b).value, description: $('#eD', b).value,
          fish: e.fish.map((f, i) => ({group: +$('#eG' + i, b).value}))});
        toast('已保存', 'success'); App.route();
      }});
    };
  },
};

function dirPicker(start, onPick) {
  const body = document.createElement('div');
  let cur = start;
  const m = modal({title: '选择目录', body, width: 620, onOk: () => onPick(cur)});
  const load = async (p) => {
    const r = await API.get('/api/browse?path=' + encodeURIComponent(p));
    cur = r.path;
    body.innerHTML = `<div class="path-bar">${esc(r.path)}</div><div class="dir-list"><div data-p="${esc(r.parent)}"><span>⬑ 上级目录</span></div>
      ${r.items.map(i => `<div data-p="${esc(i.path)}"><span>📁 ${esc(i.name)}</span><span class="muted small">${i.files ? i.files + ' 个数据文件' : ''}</span></div>`).join('')}</div>`;
    $$('[data-p]', body).forEach(el => el.onclick = () => load(el.dataset.p));
  };
  load(start);
}
