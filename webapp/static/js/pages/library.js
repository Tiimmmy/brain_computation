/* 个体特征库：样本列表、批量导入脑区信号矩阵、组间统计、样本详情 */
Pages.library = {
  async render(root) {
    const [list, stats] = await Promise.all([API.get('/api/library'), API.get('/api/library/stats')]);
    root.innerHTML = `
      <div class="page-head"><div><h2>个体特征库</h2><p>汇集每条鱼的神经活动特征作为建模样本。样本可来自实验流水线，也可通过批量导入已提取的脑区荧光信号矩阵（脑区 × 时间，.npy）自动完成预处理与特征计算。</p></div>
        <div class="row"><button class="btn" id="lExp">导出特征库 CSV</button><button class="btn primary" id="lImp">＋ 批量导入信号矩阵</button></div></div>
      <div id="lTask"></div>
      <div class="grid g4">
        <div class="kpi"><div class="l">样本总数</div><div class="v">${stats.count || 0}</div></div>
        <div class="kpi"><div class="l">对照组 / 药物处理组</div><div class="v">${stats.ctl || 0} / ${stats.drug || 0}</div></div>
        <div class="kpi"><div class="l">样本来源</div><div class="v" style="font-size:14px;padding-top:6px">${Object.entries(stats.sources || {}).map(([k, v]) => `${k} ${v}`).join('；') || '-'}</div></div>
        <div class="kpi"><div class="l">组间显著差异特征（p &lt; 0.05）</div><div class="v">${(stats.features || []).filter(f => f.p !== null && f.p < 0.05).length} / 10</div></div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h3>组间特征对比（箱线图，脑区平均）</h3><select id="bxK" class="inline-select">${FEATURES.map(([k, n]) => `<option value="${k}">${n}</option>`).join('')}</select></div>
          <div class="card-body"><div class="chart" id="bx" style="height:280px"></div></div></div>
        <div class="card"><div class="card-head"><h3>组间差异检验（Welch t 检验）</h3></div><div class="card-body" style="padding:0">
          <table class="tbl compact"><thead><tr><th>特征</th><th>对照组均值</th><th>药物组均值</th><th>变化</th><th>p 值</th><th>显著性</th></tr></thead><tbody>
          ${(stats.features || []).map(f => `<tr><td>${f.name}</td><td>${fmt(f.ctl_mean, 3)}</td><td>${fmt(f.drug_mean, 3)}</td>
            <td>${f.ctl_mean ? ((f.drug_mean / f.ctl_mean - 1) * 100).toFixed(1) + '%' : '-'}</td><td>${f.p === null ? '-' : f.p.toExponential(2)}</td>
            <td>${f.p === null ? '-' : f.p < 0.001 ? '<span class="tag red">***</span>' : f.p < 0.01 ? '<span class="tag orange">**</span>' : f.p < 0.05 ? '<span class="tag blue">*</span>' : '<span class="tag">n.s.</span>'}</td></tr>`).join('')}
          </tbody></table></div></div>
      </div>
      <div class="card"><div class="card-head"><h3>样本列表</h3><div class="row">
          <select id="fG" class="inline-select" style="min-width:120px"><option value="">全部分组</option><option value="0">对照组</option><option value="1">药物处理组</option></select>
          <input type="text" id="fK" placeholder="样本编号检索" style="width:160px"><button class="btn sm danger" id="lDel">删除选中</button></div></div>
        <div class="card-body" style="padding:0"><div class="tbl-wrap" style="max-height:440px"><table class="tbl compact"><thead><tr><th><input type="checkbox" id="all"></th><th>样本编号</th><th>分组</th><th>来源</th><th>异常帧</th>
          <th>钙峰总数</th><th>平均峰面积</th><th>平均占空比</th><th>平均峰高</th><th>同步指数</th><th>入库时间</th><th>操作</th></tr></thead><tbody id="lTb"></tbody></table></div></div></div>`;
    const draw = () => {
      const g = $('#fG').value, k = $('#fK').value.trim();
      const rows = list.filter(e => (g === '' || e.label == g) && (!k || e.fish_id.includes(k)));
      $('#lTb').innerHTML = rows.map(e => `<tr><td><input type="checkbox" class="ck" value="${e.fish_id}"></td><td><b>${e.fish_id}</b></td><td>${groupTag(e.label)}</td>
        <td><span class="tag ${e.source === '实验流水线' ? 'purple' : 'blue'}">${e.source}</span></td><td>${e.fly_frames.length}</td><td>${e.summary.total_peaks}</td>
        <td>${fmt(e.summary.areas_mean.mean, 2)}</td><td>${pct(e.summary.dr.mean)}</td><td>${fmt(e.summary.height_mean.mean, 3)}</td><td>${fmt(e.summary.sync_index, 4)}</td>
        <td class="small muted">${e.created}</td><td><button class="btn sm" data-v="${e.fish_id}">查看</button></td></tr>`).join('') || '<tr><td colspan="12"><div class="empty">暂无样本</div></td></tr>';
      $$('[data-v]').forEach(b => b.onclick = () => this.detail(b.dataset.v));
    };
    $('#fG').onchange = draw; $('#fK').oninput = draw; draw();
    $('#all').onchange = e => $$('.ck').forEach(c => c.checked = e.target.checked);
    $('#lDel').onclick = () => {
      const ids = $$('.ck').filter(c => c.checked).map(c => c.value);
      if (!ids.length) return toast('请先勾选样本', 'error');
      confirmBox(`确认删除选中的 ${ids.length} 个样本？`, async () => { await API.post('/api/library/delete', {ids}); App.route(); });
    };
    $('#lExp').onclick = () => { location.href = '/api/export/library.csv'; };
    $('#lImp').onclick = () => this.importDlg();
    const drawBox = () => {
      const f = (stats.features || []).find(x => x.key === $('#bxK').value);
      if (!f) return;
      chart('#bx', {tooltip: {trigger: 'item'}, grid: {left: 60, right: 20, top: 20, bottom: 30},
        xAxis: {type: 'category', data: ['对照组', '药物处理组'], ...axisBase()}, yAxis: {type: 'value', scale: true, ...axisBase(f.name)},
        series: [{type: 'boxplot', data: [{value: f.ctl_box, itemStyle: {color: '#d7f3ee', borderColor: '#12a594'}}, {value: f.drug_box, itemStyle: {color: '#fde6cf', borderColor: '#f08c2b'}}], boxWidth: [40, 70]}]});
    };
    $('#bxK').onchange = drawBox; drawBox();
  },
  importDlg() {
    const body = document.createElement('div');
    body.innerHTML = `<div class="field"><label>信号矩阵目录（每个样本一个 .npy 文件，形状 100 × T；可附 labels.csv 标注分组）</label>
      <div class="row" style="flex-wrap:nowrap"><input type="text" id="iP"><button class="btn" id="iB">浏览</button><button class="btn primary" id="iS">扫描</button></div></div>
      <div class="row" style="margin-top:10px"><label class="check"><input type="checkbox" id="iO">覆盖已存在样本</label><span class="small muted">并行进程</span><input type="number" id="iW" value="48" style="width:80px"></div>
      <div id="iR" style="margin-top:10px" class="small muted">请扫描目录</div>`;
    API.get('/api/browse').then(r => { $('#iP', body).value = r.path + '/demo/signal_library'; });
    let scanned = false;
    modal({title: '批量导入脑区信号矩阵', body, width: 720, okText: '开始分析', onOk: async () => {
      if (!scanned) { toast('请先扫描目录', 'error'); return false; }
      const r = await API.post('/api/library/import', {folder: $('#iP', body).value, overwrite: $('#iO', body).checked, workers: +$('#iW', body).value});
      taskWatcher($('#lTask'), r.task_id, {onDone: () => App.route()});
    }});
    $('#iB', body).onclick = () => dirPicker($('#iP', body).value, p => { $('#iP', body).value = p; });
    $('#iS', body).onclick = async () => {
      const r = await API.post('/api/library/scan', {folder: $('#iP', body).value});
      scanned = true;
      $('#iR', body).innerHTML = `发现 <b>${r.count}</b> 个信号矩阵，${r.has_labels ? '<span class="tag green">已读取 labels.csv 分组标注</span>' : '<span class="tag orange">无分组标注，默认对照组</span>'}
        <div class="tbl-wrap" style="max-height:240px;margin-top:8px"><table class="tbl compact"><thead><tr><th>文件</th><th>样本编号</th><th>矩阵形状</th><th>分组</th></tr></thead>
        <tbody>${r.files.slice(0, 200).map(f => `<tr><td class="mono">${f.file}</td><td>${f.fish_id}</td><td>${f.shape.join(' × ')}</td><td>${f.group}</td></tr>`).join('')}</tbody></table></div>`;
    };
  },
  async detail(fid) {
    const d = await API.get('/api/library/' + fid);
    const body = document.createElement('div');
    body.innerHTML = `<div class="row" style="margin-bottom:10px"><b>${fid}</b>${groupTag(d.meta.label)}<span class="tag">${d.meta.source}</span>
      <span class="small muted">异常帧：${d.meta.fly_frames.join(', ') || '无'}</span>
      <select id="dL" class="inline-select right" style="min-width:130px"><option value="0" ${d.meta.label == 0 ? 'selected' : ''}>对照组</option><option value="1" ${d.meta.label == 1 ? 'selected' : ''}>药物处理组</option></select><button class="btn sm" id="dSave">修改分组</button></div>
      <div class="chart" id="dT" style="height:260px"></div><div class="grid g2"><div class="chart" id="dF" style="height:260px"></div><div class="chart" id="dC" style="height:300px"></div></div>`;
    modal({title: '样本详情', body, width: 1000, noFoot: true});
    $('#dSave', body).onclick = async () => { await API.put('/api/library/' + fid, {label: +$('#dL', body).value}); toast('分组已更新', 'success'); };
    chart($('#dT', body), {tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 50, right: 20, top: 30, bottom: 30},
      xAxis: {type: 'value', max: 'dataMax', ...axisBase('帧')}, yAxis: {type: 'value', ...axisBase('ΔF/F')},
      series: d.regions.map((g, i) => ({name: '脑区 ' + g, type: 'line', showSymbol: false, data: d.dff[i].map((v, t) => [t, v + i * 0.5]), lineStyle: {width: 1}}))});
    chart($('#dF', body), {tooltip: {trigger: 'axis'}, grid: {left: 50, right: 20, top: 20, bottom: 30}, title: {text: '各脑区钙峰数量', left: 'center', textStyle: {fontSize: 12}},
      xAxis: {type: 'category', data: d.table.num_peaks.map((_, i) => i + 1)}, yAxis: {type: 'value'}, series: [{type: 'bar', data: d.table.num_peaks, color: d.meta.label ? '#f08c2b' : '#12a594'}]});
    chart($('#dC', body), heatmapOption(d.connection_corr, {title: '放电同步连接矩阵'}));
  },
};
