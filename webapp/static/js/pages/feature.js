/* 神经活动特征：钙峰检测、13 类特征、脑区特征分布图、同步连接矩阵 */
Pages.feature = {
  async render(root) {
    root.innerHTML = `<div class="page-head"><div><h2>神经活动特征</h2><p>对 ΔF/F 信号进行四分位距自适应阈值的钙峰检测，定位峰左右拐点，计算钙峰数量、峰面积、持续时间、占空比、峰高、动作电位强度、峰间间隔、峰距及脑区间同步连接等特征。</p></div></div>`;
    const wrap = document.createElement('div');
    await expToolbar(root, {sides: true, onChange: s => this.load(wrap, s)});
    root.appendChild(wrap);
  },
  async load(wrap, st) {
    const e = await API.get('/api/experiments/' + st.exp.id);
    if (stepGuard(wrap, e, 'signal', '钙信号提取', 'signal')) return;
    wrap.innerHTML = `<div class="card"><div class="card-body"><div class="row">
      <span class="small muted">峰检测：</span><span class="tag blue">高度阈值 Q3 + 2·IQR</span><span class="tag blue">最小峰宽 5 帧</span><span class="tag teal">拐点：一阶差分符号变化</span>
      <div class="right row"><button class="btn" id="fExp">导出特征 CSV</button><button class="btn" id="fLib">加入个体特征库</button><button class="btn primary" id="fRun">${e.status.feature ? '重新计算' : '开始计算'}</button></div></div>
      <div id="fTask" style="margin-top:10px"></div></div></div><div id="fRes"></div>`;
    $('#fRun').onclick = async () => { const r = await API.post(`/api/experiments/${st.exp.id}/feature`); taskWatcher($('#fTask'), r.task_id, {onDone: () => this.load(wrap, st)}); };
    $('#fExp').onclick = () => { location.href = `/api/export/experiment/${st.exp.id}/${st.side}.csv`; };
    $('#fLib').onclick = async () => { const ids = await API.post('/api/library/from_experiment', {exp_id: st.exp.id}); toast(`已加入特征库：${ids.join('、')}`, 'success'); };
    if (!e.status.feature) { $('#fRes').innerHTML = '<div class="card"><div class="empty"><div class="big">▥</div>尚未计算特征</div></div>'; return; }
    const r = await API.get(`/api/experiments/${st.exp.id}/feature?side=${st.side}`);
    const sig = await API.get(`/api/experiments/${st.exp.id}/signal?side=${st.side}&regions=1,35,60`);
    const s = r.summary;
    $('#fRes').innerHTML = `
      <div class="grid g5">
        <div class="kpi"><div class="l">全脑钙峰总数</div><div class="v">${s.total_peaks}</div></div>
        <div class="kpi"><div class="l">平均钙峰数 / 脑区</div><div class="v">${fmt(s.num_peaks.mean, 2)}</div></div>
        <div class="kpi"><div class="l">平均占空比</div><div class="v">${pct(s.dr.mean)}</div></div>
        <div class="kpi"><div class="l">平均峰高（×10 ΔF/F）</div><div class="v">${fmt(s.height_mean.mean, 3)}</div></div>
        <div class="kpi"><div class="l">全脑同步指数</div><div class="v">${fmt(s.sync_index, 4)}</div></div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h3>钙峰检测结果</h3><div class="row"><span class="small muted">脑区</span><select id="pkR" class="inline-select" style="min-width:110px">${[1, 35, 60].map(x => `<option>${x}</option>`).join('')}</select></div></div>
        <div class="card-body"><div class="chart" id="pk" style="height:260px"></div></div></div>
      <div class="grid" style="grid-template-columns: 1fr 1.25fr">
        <div class="card"><div class="card-head"><h3>特征脑区分布图</h3><select id="mapK" class="inline-select">${FEATURES.map(([k, n]) => `<option value="${k}">${n}</option>`).join('')}</select></div>
          <div class="card-body"><div class="img-panel light" style="height:420px"><img id="mapImg" src="${r.maps.num_peaks}"></div>
          <div class="row small muted" style="margin-top:6px"><span>低</span><div class="legend-bar" style="flex:1"></div><span>高</span></div></div></div>
        <div class="card"><div class="card-head"><h3>脑区间放电同步连接矩阵</h3><select id="ccK" class="inline-select" style="min-width:150px"><option value="connection_corr">放电同步连接</option><option value="value_corr">峰值强度相关</option></select></div>
          <div class="card-body"><div class="chart" id="cc" style="height:450px"></div></div></div>
      </div>
      <div class="card"><div class="card-head"><h3>逐帧活跃脑区数</h3><span class="muted small">同一时刻出现钙峰的脑区数量，反映全脑同步放电事件</span></div><div class="card-body"><div class="chart" id="ar" style="height:220px"></div></div></div>
      <div class="card"><div class="card-head"><h3>脑区特征明细表</h3><span class="muted small">100 脑区 × 10 类标量特征</span></div><div class="card-body" style="padding:0"><div class="tbl-wrap" style="max-height:380px">
        <table class="tbl compact"><thead><tr><th>脑区</th>${FEATURES.map(([, n]) => `<th>${n}</th>`).join('')}</tr></thead>
        <tbody>${r.table.num_peaks.map((_, i) => `<tr><td>${i + 1}</td>${FEATURES.map(([k]) => `<td>${fmt(r.table[k][i], 3)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div></div>`;
    $('#mapK').onchange = () => { $('#mapImg').src = r.maps[$('#mapK').value]; };
    const drawPk = () => {
      const idx = [1, 35, 60].indexOf(+$('#pkR').value);
      const reg = +$('#pkR').value;
      const y = sig.dff[idx];
      const peaks = r.peaks[reg - 1] || [];
      chart('#pk', {tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 50, right: 20, top: 30, bottom: 45},
        dataZoom: [{type: 'slider', height: 14, bottom: 6}, {type: 'inside'}],
        xAxis: {type: 'value', max: y.length, ...axisBase('帧')}, yAxis: {type: 'value', ...axisBase('ΔF/F')},
        series: [{name: `脑区 ${reg} ΔF/F`, type: 'line', data: y.map((v, t) => [t, v]), showSymbol: false, lineStyle: {width: 1.2}},
          {name: `检测到的钙峰（${peaks.length} 个）`, type: 'scatter', data: peaks.map(p => [p, y[p]]), symbol: 'triangle', symbolSize: 10, itemStyle: {color: '#e5484d'}}]});
    };
    $('#pkR').onchange = drawPk; drawPk();
    const drawCc = () => chart('#cc', heatmapOption(r[$('#ccK').value]));
    $('#ccK').onchange = drawCc; drawCc();
    chart('#ar', {tooltip: {trigger: 'axis'}, grid: {left: 50, right: 20, top: 15, bottom: 30},
      xAxis: {type: 'category', data: r.active_regions.map((_, i) => i), ...axisBase('帧')}, yAxis: {type: 'value', ...axisBase('活跃脑区数')},
      series: [{type: 'bar', data: r.active_regions, itemStyle: {color: '#7c5cdb'}, barCategoryGap: 0}]});
  },
};
