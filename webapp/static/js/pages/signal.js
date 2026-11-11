/* 钙信号提取：异常帧检测、脑区平均荧光、滤波插值、基线校正与 ΔF/F */
Pages.signal = {
  async render(root) {
    root.innerHTML = `<div class="page-head"><div><h2>钙信号提取</h2><p>基于映射的脑区标签提取 100 个脑区的平均荧光时间序列，自动识别甩动异常帧与死鱼样本，经 Butterworth 低通滤波、异常帧插值、IModPoly 多项式基线估计后计算 ΔF/F。</p></div></div>`;
    const wrap = document.createElement('div');
    await expToolbar(root, {sides: true, onChange: s => this.load(wrap, s)});
    root.appendChild(wrap);
  },
  async load(wrap, st) {
    const e = await API.get('/api/experiments/' + st.exp.id);
    if (stepGuard(wrap, e, 'atlas', '脑区模板映射', 'atlas')) return;
    wrap.innerHTML = `
      <div class="card"><div class="card-body"><div class="row">
        <span class="small muted">处理链：</span>
        ${['异常帧识别（IQR 围栏）', '异常像素剔除（时程极差 > 500）', '脑区平均荧光', 'Butterworth 低通（30 阶, 0.3）', '线性插值', 'IModPoly 基线（6 次）', 'ΔF/F 归一化'].map((s, i) => `<span class="tag ${i % 2 ? 'teal' : 'blue'}">${i + 1}. ${s}</span>`).join('<span class="muted">→</span>')}
        <button class="btn primary right" id="sRun">${e.status.signal ? '重新提取' : '开始提取'}</button></div><div id="sTask" style="margin-top:10px"></div></div></div>
      <div id="sRes"></div>`;
    $('#sRun').onclick = async () => {
      const r = await API.post(`/api/experiments/${st.exp.id}/signal`);
      taskWatcher($('#sTask'), r.task_id, {onDone: () => this.load(wrap, st)});
    };
    if (!e.status.signal) { $('#sRes').innerHTML = '<div class="card"><div class="empty"><div class="big">∿</div>尚未提取钙信号</div></div>'; return; }
    this.show(st, '1,12,35,60,88');
  },
  async show(st, regions) {
    const r = await API.get(`/api/experiments/${st.exp.id}/signal?side=${st.side}&regions=${regions}`);
    const s = r.summary;
    $('#sRes').innerHTML = `
      <div class="grid g5">
        <div class="kpi"><div class="l">异常(甩动)帧</div><div class="v">${s.fly_frames.length} 帧</div></div>
        <div class="kpi"><div class="l">剔除异常像素</div><div class="v">${s.outlier_pixels.toLocaleString()}</div></div>
        <div class="kpi"><div class="l">死鱼判定</div><div class="v" style="color:${s.dead ? '#e5484d' : '#2fa36b'}">${s.dead ? '是' : '否（存活）'}</div></div>
        <div class="kpi"><div class="l">有效分析帧数</div><div class="v">${s.frames_used}</div></div>
        <div class="kpi"><div class="l">ΔF/F 均值 / 最大值</div><div class="v">${fmt(s.dff_mean, 3)} / ${fmt(s.dff_max, 3)}</div></div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h3>鱼体平均荧光与异常帧</h3><span class="muted small">红点：自动识别的甩动帧（已剔除并插值）</span></div><div class="card-body"><div class="chart" id="fm" style="height:250px"></div></div></div>
        <div class="card"><div class="card-head"><h3>像素时程极差分布</h3><span class="muted small">极差 > 500 视为异常像素；> 1000 像素过多判定死鱼</span></div><div class="card-body"><div class="chart" id="tr" style="height:250px"></div></div></div>
      </div>
      <div class="card"><div class="card-head"><h3>脑区信号处理过程</h3><div class="row"><span class="small muted">脑区编号（逗号分隔）</span><input type="text" id="rg" value="${r.regions.join(',')}" style="width:180px"><button class="btn sm" id="rgGo">更新</button>
          <select id="stage" class="inline-select" style="min-width:150px"><option value="raw">原始平均荧光</option><option value="filtered">低通滤波后</option><option value="interp" selected>插值 + 基线</option><option value="dff">ΔF/F 归一化</option></select></div></div>
        <div class="card-body"><div class="chart tall" id="tr2"></div></div></div>
      <div class="card"><div class="card-head"><h3>全脑 ΔF/F 活动栅格图（100 脑区 × 时间）</h3></div><div class="card-body">
        <div class="img-panel" style="height:280px"><img src="${r.raster}" style="width:100%;height:100%;object-fit:fill"></div>
        <div class="row small muted" style="margin-top:6px"><span>纵轴：脑区 1 → 100</span><span class="right">横轴：时间帧 → 　色标：Inferno（暗 → 亮 表示 ΔF/F 低 → 高）</span></div></div></div>
      <div class="card"><div class="card-head"><h3>各脑区 ΔF/F 统计</h3></div><div class="card-body"><div class="chart" id="rs" style="height:240px"></div></div></div>`;
    chart('#fm', {tooltip: {trigger: 'axis'}, grid: {left: 55, right: 20, top: 20, bottom: 30},
      xAxis: {type: 'category', data: r.frame_mean.map((_, i) => i), ...axisBase('帧号')}, yAxis: {type: 'value', scale: true, ...axisBase('平均灰度')},
      series: [{type: 'line', data: r.frame_mean, showSymbol: false, lineStyle: {width: 1}},
        {type: 'scatter', data: r.need_move.map(i => [String(i), r.frame_mean[i]]), symbolSize: 8, itemStyle: {color: '#e5484d'}}]});
    chart('#tr', {tooltip: {trigger: 'axis'}, grid: {left: 60, right: 20, top: 20, bottom: 30},
      xAxis: {type: 'category', data: r.time_range.map((_, i) => i * 25), ...axisBase('像素时程极差')}, yAxis: {type: 'log', ...axisBase('像素数')},
      series: [{type: 'bar', data: r.time_range.map((v, i) => ({value: v || null, itemStyle: {color: i * 25 >= 500 ? '#e5484d' : '#12a594'}})), barCategoryGap: 0}],
      markLine: {}});
    const draw = () => {
      const stage = $('#stage').value;
      const series = [];
      r.regions.forEach((g, i) => {
        const data = r[stage][i];
        series.push({name: '脑区 ' + g, type: 'line', data: data.map((v, t) => [t, stage === 'dff' ? v : v + 0]), showSymbol: false, lineStyle: {width: 1.1}, color: PALETTE[i]});
        if (stage === 'interp') series.push({name: '脑区 ' + g + ' 基线', type: 'line', data: r.baseline[i].map((v, t) => [t + 10, v]), showSymbol: false, lineStyle: {width: 1.6, type: 'dashed'}, color: PALETTE[i]});
      });
      chart('#tr2', {tooltip: {trigger: 'axis'}, legend: {top: 0, type: 'scroll'}, grid: {left: 60, right: 20, top: 34, bottom: 50},
        dataZoom: [{type: 'slider', height: 16, bottom: 8}, {type: 'inside'}],
        xAxis: {type: 'value', max: 'dataMax', ...axisBase('帧')}, yAxis: {type: 'value', scale: true, ...axisBase(stage === 'dff' ? 'ΔF/F' : '荧光强度')}, series});
    };
    $('#stage').onchange = draw; draw();
    $('#rgGo').onclick = () => this.show(st, $('#rg').value);
    chart('#rs', {tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 50, right: 20, top: 30, bottom: 30},
      xAxis: {type: 'category', data: r.region_mean_dff.map((_, i) => i + 1), ...axisBase('脑区')}, yAxis: {type: 'value', ...axisBase('ΔF/F')},
      series: [{name: '平均 ΔF/F', type: 'bar', data: r.region_mean_dff, color: '#2f7cf6'}, {name: '最大 ΔF/F', type: 'line', data: r.region_max_dff, color: '#f08c2b', showSymbol: false}]});
  },
};
