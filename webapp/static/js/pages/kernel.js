/* 高斯核空间平滑：基于脑区中心欧氏距离的高斯核，带宽自适应于最近邻距离 */
Pages.kernel = {
  async render(root) {
    const lib = await API.get('/api/library');
    root.innerHTML = `
      <div class="page-head"><div><h2>高斯核空间平滑</h2><p>依据脑区中心坐标计算两两欧氏距离，以各脑区最近邻距离乘带宽系数 H 作为高斯核宽度 σ，归一化后对脑区特征进行空间加权平滑，削弱个体间配准误差带来的脑区边界噪声。</p></div></div>
      <div class="card"><div class="card-body"><div class="row">
        <span class="small muted">带宽系数 H</span><input type="range" id="kH" min="0.1" max="2" step="0.1" value="0.5" style="width:220px"><b id="kHv" class="mono">0.5</b>
        <span class="small muted" style="margin-left:14px">中心脑区</span><input type="number" id="kR" value="25" min="1" max="100" style="width:80px">
        <span class="small muted" style="margin-left:14px">示例样本</span><select id="kF" class="inline-select">${lib.map(e => `<option value="${e.fish_id}">${e.fish_id}（${e.group}）</option>`).join('')}</select>
        <span class="small muted">特征</span><select id="kK" class="inline-select" style="min-width:140px">${FEATURES.map(([k, n]) => `<option value="${k}">${n}</option>`).join('')}</select>
        <button class="btn primary right" id="kGo">计算</button></div>
        <div class="small muted" style="margin-top:8px">公式：K<sub>ij</sub> = exp(−‖c<sub>i</sub> − c<sub>j</sub>‖² / 2σ<sub>j</sub>²)，σ<sub>j</sub> = H · d<sub>j</sub><sup>NN</sup>；列归一化后 X' = K · X</div></div></div>
      <div class="grid g4" id="kKpi"></div>
      <div class="grid" style="grid-template-columns: 1fr 1.3fr; margin-top:16px">
        <div class="card"><div class="card-head"><h3>中心脑区的核权重空间分布</h3></div><div class="card-body"><div class="img-panel light" style="height:430px"><img id="kMap"></div></div></div>
        <div class="card"><div class="card-head"><h3>高斯核矩阵（100 × 100）</h3></div><div class="card-body"><div class="chart" id="kMat" style="height:450px"></div></div></div>
      </div>
      <div class="card"><div class="card-head"><h3>平滑前后特征脑区分布对比</h3></div><div class="card-body"><div class="grid" style="grid-template-columns: 1fr 1fr 1.6fr">
        <div><div class="img-panel light" style="height:400px"><span class="img-label">平滑前</span><img id="kB"></div></div>
        <div><div class="img-panel light" style="height:400px"><span class="img-label">平滑后</span><img id="kA"></div></div>
        <div class="chart" id="kLine" style="height:400px"></div></div>
        <div class="row small muted" style="margin-top:6px;width:66%"><span>低</span><div class="legend-bar" style="flex:1"></div><span>高</span></div></div></div>`;
    $('#kH').oninput = () => { $('#kHv').textContent = $('#kH').value; };
    const go = async () => {
      const H = $('#kH').value, R = $('#kR').value;
      const [p, d] = await Promise.all([API.get(`/api/kernel/preview?H=${H}&region=${R}`),
        lib.length ? API.get(`/api/kernel/demo?H=${H}&fish=${$('#kF').value}&feature=${$('#kK').value}`) : null]);
      $('#kKpi').innerHTML = `<div class="kpi"><div class="l">带宽系数 H</div><div class="v">${H}</div></div>
        <div class="kpi"><div class="l">平均自身权重</div><div class="v">${fmt(p.self_weight, 4)}</div></div>
        <div class="kpi"><div class="l">平均有效参与脑区数</div><div class="v">${fmt(p.effective_regions, 2)}</div></div>
        <div class="kpi"><div class="l">中心脑区</div><div class="v">#${R}</div></div>`;
      $('#kMap').src = p.map;
      chart('#kMat', heatmapOption(p.matrix, {min: 0, max: Math.max(...p.matrix.flat()) * 0.6, colors: ['#ffffff', '#fde0c5', '#f08c2b', '#b03a0a', '#4a1205']}));
      if (d) {
        $('#kB').src = d.before; $('#kA').src = d.after;
        chart('#kLine', {tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 50, right: 20, top: 30, bottom: 30},
          xAxis: {type: 'category', data: d.values_before.map((_, i) => i + 1), ...axisBase('脑区')}, yAxis: {type: 'value', ...axisBase()},
          series: [{name: '平滑前', type: 'line', data: d.values_before, showSymbol: false, lineStyle: {width: 1}, color: '#9fb0c7'},
            {name: '平滑后', type: 'line', data: d.values_after, showSymbol: false, lineStyle: {width: 2}, color: '#2f7cf6'}]});
      }
    };
    $('#kGo').onclick = go; go();
  },
};
