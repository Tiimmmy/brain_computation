/* 标准脑模板库：模板图像、100 脑区图谱、脑区信息表 */
Pages.templates = {
  async render(root) {
    const list = await API.get('/api/templates');
    if (!list.length) { root.innerHTML = '<div class="card"><div class="empty">暂无模板</div></div>'; return; }
    const t = await API.get('/api/templates/' + list[0].id);
    const zones = {};
    t.regions.forEach(r => { zones[r.zone] = (zones[r.zone] || 0) + 1; });
    root.innerHTML = `
      <div class="page-head"><div><h2>标准脑模板库</h2><p>管理用于脑区映射的标准脑模板与脑区分割图谱。模板经配准映射至个体鱼脑后，用于提取各脑区的平均钙信号。</p></div>
        <div class="row"><select class="inline-select">${list.map(x => `<option>${esc(x.name)}（${x.id}）</option>`).join('')}</select></div></div>
      <div class="grid g4">
        <div class="card kpi-card"><div class="card-body"><div class="kpi"><div class="v">${t.info.height} × ${t.info.width}</div><div class="l">模板尺寸（像素）</div></div></div></div>
        <div class="card"><div class="card-body"><div class="kpi"><div class="v">${t.info.regions}</div><div class="l">脑区数量（${esc(t.info.hemisphere)}）</div></div></div></div>
        <div class="card"><div class="card-body"><div class="kpi"><div class="v">${Object.keys(zones).length}</div><div class="l">解剖大区：${Object.keys(zones).join('、')}</div></div></div></div>
        <div class="card"><div class="card-body"><div class="kpi"><div class="v">${Math.round(t.regions.reduce((a, r) => a + r.pixels, 0) / t.regions.length)}</div><div class="l">平均脑区面积（像素）</div></div></div></div>
      </div>
      <div class="grid g3">
        <div class="card"><div class="card-head"><h3>标准脑模板图像</h3></div><div class="card-body">${imgPanel(t.template, 'template.tif', 420)}</div></div>
        <div class="card"><div class="card-head"><h3>100 脑区分割图谱</h3></div><div class="card-body">${imgPanel(t.atlas, 'atlas_100.png', 420, true)}</div></div>
        <div class="card"><div class="card-head"><h3>模板与图谱叠加</h3></div><div class="card-body">${imgPanel(t.overlay, '叠加显示', 420)}</div></div>
      </div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><h3>脑区信息表</h3><span class="muted small">区域编号、所属解剖大区、面积与中心坐标</span></div>
          <div class="card-body" style="padding:0"><div class="tbl-wrap" style="max-height:360px"><table class="tbl compact"><thead><tr><th>编号</th><th>名称</th><th>解剖大区</th><th>面积(像素)</th><th>中心 X</th><th>中心 Y</th></tr></thead>
          <tbody>${t.regions.map(r => `<tr><td>${r.id}</td><td>${r.name}</td><td>${r.zone}</td><td>${r.pixels}</td><td>${r.cx}</td><td>${r.cy}</td></tr>`).join('')}</tbody></table></div></div></div>
        <div class="card"><div class="card-head"><h3>脑区中心空间分布</h3></div><div class="card-body"><div class="chart" id="sc" style="height:360px"></div></div></div>
      </div>`;
    const zoneNames = Object.keys(zones);
    chart('#sc', {
      tooltip: {formatter: p => `脑区 ${p.value[2]}（${p.seriesName}）<br>中心 (${p.value[0]}, ${p.value[1]})`},
      legend: {top: 0}, grid: {left: 40, right: 20, top: 30, bottom: 30},
      xAxis: {type: 'value', min: 0, max: 250, ...axisBase('X')}, yAxis: {type: 'value', min: 0, max: 450, inverse: true, ...axisBase('Y')},
      series: zoneNames.map(z => ({name: z, type: 'scatter', symbolSize: r => Math.sqrt(r[3]) / 1.6,
        data: t.regions.filter(r => r.zone === z).map(r => [r.cx, r.cy, r.id, r.pixels])})),
    });
  },
};
