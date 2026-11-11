/* 系统首页：数据概况、分析流程导航、最近实验与任务 */
Pages.dashboard = {
  async render(root) {
    const d = await API.get('/api/dashboard');
    const best = d.best_model;
    root.innerHTML = `
      <div class="page-head"><div><h2>系统首页</h2><p>欢迎使用斑马鱼全脑钙成像神经计算分析系统，以下为当前工作空间的数据与分析概况。</p></div>
        <div class="row"><a class="btn" href="#experiments">＋ 导入实验</a><a class="btn primary" href="#train">开始建模</a></div></div>
      <div class="grid g4">
        ${this.stat('▤', '#e7f0ff', '#2f7cf6', d.experiments, '实验数据集', `共 ${d.fish} 条鱼体样本`)}
        ${this.stat('☰', '#def6f2', '#12a594', d.library, '个体特征库样本', `对照 ${d.library_groups['0']} · 药物 ${d.library_groups['1']}`)}
        ${this.stat('▣', '#fff1e2', '#f08c2b', d.datasets, '建模数据集', `已训练模型 ${d.models} 个`)}
        ${this.stat('✓', '#efe9fd', '#7c5cdb', best ? best.metrics.accuracy + '%' : '-', '最佳模型准确率', best ? esc(best.name) : '暂无模型')}
      </div>
      <div class="card"><div class="card-head"><h3>分析流程导航</h3><span class="muted small">点击各环节进入对应功能模块</span></div>
        <div class="card-body"><div class="chart short" id="flow" style="height:150px"></div></div></div>
      <div class="grid g3">
        <div class="card span2"><div class="card-head"><h3>各分析环节完成情况</h3></div><div class="card-body"><div class="chart" id="stepChart" style="height:260px"></div></div></div>
        <div class="card"><div class="card-head"><h3>特征库样本构成</h3></div><div class="card-body"><div class="chart" id="pie" style="height:260px"></div></div></div>
      </div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><h3>最近实验</h3><a class="small" href="#experiments">全部 ›</a></div><div class="card-body" style="padding:0">
          <table class="tbl"><thead><tr><th>实验编号</th><th>帧数</th><th>鱼体</th><th>流程进度</th><th>创建时间</th></tr></thead><tbody>
          ${d.recent_experiments.map(e => `<tr><td><a href="#experiments?id=${e.id}">${esc(e.name)}</a></td><td>${e.frames}</td><td>${e.fish.map(f => groupTag(f.group)).join(' ')}</td>
            <td style="width:140px"><div class="progress"><div style="width:${e.progress * 100}%"></div></div></td><td class="muted small">${e.created}</td></tr>`).join('') || '<tr><td colspan="5" class="empty">暂无实验</td></tr>'}
          </tbody></table></div></div>
        <div class="card"><div class="card-head"><h3>最近任务</h3><a class="small" href="#tasks">任务中心 ›</a></div><div class="card-body" style="padding:0">
          <table class="tbl"><thead><tr><th>任务类型</th><th>任务名称</th><th>状态</th><th>用时</th></tr></thead><tbody>
          ${d.tasks.slice(0, 6).map(t => `<tr><td>${t.type_name}</td><td>${esc(t.title)}</td><td>${statusTag(t.status)}</td><td class="muted">${sec(t.elapsed)}</td></tr>`).join('') || '<tr><td colspan="4" class="empty">暂无任务</td></tr>'}
          </tbody></table></div></div>
      </div>`;
    const names = ['数据导入', '图像预处理', '运动校正配准', '脑区模板映射', '钙信号提取', '特征计算', '个体特征库', '高斯核平滑', '数据集构建', '模型训练', '智能预测'];
    const links = ['experiments', 'preprocess', 'register', 'atlas', 'signal', 'feature', 'library', 'kernel', 'dataset', 'train', 'predict'];
    const fc = chart('#flow', {
      tooltip: {show: false},
      series: [{type: 'graph', layout: 'none', symbol: 'roundRect', symbolSize: [92, 34], roam: false,
        label: {show: true, color: '#fff', fontSize: 12},
        edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 7, lineStyle: {color: '#9fb0c7', width: 1.5},
        data: names.map((n, i) => ({name: n, x: i * 110, y: i < 6 ? 0 : 0, itemStyle: {color: i < 1 ? '#2f7cf6' : i < 4 ? '#3d8bfd' : i < 6 ? '#12a594' : i < 9 ? '#f08c2b' : '#7c5cdb'}})),
        links: names.slice(1).map((n, i) => ({source: names[i], target: n}))}],
    });
    fc.on('click', p => { if (p.dataType === 'node') location.hash = links[names.indexOf(p.name)]; });
    const keys = Object.keys(d.step_names);
    chart('#stepChart', {
      tooltip: {trigger: 'axis'}, grid: {left: 40, right: 20, top: 20, bottom: 30},
      xAxis: {type: 'category', data: keys.map(k => d.step_names[k]), ...axisBase()},
      yAxis: {type: 'value', minInterval: 1, ...axisBase('实验数')},
      series: [{type: 'bar', barWidth: 34, data: keys.map(k => d.steps[k]), itemStyle: {borderRadius: [4, 4, 0, 0], color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#2f7cf6'}, {offset: 1, color: '#12a594'}])},
        label: {show: true, position: 'top'}}],
    });
    const src = {};
    chart('#pie', {
      tooltip: {trigger: 'item'}, legend: {bottom: 0},
      series: [{type: 'pie', radius: ['45%', '70%'], center: ['50%', '45%'], label: {formatter: '{b}\n{c} 例'},
        data: [{name: '对照组', value: d.library_groups['0'], itemStyle: {color: '#12a594'}}, {name: '药物处理组', value: d.library_groups['1'], itemStyle: {color: '#f08c2b'}}]}],
    });
  },
  stat(ico, bg, color, val, label, sub) {
    return `<div class="card stat"><div class="stat-ico" style="background:${bg};color:${color}">${ico}</div>
      <div><div class="stat-val">${val}</div><div class="stat-label">${label}</div><div class="small muted" style="margin-top:2px">${sub}</div></div></div>`;
  },
};
