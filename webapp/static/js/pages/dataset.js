/* 建模数据集：样本与特征选择、多通道归一化、高斯核平滑、PCA 预览 */
Pages.dataset = {
  async render(root) {
    const [list, lib] = await Promise.all([API.get('/api/datasets'), API.get('/api/library')]);
    const CH = [['raw', '原始特征值'], ['minmax', 'Min-Max 归一化'], ['zscore', 'Z-Score 标准化'], ['kernel_raw', '高斯核平滑'], ['kernel_minmax', '高斯核平滑 + Min-Max'], ['kernel_zscore', '高斯核平滑 + Z-Score']];
    root.innerHTML = `
      <div class="page-head"><div><h2>建模数据集</h2><p>从个体特征库选取样本与特征类型，按脑区展开为特征单元，并以多种归一化/空间平滑方式构成多通道输入张量（样本 × 特征单元 × 通道），供 BrainNet 模型训练。</p></div></div>
      <div class="grid" style="grid-template-columns: 400px 1fr">
        <div class="card"><div class="card-head"><h3>新建数据集</h3></div><div class="card-body"><div class="form">
          <div class="field"><label>数据集名称</label><input type="text" id="dN" value="成瘾表型识别数据集-${new Date().toISOString().slice(0, 10)}"></div>
          <div class="field"><label>样本范围</label><select id="dS"><option value="all">特征库全部样本（${lib.length} 例）</option><option value="import">仅信号矩阵导入样本</option></select></div>
          <div class="field"><label>特征类型</label><div>${FEATURES.map(([k, n]) => `<label class="check"><input type="checkbox" class="dF" value="${k}" checked>${n}</label>`).join('')}</div></div>
          <div class="field"><label>输入通道</label><div>${CH.map(([k, n]) => `<label class="check"><input type="checkbox" class="dC" value="${k}" ${['minmax', 'zscore', 'kernel_minmax', 'kernel_zscore'].includes(k) ? 'checked' : ''}>${n}</label>`).join('')}</div></div>
          <div class="field"><label>高斯核带宽系数 H</label><input type="number" id="dH" value="0.5" step="0.1"></div>
          <div class="field"><label>描述</label><input type="text" id="dD" placeholder="可选"></div>
          <button class="btn primary" id="dGo">构建数据集</button><div id="dTask"></div></div></div></div>
        <div>
          <div class="card"><div class="card-head"><h3>数据集列表</h3></div><div class="card-body" style="padding:0"><table class="tbl"><thead><tr><th>编号</th><th>名称</th><th>样本(对照/药物)</th><th>张量形状</th><th>通道</th><th>H</th><th>创建时间</th><th>操作</th></tr></thead>
            <tbody>${list.map(d => `<tr><td class="mono">${d.id}</td><td>${esc(d.name)}</td><td>${d.samples}（${d.labels['0']}/${d.labels['1']}）</td><td class="mono">${d.shape.join('×')}</td>
              <td>${d.channels.length}</td><td>${d.band_width}</td><td class="small muted">${d.created}</td><td><button class="btn sm" data-v="${d.id}">预览</button> <button class="btn sm danger" data-d="${d.id}">删除</button></td></tr>`).join('') || '<tr><td colspan="8"><div class="empty">暂无数据集</div></td></tr>'}</tbody></table></div></div>
          <div id="dPrev"></div>
        </div>
      </div>`;
    $('#dGo').onclick = async () => {
      const ids = $('#dS').value === 'all' ? lib.map(e => e.fish_id) : lib.filter(e => e.source === '信号矩阵导入').map(e => e.fish_id);
      const r = await API.post('/api/datasets', {name: $('#dN').value, ids, feature_keys: $$('.dF').filter(c => c.checked).map(c => c.value),
        channels: $$('.dC').filter(c => c.checked).map(c => c.value), band_width: +$('#dH').value, description: $('#dD').value});
      taskWatcher($('#dTask'), r.task_id, {showLog: true, onDone: () => App.route()});
    };
    $$('[data-d]').forEach(b => b.onclick = () => confirmBox('确认删除该数据集？', async () => { await API.del('/api/datasets/' + b.dataset.d); App.route(); }));
    $$('[data-v]').forEach(b => b.onclick = () => this.preview(b.dataset.v));
    if (list.length) this.preview(list[0].id);
  },
  async preview(id) {
    const d = await API.get('/api/datasets/' + id);
    $('#dPrev').innerHTML = `<div class="card"><div class="card-head"><h3>数据集预览 · ${esc(d.name)}</h3><span class="tag blue">${d.id}</span></div><div class="card-body">
      <div class="grid g4"><div class="kpi"><div class="l">样本数</div><div class="v">${d.samples}</div></div><div class="kpi"><div class="l">特征单元（类型 × 脑区）</div><div class="v">${d.feature_keys.length} × ${d.regions}</div></div>
        <div class="kpi"><div class="l">输入通道</div><div class="v">${d.channels.length}</div></div><div class="kpi"><div class="l">数据量</div><div class="v">${d.size_mb} MB</div></div></div>
      <div class="grid g2" style="margin-top:14px"><div class="chart" id="pca" style="height:320px"></div>
        <div><div class="small muted" style="margin-bottom:6px">特征类型</div><div>${d.feature_names.map(n => `<span class="tag teal" style="margin:2px">${n}</span>`).join('')}</div>
          <div class="small muted" style="margin:12px 0 6px">输入通道</div><div>${d.channels.map(n => `<span class="tag purple" style="margin:2px">${n}</span>`).join('')}</div>
          <div class="chart" id="lb" style="height:200px;margin-top:10px"></div></div></div></div></div>`;
    const pts = g => d.pca.x.map((x, i) => [x, d.pca.y[i], d.pca.label[i]]).filter(p => p[2] === g);
    chart('#pca', {title: {text: `PCA 二维投影（PC1 ${(d.pca.var[0] * 100).toFixed(1)}%，PC2 ${(d.pca.var[1] * 100).toFixed(1)}%）`, left: 'center', textStyle: {fontSize: 12, fontWeight: 'normal'}},
      tooltip: {}, legend: {bottom: 0}, grid: {left: 45, right: 20, top: 35, bottom: 45}, xAxis: {...axisBase('PC1'), scale: true}, yAxis: {...axisBase('PC2'), scale: true},
      series: [{name: '对照组', type: 'scatter', data: pts(0), color: '#12a594', symbolSize: 8}, {name: '药物处理组', type: 'scatter', data: pts(1), color: '#f08c2b', symbolSize: 8}]});
    chart('#lb', {tooltip: {}, series: [{type: 'pie', radius: ['40%', '68%'], label: {formatter: '{b} {c}'},
      data: [{name: '对照组', value: d.labels['0'], itemStyle: {color: '#12a594'}}, {name: '药物处理组', value: d.labels['1'], itemStyle: {color: '#f08c2b'}}]}]});
  },
};
