/* 模型评估：模型列表、交叉验证指标、混淆矩阵、ROC 曲线、显著性脑区 */
Pages.evaluate = {
  async render(root, params) {
    const list = await API.get('/api/models');
    root.innerHTML = `<div class="page-head"><div><h2>模型评估</h2><p>查看已训练模型的交叉验证性能：准确率、精确率、召回率、F1、AUC、混淆矩阵与 ROC 曲线，并基于输入梯度显著性分析对分类贡献最大的特征类型与脑区。</p></div></div>
      <div class="card"><div class="card-head"><h3>模型列表</h3></div><div class="card-body" style="padding:0"><table class="tbl"><thead><tr><th>模型编号</th><th>名称</th><th>数据集</th><th>参数量</th><th>折数/轮数</th><th>准确率</th><th>精确率</th><th>召回率</th><th>AUC</th><th>训练用时</th><th>创建时间</th><th>操作</th></tr></thead>
        <tbody>${list.map(m => `<tr><td class="mono">${m.id}</td><td>${esc(m.name)}</td><td>${esc(m.dataset_name)}</td><td>${(m.params / 1e6).toFixed(2)} M</td><td>${m.hyper.k_folds} / ${m.hyper.epochs}</td>
          <td><b style="color:#2fa36b">${m.metrics.accuracy}%</b></td><td>${m.metrics.precision}%</td><td>${m.metrics.recall}%</td><td>${fmt(m.metrics.auc, 4)}</td><td>${sec(m.seconds)}</td><td class="small muted">${m.created}</td>
          <td><button class="btn sm" data-v="${m.id}">评估详情</button> <button class="btn sm danger" data-d="${m.id}">删除</button></td></tr>`).join('') || '<tr><td colspan="12"><div class="empty">暂无模型，请先在「模型训练」中训练</div></td></tr>'}</tbody></table></div></div>
      <div id="eDet"></div>`;
    $$('[data-v]').forEach(b => b.onclick = () => this.detail(b.dataset.v));
    $$('[data-d]').forEach(b => b.onclick = () => confirmBox('确认删除该模型？', async () => { await API.del('/api/models/' + b.dataset.d); App.route(); }));
    if (list.length) this.detail(params.id || list[0].id);
  },
  async detail(id) {
    const m = await API.get('/api/models/' + id);
    const c = m.metrics.confusion;
    $('#eDet').innerHTML = `
      <div class="grid g5">${[['交叉验证准确率', m.metrics.accuracy + '%'], ['精确率 Precision', m.metrics.precision + '%'], ['召回率 Recall', m.metrics.recall + '%'], ['F1 分数', m.metrics.f1 + '%'], ['AUC', fmt(m.metrics.auc, 4)]]
        .map(([l, v]) => `<div class="kpi up"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('')}</div>
      <div class="grid g3" style="margin-top:16px">
        <div class="card"><div class="card-head"><h3>混淆矩阵（汇总各折最佳）</h3></div><div class="card-body"><div class="chart" id="cm" style="height:300px"></div></div></div>
        <div class="card"><div class="card-head"><h3>ROC 曲线</h3></div><div class="card-body"><div class="chart" id="roc" style="height:300px"></div></div></div>
        <div class="card"><div class="card-head"><h3>各折验证准确率</h3></div><div class="card-body"><div class="chart" id="fold" style="height:300px"></div></div></div>
      </div>
      <div class="grid" style="grid-template-columns: 1fr 1fr 1.2fr">
        <div class="card"><div class="card-head"><h3>特征类型重要性</h3></div><div class="card-body"><div class="chart" id="impF" style="height:360px"></div></div></div>
        <div class="card"><div class="card-head"><h3>脑区重要性分布图</h3></div><div class="card-body"><div class="img-panel light" style="height:360px"><img src="${m.importance_map}"></div></div></div>
        <div class="card"><div class="card-head"><h3>特征类型 × 脑区 显著性热图</h3></div><div class="card-body"><div class="chart" id="impM" style="height:360px"></div></div></div>
      </div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><h3>训练过程（各折验证准确率）</h3></div><div class="card-body"><div class="chart" id="hist" style="height:280px"></div></div></div>
        <div class="card"><div class="card-head"><h3>模型信息</h3></div><div class="card-body"><dl class="desc-list">
          <dt>模型编号</dt><dd class="mono">${m.id}</dd><dt>网络结构</dt><dd>${m.arch}（输入 ${m.num_features} 特征单元 × ${m.in_dim} 通道）</dd><dt>参数量</dt><dd>${m.params.toLocaleString()}</dd>
          <dt>训练数据集</dt><dd>${esc(m.dataset_name)}（${m.dataset}）</dd><dt>输入通道</dt><dd>${m.channels.join('，')}</dd>
          <dt>优化器</dt><dd>${m.hyper.optimizer}，lr=${m.hyper.lr}，momentum=${m.hyper.momentum}，weight_decay=${m.hyper.weight_decay}</dd>
          <dt>交叉验证</dt><dd>${m.hyper.k_folds} 折 × ${m.hyper.epochs} 轮</dd><dt>计算设备</dt><dd>${m.hyper.device}</dd><dt>训练用时</dt><dd>${sec(m.seconds)}</dd>
          <dt>各折明细</dt><dd>${m.folds.map(f => `第${f.fold}折：${f.acc}%（Epoch ${f.best_epoch}）`).join('<br>')}</dd></dl></div></div>
      </div>`;
    chart('#cm', {tooltip: {}, grid: {left: 90, right: 30, top: 20, bottom: 50},
      xAxis: {type: 'category', data: ['预测：对照', '预测：药物'], position: 'bottom'}, yAxis: {type: 'category', data: ['真实：药物', '真实：对照']},
      visualMap: {show: false, min: 0, max: Math.max(c.tp, c.tn, c.fp, c.fn), inRange: {color: ['#eef4ff', '#2f7cf6']}},
      series: [{type: 'heatmap', data: [[0, 1, c.tn], [1, 1, c.fp], [0, 0, c.fn], [1, 0, c.tp]], label: {show: true, fontSize: 20, fontWeight: 'bold'}}]});
    chart('#roc', {tooltip: {trigger: 'axis'}, grid: {left: 50, right: 20, top: 20, bottom: 40},
      xAxis: {type: 'value', min: 0, max: 1, ...axisBase('假阳性率 FPR')}, yAxis: {type: 'value', min: 0, max: 1, ...axisBase('真阳性率 TPR')},
      series: [{type: 'line', data: m.roc.fpr.map((x, i) => [x, m.roc.tpr[i]]), areaStyle: {opacity: .12}, showSymbol: false, name: `AUC=${fmt(m.metrics.auc, 3)}`, lineStyle: {width: 2}},
        {type: 'line', data: [[0, 0], [1, 1]], lineStyle: {type: 'dashed', color: '#b0b8c4'}, showSymbol: false}],
      graphic: [{type: 'text', right: 40, bottom: 60, style: {text: `AUC = ${fmt(m.metrics.auc, 4)}`, fontSize: 14, fill: '#2f7cf6'}}]});
    chart('#fold', {tooltip: {}, grid: {left: 45, right: 20, top: 20, bottom: 30}, xAxis: {type: 'category', data: m.folds.map(f => '第' + f.fold + '折')}, yAxis: {type: 'value', max: 100, ...axisBase('%')},
      series: [{type: 'bar', data: m.folds.map(f => f.acc), barWidth: 30, itemStyle: {color: '#12a594', borderRadius: [4, 4, 0, 0]}, label: {show: true, position: 'top'},
        markLine: {data: [{type: 'average', name: '平均'}], lineStyle: {color: '#f08c2b'}}}]});
    const fi = m.importance.by_feature.map((v, i) => [m.feature_names[i], v]).sort((a, b) => a[1] - b[1]);
    chart('#impF', {tooltip: {}, grid: {left: 110, right: 30, top: 10, bottom: 30}, xAxis: {type: 'value', ...axisBase('相对重要性')}, yAxis: {type: 'category', data: fi.map(x => x[0])},
      series: [{type: 'bar', data: fi.map(x => x[1]), itemStyle: {color: '#7c5cdb', borderRadius: [0, 4, 4, 0]}}]});
    const data = [];
    m.importance.matrix.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
    chart('#impM', {tooltip: {formatter: p => `${m.feature_names[p.value[1]]} · 脑区 ${p.value[0] + 1}<br>${fmt(p.value[2], 4)}`}, grid: {left: 110, right: 60, top: 10, bottom: 30},
      xAxis: {type: 'category', data: [...Array(100).keys()].map(i => i + 1), axisLabel: {interval: 9}}, yAxis: {type: 'category', data: m.feature_names},
      visualMap: {min: 0, max: 1, right: 0, top: 'center', itemHeight: 140, inRange: {color: ['#ffffff', '#fcd9a8', '#f08c2b', '#a3300c']}}, series: [{type: 'heatmap', data}]});
    chart('#hist', {tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 45, right: 20, top: 30, bottom: 30},
      xAxis: {type: 'category', data: m.history[0].test_acc.map((_, i) => i + 1), ...axisBase('Epoch')}, yAxis: {type: 'value', max: 100, ...axisBase('%')},
      series: m.history.map(h => ({name: `第${h.fold}折`, type: 'line', data: h.test_acc, showSymbol: false, lineStyle: {width: 1.2}}))});
  },
};
