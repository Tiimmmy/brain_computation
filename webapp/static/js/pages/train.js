/* 模型训练：BrainNet 网络结构、超参数设置、K 折交叉验证实时训练曲线 */
Pages.train = {
  async render(root, params) {
    const [ds, sys] = await Promise.all([API.get('/api/datasets'), API.get('/api/system')]);
    const t = sys.settings.training;
    root.innerHTML = `
      <div class="page-head"><div><h2>模型训练</h2><p>采用 BrainNet 深度网络：每个特征单元由独立子网络（Linear-BN-ReLU-Dropout × 2）编码为标量，拼接后经 4096-2048-1024-512 全连接分类头输出对照/药物处理二分类结果，K 折交叉验证评估泛化性能。</p></div></div>
      <div class="grid" style="grid-template-columns: 380px 1fr">
        <div class="card"><div class="card-head"><h3>训练配置</h3></div><div class="card-body"><div class="form">
          <div class="field"><label>模型名称</label><input type="text" id="tN" value="BrainNet-成瘾识别-${new Date().toISOString().slice(5, 10)}"></div>
          <div class="field"><label>训练数据集</label><select id="tD">${ds.map(d => `<option value="${d.id}">${esc(d.name)}（${d.shape.join('×')}）</option>`).join('')}</select></div>
          <div class="form g2">
            <div class="field"><label>交叉验证折数</label><input type="number" id="tK" value="${t.k_folds}"></div>
            <div class="field"><label>训练轮数 Epoch</label><input type="number" id="tE" value="${t.epochs}"></div>
            <div class="field"><label>优化器</label><select id="tO"><option ${t.optimizer === 'SGD' ? 'selected' : ''}>SGD</option><option ${t.optimizer === 'Adam' ? 'selected' : ''}>Adam</option></select></div>
            <div class="field"><label>学习率</label><input type="number" id="tL" value="${t.lr}" step="0.001"></div>
            <div class="field"><label>动量</label><input type="number" id="tM" value="${t.momentum}" step="0.05"></div>
            <div class="field"><label>权重衰减</label><input type="text" id="tW" value="${t.weight_decay}"></div>
            <div class="field"><label>计算设备</label><select id="tDev"><option value="cuda">GPU（${sys.device.gpus[0] || '不可用'}）</option><option value="cpu">CPU</option></select></div>
            <div class="field"><label>随机种子</label><input type="number" id="tS" value="42"></div>
          </div>
          <div class="field"><label>损失函数</label><input type="text" value="交叉熵损失 CrossEntropyLoss" disabled></div>
          <button class="btn primary" id="tGo" ${ds.length ? '' : 'disabled'}>开始训练</button></div></div></div>
        <div>
          <div class="card"><div class="card-head"><h3>网络结构</h3></div><div class="card-body"><div class="chart" id="arch" style="height:170px"></div></div></div>
          <div class="card"><div class="card-head"><h3>训练监控</h3><span class="muted small" id="tInfo">提交训练后实时显示各折损失与准确率曲线</span></div><div class="card-body">
            <div id="tTask"></div>
            <div class="grid g2" style="margin-top:12px"><div class="chart" id="lossC" style="height:280px"></div><div class="chart" id="accC" style="height:280px"></div></div></div></div>
        </div>
      </div>`;
    const blocks = [['输入张量', 'N × F × C'], ['Cell 子网络 ×F', 'C→32→16→1'], ['拼接', 'F 维'], ['FC 4096', 'BN·ReLU·Drop'], ['FC 2048', 'BN·ReLU·Drop'], ['FC 1024', 'BN·ReLU'], ['FC 512', 'BN·ReLU'], ['输出 2 类', 'Softmax']];
    chart('#arch', {series: [{type: 'graph', layout: 'none', symbol: 'roundRect', symbolSize: [100, 50], label: {show: true, color: '#fff', fontSize: 11.5, formatter: p => `{a|${p.data.name}}\n{b|${p.data.sub}}`, rich: {a: {fontWeight: 'bold', fontSize: 12, color: '#fff'}, b: {fontSize: 10.5, color: '#e6eefc', padding: [3, 0, 0, 0]}}},
      edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 7, lineStyle: {color: '#9fb0c7', width: 1.5},
      data: blocks.map(([n, s], i) => ({name: n, sub: s, x: i * 120, y: 0, itemStyle: {color: i === 0 ? '#64748b' : i < 3 ? '#12a594' : i < 7 ? '#2f7cf6' : '#f08c2b'}})),
      links: blocks.slice(1).map((b, i) => ({source: blocks[i][0], target: b[0]}))}]});
    const drawLive = (hist) => {
      if (!hist || !hist.length) return;
      const ser = (key, dashed) => hist.map((h, i) => ({name: `第${h.fold}折 ${key.startsWith('train') ? '训练' : '验证'}`, type: 'line', showSymbol: false, data: h[key], color: PALETTE[i % 8], lineStyle: {width: 1.3, type: dashed ? 'dashed' : 'solid'}}));
      const ep = Math.max(...hist.map(h => h.train_loss.length));
      const base = {tooltip: {trigger: 'axis'}, legend: {top: 0, type: 'scroll'}, grid: {left: 50, right: 15, top: 34, bottom: 30}, xAxis: {type: 'category', data: [...Array(ep).keys()].map(i => i + 1), ...axisBase('Epoch')}};
      chart('#lossC', {...base, title: {show: false}, yAxis: {type: 'value', ...axisBase('损失')}, series: [...ser('train_loss', true), ...ser('test_loss')]});
      chart('#accC', {...base, yAxis: {type: 'value', max: 100, ...axisBase('准确率 %')}, series: [...ser('train_acc', true), ...ser('test_acc')]});
    };
    const watch = (tid) => taskWatcher($('#tTask'), tid, {onTick: async (t) => { const full = await API.get('/api/tasks/' + tid); drawLive(full.extra && full.extra.live); },
      onDone: t => { if (t.status === 'success') $('#tInfo').innerHTML = `训练完成，<a href="#evaluate?id=${t.result.model_id}">查看模型评估 ›</a>`; }});
    $('#tGo').onclick = async () => {
      const r = await API.post('/api/models/train', {name: $('#tN').value, dataset: $('#tD').value, k_folds: +$('#tK').value, epochs: +$('#tE').value,
        optimizer: $('#tO').value, lr: +$('#tL').value, momentum: +$('#tM').value, weight_decay: +$('#tW').value, device: $('#tDev').value, seed: +$('#tS').value});
      watch(r.task_id);
    };
    const tasks = await API.get('/api/tasks');
    const last = params.task ? {id: params.task} : tasks.find(x => x.type === 'train');
    if (last) watch(last.id);
  },
};
