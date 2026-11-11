/* 智能预测：选择模型与待测样本，输出药物处理（成瘾）概率与判定结果 */
Pages.predict = {
  async render(root) {
    const [models, lib] = await Promise.all([API.get('/api/models'), API.get('/api/library')]);
    root.innerHTML = `
      <div class="page-head"><div><h2>智能预测</h2><p>调用已训练的 BrainNet 模型，对个体特征库中的待测样本进行表型判别，输出属于药物处理（成瘾）表型的概率及判定结果，支持批量预测与结果导出。</p></div></div>
      <div class="grid" style="grid-template-columns: 380px 1fr">
        <div class="card"><div class="card-head"><h3>预测设置</h3></div><div class="card-body"><div class="form">
          <div class="field"><label>预测模型</label><select id="pM">${models.map(m => `<option value="${m.id}">${esc(m.name)}（准确率 ${m.metrics.accuracy}%）</option>`).join('')}</select></div>
          <div class="field"><label>判定阈值</label><input type="text" value="P(药物处理) ≥ 0.5 判定为药物处理表型" disabled></div>
          <div class="field"><label>待测样本（可多选）</label><div class="row" style="margin-bottom:6px"><button class="btn sm" id="sA">全选</button><button class="btn sm" id="sE">仅实验流水线样本</button><button class="btn sm" id="sR">随机 20 例</button><button class="btn sm" id="sN">清空</button></div>
            <div class="dir-list" style="max-height:330px">${lib.map(e => `<div><label class="check" style="margin:0"><input type="checkbox" class="pS" value="${e.fish_id}" data-src="${e.source}">${e.fish_id}</label>${groupTag(e.label)}</div>`).join('')}</div></div>
          <button class="btn primary" id="pGo" ${models.length ? '' : 'disabled'}>开始预测</button></div></div></div>
        <div id="pRes"><div class="card"><div class="empty"><div class="big">➤</div>请选择模型与待测样本后点击「开始预测」</div></div></div>
      </div>`;
    $('#sA').onclick = () => $$('.pS').forEach(c => c.checked = true);
    $('#sN').onclick = () => $$('.pS').forEach(c => c.checked = false);
    $('#sE').onclick = () => $$('.pS').forEach(c => c.checked = c.dataset.src === '实验流水线');
    $('#sR').onclick = () => { const cs = $$('.pS'); cs.forEach(c => c.checked = false); cs.sort(() => Math.random() - .5).slice(0, 20).forEach(c => c.checked = true); };
    $('#pGo').onclick = async () => {
      const ids = $$('.pS').filter(c => c.checked).map(c => c.value);
      if (!ids.length) return toast('请选择待测样本', 'error');
      this.show(await API.post(`/api/models/${$('#pM').value}/predict`, {ids}));
    };
    if (models.length) {
      const h = await API.get(`/api/models/${models[0].id}/predictions`);
      if (h.length) this.show(h[0]);
    }
  },
  show(rec) {
    const R = rec.results;
    const correct = R.filter(r => r.correct).length;
    const drug = R.filter(r => r.pred === 1).length;
    $('#pRes').innerHTML = `
      <div class="grid g4"><div class="kpi"><div class="l">预测样本数</div><div class="v">${R.length}</div></div>
        <div class="kpi"><div class="l">判定为药物处理表型</div><div class="v" style="color:#f08c2b">${drug}</div></div>
        <div class="kpi"><div class="l">判定为对照表型</div><div class="v" style="color:#12a594">${R.length - drug}</div></div>
        <div class="kpi up"><div class="l">与已知分组一致率</div><div class="v">${pct(correct / R.length)}</div></div></div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h3>预测概率分布</h3><span class="small muted">预测时间 ${rec.time} · 模型 ${rec.model}</span></div><div class="card-body"><div class="chart" id="pc" style="height:260px"></div></div></div>
      <div class="card"><div class="card-head"><h3>预测结果明细</h3><a class="btn sm" href="/api/export/model/${rec.model}/predictions.csv">导出 CSV</a></div><div class="card-body" style="padding:0"><div class="tbl-wrap" style="max-height:420px">
        <table class="tbl compact"><thead><tr><th>样本编号</th><th>样本来源</th><th>已知分组</th><th>药物处理概率</th><th>预测结果</th><th>是否一致</th></tr></thead><tbody>
        ${R.map(r => `<tr><td><b>${r.fish_id}</b></td><td>${r.source}${r.in_training ? ' <span class="tag">训练集</span>' : ' <span class="tag blue">新样本</span>'}</td><td>${groupTag(r.true)}</td>
          <td style="width:220px"><div class="row" style="flex-wrap:nowrap"><div class="progress" style="flex:1"><div style="width:${r.prob * 100}%;background:${r.prob >= .5 ? '#f08c2b' : '#12a594'}"></div></div><span class="mono">${r.prob.toFixed(3)}</span></div></td>
          <td>${groupTag(r.pred)}</td><td>${r.correct ? '<span class="tag green">✓ 一致</span>' : '<span class="tag red">✗ 不一致</span>'}</td></tr>`).join('')}</tbody></table></div></div></div>`;
    chart('#pc', {tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 50, right: 20, top: 30, bottom: 60},
      xAxis: {type: 'category', data: R.map(r => r.fish_id), axisLabel: {rotate: 45, fontSize: 10}}, yAxis: {type: 'value', min: 0, max: 1, ...axisBase('P(药物处理)')},
      series: [{name: '药物处理概率', type: 'bar', data: R.map(r => ({value: r.prob, itemStyle: {color: r.true ? '#f08c2b' : '#12a594'}})),
        markLine: {silent: true, data: [{yAxis: 0.5, name: '阈值'}], lineStyle: {color: '#e5484d', type: 'dashed'}}}]});
  },
};
