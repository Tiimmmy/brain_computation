/* 脑区模板映射：模板→个体配准，逆变换映射 100 脑区标签 */
Pages.atlas = {
  async render(root) {
    root.innerHTML = `<div class="page-head"><div><h2>脑区模板映射</h2><p>将标准脑模板与个体鱼脑参考图像配准，并利用逆变换（InverseComposite）将 100 脑区分割图谱映射到个体空间，实现脑区自动标注。</p></div></div>`;
    const wrap = document.createElement('div');
    await expToolbar(root, {sides: false, onChange: s => this.load(wrap, s.exp)});
    root.appendChild(wrap);
  },
  async load(wrap, exp) {
    const e = await API.get('/api/experiments/' + exp.id);
    if (stepGuard(wrap, e, 'register', '运动校正配准', 'register')) return;
    const tps = await API.get('/api/templates');
    wrap.innerHTML = `
      <div class="card"><div class="card-body"><div class="row">
        <span class="small muted">标准脑模板</span><select id="aTpl" class="inline-select">${tps.map(t => `<option value="${t.id}">${esc(t.name)}</option>`).join('')}</select>
        <span class="small muted">配准算法</span><span class="tag blue">ElasticSyN</span><span class="small muted">标签插值</span><span class="tag">最近邻</span>
        <button class="btn primary right" id="aRun">${e.status.atlas ? '重新映射' : '开始映射'}</button></div><div id="aTask" style="margin-top:10px"></div></div></div>
      <div id="aRes"></div>`;
    $('#aRun').onclick = async () => {
      const r = await API.post(`/api/experiments/${exp.id}/atlas`, {template: $('#aTpl').value});
      taskWatcher($('#aTask'), r.task_id, {onDone: () => this.load(wrap, exp)});
    };
    if (!e.status.atlas) { $('#aRes').innerHTML = '<div class="card"><div class="empty"><div class="big">▦</div>尚未执行脑区映射</div></div>'; return; }
    const r = await API.get(`/api/experiments/${exp.id}/atlas`);
    const s = r.summary.summary;
    $('#aRes').innerHTML = r.fish.map(f => `
      <div class="card"><div class="card-head"><h3>${f.side === 'l' ? '左侧鱼' : '右侧鱼'}映射结果</h3>
        <div class="row"><span class="tag">Dice 映射前 ${fmt(s[f.side].dice_before, 3)}</span><span class="tag green">Dice 映射后 ${fmt(s[f.side].dice_after, 3)}</span><span class="tag blue">覆盖脑区 ${s[f.side].regions}/100</span></div></div>
        <div class="card-body"><div class="grid g5">
          <div>${imgPanel(f.fish, '个体参考图像', 320)}</div>
          <div>${imgPanel(f.before, '映射前轮廓', 320)}<div class="img-cap">黄：个体脑轮廓　品红：模板轮廓</div></div>
          <div>${imgPanel(f.warped, '模板变形结果', 320)}</div>
          <div>${imgPanel(f.after, '映射后轮廓', 320)}<div class="img-cap">轮廓高度吻合</div></div>
          <div>${imgPanel(f.labels, '个体脑区标注', 320)}</div>
        </div>
        <div class="chart short" id="sz_${f.side}" style="margin-top:12px"></div></div></div>`).join('');
    r.fish.forEach(f => chart('#sz_' + f.side, {
      tooltip: {trigger: 'axis'}, grid: {left: 50, right: 20, top: 30, bottom: 30}, title: {text: '各脑区映射像素数', left: 'center', textStyle: {fontSize: 12, fontWeight: 'normal'}},
      xAxis: {type: 'category', data: f.sizes.map((_, i) => i + 1), ...axisBase('脑区编号')}, yAxis: {type: 'value', ...axisBase('像素数')},
      series: [{type: 'bar', data: f.sizes, itemStyle: {color: f.side === 'l' ? '#2f7cf6' : '#f08c2b'}}]}));
  },
};
