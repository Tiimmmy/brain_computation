/* 运动校正配准：ANTs 逐帧配准任务、配准质量曲线、配准前后对比 */
Pages.register = {
  async render(root) {
    root.innerHTML = `<div class="page-head"><div><h2>运动校正配准</h2><p>以参考帧为固定图像，采用 ANTs 弹性对称归一化（ElasticSyN）等算法对每一帧进行非刚性配准，消除鱼体漂移与形变，生成对齐的时空矩阵。</p></div></div>`;
    const wrap = document.createElement('div');
    await expToolbar(root, {sides: false, onChange: s => this.load(wrap, s.exp)});
    root.appendChild(wrap);
  },
  async load(wrap, exp) {
    const e = await API.get('/api/experiments/' + exp.id);
    if (stepGuard(wrap, e, 'preprocess', '图像预处理', 'preprocess')) return;
    const rp = e.params.registration;
    const sum = e.results.register;
    wrap.innerHTML = `
      <div class="grid" style="grid-template-columns: 330px 1fr">
        <div class="card"><div class="card-head"><h3>配准参数</h3></div><div class="card-body"><div class="form">
          <div class="field"><label>配准变换模型</label><select id="rTr">
            ${[['ElasticSyN', 'ElasticSyN（仿射 + 弹性对称归一化）'], ['SyN', 'SyN（仿射 + 对称微分同胚）'], ['Affine', 'Affine（仿射）'], ['Rigid', 'Rigid（刚体）']]
              .map(([v, n]) => `<option value="${v}" ${rp.transform === v ? 'selected' : ''}>${n}</option>`).join('')}</select></div>
          <div class="field"><label>配准降采样尺寸</label><input type="text" value="${rp.resample_h} × ${rp.resample_w}" disabled><div class="hint">加速配准；形变场在全分辨率图像上应用</div></div>
          <div class="field"><label>多分辨率平滑 σ / 收缩因子</label><input type="text" value="(4,3,2,1) / (8,4,2,1)" disabled></div>
          <div class="field"><label>并行进程数</label><input type="number" id="rW" value="${rp.workers}" min="1" max="112"></div>
          <div class="field"><label>参考帧</label><input type="text" value="#${e.params.preprocess.ref_frame}" disabled></div>
          <button class="btn primary" id="rRun">${e.status.register ? '重新配准' : '开始配准'}</button>
          <div id="rTask"></div>
        </div></div></div>
        <div id="rRes">${sum ? '' : '<div class="card"><div class="empty"><div class="big">⇄</div>尚未执行配准，请设置参数后点击「开始配准」</div></div>'}</div>
      </div>`;
    $('#rRun').onclick = async () => {
      const r = await API.post(`/api/experiments/${exp.id}/register`, {transform: $('#rTr').value, workers: +$('#rW').value});
      taskWatcher($('#rTask'), r.task_id, {onDone: () => this.load(wrap, exp)});
    };
    if (sum) this.result(exp.id, 100);
  },
  async result(id, frame) {
    const r = await API.get(`/api/experiments/${id}/register?frame=${frame}`);
    if (!r.ready) return;
    const s = r.summary;
    $('#rRes').innerHTML = `
      <div class="grid g4">
        ${r.fish.map(f => `<div class="kpi"><div class="l">${f.side === 'l' ? '左侧鱼' : '右侧鱼'} 平均相关系数（配准前 → 后）</div><div class="v">${fmt(s.summary[f.side].corr_before, 4)} → <span style="color:#2fa36b">${fmt(s.summary[f.side].corr_after, 4)}</span></div></div>`).join('')}
        <div class="kpi"><div class="l">配准帧数 / 算法</div><div class="v">${s.frames} 帧 · ${s.transform}</div></div>
        <div class="kpi"><div class="l">计算用时</div><div class="v">${sec(s.seconds)}</div></div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h3>逐帧配准质量（与参考帧的相关系数）</h3><span class="muted small">点击曲线查看对应帧配准效果</span></div>
        <div class="card-body"><div class="chart" id="curve" style="height:250px"></div></div></div>
      <div class="card"><div class="card-head"><h3>单帧配准效果对比 · 帧 #${r.frame}</h3><div class="row"><span class="small muted">帧号</span><input type="number" id="cf" value="${r.frame}" style="width:90px"><button class="btn sm" id="cfGo">查看</button></div></div>
        <div class="card-body"><div class="small muted" style="margin-bottom:8px">品红：参考帧　绿色：当前帧　灰白：两者重合（错位越小，彩色边缘越少）</div>
          ${r.fish.map(f => `<div class="grid g4" style="margin-bottom:12px">
            <div>${imgPanel(f.before, (f.side === 'l' ? '左' : '右') + '·配准前叠加', 300)}<div class="img-cap">r = ${fmt(f.frame_metric.before, 4)}</div></div>
            <div>${imgPanel(f.after, (f.side === 'l' ? '左' : '右') + '·配准后叠加', 300)}<div class="img-cap">r = ${fmt(f.frame_metric.after, 4)}</div></div>
            <div>${imgPanel(f.checker, '棋盘格对比', 300)}<div class="img-cap">参考帧 / 配准后交替拼接</div></div>
            <div>${imgPanel(f.std_after, '时间标准差投影', 300)}<div class="img-cap">配准后全时程像素波动</div></div></div>`).join('')}
        </div></div>
      <div class="card"><div class="card-head"><h3>时间平均投影（配准前 vs 配准后）</h3></div><div class="card-body"><div class="grid g4">
        ${r.fish.map(f => `<div>${imgPanel(f.mean_before, (f.side === 'l' ? '左' : '右') + '·配准前', 300)}</div><div>${imgPanel(f.mean_after, (f.side === 'l' ? '左' : '右') + '·配准后', 300)}</div>`).join('')}
      </div><div class="img-cap">配准后平均投影边界更清晰，表明运动伪影被有效消除</div></div></div>`;
    chart('#curve', {
      tooltip: {trigger: 'axis'}, legend: {top: 0}, grid: {left: 50, right: 20, top: 30, bottom: 30},
      xAxis: {type: 'category', data: r.fish[0].curve.frames, ...axisBase('帧号')}, yAxis: {type: 'value', min: v => Math.max(0, Math.floor(v.min * 20) / 20), max: 1, ...axisBase('相关系数')},
      series: r.fish.flatMap((f, i) => [
        {name: `${f.side === 'l' ? '左' : '右'}·配准前`, type: 'line', data: f.curve.before, showSymbol: false, lineStyle: {width: 1, type: 'dashed'}, color: i ? '#f5b06b' : '#9cc2fb'},
        {name: `${f.side === 'l' ? '左' : '右'}·配准后`, type: 'line', data: f.curve.after, showSymbol: false, lineStyle: {width: 1.4}, color: i ? '#f08c2b' : '#2f7cf6'}]),
    }).on('click', p => this.result(id, +p.name));
    $('#cfGo').onclick = () => this.result(id, +$('#cf').value);
  },
};
