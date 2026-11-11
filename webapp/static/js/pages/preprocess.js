/* 图像预处理：双鱼拆分、旋转摆正、鱼体中心定位、居中裁剪 */
Pages.preprocess = {
  async render(root) {
    const wrap = document.createElement('div');
    root.innerHTML = `<div class="page-head"><div><h2>图像预处理</h2><p>对双鱼并排成像的原始帧进行左右拆分、按设定角度旋转摆正，基于阈值分割定位鱼体中心并居中裁剪，生成标准化鱼脑图像。</p></div></div>`;
    const st = await expToolbar(root, {sides: false, onChange: s => this.load(wrap, s.exp)});
    root.appendChild(wrap);
  },
  async load(wrap, exp) {
    const e = await API.get('/api/experiments/' + exp.id);
    const p = e.params.preprocess;
    wrap.innerHTML = `
      <div class="grid" style="grid-template-columns: 330px 1fr">
        <div class="card"><div class="card-head"><h3>预处理参数</h3>${e.status.preprocess ? '<span class="tag green">已确认</span>' : '<span class="tag orange">未确认</span>'}</div>
          <div class="card-body"><div class="form">
            <div class="field"><label>预览帧号</label><input type="number" id="pFrame" value="${p.ref_frame}"></div>
            <div class="field"><label>参考帧号（配准模板帧）</label><input type="number" id="ref_frame" value="${p.ref_frame}"><div class="hint">逐帧配准以该帧为固定图像</div></div>
            <div class="field"><label>左侧鱼旋转角度（°）</label><input type="number" id="rotate_left" value="${p.rotate_left}" step="0.5"></div>
            <div class="field"><label>右侧鱼旋转角度（°）</label><input type="number" id="rotate_right" value="${p.rotate_right}" step="0.5"></div>
            <div class="field"><label>裁剪高度 × 宽度（像素）</label><div class="row" style="flex-wrap:nowrap"><input type="number" id="crop_h" value="${p.crop_h}"><input type="number" id="crop_w" value="${p.crop_w}"></div></div>
            <div class="field"><label>鱼体分割阈值（8 bit）</label><input type="number" id="threshold" value="${p.threshold}"><div class="hint">16 位图像线性映射至 8 位后大于该阈值的像素视为鱼体</div></div>
            <div class="row"><button class="btn" id="btnPrev">预览效果</button><button class="btn primary" id="btnApply">确认参数</button></div>
          </div></div></div>
        <div id="pv"><div class="card"><div class="empty">加载预览中…</div></div></div>
      </div>`;
    const params = () => Object.fromEntries(['ref_frame', 'rotate_left', 'rotate_right', 'crop_h', 'crop_w', 'threshold'].map(k => [k, +$('#' + k).value]));
    const show = (r) => {
      $('#pv').innerHTML = `
        <div class="card"><div class="card-head"><h3>① 原始帧与左右拆分</h3><span class="muted small">帧 #${r.frame}</span></div>
          <div class="card-body">${imgPanel(r.split || r.raw, '拆分线', 300)}</div></div>
        <div class="grid g2">${r.fish.map(f => `
          <div class="card"><div class="card-head"><h3>② ${f.side === 'l' ? '左侧' : '右侧'}鱼 · 旋转摆正与中心定位</h3></div>
            <div class="card-body"><div class="grid" style="grid-template-columns:1.25fr 1fr">
              <div>${imgPanel(f.rotated, '旋转后', 330)}<div class="img-cap">绿色：鱼体轮廓　红十字：中心　橙框：裁剪区域</div></div>
              <div>${imgPanel(f.crop, '③ 裁剪结果', 330)}<div class="img-cap">${f.crop_shape.join(' × ')} 像素</div></div></div>
              <dl class="desc-list small" style="margin-top:10px"><dt>旋转后尺寸</dt><dd>${f.rotated_shape.join(' × ')}</dd><dt>鱼体中心</dt><dd>(${f.center.join(', ')})</dd>
                <dt>鱼体像素数</dt><dd>${f.fish_pixels.toLocaleString()}</dd><dt>裁剪框</dt><dd>[${f.crop_box.join(', ')}]</dd></dl></div></div>`).join('')}</div>`;
    };
    const preview = async () => show(await API.post(`/api/experiments/${exp.id}/preprocess/preview`, {frame: +$('#pFrame').value, params: params()}));
    $('#btnPrev').onclick = preview;
    $('#btnApply').onclick = async () => { show(await API.post(`/api/experiments/${exp.id}/preprocess/apply`, {params: params()})); toast('预处理参数已确认并保存', 'success'); this.load(wrap, exp); };
    preview();
  },
};
