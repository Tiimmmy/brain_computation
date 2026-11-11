/* 系统设置：默认算法参数、运行环境、源程序统计、账户密码 */
Pages.settings = {
  async render(root) {
    const s = await API.get('/api/system');
    const st = s.settings;
    const f = (grp, key, label, step = 'any') => `<div class="field"><label>${label}</label><input type="number" step="${step}" data-g="${grp}" data-k="${key}" value="${st[grp][key]}"></div>`;
    root.innerHTML = `
      <div class="page-head"><div><h2>系统设置</h2><p>配置各分析模块的默认参数，查看系统运行环境与软件信息，修改登录密码。</p></div><button class="btn primary" id="sSave">保存设置</button></div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><h3>图像预处理默认参数</h3></div><div class="card-body"><div class="form g3">
          ${f('preprocess', 'rotate_left', '左侧鱼旋转角(°)')}${f('preprocess', 'rotate_right', '右侧鱼旋转角(°)')}${f('preprocess', 'ref_frame', '参考帧号', 1)}
          ${f('preprocess', 'crop_h', '裁剪高度', 1)}${f('preprocess', 'crop_w', '裁剪宽度', 1)}${f('preprocess', 'threshold', '鱼体阈值', 1)}</div></div></div>
        <div class="card"><div class="card-head"><h3>配准默认参数</h3></div><div class="card-body"><div class="form g3">
          <div class="field"><label>变换模型</label><select data-g="registration" data-k="transform">${['ElasticSyN', 'SyN', 'Affine', 'Rigid'].map(x => `<option ${st.registration.transform === x ? 'selected' : ''}>${x}</option>`).join('')}</select></div>
          ${f('registration', 'resample_h', '降采样高', 1)}${f('registration', 'resample_w', '降采样宽', 1)}${f('registration', 'workers', '并行进程数', 1)}</div></div></div>
        <div class="card"><div class="card-head"><h3>模型训练默认参数</h3></div><div class="card-body"><div class="form g3">
          ${f('training', 'k_folds', '交叉验证折数', 1)}${f('training', 'epochs', '训练轮数', 1)}${f('training', 'lr', '学习率')}${f('training', 'momentum', '动量')}${f('training', 'weight_decay', '权重衰减')}
          <div class="field"><label>优化器</label><select data-g="training" data-k="optimizer">${['SGD', 'Adam'].map(x => `<option ${st.training.optimizer === x ? 'selected' : ''}>${x}</option>`).join('')}</select></div>
          ${f('kernel', 'band_width', '高斯核带宽系数 H')}</div></div></div>
        <div class="card"><div class="card-head"><h3>信号处理参数（算法内置）</h3></div><div class="card-body"><dl class="desc-list">
          <dt>低通滤波</dt><dd>Butterworth ${st.signal.lowpass_order} 阶，归一化截止频率 ${st.signal.lowpass_cutoff}，零相位 filtfilt</dd>
          <dt>基线估计</dt><dd>IModPoly 迭代修正多项式拟合，${st.signal.poly_degree} 次，最多 100 次迭代，收敛阈值 0.001</dd>
          <dt>边界裁剪</dt><dd>滤波与基线校正后两端各裁剪 ${st.signal.trim} 帧</dd><dt>异常像素</dt><dd>时程极差 &gt; ${st.signal.outlier_range}</dd>
          <dt>甩动帧判据</dt><dd>像素值低于 Q20 − 1.5·IQR 的像素数 &gt; 1000</dd><dt>死鱼判据</dt><dd>时程极差 &gt; 1000 的像素数 &gt; 10000</dd></dl></div></div>
      </div>
      <div class="grid g3">
        <div class="card"><div class="card-head"><h3>软件信息</h3></div><div class="card-body"><dl class="desc-list">
          <dt>软件名称</dt><dd>${s.software}</dd><dt>版本号</dt><dd>${s.version}</dd><dt>源程序总量</dt><dd>${s.source_total.toLocaleString()} 行</dd>
          ${Object.entries(s.source_by_lang).map(([k, v]) => `<dt>${k}</dt><dd>${v.toLocaleString()} 行</dd>`).join('')}
          <dt>源文件数</dt><dd>${s.source.length} 个</dd></dl></div></div>
        <div class="card"><div class="card-head"><h3>运行环境</h3></div><div class="card-body"><dl class="desc-list">
          <dt>操作系统</dt><dd class="small">${esc(s.platform)}</dd><dt>Python</dt><dd>${s.python}</dd><dt>CPU 核数</dt><dd>${s.cpu}</dd>
          <dt>PyTorch</dt><dd>${s.device.torch || '未安装'}</dd><dt>GPU</dt><dd>${s.device.gpus.join('<br>') || '不可用'}</dd>
          ${Object.entries(s.packages).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}
          <dt>工作空间</dt><dd class="mono small" style="word-break:break-all">${esc(s.workspace)}（${s.workspace_gb} GB）</dd></dl></div></div>
        <div class="card"><div class="card-head"><h3>修改密码</h3></div><div class="card-body"><div class="form">
          <div class="field"><label>当前用户</label><input type="text" value="${App.me.name}（${App.me.username}）" disabled></div>
          <div class="field"><label>原密码</label><input type="password" id="pO"></div><div class="field"><label>新密码</label><input type="password" id="pN"></div>
          <div class="field"><label>确认新密码</label><input type="password" id="pC"></div><button class="btn primary" id="pGo">修改密码</button></div></div></div>
      </div>
      <div class="card"><div class="card-head"><h3>源程序文件清单</h3></div><div class="card-body" style="padding:0"><div class="tbl-wrap" style="max-height:320px"><table class="tbl compact"><thead><tr><th>文件</th><th>语言</th><th>行数</th></tr></thead>
        <tbody>${s.source.map(x => `<tr><td class="mono">${x.file}</td><td>${x.lang}</td><td>${x.lines}</td></tr>`).join('')}</tbody></table></div></div></div>`;
    $('#sSave').onclick = async () => {
      const body = {};
      $$('[data-g]').forEach(el => { (body[el.dataset.g] = body[el.dataset.g] || {})[el.dataset.k] = el.tagName === 'SELECT' ? el.value : +el.value; });
      await API.post('/api/settings', body); toast('设置已保存', 'success');
    };
    $('#pGo').onclick = async () => {
      if ($('#pN').value !== $('#pC').value) return toast('两次输入的新密码不一致', 'error');
      await API.post('/api/password', {old: $('#pO').value, new: $('#pN').value}); toast('密码已修改', 'success');
    };
  },
};
