/* 任务中心：后台计算任务列表、进度、日志查看 */
Pages.tasks = {
  async render(root, params) {
    const list = await API.get('/api/tasks');
    const cnt = s => list.filter(t => t.status === s).length;
    root.innerHTML = `
      <div class="page-head"><div><h2>任务中心</h2><p>所有耗时计算均以后台任务方式运行，可在此查看任务状态、进度与运行日志。</p></div><button class="btn" id="tR">刷新</button></div>
      <div class="grid g4"><div class="kpi"><div class="l">任务总数</div><div class="v">${list.length}</div></div><div class="kpi"><div class="l">运行中</div><div class="v" style="color:#2f7cf6">${cnt('running')}</div></div>
        <div class="kpi"><div class="l">已完成</div><div class="v" style="color:#2fa36b">${cnt('success')}</div></div><div class="kpi"><div class="l">失败</div><div class="v" style="color:#e5484d">${cnt('failed')}</div></div></div>
      <div class="grid" style="grid-template-columns: 1.3fr 1fr; margin-top:16px">
        <div class="card"><div class="card-head"><h3>任务列表</h3><select id="tF" class="inline-select" style="min-width:130px"><option value="">全部类型</option>${[...new Set(list.map(t => t.type_name))].map(n => `<option>${n}</option>`).join('')}</select></div>
          <div class="card-body" style="padding:0"><div class="tbl-wrap" style="max-height:600px"><table class="tbl compact"><thead><tr><th>任务编号</th><th>类型</th><th>名称</th><th>状态</th><th>进度</th><th>开始时间</th><th>用时</th><th>操作</th></tr></thead><tbody id="tTb"></tbody></table></div></div></div>
        <div class="card"><div class="card-head"><h3>任务详情</h3></div><div class="card-body" id="tDet"><div class="empty">点击左侧任务查看详情</div></div></div>
      </div>`;
    const draw = () => {
      const f = $('#tF').value;
      $('#tTb').innerHTML = list.filter(t => !f || t.type_name === f).map(t => `<tr><td class="mono small">${t.id}</td><td>${t.type_name}</td><td>${esc(t.title)}</td><td>${statusTag(t.status)}</td>
        <td style="width:110px"><div class="progress"><div style="width:${t.progress * 100}%"></div></div></td><td class="small">${t.created}</td><td>${sec(t.elapsed)}</td>
        <td><button class="btn sm" data-v="${t.id}">查看</button> ${t.status !== 'running' ? `<button class="btn sm danger" data-d="${t.id}">删除</button>` : ''}</td></tr>`).join('') || '<tr><td colspan="8"><div class="empty">暂无任务</div></td></tr>';
      $$('[data-v]').forEach(b => b.onclick = () => taskWatcher($('#tDet'), b.dataset.v));
      $$('[data-d]').forEach(b => b.onclick = async () => { await API.del('/api/tasks/' + b.dataset.d); App.route(); });
    };
    $('#tF').onchange = draw; draw();
    $('#tR').onclick = () => App.route();
    const sel = params.id || (list[0] && list[0].id);
    if (sel) taskWatcher($('#tDet'), sel);
  },
};
