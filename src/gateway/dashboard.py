"""Single-file ops dashboard. Polls /api/stats every 2s; no JS dependencies."""
from __future__ import annotations


def dashboard_html() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Inference Gateway</title>
<style>
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#8b949e;font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card .k{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:24px;font-weight:600;margin-top:4px}
.card .u{font-size:12px;color:#8b949e}
.ok{color:#3fb950}.bad{color:#f85149}
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px}
th{text-align:left;color:#8b949e;font-weight:500;padding:8px;border-bottom:1px solid #30363d}
td{padding:8px;border-bottom:1px solid #21262d}
h2{font-size:15px;margin:24px 0 10px}
#events{font-family:monospace;font-size:12px;color:#a5d6ff}
#events div{padding:3px 0;border-bottom:1px solid #21262d}
canvas{background:#161b22;border:1px solid #30363d;border-radius:8px;width:100%;height:180px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge.ok{background:rgba(63,185,80,.15);color:#3fb950}
.badge.bad{background:rgba(248,81,73,.15);color:#f85149}
</style></head><body>
<h1>Inference Gateway</h1>
<div class="sub">multi-tenant LLM serving &middot; live &middot; auto-refresh 2s</div>
<div class="grid" id="kpis"></div>
<h2>Latency p99 (ms) &mdash; TTFT vs end-to-end</h2>
<canvas id="chart" width="900" height="180"></canvas>
<h2>Tenants</h2><table><thead><tr><th>Tenant</th><th>Tier</th><th>Active</th><th>Tokens in/out</th><th>Cost</th><th>OK</th><th>Rejected</th></tr></thead><tbody id="tenants"></tbody></table>
<h2>Workers</h2><table><thead><tr><th>Worker</th><th>Profile</th><th>Batch</th><th>State</th></tr></thead><tbody id="workers"></tbody></table>
<h2>Events</h2><div id="events"></div>
<script>
async function tick(){
  const s=await (await fetch('/api/stats')).json();
  const L=s.latency, C=s.counts, G=s.gauges;
  const slo=(name)=>{const x=s.slos[name];return `<span class="badge ${x.met?'ok':'bad'}">${x.met?'MET':'MISS'}</span>`};
  document.getElementById('kpis').innerHTML=`
    <div class="card"><div class="k">Throughput</div><div class="v">${C.requests_completed||0}</div><div class="u">completed &middot; ${(C.requests_shed||0)} shed</div></div>
    <div class="card"><div class="k">TTFT p99 ${slo('ttft_p99_ms')}</div><div class="v">${L.ttft_ms.p99}</div><div class="u">ms &middot; target ${s.slos.ttft_p99_ms.target}</div></div>
    <div class="card"><div class="k">E2E p99 ${slo('e2e_p99_ms')}</div><div class="v">${L.e2e_ms.p99}</div><div class="u">ms &middot; target ${s.slos.e2e_p99_ms.target}</div></div>
    <div class="card"><div class="k">Queue depth</div><div class="v">${G.queue_depth}</div><div class="u">wait p99 ${L.queue_wait_ms.p99} ms</div></div>
    <div class="card"><div class="k">Workers</div><div class="v">${G.workers_large+G.workers_small}</div><div class="u">${G.workers_large} large &middot; ${G.workers_small} small</div></div>
    <div class="card"><div class="k">Availability ${slo('availability')}</div><div class="v">${(s.slos.availability.actual*100).toFixed(2)}%</div><div class="u">preemptions ${C.preemptions||0} &middot; fallbacks ${C.fallbacks||0}</div></div>`;
  document.getElementById('tenants').innerHTML=s.tenants.map(t=>
    `<tr><td>${t.id}</td><td>${t.tier}</td><td>${t.active}</td><td>${t.prompt_tokens}/${t.completion_tokens}</td><td>$${t.cost_usd}</td><td>${t.requests_ok}</td><td>${t.requests_rejected}</td></tr>`).join('');
  document.getElementById('workers').innerHTML=s.workers.map(w=>
    `<tr><td>${w.id}</td><td>${w.profile}</td><td>${w.batch}</td><td>${!w.alive?'<span class="bad">dead</span>':w.draining?'draining':'<span class="ok">serving</span>'}</td></tr>`).join('');
  document.getElementById('events').innerHTML=s.events.map(e=>`<div>${e}</div>`).join('');
  draw(s.history);
}
function draw(h){
  const c=document.getElementById('chart'),x=c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  if(h.length<2)return;
  const series=[['ttft_p99','#58a6ff'],['e2e_p99','#f85149']];
  const mx=Math.max(1,...h.map(p=>p.e2e_p99));
  x.strokeStyle='#30363d';x.beginPath();x.moveTo(0,c.height-1);x.lineTo(c.width,c.height-1);x.stroke();
  for(const [k,col] of series){
    x.strokeStyle=col;x.lineWidth=1.5;x.beginPath();
    h.forEach((p,i)=>{const px=i/(h.length-1)*c.width, py=c.height-4-(p[k]/mx)*(c.height-10);
      i?x.lineTo(px,py):x.moveTo(px,py)});
    x.stroke();
  }
  x.fillStyle='#8b949e';x.font='11px sans-serif';
  x.fillStyle='#58a6ff';x.fillText('— TTFT p99',8,14);
  x.fillStyle='#f85149';x.fillText('— E2E p99',90,14);
  x.fillStyle='#8b949e';x.fillText('max '+mx.toFixed(0)+' ms',c.width-90,14);
}
tick();setInterval(tick,2000);
</script></body></html>"""
