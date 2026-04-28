"""
status_server.py — tiny web dashboard for monitoring kitchen_sink remotely.

Run this on the EC2 instance alongside kitchen_sink. Open the resulting
URL in your phone browser — no SSH, no app needed.

USAGE (on the EC2 server, after starting kitchen_sink):
    nohup python status_server.py > status.log 2>&1 &
    disown
    echo "Open in phone browser: http://$(curl -s ifconfig.me):8000/"

You also need to OPEN PORT 8000 in your EC2 Security Group:
    EC2 Console -> Security Groups -> select the one tied to your instance
    -> Inbound Rules -> Edit -> Add Rule -> Custom TCP -> Port 8000
    -> Source: Anywhere (0.0.0.0/0)  -> Save.

The dashboard auto-refreshes every 5 seconds. Pull-to-refresh works on
mobile too.
"""

import http.server
import socketserver
import json
import time
import os
import subprocess
from pathlib import Path

PORT = 8000
HERE = Path(__file__).parent.resolve()
LOG_FILE     = HERE / "aurexis.log"
ARCHIVE_FILE = HERE / "aurexis_archive" / "_measurements.json"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass  # silence access logs

    def do_GET(self):
        try:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(self.render_dashboard().encode("utf-8"))
            elif self.path.startswith("/api/status"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                payload = json.dumps(self.gather_stats()).encode("utf-8")
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            try:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"err: {e}".encode())
            except Exception:
                pass

    def gather_stats(self):
        stats = {"timestamp": time.time(), "host": os.uname().nodename}

        # Recent log lines
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                stats["last_lines"] = [l.rstrip()[:200] for l in lines[-40:]]
                stats["total_log_lines"] = len(lines)
        except Exception:
            stats["last_lines"] = ["log not yet available"]
            stats["total_log_lines"] = 0

        # Count images + sources from the archive
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats["images"] = len(data)
            sources = {}
            for m in data.values():
                s = m.get("_source", "?")
                sources[s] = sources.get(s, 0) + 1
            stats["sources"] = dict(sorted(sources.items(),
                                            key=lambda kv: -kv[1])[:12])
            # Train/test split
            splits = {"train": 0, "test": 0}
            for m in data.values():
                splits[m.get("_split", "train")] = \
                    splits.get(m.get("_split", "train"), 0) + 1
            stats["splits"] = splits
        except Exception:
            stats["images"] = 0
            stats["sources"] = {}
            stats["splits"] = {}

        # Recent NEW counts
        new_count = sum(1 for l in stats.get("last_lines", []) if "NEW" in l)
        dup_count = sum(1 for l in stats.get("last_lines", []) if "dup" in l)
        stats["recent_new"] = new_count
        stats["recent_dup"] = dup_count

        # Is kitchen_sink running?
        try:
            r = subprocess.run(["pgrep", "-f", "kitchen_sink"],
                                capture_output=True, text=True, timeout=5)
            stats["running"] = bool(r.stdout.strip())
        except Exception:
            stats["running"] = None

        # Find the most recent phase header
        phase = "—"
        for l in reversed(stats.get("last_lines", [])):
            if "PHASE" in l and "#" in l:
                phase = l.replace("#", "").strip()
                break
        stats["current_phase"] = phase

        return stats

    def render_dashboard(self):
        return """<!DOCTYPE html>
<html><head>
<title>Aurexis Live</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta http-equiv="refresh" content="30">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'SF Mono', 'Consolas', monospace;
  background: #00000A; color: #BEC6D3;
  padding: 16px; font-size: 14px; line-height: 1.5;
  max-width: 720px; margin: 0 auto;
}
h1 {
  font-size: 22px; color: #006076; letter-spacing: 0.05em;
  padding-bottom: 12px; border-bottom: 1px solid #006076;
  margin-bottom: 16px; text-transform: uppercase;
}
.box {
  background: #0a0a14;
  border-left: 2px solid #006076;
  padding: 14px; margin-bottom: 14px;
}
.row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 4px 0;
}
.label {
  color: #006076; text-transform: uppercase;
  font-size: 11px; letter-spacing: 0.08em;
}
.value { font-size: 15px; color: #BEC6D3; }
.value.big { font-size: 32px; color: #00a0c0; font-weight: 500; }
.log {
  max-height: 60vh; overflow-y: auto;
  font-size: 12px; line-height: 1.4;
  background: #00000A;
  padding: 8px; margin-top: 8px;
  border: 1px solid #1a2230;
}
.log-line {
  padding: 1px 0; word-break: break-all;
  white-space: pre-wrap;
}
.log-line.new   { color: #50ff80; }
.log-line.dup   { color: #606870; }
.log-line.err   { color: #ff8080; }
.log-line.phase { color: #00a0c0; font-weight: 500; padding: 4px 0; }
.indicator {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 6px;
}
.run  { background: #50ff80; box-shadow: 0 0 8px #50ff80; }
.stop { background: #ff5050; }
.unk  { background: #808080; }
.refresh {
  color: #006076; font-size: 11px;
  text-align: center; margin-top: 16px;
  letter-spacing: 0.05em;
}
.phase-banner {
  color: #00a0c0; padding: 6px 0; font-size: 13px;
}
.source-row { font-size: 12px; }
</style>
</head><body>
<h1>aurexis // live</h1>

<div class="box">
  <div class="row">
    <span class="label">status</span>
    <span><span class="indicator unk" id="ind"></span><span id="run">connecting…</span></span>
  </div>
  <div class="row">
    <span class="label">archive size</span>
    <span class="value big" id="imgs">—</span>
  </div>
  <div class="row">
    <span class="label">train / test</span>
    <span class="value" id="splits">—</span>
  </div>
  <div class="row">
    <span class="label">last 40 lines: NEW / dup</span>
    <span class="value"><span id="new">—</span> / <span id="dup">—</span></span>
  </div>
  <div class="phase-banner" id="phase">—</div>
</div>

<div class="box">
  <div class="label">by source (top 12)</div>
  <div id="sources"></div>
</div>

<div class="box">
  <div class="label">recent log</div>
  <div class="log" id="log"></div>
</div>

<div class="refresh">live • polling every 5 sec • pull down to refresh</div>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/status', {cache: 'no-store'});
    const s = await r.json();
    document.getElementById('imgs').textContent =
      (s.images || 0).toLocaleString();
    document.getElementById('new').textContent = s.recent_new;
    document.getElementById('dup').textContent = s.recent_dup;
    const tr = (s.splits && s.splits.train) || 0;
    const te = (s.splits && s.splits.test) || 0;
    document.getElementById('splits').textContent =
      tr.toLocaleString() + ' / ' + te.toLocaleString();
    document.getElementById('phase').textContent = s.current_phase || '—';

    let cls = 'unk', txt = 'unknown';
    if (s.running === true)  { cls = 'run';  txt = 'RUNNING'; }
    if (s.running === false) { cls = 'stop'; txt = 'STOPPED'; }
    document.getElementById('ind').className = 'indicator ' + cls;
    document.getElementById('run').textContent = txt;

    const src = document.getElementById('sources');
    src.innerHTML = '';
    for (const [k, v] of Object.entries(s.sources || {})) {
      const div = document.createElement('div');
      div.className = 'row source-row';
      div.innerHTML =
        '<span>' + escapeHtml(k) + '</span>' +
        '<span class="value">' + v.toLocaleString() + '</span>';
      src.appendChild(div);
    }

    const log = document.getElementById('log');
    log.innerHTML = (s.last_lines || []).map(l => {
      let cls = '';
      if (/NEW/.test(l))   cls = 'new';
      else if (/dup/.test(l)) cls = 'dup';
      else if (/error|fail/i.test(l)) cls = 'err';
      else if (/PHASE|####/.test(l))  cls = 'phase';
      return '<div class="log-line ' + cls + '">' + escapeHtml(l) + '</div>';
    }).join('');
    log.scrollTop = log.scrollHeight;
  } catch (e) {
    document.getElementById('run').textContent = 'connection lost';
    document.getElementById('ind').className = 'indicator stop';
  }
}
function escapeHtml(s) {
  return String(s || '').replace(/[&<>]/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;'})[c]);
}
refresh();
setInterval(refresh, 5000);
</script>
</body></html>"""


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"\n  Aurexis status server running on port {PORT}")
        print(f"  Find your EC2 public IP, then open in browser:")
        print(f"     http://YOUR_EC2_IP:{PORT}/\n")
        httpd.serve_forever()
