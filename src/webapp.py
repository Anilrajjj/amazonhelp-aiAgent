"""Chat UI for testing the agent.

    python src/webapp.py          then open http://localhost:8000

A reviewer talks to the agent like a customer would. Each message gets the
brand's drafted reply, and a note saying whether that reply would have gone out
automatically or been held for a human -- which is the decision the whole system
exists to make.

Built on http.server from the standard library rather than Flask or FastAPI, so
the project still installs with two packages. The precedent index is built once
at startup.

Note on scope: every message is handled independently, as a fresh incoming
tweet. The agent is a triage system for opening messages and does not carry
conversation state (see README, "what I chose not to build"). The transcript is
a display convenience, not a multi-turn dialogue.
"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import console
import llm
import agent as agent_mod
from dataset import load_split
from retrieval import PrecedentIndex

INDEX = None
INDEX_ERROR = None
READY = threading.Event()

# The agent is required to write the placeholder "<link>" and never to invent a
# URL -- inventing one is the fastest way to send a customer somewhere harmful.
# Resolving that placeholder to a real destination is a presentation decision,
# made here in the UI layer, not by the model. One canonical destination, so
# there is no chance of pointing someone at a plausible-looking wrong page.
HELP_URL = "https://www.amazon.com/gp/help/customer/display.html"
HELP_LABEL = "amazon.com/help"

EXAMPLES = [
    "my parcel was supposed to arrive yesterday and it still hasn't turned up",
    "this is the third time my order has been late and no one has replied to me",
    "there are unauthorised transactions on my saved card, someone has used it",
    "just got my order in 6 hours, honestly incredible service",
    "i ordered an SSD and you sent me a stick of RAM instead",
    "this is absolute bullshit, i'm cancelling my prime membership today",
]

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AmazonHelp Support</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#fff; --fg:#1b1b19; --mut:#70706a; --line:#e4e4df;
  --me:#2f6f8f; --mefg:#fff; --bot:#fff;
  --ok:#1f7a4d; --okbg:#e9f5ee; --esc:#a8410f; --escbg:#fdf0e6;
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#141417; --panel:#1d1d21; --fg:#ececeb; --mut:#9b9b95; --line:#2f2f35;
  --me:#3d7f9e; --mefg:#fff; --bot:#25252b;
  --ok:#61d09a; --okbg:#15301f80; --esc:#f2a67f; --escbg:#3b241680;
}}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  display:flex;flex-direction:column;align-items:center}
.app{width:100%;max-width:720px;height:100dvh;display:flex;flex-direction:column;padding:0 16px}
header{padding:18px 2px 14px;border-bottom:1px solid var(--line)}
header h1{margin:0;font-size:17px;display:flex;align-items:center;gap:9px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);flex:none}
header p{margin:3px 0 0 17px;color:var(--mut);font-size:13px}

#log{flex:1;overflow-y:auto;padding:20px 2px;display:flex;flex-direction:column;gap:14px}
.turn{display:flex;flex-direction:column;max-width:86%}
.turn.me{align-self:flex-end;align-items:flex-end}
.turn.bot{align-self:flex-start}
.bub{padding:11px 15px;border-radius:16px;white-space:pre-wrap;word-wrap:break-word}
.me .bub{background:var(--me);color:var(--mefg);border-bottom-right-radius:5px}
.bot .bub{background:var(--bot);border:1px solid var(--line);border-bottom-left-radius:5px}
.bub a{color:inherit;text-decoration:underline;text-underline-offset:2px}
.bot .bub a{color:var(--me);font-weight:600}

.note{font-size:12.5px;margin-top:6px;display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.pill{padding:2.5px 9px;border-radius:999px;font-weight:650;font-size:11.5px;
  letter-spacing:.03em;text-transform:uppercase}
.pill.ok{background:var(--okbg);color:var(--ok)}
.pill.esc{background:var(--escbg);color:var(--esc)}
.note .why{color:var(--mut);cursor:pointer;text-decoration:underline;
  text-underline-offset:2px;background:none;border:0;font:inherit;font-size:12.5px;padding:0}
.det{margin-top:8px;font-size:12.5px;color:var(--mut);background:var(--panel);
  border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.det .r{margin-bottom:9px}
.det .r:last-child{margin-bottom:0}
.det b{color:var(--fg);font-weight:600}
.det code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:11.5px}
.hide{display:none}

.chips{display:flex;flex-wrap:wrap;gap:7px;padding:0 2px 14px}
.chips button{font:inherit;font-size:13px;cursor:pointer;color:var(--mut);
  background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:6px 13px;
  text-align:left}
.chips button:hover{border-color:var(--me);color:var(--fg)}

form{display:flex;gap:9px;padding:12px 2px 18px;border-top:1px solid var(--line)}
textarea{flex:1;min-height:44px;max-height:150px;padding:11px 14px;font:inherit;
  color:var(--fg);background:var(--panel);border:1px solid var(--line);
  border-radius:22px;resize:none}
textarea:focus{outline:2px solid var(--me);outline-offset:-1px}
button.send{flex:none;width:44px;height:44px;border-radius:50%;border:0;cursor:pointer;
  background:var(--me);color:#fff;font-size:17px;line-height:1}
button.send[disabled]{opacity:.45;cursor:default}
.typing span{display:inline-block;width:6px;height:6px;margin-right:3px;border-radius:50%;
  background:var(--mut);animation:b 1.2s infinite}
.typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}
@keyframes b{0%,60%,100%{opacity:.3}30%{opacity:1}}
.err{color:var(--esc)}
</style></head><body>
<div class="app">
  <header>
    <h1><span class="dot"></span> AmazonHelp</h1>
    <p>Support assistant — replies are drafted from how this brand actually answered similar messages.</p>
  </header>

  <div id="log"></div>
  <div class="chips" id="chips"></div>

  <form id="f">
    <textarea id="msg" rows="1" placeholder="Message AmazonHelp…" autofocus></textarea>
    <button class="send" id="send" type="submit" aria-label="Send">↑</button>
  </form>
</div>

<script>
const $=s=>document.querySelector(s);
const log=$('#log'); let busy=false, n=0;

fetch('/api/meta').then(r=>r.json()).then(m=>{
  $('#chips').innerHTML=m.examples.map(t=>
    `<button type="button" data-t="${t.replace(/"/g,'&quot;')}">${t.length>52?t.slice(0,52)+'…':t}</button>`).join('');
  $('#chips').onclick=e=>{const t=e.target.dataset.t; if(t){$('#msg').value=t; send();}};
  if(!m.llm_ready) bot('No LLM key configured. Create <code>.env</code> in the repo root — see <code>.env.example</code>.',null,true);
});

function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function linkify(s){  // the agent writes "<link>"; the UI resolves it to a real URL
  return esc(s).replace(/&lt;link&gt;/g,
    `<a href="${LINK_URL}" target="_blank" rel="noopener">${LINK_LABEL}</a>`);
}
let LINK_URL='#', LINK_LABEL='link';
fetch('/api/meta').then(r=>r.json()).then(m=>{LINK_URL=m.help_url; LINK_LABEL=m.help_label;});

function scroll(){log.scrollTop=log.scrollHeight;}
function me(t){
  const d=document.createElement('div'); d.className='turn me';
  d.innerHTML=`<div class="bub">${esc(t)}</div>`; log.appendChild(d); scroll();
}
function bot(html,meta,isErr){
  const d=document.createElement('div'); d.className='turn bot';
  d.innerHTML=`<div class="bub${isErr?' err':''}">${html}</div>`+(meta||'');
  log.appendChild(d); scroll(); return d;
}
function typing(){
  const d=document.createElement('div'); d.className='turn bot';
  d.innerHTML='<div class="bub typing"><span></span><span></span><span></span></div>';
  log.appendChild(d); scroll(); return d;
}

async function send(){
  const t=$('#msg').value.trim(); if(!t||busy) return;
  busy=true; $('#send').disabled=true; $('#msg').value='';
  $('#chips').classList.add('hide');
  me(t);
  const tip=typing();
  try{
    const r=await fetch('/api/handle',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:t})});
    const d=await r.json();
    tip.remove();
    if(d.error){ bot(esc(d.error),null,true); return; }
    const auto=d.route==='auto', id='d'+(n++);
    const meta=`<div class="note">
        <span class="pill ${auto?'ok':'esc'}">${auto?'sent automatically':'held for a human'}</span>
        ${d.escalation_reason?`<span style="color:var(--mut)">${esc(d.escalation_reason.replace(/_/g,' '))}</span>`:''}
        <button class="why" onclick="document.getElementById('${id}').classList.toggle('hide')">details</button>
      </div>
      <div class="det hide" id="${id}">
        <div class="r"><b>Classified as</b> <code>${esc(d.intent)}</code> (confidence ${d.confidence})</div>
        ${d.rules_fired&&d.rules_fired.length?`<div class="r"><b>Rules triggered</b> ${d.rules_fired.map(x=>`<code>${esc(x)}</code>`).join(' ')}</div>`:''}
        <div class="r"><b>Grounded on past replies</b>${d.precedents.map(p=>
          `<div style="margin-top:6px">↳ <i>“${esc(p.customer_message.slice(0,90))}”</i><br>
             &nbsp;&nbsp;&nbsp;“${esc((p.brand_reply||'').slice(0,90))}” <span style="opacity:.7">(similarity ${p.score.toFixed(2)})</span></div>`).join('')}
        </div>
      </div>`;
    bot(linkify(d.reply)||'<i>no reply drafted</i>',meta);
  }catch(e){ tip.remove(); bot(esc(e.message),null,true); }
  finally{ busy=false; $('#send').disabled=false; $('#msg').focus(); }
}

$('#f').onsubmit=e=>{e.preventDefault();send();};
$('#msg').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
$('#msg').addEventListener('input',e=>{
  e.target.style.height='auto'; e.target.style.height=Math.min(e.target.scrollHeight,150)+'px';
});
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass   # keep the console clean; errors still reach the browser

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html")
        if path == "/api/meta":
            return self._send(200, json.dumps({
                "examples": EXAMPLES,
                "help_url": HELP_URL,
                "help_label": HELP_LABEL,
                "llm_ready": llm.is_available(),
            }))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/handle":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length") or 0)
            msg = (json.loads(self.rfile.read(n) or b"{}").get("message") or "").strip()
            if not msg:
                return self._send(400, json.dumps({"error": "empty message"}))
            if not READY.is_set():
                return self._send(503, json.dumps(
                    {"error": "still loading past conversations, try again in a moment"}))
            if INDEX_ERROR:
                return self._send(500, json.dumps({"error": INDEX_ERROR}))
            out = agent_mod.handle(msg, INDEX)
            out["precedents"] = INDEX.search(msg, k=3)
            self._send(200, json.dumps(out, ensure_ascii=False))
        except Exception as e:                    # surface it in the chat, never fail silently
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def build_index():
    global INDEX, INDEX_ERROR
    try:
        corpus, _ = load_split()
        print(f"indexing {len(corpus):,} past conversations ...", flush=True)
        INDEX = PrecedentIndex(corpus)
        print("index ready.", flush=True)
    except Exception as e:
        INDEX_ERROR = f"{type(e).__name__}: {e}"
        print(f"index FAILED: {INDEX_ERROR}", flush=True)
    finally:
        READY.set()


def main():
    console.init()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not llm.is_available():
        print("WARNING: no LLM key configured -- the chat will load but cannot "
              "reply. Create .env (see .env.example).\n")
    threading.Thread(target=build_index, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n  open  http://localhost:{port}\n  (first reply waits for the index, ~1 min)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
