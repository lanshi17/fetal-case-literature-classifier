#!/usr/bin/env python3
"""Gold-standard labeling UI: one paper at a time, multiple-choice answers.

Launch:
  .venv/bin/python scripts/gold_review.py
  .venv/bin/python scripts/gold_review.py --port 8765 --reviewer alice
  .venv/bin/python scripts/gold_review.py --queue fn_fp
  .venv/bin/python scripts/gold_review.py --cli          # terminal mode

Open http://127.0.0.1:8765/  Keys: 1=纳入  0=排除  s=跳过  u=不确定

Writes/appends: data/gold_standard.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"
DEFAULT_PRED = ROOT / "artifacts" / "test_predictions.csv"
DEFAULT_FN = ROOT / "artifacts" / "optimization" / "fn_audit.csv"
DEFAULT_FP = ROOT / "artifacts" / "optimization" / "fp_audit.csv"

# Multiple-choice label options (value, button label, description)
LABEL_CHOICES: list[tuple[str, str, str]] = [
    ("1", "纳入", "原始研究报告具体患者/家系的遗传变异及临床表型（病例报告/病例系列/符合标准的家系）"),
    ("0", "排除", "不满足纳入：综述、社论、纯算法/数据库、仅预测分数、不合格队列等"),
    ("uncertain", "不确定", "证据不足或研究类型模糊，暂不写入金标准正/负，记为待议"),
    ("skip", "跳过", "本次不审，稍后再来（不写入判定）"),
]

# Multiple-choice reason presets (code, text shown)
REASON_CHOICES: list[tuple[str, str]] = [
    ("case_report_variant_phenotype", "病例/病例系列：有具体变异 + 患者表型"),
    ("family_segregation", "家系研究：个体级变异与表型清楚"),
    ("cohort_case_level", "队列/系列中仍有可核对的病例级证据，可纳入"),
    ("review_or_editorial", "综述 / 社论 / 指南 / 通讯"),
    ("method_algorithm_db", "纯方法、算法、数据库或工具文"),
    ("prediction_only", "仅有预测分数 / 频率 / 计算致病性，无合格病例证据"),
    ("gwas_large_sequencing", "GWAS / 大规模测序 / 筛查准确性研究，无病例级纳入依据"),
    ("wrong_scope", "主题不符（非本筛查范围）"),
    ("insufficient_info", "摘要/全文信息不足，无法判定"),
    ("label_noise_disagree", "不同意现有自动/LLM 标签"),
    ("other", "其他（请在备注填写）"),
]

GOLD_FIELDS = [
    "pmid",
    "title",
    "year",
    "gold_label",
    "gold_reason_code",
    "gold_reason_text",
    "gold_note",
    "reviewer",
    "reviewed_at",
    "queue",
    "prior_label",
    "prior_label_source",
    "prior_design",
    "prior_reason",
    "model_score",
    "model_decision",
]


class SubmitIn(BaseModel):
    """Request body for POST /api/submit.

    Defined at module scope: ``from __future__ import annotations`` stringifies
    annotations, and FastAPI resolves them against module globals — a class
    local to ``make_app`` would not be found and the body would be parsed as a
    query parameter (HTTP 422).
    """

    gold_label: str
    reason_code: str = ""
    note: str = ""


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _append_gold(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GOLD_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in GOLD_FIELDS})


def load_papers(data_path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(data_path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        pmid = str(r.get("pmid") or r.get("doc_id") or "").strip()
        if pmid:
            out[pmid] = r
    return out


def load_predictions(pred_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in _read_csv(pred_path):
        pmid = str(r.get("doc_id") or r.get("pmid") or "").strip()
        if pmid:
            out[pmid] = r
    return out


def load_done_pmids(gold_path: Path) -> set[str]:
    """PMIDs that already have a definitive gold label (0/1), not uncertain."""
    done: set[str] = set()
    for r in _read_csv(gold_path):
        lab = str(r.get("gold_label") or "").strip()
        pmid = str(r.get("pmid") or "").strip()
        if pmid and lab in {"0", "1"}:
            done.add(pmid)
    return done


def build_queue(
    papers: dict[str, dict[str, str]],
    preds: dict[str, dict[str, str]],
    done: set[str],
    mode: str,
    limit: int | None,
    *,
    fn_path: Path | None = None,
    fp_path: Path | None = None,
) -> list[str]:
    """Return ordered PMIDs still needing gold review."""
    all_ids = list(papers.keys())

    def pending(ids: list[str]) -> list[str]:
        return [i for i in ids if i not in done]

    if mode == "fn_fp":
        fn_csv = fn_path or DEFAULT_FN
        fp_csv = fp_path or DEFAULT_FP
        fn = [str(r.get("doc_id") or r.get("pmid") or "").strip() for r in _read_csv(fn_csv)]
        fp = [str(r.get("doc_id") or r.get("pmid") or "").strip() for r in _read_csv(fp_csv)]
        ordered = pending([i for i in fn + fp if i in papers])
    elif mode == "test_set":
        ordered = pending([i for i in preds if i in papers])
    elif mode == "disagree":
        ids = []
        for pmid, pr in preds.items():
            if pmid not in papers:
                continue
            prior = str(papers[pmid].get("label") or "").strip()
            dec = str(pr.get("decision") or "").strip()
            # relevant≈1, irrelevant≈0; review counts as soft disagree with prior
            model_pos = dec == "relevant"
            prior_pos = prior == "1"
            if model_pos != prior_pos or dec == "review":
                ids.append(pmid)
        ordered = pending(ids)
    elif mode == "weak_model":
        ids = [
            p
            for p, r in papers.items()
            if _as_bool(r.get("weak")) or str(r.get("label_source") or "").strip().lower() == "model"
        ]
        ordered = pending(ids)
    elif mode == "no_gold":
        ordered = pending(all_ids)
    else:  # all
        ordered = pending(all_ids)

    # stable order: prefer existing prediction order then pmid
    if mode in {"fn_fp", "test_set", "disagree"}:
        pass
    else:
        ordered = sorted(ordered, key=lambda x: (papers[x].get("year") or "", x))

    if limit is not None and limit > 0:
        ordered = ordered[:limit]
    return ordered


def reason_text(code: str) -> str:
    for c, t in REASON_CHOICES:
        if c == code:
            return t
    return code


def clip(text: str, n: int = 4000) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n] + f"\n\n… [截断，全文共 {len(text)} 字]"


# ---------------------------------------------------------------------------
# Shared review state
# ---------------------------------------------------------------------------


class ReviewState:
    def __init__(
        self,
        papers: dict[str, dict[str, str]],
        preds: dict[str, dict[str, str]],
        queue: list[str],
        gold_path: Path,
        reviewer: str,
        queue_name: str,
    ) -> None:
        self.papers = papers
        self.preds = preds
        self.queue = queue
        self.gold_path = gold_path
        self.reviewer = reviewer
        self.queue_name = queue_name
        self.index = 0

    @property
    def remaining(self) -> int:
        return max(0, len(self.queue) - self.index)

    @property
    def total(self) -> int:
        return len(self.queue)

    def current(self) -> dict[str, Any] | None:
        if self.index >= len(self.queue):
            return None
        pmid = self.queue[self.index]
        paper = self.papers[pmid]
        pred = self.preds.get(pmid, {})
        return {
            "position": self.index + 1,
            "total": self.total,
            "remaining": self.remaining,
            "pmid": pmid,
            "title": paper.get("title") or "",
            "year": paper.get("year") or "",
            "pmcid": paper.get("pmcid") or "",
            "design": paper.get("design") or "",
            "tests": paper.get("tests") or "",
            "methods": paper.get("methods") or "",
            "prior_label": paper.get("label") or "",
            "prior_label_source": paper.get("label_source") or "",
            "prior_confidence": paper.get("confidence") or "",
            "prior_weak": paper.get("weak") or "",
            "prior_reason": paper.get("label_reason") or paper.get("reason") or "",
            "abstract": paper.get("abstract") or "",
            "text_head": clip(paper.get("text") or "", 6000),
            "text_chars": paper.get("text_chars") or str(len(paper.get("text") or "")),
            "text_source": paper.get("text_source") or "",
            "abstract_available": paper.get("abstract_available") or "",
            "full_text_available": paper.get("full_text_available") or "",
            "model_score": pred.get("score") or "",
            "model_decision": pred.get("decision") or "",
            "model_threshold": pred.get("threshold") or "",
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    def submit(
        self,
        gold_label: str,
        reason_code: str,
        note: str = "",
    ) -> dict[str, Any]:
        cur = self.current()
        if cur is None:
            return {"ok": False, "error": "queue empty"}

        if gold_label == "skip":
            self.index += 1
            return {"ok": True, "saved": False, "action": "skip", "next": self.current()}

        if gold_label not in {"0", "1", "uncertain"}:
            return {"ok": False, "error": f"invalid label {gold_label}"}

        if not reason_code:
            return {"ok": False, "error": "请选择题干原因（reason）"}

        row = {
            "pmid": cur["pmid"],
            "title": cur["title"],
            "year": cur["year"],
            "gold_label": gold_label,
            "gold_reason_code": reason_code,
            "gold_reason_text": reason_text(reason_code),
            "gold_note": note.strip(),
            "reviewer": self.reviewer,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "queue": self.queue_name,
            "prior_label": cur["prior_label"],
            "prior_label_source": cur["prior_label_source"],
            "prior_design": cur["design"],
            "prior_reason": cur["prior_reason"],
            "model_score": cur["model_score"],
            "model_decision": cur["model_decision"],
        }
        _append_gold(self.gold_path, row)
        # remove from queue progression
        self.index += 1
        return {"ok": True, "saved": True, "row": row, "next": self.current()}


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>金标准审核</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1f2933; --muted:#62707c; --line:#d9e2ec;
          --ok:#0b6e4f; --bad:#9b2226; --warn:#9a6b00; --accent:#245bdb; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "IBM Plex Sans", "Noto Sans SC", system-ui, sans-serif;
         background: var(--bg); color: var(--ink); line-height:1.45; }
  header { background:#102a43; color:#fff; padding:14px 20px; display:flex; gap:16px;
           align-items:center; justify-content:space-between; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; font-weight:600; }
  header .meta { font-size:13px; opacity:.85; }
  main { max-width: 980px; margin: 0 auto; padding: 18px 16px 80px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px 18px; margin-bottom:14px; }
  .progress { height:8px; background:#e4eaf1; border-radius:99px; overflow:hidden; }
  .progress > span { display:block; height:100%; background:var(--accent); }
  h2 { font-size:20px; margin:0 0 8px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 0; }
  .chip { font-size:12px; padding:4px 8px; border-radius:999px; background:#eef2f6; color:var(--muted); }
  .chip.pos { background:#e3f6ee; color:var(--ok); }
  .chip.neg { background:#fde8e8; color:var(--bad); }
  .chip.warn { background:#fff3d6; color:var(--warn); }
  .section-title { font-size:13px; font-weight:600; color:var(--muted); text-transform:uppercase;
                   letter-spacing:.04em; margin:0 0 6px; }
  .body { white-space: pre-wrap; font-size:14px; max-height: 280px; overflow:auto;
          background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .body.tall { max-height: 420px; }
  a { color: var(--accent); }
  .choices { display:grid; gap:10px; }
  .choice { display:flex; gap:12px; align-items:flex-start; border:1px solid var(--line);
            border-radius:10px; padding:12px 14px; cursor:pointer; background:#fff; }
  .choice:hover { border-color:#9fb3c8; }
  .choice.selected { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  .choice input { margin-top:3px; }
  .choice .key { font-family: ui-monospace, monospace; font-size:12px; color:var(--muted);
                 min-width:1.4em; }
  .choice strong { display:block; }
  .choice span.desc { font-size:13px; color:var(--muted); }
  textarea, input[type=text] { width:100%; border:1px solid var(--line); border-radius:8px;
                               padding:10px 12px; font: inherit; }
  .actions { position:sticky; bottom:0; background:rgba(246,247,249,.92); backdrop-filter:blur(6px);
             border-top:1px solid var(--line); padding:12px 16px; display:flex; gap:10px;
             justify-content:flex-end; flex-wrap:wrap; }
  button { border:0; border-radius:8px; padding:10px 16px; font:inherit; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; }
  button.ghost { background:#fff; border:1px solid var(--line); }
  .done { text-align:center; padding:48px 16px; }
  .err { color:var(--bad); font-size:13px; min-height:1.2em; }
  .hint { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <div>
    <h1>金标准逐篇审核（选择题）</h1>
    <div class="meta">审稿人：{{ reviewer }} · 队列：{{ queue }} · 输出：{{ gold_path }}</div>
  </div>
  <div class="meta" id="progress-text">—</div>
</header>
<main id="app">
  <div class="card hint">加载中…</div>
</main>
<div class="actions" id="actions" style="display:none">
  <div class="err" id="err" style="flex:1;text-align:left"></div>
  <button class="ghost" type="button" onclick="submit('skip')">跳过 (S)</button>
  <button class="primary" type="button" onclick="submitCurrent()">提交判定</button>
</div>
<script>
const LABEL_CHOICES = {{ label_choices_json }};
const REASON_CHOICES = {{ reason_choices_json }};
let state = null;

function esc(s){ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }

function chip(label, cls=''){ return `<span class="chip ${cls}">${esc(label)}</span>`; }

function renderDone(){
  document.getElementById('actions').style.display='none';
  document.getElementById('app').innerHTML = `
    <div class="card done">
      <h2>本队列已审完</h2>
      <p>金标准已写入服务器端 CSV。可更换 --queue 继续，或导出核对。</p>
    </div>`;
  document.getElementById('progress-text').textContent = '完成';
}

function render(item){
  if(!item){ renderDone(); return; }
  const pct = item.total ? Math.round((item.position-1)/item.total*100) : 0;
  document.getElementById('progress-text').textContent =
    `${item.position} / ${item.total}（剩余 ${item.remaining}）`;
  const priorCls = item.prior_label==='1'?'pos':(item.prior_label==='0'?'neg':'warn');
  const modelCls = item.model_decision==='relevant'?'pos':(item.model_decision==='irrelevant'?'neg':'warn');
  const labelHtml = LABEL_CHOICES.map(([val,name,desc])=>{
    const key = val==='1'?'1':val==='0'?'0':(val==='uncertain'?'U':'S');
    return `<label class="choice" data-group="label">
      <input type="radio" name="gold_label" value="${esc(val)}">
      <div class="key">${key}</div>
      <div><strong>${esc(name)}</strong><span class="desc">${esc(desc)}</span></div>
    </label>`;
  }).join('');
  const reasonHtml = REASON_CHOICES.map(([code,text],i)=>`
    <label class="choice" data-group="reason">
      <input type="radio" name="reason_code" value="${esc(code)}">
      <div class="key">${i+1}</div>
      <div><strong>${esc(text)}</strong></div>
    </label>`).join('');

  document.getElementById('app').innerHTML = `
    <div class="card">
      <div class="progress"><span style="width:${pct}%"></span></div>
      <p class="hint" style="margin:8px 0 0">快捷键：1 纳入 · 0 排除 · U 不确定 · S 跳过 · Enter 提交</p>
    </div>
    <div class="card">
      <h2>${esc(item.title)||'(无标题)'}</h2>
      <div class="chips">
        ${chip('PMID '+item.pmid)}
        ${chip(item.year||'年未知')}
        ${chip('先验标签 '+item.prior_label, priorCls)}
        ${chip(item.prior_label_source||'来源未知')}
        ${item.model_decision?chip('模型 '+item.model_decision+' @ '+item.model_score, modelCls):''}
        ${chip(item.text_source||'')}
        ${chip((item.text_chars||'?')+' 字')}
      </div>
      <p style="margin:10px 0 0"><a href="${esc(item.pubmed_url)}" target="_blank" rel="noopener">PubMed</a>
        ${item.pmcid?` · ${esc(item.pmcid)}`:''}</p>
      <p class="hint" style="margin:8px 0 0"><b>design:</b> ${esc(item.design)||'—'}<br/>
        <b>tests/methods:</b> ${esc(item.tests)} / ${esc(item.methods)}<br/>
        <b>先验理由:</b> ${esc(item.prior_reason)||'—'}</p>
    </div>
    <div class="card">
      <div class="section-title">摘要</div>
      <div class="body">${esc(item.abstract)||'（无摘要）'}</div>
    </div>
    <div class="card">
      <div class="section-title">正文摘录</div>
      <div class="body tall">${esc(item.text_head)||'（无正文）'}</div>
    </div>
    <div class="card">
      <div class="section-title">1. 判定（单选）</div>
      <div class="choices" id="labels">${labelHtml}</div>
    </div>
    <div class="card">
      <div class="section-title">2. 原因（单选）</div>
      <div class="choices" id="reasons">${reasonHtml}</div>
    </div>
    <div class="card">
      <div class="section-title">3. 备注（可选）</div>
      <textarea id="note" rows="3" placeholder="补充说明、原文关键句、分歧点…"></textarea>
    </div>`;
  document.getElementById('actions').style.display='flex';
  document.getElementById('err').textContent='';
  document.querySelectorAll('.choice').forEach(el=>{
    el.addEventListener('click', ()=>{
      const group = el.getAttribute('data-group');
      document.querySelectorAll(`.choice[data-group="${group}"]`).forEach(x=>x.classList.remove('selected'));
      el.classList.add('selected');
      el.querySelector('input').checked = true;
    });
  });
}

async function loadCurrent(){
  const r = await fetch('/api/current');
  const data = await r.json();
  state = data;
  render(data.item);
}

async function postSubmit(payload){
  const r = await fetch('/api/submit', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)});
  const data = await r.json().catch(()=>({}));
  if(!r.ok || !data.ok){
    const detail = data.error || (data.detail ? JSON.stringify(data.detail) : `提交失败 (HTTP ${r.status})`);
    throw new Error(detail);
  }
  return data;
}

async function submit(forcedLabel){
  const err = document.getElementById('err');
  err.textContent='';
  let label = forcedLabel;
  if(!label){
    const el = document.querySelector('input[name="gold_label"]:checked');
    if(!el){ err.textContent='请先选择判定（纳入/排除/不确定）'; return; }
    label = el.value;
  }
  if(label==='skip'){
    try {
      const data = await postSubmit({gold_label:'skip', reason_code:'', note:''});
      render(data.next);
    } catch(e){ err.textContent = e.message; }
    return;
  }
  const reasonEl = document.querySelector('input[name="reason_code"]:checked');
  if(!reasonEl){ err.textContent='请选择题干原因'; return; }
  const note = document.getElementById('note')?.value || '';
  try {
    const data = await postSubmit({gold_label:label, reason_code:reasonEl.value, note});
    render(data.next);
  } catch(e){ err.textContent = e.message; }
}

function submitCurrent(){ submit(null); }

document.addEventListener('keydown', (e)=>{
  if(e.target && (e.target.tagName==='TEXTAREA' || e.target.tagName==='INPUT')){
    if(e.key!=='Enter' || e.shiftKey) return;
  }
  if(e.key==='1'){ const i=document.querySelector('input[name="gold_label"][value="1"]'); if(i){i.checked=true; i.closest('.choice').click();} }
  if(e.key==='0'){ const i=document.querySelector('input[name="gold_label"][value="0"]'); if(i){i.checked=true; i.closest('.choice').click();} }
  if(e.key==='u' || e.key==='U'){ const i=document.querySelector('input[name="gold_label"][value="uncertain"]'); if(i){i.checked=true; i.closest('.choice').click();} }
  if(e.key==='s' || e.key==='S'){ e.preventDefault(); submit('skip'); }
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); submitCurrent(); }
});

loadCurrent();
</script>
</body>
</html>
"""


def make_app(state: ReviewState):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Gold standard review")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = PAGE_HTML
        page = page.replace("{{ reviewer }}", html.escape(state.reviewer))
        page = page.replace("{{ queue }}", html.escape(state.queue_name))
        page = page.replace("{{ gold_path }}", html.escape(str(state.gold_path)))
        page = page.replace("{{ label_choices_json }}", json.dumps(LABEL_CHOICES, ensure_ascii=False))
        page = page.replace("{{ reason_choices_json }}", json.dumps(REASON_CHOICES, ensure_ascii=False))
        return page

    @app.get("/api/current")
    def current() -> dict[str, Any]:
        return {
            "item": state.current(),
            "reviewer": state.reviewer,
            "queue": state.queue_name,
            "gold_path": str(state.gold_path),
        }

    @app.post("/api/submit")
    def submit(body: SubmitIn) -> JSONResponse:
        result = state.submit(body.gold_label, body.reason_code, body.note)
        code = 200 if result.get("ok") else 400
        return JSONResponse(result, status_code=code)

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        gold = _read_csv(state.gold_path)
        n1 = sum(1 for r in gold if r.get("gold_label") == "1")
        n0 = sum(1 for r in gold if r.get("gold_label") == "0")
        nu = sum(1 for r in gold if r.get("gold_label") == "uncertain")
        return {
            "queue_total": state.total,
            "queue_index": state.index,
            "gold_rows": len(gold),
            "gold_pos": n1,
            "gold_neg": n0,
            "gold_uncertain": nu,
        }

    return app


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------


def run_cli(state: ReviewState) -> None:
    print("金标准 CLI 审核。命令: 1 / 0 / u / s / q")
    print(f"输出: {state.gold_path}\n")
    while True:
        item = state.current()
        if item is None:
            print("队列已空。")
            break
        print("=" * 72)
        print(f"[{item['position']}/{item['total']}] PMID {item['pmid']} ({item['year']})")
        print(item["title"])
        print(f"先验 label={item['prior_label']} source={item['prior_label_source']} design={item['design']}")
        print(f"模型 decision={item['model_decision']} score={item['model_score']}")
        print(f"PubMed: {item['pubmed_url']}")
        print("- 摘要 -")
        print((item["abstract"] or "（无）")[:1200])
        print("- 正文头 -")
        print((item["text_head"] or "（无）")[:1500])
        print("- 判定 -")
        for val, name, desc in LABEL_CHOICES:
            print(f"  {val}: {name} — {desc}")
        print("- 原因编号 -")
        for i, (code, text) in enumerate(REASON_CHOICES, 1):
            print(f"  {i}. {code} — {text}")
        raw = input("输入判定 (1/0/u/s/q): ").strip().lower()
        if raw == "q":
            break
        if raw in {"s", "skip"}:
            state.submit("skip", "")
            continue
        label = {"1": "1", "0": "0", "u": "uncertain", "uncertain": "uncertain"}.get(raw)
        if not label:
            print("无效输入")
            continue
        rraw = input("原因编号或 code: ").strip()
        if rraw.isdigit():
            idx = int(rraw) - 1
            if not (0 <= idx < len(REASON_CHOICES)):
                print("原因编号超出范围")
                continue
            reason_code = REASON_CHOICES[idx][0]
        else:
            reason_code = rraw
        note = input("备注（可空）: ").strip()
        result = state.submit(label, reason_code, note)
        if not result.get("ok"):
            print("失败:", result.get("error"))
        else:
            print("已保存" if result.get("saved") else "已跳过")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="金标准输出 CSV")
    p.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    p.add_argument("--fn", type=Path, default=None, help="FN audit CSV（默认 artifacts/optimization/fn_audit.csv）")
    p.add_argument("--fp", type=Path, default=None, help="FP audit CSV（默认 artifacts/optimization/fp_audit.csv）")
    p.add_argument(
        "--queue",
        choices=["fn_fp", "test_set", "disagree", "weak_model", "no_gold", "all"],
        default="fn_fp",
        help="审核队列（默认优先 FN/FP）",
    )
    p.add_argument("--limit", type=int, default=None, help="最多审核 N 篇")
    p.add_argument("--reviewer", type=str, default=os.environ.get("USER") or "reviewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cli", action="store_true", help="终端选择题模式")
    p.add_argument("--reset-index", action="store_true", help="忽略进度，从队列头开始（已写入金标准的仍会跳过）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"缺少数据: {args.data}（请先运行 build_dataset.py）")

    papers = load_papers(args.data)
    preds = load_predictions(args.predictions) if args.predictions.exists() else {}
    done = load_done_pmids(args.gold)
    queue = build_queue(
        papers,
        preds,
        done,
        args.queue,
        args.limit,
        fn_path=args.fn,
        fp_path=args.fp,
    )

    print(
        f"papers={len(papers)} gold_done={len(done)} queue={args.queue} pending={len(queue)} -> {args.gold}",
        flush=True,
    )
    if not queue:
        raise SystemExit("队列为空：该模式下没有待审文献（或已全部写入金标准 0/1）。")

    state = ReviewState(papers, preds, queue, args.gold, args.reviewer, args.queue)

    if args.cli:
        run_cli(state)
        return

    import uvicorn

    app = make_app(state)
    print(f"打开 http://{args.host}:{args.port}/  开始选择题审核", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
