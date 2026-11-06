#!/usr/bin/env python3
"""Run three light expert models over corpus.jsonl with per-model JSONL cache.

Defaults (overridable via env / flags):
  Claude  claude-haiku-5-5     linxi OpenAI-compat (JUDGE_*) or ANTHROPIC_API_KEY
  GPT     gpt-6-luna           linxi OpenAI-compat (fallback gpt-6-sol / gpt-6-astra)
  Gemini  gemini-3.8-flash     Vertex AI or GOOGLE_API_KEY
                               (linxi token may lack a Gemini channel)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from criteria import build_prompt, parse_reply  # noqa: E402

logger = logging.getLogger("ahp_judge")

DEFAULT_CORPUS = ROOT / "data" / "ahp_wave1" / "corpus.jsonl"
DEFAULT_CACHE = ROOT / "data" / "ahp_wave1" / "cache"

EXPERTS = {
    "claude": {
        "env_model": "AHP_CLAUDE_MODEL",
        "default_model": "claude-haiku-5-5",
        "key_envs": ("ANTHROPIC_API_KEY", "JUDGE_API_KEY", "OPENAI_API_KEY", "RAG_API_KEY"),
    },
    "gpt": {
        "env_model": "AHP_GPT_MODEL",
        "default_model": "gpt-6-luna",
        "key_envs": ("JUDGE_API_KEY", "OPENAI_API_KEY", "RAG_API_KEY"),
    },
    "gemini": {
        "env_model": "AHP_GEMINI_MODEL",
        "default_model": "gemini-3.8-flash",
        "key_envs": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    },
}


def _key_present(envs: tuple[str, ...]) -> bool:
    return any(os.environ.get(k) for k in envs)


def _linxi_client():
    """OpenAI-compatible client for linxi / OpenAI gateways."""
    from openai import OpenAI

    base = (
        os.environ.get("JUDGE_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("RAG_API_BASE")
    )
    if os.environ.get("JUDGE_API_BASE") and os.environ.get("JUDGE_API_KEY"):
        api_key = os.environ["JUDGE_API_KEY"]
    elif os.environ.get("RAG_API_BASE") and os.environ.get("RAG_API_KEY"):
        api_key = os.environ["RAG_API_KEY"]
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY") or os.environ.get("RAG_API_KEY")
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs), base


def _openai_compat_ask(client, candidates: list[str]) -> Callable[[str], dict]:
    disabled: set[str] = set()

    def ask(prompt: str) -> dict:
        last_err: Exception | None = None
        for mid in candidates:
            if mid in disabled:
                continue
            try:
                resp = client.chat.completions.create(
                    model=mid,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500,
                )
                text = resp.choices[0].message.content or ""
                if not text.strip():
                    if mid in {"gpt-5", "gpt-6-luna", "gpt-5-mini"}:
                        disabled.add(mid)
                    last_err = RuntimeError(f"empty_content model={mid}")
                    time.sleep(0.5)
                    continue
                parsed = parse_reply(text)
                if parsed is None:
                    return {"error": "parse_failed", "raw": text[:500], "model_used": mid}
                parsed["_model_used"] = mid
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                if "404" in msg or "not found" in msg or "no available channel" in msg or "does not exist" in msg:
                    disabled.add(mid)
                    logger.warning("disable model %s: %s", mid, str(exc)[:160])
                    continue
                if "429" in msg or "rate" in msg:
                    time.sleep(3.0)
                    continue
                if "503" in msg or "502" in msg or "timeout" in msg:
                    time.sleep(1.5)
                    continue
                return {"error": f"{type(exc).__name__}: {exc}"[:300], "model_used": mid}
        return {"error": f"{type(last_err).__name__}: {last_err}"[:300] if last_err else "compat_failed"}

    return ask


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        pmid = str(d.get("pmid") or "")
        if pmid:
            cache[pmid] = d
    return cache


def append_cache(path: Path, pmid: str, result: dict, *, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"pmid": pmid, "model": model, "result": result, "ts": time.time()},
                ensure_ascii=False,
            )
            + "\n"
        )


def make_claude(model: str) -> Callable[[str], dict]:
    """Prefer linxi OpenAI-compat; fall back to Anthropic SDK if only ANTHROPIC_API_KEY."""
    primary = os.environ.get("AHP_CLAUDE_MODEL", model)
    client, base = _linxi_client()
    if base or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        logger.info("Claude via OpenAI-compat base=%s model=%s", base, primary)
        return _openai_compat_ask(client, [primary])

    import anthropic

    aclient = anthropic.Anthropic()

    def ask(prompt: str) -> dict:
        msg = aclient.messages.create(
            model=primary,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "text", None))
        parsed = parse_reply(text)
        if parsed is None:
            return {"error": "parse_failed", "raw": text[:500]}
        parsed["_model_used"] = primary
        return parsed

    return ask


def make_gpt(model: str) -> Callable[[str], dict]:
    client, base = _linxi_client()
    primary = os.environ.get("AHP_GPT_MODEL", model)
    # Current linxi token: gpt-6-luna has no channel; gpt-6-sol / gpt-6-astra work.
    fallbacks = [
        os.environ.get("AHP_GPT_FALLBACK_MODEL"),
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-4o-mini",
    ]
    candidates: list[str] = []
    for mid in [primary, *fallbacks]:
        if mid and mid not in candidates:
            candidates.append(mid)
    logger.info("GPT via OpenAI-compat base=%s candidates=%s", base, candidates)
    return _openai_compat_ask(client, candidates)


def make_gemini(model: str) -> Callable[[str], dict]:
    """Gemini via Vertex AI (preferred) or API key."""
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
    from google import genai

    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini needs GOOGLE_API_KEY/GEMINI_API_KEY or Vertex AI env")
        client = genai.Client(api_key=api_key)

    def ask(prompt: str) -> dict:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None) or str(resp)
        parsed = parse_reply(text)
        if parsed is None:
            return {"error": "parse_failed", "raw": text[:500]}
        return parsed

    return ask


FACTORY = {"claude": make_claude, "gpt": make_gpt, "gemini": make_gemini}


def load_dotenv_files(paths: list[Path]) -> None:
    """Load KEY=VALUE from dotenv files without overriding existing env."""
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def gemini_available() -> bool:
    if _key_present(("GOOGLE_API_KEY", "GEMINI_API_KEY")):
        return True
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
    if use_vertex and (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")):
        return True
    return False


def iter_corpus(path: Path, limit: int | None = None):
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield rec
            n += 1
            if limit and n >= limit:
                break


def run_expert(
    name: str,
    corpus: Path,
    cache_path: Path,
    *,
    model: str,
    limit: int | None,
    workers: int,
    max_chars: int,
    skip_ocr: bool,
) -> dict[str, Any]:
    ask = FACTORY[name](model)
    cache = load_cache(cache_path)
    todo: list[dict] = []
    for rec in iter_corpus(corpus, limit=limit):
        pmid = str(rec["pmid"])
        prev = cache.get(pmid, {}).get("result") if pmid in cache else None
        if prev and not (isinstance(prev, dict) and prev.get("error")):
            continue
        if skip_ocr and rec.get("ocr_needed"):
            continue
        if rec.get("extract_error") or int(rec.get("text_chars") or 0) < 100:
            append_cache(
                cache_path,
                pmid,
                {"error": "insufficient_text", "qualifies": False, "confidence": 0.0},
                model=model,
            )
            continue
        todo.append(rec)

    logger.info("%s: cached_ok=%d todo=%d model=%s", name, len(cache) - len(todo), len(todo), model)
    ok = err = 0

    def one(rec: dict) -> tuple[str, dict]:
        prompt = build_prompt(rec.get("title") or "", rec.get("text") or "", max_chars=max_chars)
        try:
            return str(rec["pmid"]), ask(prompt)
        except Exception as exc:  # noqa: BLE001
            return str(rec["pmid"]), {"error": f"{type(exc).__name__}: {exc}"[:300]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            pmid, result = fut.result()
            append_cache(cache_path, pmid, result, model=model)
            if result.get("error"):
                err += 1
            else:
                ok += 1
            if (ok + err) % 50 == 0:
                logger.info("%s progress ok=%d err=%d", name, ok, err)

    return {"expert": name, "model": model, "newly_ok": ok, "newly_err": err, "todo": len(todo)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--experts", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--limit", type=int, default=None, help="pilot size (e.g. 100)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=16000)
    ap.add_argument("--skip-ocr", action="store_true", default=True)
    ap.add_argument("--no-skip-ocr", action="store_false", dest="skip_ocr")
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"missing corpus: {args.corpus}")

    # PACE_RAG first so JUDGE_/OPENAI_API_BASE + matching keys win over bare OpenAI keys.
    load_dotenv_files(
        [
            Path("/home/yangzs/.zshenv"),
            Path("/data/yangzs/Projects/01_Platform/06_PACE_RAG/benchmark/.env"),
            ROOT / ".env",
            Path("/data/yangzs/Projects/01_Platform/04_UltraPhenoCards/backend/.env.local"),
        ]
    )

    summary = []
    for name in args.experts:
        if name not in EXPERTS:
            raise SystemExit(f"unknown expert {name}")
        meta = EXPERTS[name]
        ready = gemini_available() if name == "gemini" else _key_present(meta["key_envs"])
        if not ready:
            logger.warning("skip %s: credentials missing (%s)", name, meta["key_envs"])
            summary.append({"expert": name, "skipped": "no_api_key"})
            continue
        model = os.environ.get(meta["env_model"], meta["default_model"])
        cache_path = args.cache_dir / f"{name}.jsonl"
        try:
            summary.append(
                run_expert(
                    name,
                    args.corpus,
                    cache_path,
                    model=model,
                    limit=args.limit,
                    workers=args.workers,
                    max_chars=args.max_chars,
                    skip_ocr=args.skip_ocr,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("expert %s failed", name)
            summary.append({"expert": name, "error": str(exc)})

    out = args.cache_dir / "judge_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
