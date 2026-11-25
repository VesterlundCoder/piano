#!/usr/bin/env python3
"""
diarize.py: textbaserad talardiarisering av transkript med en lokal LLM.

Flöde per dokument:
  1. Läs in (.txt, .srt, .vtt, .jsonl)
  2. Dela upp i meningar
  3. Filtrera bort ASR-hallucinationer och loopar
  4. Låt LLM:en märka varje mening med en talare (texten skickas aldrig tillbaka och kan inte ändras)
  5. Validera och bestäm dokumenttyp: monolog / dialog / ej_separerad
  6. Formatera med rubrik och talaretiketter och skriv JSONL (kan återupptas)

Kräver Python 3.11 eller senare och inga externa paket.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Datatyper
# --------------------------------------------------------------------------- #

@dataclass
class Doc:
    id: str
    source: str
    file: str
    segments: list[str]                 # råa segment (stycken eller undertextrader)
    seg_kind: str = "paragraph"         # "paragraph" (behåll stycken) eller "cue" (undertexter)
    language: str | None = None
    labeled_turns: list[tuple[str, str]] | None = None   # om texten redan har talaretiketter


@dataclass
class Sentence:
    text: str
    seg_idx: int


# --------------------------------------------------------------------------- #
# Inläsning
# --------------------------------------------------------------------------- #

LABEL_LINE = re.compile(r"^\s*([A-ZÅÄÖÆØÉÜ][\wÅÄÖåäöÆØæøÉéÜü.\- ]{0,40}?)\s*:\s+(\S.*)$")


def detect_labeled(lines: list[str]) -> list[tuple[str, str]] | None:
    """Returnera [(talare, text)] om texten redan har talaretiketter, annars None."""
    nonempty = [l for l in lines if l.strip()]
    if len(nonempty) < 4:
        return None
    turns: list[tuple[str, str]] = []
    hits = 0
    for l in nonempty:
        m = LABEL_LINE.match(l)
        if m:
            hits += 1
            turns.append((m.group(1).strip(), m.group(2).strip()))
        elif turns:
            sp, tx = turns[-1]
            turns[-1] = (sp, tx + " " + l.strip())
    labels = Counter(sp for sp, _ in turns)
    repeating = [s for s, c in labels.items() if c >= 2]
    if hits / len(nonempty) >= 0.5 and len(labels) >= 2 and len(repeating) >= 1 and len(labels) <= 20:
        return turns
    return None


def read_subtitles(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    segs: list[str] = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [l.strip() for l in block.strip().splitlines()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        text_lines = []
        for l in lines:
            if not l or l.isdigit() or "-->" in l:
                continue
            l = re.sub(r"<[^>]+>", "", l)          # ta bort <i>, <v Talare> osv.
            l = re.sub(r"\{\\[^}]*\}", "", l)       # ASS-taggar
            if l:
                text_lines.append(l)
        # Repliker som börjar med tankstreck är egna segment (vanlig undertextkonvention för talarbyte)
        current: list[str] = []
        for l in text_lines:
            if re.match(r"^[-–]\s*\S", l) and current:
                segs.append(" ".join(current))
                current = [l]
            else:
                current.append(l)
        if current:
            segs.append(" ".join(current))
    return segs


def iter_docs(inputs: list[Path], cfg: dict):
    source_from = cfg["output"].get("source_from", "parent_dir")
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in {".txt", ".srt", ".vtt", ".jsonl"})
        elif p.exists():
            files.append(p)
        else:
            print(f"VARNING: hittar inte {p}", file=sys.stderr)

    for f in files:
        default_source = f.parent.name if source_from == "parent_dir" else "okänd"
        suf = f.suffix.lower()
        try:
            if suf in {".srt", ".vtt"}:
                yield Doc(id=str(f), source=default_source, file=str(f),
                          segments=read_subtitles(f), seg_kind="cue")
            elif suf == ".txt":
                raw = f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
                labeled = detect_labeled(raw.splitlines())
                paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
                yield Doc(id=str(f), source=default_source, file=str(f),
                          segments=[p for p in paras if p], labeled_turns=labeled)
            elif suf == ".jsonl":
                with f.open(encoding="utf-8", errors="replace") as fh:
                    for ln, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"VARNING: ogiltig JSON i {f}:{ln}", file=sys.stderr)
                            continue
                        doc_id = str(obj.get("id", f"{f}#{ln}"))
                        src = str(obj.get("source", default_source))
                        if isinstance(obj.get("segments"), list):
                            segs = [s["text"] if isinstance(s, dict) else str(s) for s in obj["segments"]]
                            segs = [re.sub(r"\s+", " ", s).strip() for s in segs]
                            yield Doc(id=doc_id, source=src, file=str(f), segments=[s for s in segs if s],
                                      seg_kind="cue", language=obj.get("lang") or obj.get("language"))
                        else:
                            text = str(obj.get("text", ""))
                            labeled = detect_labeled(text.splitlines())
                            paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
                            yield Doc(id=doc_id, source=src, file=str(f), segments=[p for p in paras if p],
                                      language=obj.get("lang") or obj.get("language"), labeled_turns=labeled)
        except OSError as e:
            print(f"VARNING: kunde inte läsa {f}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Meningsdelning och filtrering
# --------------------------------------------------------------------------- #

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"“«'(\-–]?[A-ZÅÄÖÆØÉÜ0-9])")


def split_sentences(segments: list[str], max_words: int) -> list[Sentence]:
    out: list[Sentence] = []
    for si, seg in enumerate(segments):
        for part in SENT_SPLIT.split(seg.strip()):
            words = part.split()
            for k in range(0, len(words), max_words):
                chunk = " ".join(words[k:k + max_words])
                if chunk:
                    out.append(Sentence(chunk, si))
    return out


def compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def filter_sentences(sents: list[Sentence], cfg: dict) -> tuple[list[Sentence], dict]:
    fc = cfg["filter"]
    pats = [re.compile(p, re.IGNORECASE) for p in fc.get("hallucination_patterns", [])]
    out: list[Sentence] = []
    stats = Counter()
    prev_norm = None
    for s in sents:
        norm = re.sub(r"[^\w]+", " ", s.text.lower()).strip()
        nwords = len(norm.split())
        if not norm:
            stats["tomma"] += 1
            continue
        if nwords <= 12 and any(p.search(s.text) for p in pats):
            stats["hallucination"] += 1
            continue
        if len(s.text) >= 30 and compression_ratio(s.text) > fc.get("compression_ratio_max", 2.4):
            stats["loop_i_mening"] += 1
            continue
        if norm == prev_norm and nwords >= fc.get("min_words_for_duplicate_check", 4):
            stats["upprepad_mening"] += 1
            continue
        out.append(s)
        prev_norm = norm
    return out, dict(stats)


# --------------------------------------------------------------------------- #
# LLM-klient (OpenAI-kompatibel, endast standardbiblioteket)
# --------------------------------------------------------------------------- #

class LLM:
    def __init__(self, c: dict):
        self.url = c["base_url"].rstrip("/") + "/chat/completions"
        self.model = c["model"]
        self.key = c.get("api_key", "lokal")
        self.temperature = c.get("temperature", 0.0)
        self.max_tokens = c.get("max_tokens", 2048)
        self.timeout = c.get("timeout_s", 300)
        self.json_mode = c.get("json_mode", True)
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and self.json_mode:
                print("INFO: servern stöder inte json_mode, fortsätter utan.", file=sys.stderr)
                self.json_mode = False
                return self.chat(system, user)
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} från LLM-servern: {detail}") from e
        usage = resp.get("usage") or {}
        with self.lock:
            self.calls += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
        return resp["choices"][0]["message"].get("content") or ""


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("inget JSON-objekt i svaret")
    return json.loads(text[a:b + 1])


def norm_label(s) -> str:
    s = str(s).strip().upper()
    if s in {"?", "OKÄND", "UNKNOWN", ""}:
        return "?"
    m = re.fullmatch(r"(?:T|TALARE|SPEAKER)?\s*(\d+)", s)
    return f"T{int(m.group(1))}" if m else s


def validate(obj, lo: int, hi: int, max_speakers: int) -> tuple[dict[int, str], dict[str, str], str, str]:
    """lo/hi är 1-baserade ID:n (inklusive). Returnerar (etiketter, talare, språk, osäkerhet)."""
    if not isinstance(obj, dict):
        raise ValueError("svaret är inte ett JSON-objekt")
    items = obj.get("labels")
    if not isinstance(items, list):
        raise ValueError('nyckeln "labels" saknas eller är inte en lista')
    allowed = {f"T{k}" for k in range(1, max_speakers + 1)} | {"?"}
    out: dict[int, str] = {}
    for it in items:
        if not isinstance(it, dict) or "id" not in it:
            raise ValueError("varje post i labels måste vara {\"id\": ..., \"s\": ...}")
        try:
            i = int(it["id"])
        except (TypeError, ValueError):
            raise ValueError(f"ogiltigt id {it.get('id')!r}")
        if not lo <= i <= hi:
            continue                                  # modellen märkte sammanhanget igen: ignorera
        s = norm_label(it.get("s", "?"))
        if s not in allowed:
            raise ValueError(f"ogiltig etikett {s!r} för id {i} (tillåtna: T1–T{max_speakers} eller ?)")
        out[i] = s
    missing = [i for i in range(lo, hi + 1) if i not in out]
    if missing:
        if len(missing) > max(1, (hi - lo + 1) // 10):
            raise ValueError(f"etiketter saknas för id {missing[:15]}")
        for i in missing:                             # små luckor: samma talare fortsätter
            out[i] = out.get(i - 1, "?")
    speakers = {}
    if isinstance(obj.get("speakers"), dict):
        for k, v in obj["speakers"].items():
            k = norm_label(k)
            if k in allowed and k != "?":
                speakers[k] = str(v)[:60]
    lang = str(obj.get("language", "")).lower()[:5]
    unc = str(obj.get("uncertainty", "medium")).lower()
    if unc not in {"low", "medium", "high"}:
        unc = "medium"
    return out, speakers, lang, unc


# --------------------------------------------------------------------------- #
# Diarisering av ett dokument
# --------------------------------------------------------------------------- #

def build_user_msg(source: str, roster: dict, sents: list[Sentence], labels: list,
                   ctx_from: int, lo0: int, hi0: int) -> str:
    parts = [f"KÄLLA: {source}"]
    if roster:
        parts.append("TALARLISTA (hittills):\n" + "\n".join(f"{k}: {v}" for k, v in sorted(roster.items())))
    else:
        parts.append("TALARLISTA (hittills):\n(tom: detta är början av transkriptet)")
    if ctx_from < lo0:
        parts.append("TIDIGARE SAMMANHANG (redan märkt, märk inte igen):\n" +
                     "\n".join(f"[{labels[k]}] {sents[k].text}" for k in range(ctx_from, lo0)))
    parts.append(f"AKTUELLT AVSNITT – märk varje ID från {lo0 + 1} till {hi0}:\n" +
                 "\n".join(f"{k + 1}: {sents[k].text}" for k in range(lo0, hi0)))
    parts.append("Svara endast med JSON enligt formatet.")
    return "\n\n".join(parts)


def diarize(sents: list[Sentence], source: str, llm: LLM, system: str, cfg: dict) -> dict:
    ch, dz = cfg["chunking"], cfg["diarization"]
    n = len(sents)
    labels: list[str | None] = [None] * n
    roster: dict[str, str] = {}
    lang_votes: Counter = Counter()
    unc: list[str] = []
    failed = chunks = 0
    i = 0
    while i < n:
        j, words = i, 0
        while j < n and j - i < ch["chunk_sentences"]:
            w = len(sents[j].text.split())
            if j > i and words + w > ch["max_chunk_words"]:
                break
            words += w
            j += 1
        chunks += 1
        user = build_user_msg(source, roster, sents, labels, max(0, i - ch["context_sentences"]), i, j)
        result, err = None, None
        for attempt in range(cfg["llm"].get("retries", 2) + 1):
            msg = user if err is None else (
                user + f"\n\nDITT FÖREGÅENDE SVAR VAR OGILTIGT: {err}. Följ formatet exakt och märk alla ID.")
            try:
                raw = llm.chat(system, msg)
                result = validate(parse_json(raw), i + 1, j, dz["max_speakers"])
                break
            except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                err = str(e)[:200]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt >= cfg["llm"].get("retries", 2):
                    raise RuntimeError(f"LLM-servern svarar inte ({e}). Kör den? Kolla base_url.") from e
                time.sleep(5 * (attempt + 1))
        if result is None:
            failed += 1
            for k in range(i, j):
                labels[k] = "?"
        else:
            labs, spk, lang, u = result
            for k1, s in labs.items():
                labels[k1 - 1] = s
            for k, v in spk.items():
                roster.setdefault(k, v)
            if lang:
                lang_votes[lang] += j - i
            unc.append(u)
        i = j
    return {"labels": labels, "roster": roster, "lang_votes": lang_votes,
            "uncertainty": unc, "failed_chunks": failed, "chunks": chunks}


def decide_type(labels: list[str], res: dict, dz: dict) -> str:
    n = len(labels)
    unknown = labels.count("?") / n
    unc = res["uncertainty"]
    high = unc.count("high") / len(unc) if unc else 1.0
    if (res["failed_chunks"] / max(1, res["chunks"]) > dz["max_failed_chunk_share"]
            or unknown > dz["max_unknown_share"] or high > dz["max_high_uncertainty_share"]):
        return "ej_separerad"
    counts = Counter(l for l in labels if l != "?")
    if not counts:
        return "ej_separerad"
    top = counts.most_common(1)[0][1] / sum(counts.values())
    if len(counts) == 1 or top >= dz["monolog_share"]:
        return "monolog"
    return "dialog"


# --------------------------------------------------------------------------- #
# Formatering
# --------------------------------------------------------------------------- #

def strip_dash(t: str) -> str:
    return re.sub(r"^[-–]\s*", "", t)


def format_plain(sents: list[Sentence], seg_kind: str, para_n: int) -> str:
    paras, cur, last_seg = [], [], None
    for k, s in enumerate(sents):
        brk = (seg_kind == "paragraph" and last_seg is not None and s.seg_idx != last_seg) or \
              (seg_kind == "cue" and len(cur) >= para_n)
        if brk and cur:
            paras.append(" ".join(cur))
            cur = []
        cur.append(strip_dash(s.text))
        last_seg = s.seg_idx
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def format_dialog(texts: list[str], labels: list[str]) -> tuple[str, dict[str, str], int]:
    """Slår ihop meningar i följd per talare och numrerar om talarna i ordning de dyker upp."""
    # "?" mellan två meningar av samma talare hör till den talaren
    labs = list(labels)
    for k, l in enumerate(labs):
        if l == "?":
            prev = next((labs[x] for x in range(k - 1, -1, -1) if labs[x] != "?"), None)
            nxt = next((labs[x] for x in range(k + 1, len(labs)) if labs[x] != "?"), None)
            if prev and prev == nxt:
                labs[k] = prev
    mapping: dict[str, str] = {}
    for l in labs:
        if l != "?" and l not in mapping:
            mapping[l] = f"Talare {len(mapping) + 1}"
    turns: list[tuple[str, list[str]]] = []
    for t, l in zip(texts, labs):
        name = mapping.get(l, "Okänd")
        if turns and turns[-1][0] == name:
            turns[-1][1].append(strip_dash(t))
        else:
            turns.append((name, [strip_dash(t)]))
    body = "\n".join(f"{name}: {' '.join(ts)}" for name, ts in turns)
    return body, mapping, len(mapping)


# --------------------------------------------------------------------------- #
# Process för ett dokument
# --------------------------------------------------------------------------- #

def process_doc(doc: Doc, llm: LLM | None, system: str, cfg: dict) -> tuple[str, dict]:
    t0 = time.time()
    ch, dz, oc = cfg["chunking"], cfg["diarization"], cfg["output"]
    base = {"id": doc.id, "source": doc.source, "file": doc.file}

    # A) Redan märkt text: använd befintliga etiketter
    if doc.labeled_turns:
        names: dict[str, str] = {}
        lines = []
        for sp, tx in doc.labeled_turns:
            if sp not in names:
                names[sp] = sp if dz.get("keep_existing_names") else f"Talare {len(names) + 1}"
            lines.append(f"{names[sp]}: {tx}")
        n_sp = len(names)
        typ = "dialog" if n_sp > 1 else "monolog"
        lang = doc.language or oc["default_language"]
        header = oc["header_template"].format(source=doc.source, type=typ, n_speakers=n_sp, language=lang)
        body = "\n".join(lines)
        if len(body.split()) < cfg["filter"]["min_words"]:
            return "reject", {**base, "reason": "för_kort"}
        return "ok", {**base, "type": typ, "n_speakers": n_sp, "language": lang, "speaker_roles": {},
                      "method": "befintliga_etiketter", "text": header + "\n\n" + body,
                      "stats": {"seconds": round(time.time() - t0, 2)}}

    # B) Meningsdelning och filtrering
    sents = split_sentences(doc.segments, ch["max_sentence_words"])
    kept, fstats = filter_sentences(sents, cfg)
    n_in = len(sents)
    dropped = (n_in - len(kept)) / max(1, n_in)
    n_words = sum(len(s.text.split()) for s in kept)
    stats = {"sentences_in": n_in, "sentences_kept": len(kept), "words": n_words, "filtered": fstats}
    if n_words < cfg["filter"]["min_words"]:
        return "reject", {**base, "reason": "för_kort", "stats": stats}
    if dropped > cfg["filter"]["max_dropped_share"]:
        return "reject", {**base, "reason": "för_mycket_hallucination_eller_loopar", "stats": stats}

    # C) Diarisering (eller känd monologkälla)
    src_l = doc.source.lower()
    roles: dict[str, str] = {}
    if any(m.lower() in src_l for m in dz.get("monolog_sources", [])):
        typ, method, lang = "monolog", "källa_är_monolog", doc.language or oc["default_language"]
        labels = ["T1"] * len(kept)
    else:
        assert llm is not None
        res = diarize(kept, doc.source, llm, system, cfg)
        labels = res["labels"]
        typ = decide_type(labels, res, dz)
        method = "llm"
        lang = doc.language or (res["lang_votes"].most_common(1)[0][0] if res["lang_votes"] else oc["default_language"])
        stats.update({"chunks": res["chunks"], "failed_chunks": res["failed_chunks"],
                      "unknown_share": round(labels.count("?") / len(labels), 3),
                      "uncertainty": dict(Counter(res["uncertainty"]))})
        roles = res["roster"]

    # D) Formatering
    if typ == "dialog":
        body, mapping, n_sp = format_dialog([s.text for s in kept], labels)
        speaker_roles = {mapping[k]: v for k, v in roles.items() if k in mapping}
    else:
        body = format_plain(kept, doc.seg_kind, oc.get("monolog_paragraph_sentences", 6))
        n_sp = 1 if typ == "monolog" else "okänt"
        speaker_roles = {}
    header = oc["header_template"].format(source=doc.source, type=typ, n_speakers=n_sp, language=lang)
    stats["seconds"] = round(time.time() - t0, 2)
    return "ok", {**base, "type": typ, "n_speakers": n_sp, "language": lang, "speaker_roles": speaker_roles,
                  "method": method, "text": header + "\n\n" + body, "stats": stats}


# --------------------------------------------------------------------------- #
# Uppskattning, granskningsurval, main
# --------------------------------------------------------------------------- #

def estimate(docs, cfg: dict, gen_tps: float, prompt_tps: float, system: str):
    ch = cfg["chunking"]
    n_docs = n_words = n_chunks = n_sents = skipped = 0
    sys_tok = int(len(system.split()) * 1.6)
    for d in docs:
        n_docs += 1
        if d.labeled_turns or any(m.lower() in d.source.lower() for m in cfg["diarization"]["monolog_sources"]):
            skipped += 1
            continue
        s, _ = filter_sentences(split_sentences(d.segments, ch["max_sentence_words"]), cfg)
        w = sum(len(x.text.split()) for x in s)
        if w < cfg["filter"]["min_words"]:
            continue
        n_words += w
        n_sents += len(s)
        n_chunks += max(1, -(-len(s) // ch["chunk_sentences"]), -(-w // ch["max_chunk_words"]))
    in_tok = n_chunks * sys_tok + n_words * 1.6 * 1.3     # text + ID och sammanhang
    out_tok = n_sents * 11 + n_chunks * 60
    hours = (out_tok / gen_tps + in_tok / prompt_tps) / 3600
    print(f"Dokument: {n_docs:,} (varav {skipped:,} utan LLM-behov)")
    print(f"Ord att diarisera: {n_words:,}  (~{int(n_words * 1.6):,} tokens)")
    print(f"Meningar: {n_sents:,}   LLM-anrop: {n_chunks:,}")
    print(f"Uppskattade tokens: in ~{int(in_tok):,}, ut ~{int(out_tok):,}")
    print(f"Uppskattad tid med 1 arbetare vid {gen_tps:.0f} tok/s generering och {prompt_tps:.0f} tok/s prompt: "
          f"~{hours:,.1f} timmar")
    if hours > 72:
        print("\nDet här är för mycket för en laptop. Kör ett urval lokalt för att kontrollera kvaliteten "
              "och kör sedan helheten mot en vLLM-server (t.ex. på LUMI) med --workers 16–64. Se LÄSMIG.md.")


def write_review(out_path: Path, review_path: Path, k: int, seed: int):
    if not out_path.exists():
        return
    rng = random.Random(seed)
    sample: list[dict] = []
    with out_path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            rec = json.loads(line)
            if len(sample) < k:
                sample.append(rec)
            else:
                r = rng.randint(0, n)
                if r < k:
                    sample[r] = rec
    types = Counter(r["type"] for r in sample)
    lines = ["# Granskningsurval", "",
             f"{len(sample)} slumpvis valda dokument. Typfördelning i urvalet: {dict(types)}", "",
             "För varje dokument: kryssa i om typen stämmer och om talarbytena ser rimliga ut.",
             "Räkna fel per dokument: missat byte, påhittat byte, fel talare tillbaka.", ""]
    for i, r in enumerate(sample, 1):
        txt = r["text"]
        if len(txt) > 3000:
            txt = txt[:3000] + "\n[... avkortat ...]"
        lines += [f"## {i}. {r['id']}", "",
                  f"- Källa: {r['source']} | Typ: **{r['type']}** | Talare: {r['n_speakers']} | "
                  f"Språk: {r['language']} | Metod: {r['method']}",
                  f"- Roller (LLM:ens gissning, ej i träningstexten): {r.get('speaker_roles') or '–'}",
                  f"- Statistik: {json.dumps(r.get('stats', {}), ensure_ascii=False)}", "",
                  "- [ ] Typen stämmer", "- [ ] Talarbytena är rimliga", "- Antal fel: ___", "",
                  "```", txt, "```", ""]
    review_path.write_text("\n".join(lines), encoding="utf-8")


def load_done(paths: list[Path]) -> set[str]:
    done = set()
    for p in paths:
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def main():
    ap = argparse.ArgumentParser(description="Textbaserad talardiarisering av transkript med lokal LLM.")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--prompt", default=None, help="systemprompt (standard: prompt_sv.txt bredvid skriptet)")
    ap.add_argument("--input", nargs="+", required=True, help="filer eller mappar (.txt .srt .vtt .jsonl)")
    ap.add_argument("--output", default="ut")
    ap.add_argument("--workers", type=int, default=1, help="parallella anrop (1 på laptop, 16–64 mot vLLM)")
    ap.add_argument("--limit", type=int, default=0, help="max antal dokument (0 = alla)")
    ap.add_argument("--sample", type=float, default=1.0, help="andel dokument att ta med, t.ex. 0.01")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--estimate", action="store_true", help="uppskatta tid utan att anropa LLM:en")
    ap.add_argument("--gen-tps", type=float, default=30.0, help="genereringshastighet för --estimate")
    ap.add_argument("--prompt-tps", type=float, default=500.0, help="promptbearbetning för --estimate")
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    prompt_path = Path(args.prompt) if args.prompt else Path(__file__).with_name("prompt_sv.txt")
    system = prompt_path.read_text(encoding="utf-8").replace(
        "{MAX_SPEAKERS}", str(cfg["diarization"]["max_speakers"]))

    rng = random.Random(args.seed)

    def selected():
        n = 0
        for d in iter_docs([Path(p) for p in args.input], cfg):
            if args.sample < 1.0 and rng.random() >= args.sample:
                continue
            yield d
            n += 1
            if args.limit and n >= args.limit:
                return

    if args.estimate:
        estimate(selected(), cfg, args.gen_tps, args.prompt_tps, system)
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_path, rej_path = out_dir / "diariserat.jsonl", out_dir / "kasserat.jsonl"
    done = load_done([ok_path, rej_path])
    if done:
        print(f"Återupptar: {len(done):,} dokument redan klara hoppas över.")

    llm = LLM(cfg["llm"])
    wlock = threading.Lock()
    counts: Counter = Counter()
    t_start = time.time()
    ok_fh = ok_path.open("a", encoding="utf-8")
    rej_fh = rej_path.open("a", encoding="utf-8")

    def handle(doc: Doc):
        status, rec = process_doc(doc, llm, system, cfg)
        with wlock:
            fh = ok_fh if status == "ok" else rej_fh
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            key = rec["type"] if status == "ok" else f"kasserat:{rec['reason']}"
            counts[key] += 1
            total = sum(counts.values())
            el = time.time() - t_start
            tps = llm.completion_tokens / el if el > 0 else 0
            print(f"[{total:>6}] {key:<38} {doc.id[:70]}  ({rec.get('stats', {}).get('seconds', 0)} s, "
                  f"{tps:.1f} gen-tok/s totalt)", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            pending = []
            for doc in selected():
                if doc.id in done:
                    continue
                pending.append(pool.submit(handle, doc))
                if len(pending) >= args.workers * 4:
                    pending.pop(0).result()
            for f in pending:
                f.result()
    except KeyboardInterrupt:
        print("\nAvbrutet. Kör samma kommando igen för att fortsätta där du slutade.")
    except RuntimeError as e:
        print(f"\nAVBRYTER: {e}\nKlara dokument är sparade. Kör samma kommando igen när felet är åtgärdat.",
              file=sys.stderr)
        sys.exit(1)
    finally:
        ok_fh.close()
        rej_fh.close()

    el = time.time() - t_start
    print("\n=== Sammanfattning ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:<40} {v:>7}")
    print(f"  LLM-anrop: {llm.calls:,}   tokens in/ut: {llm.prompt_tokens:,}/{llm.completion_tokens:,}   "
          f"tid: {el / 60:.1f} min")
    write_review(ok_path, out_dir / "granskning.md", cfg["output"].get("review_sample_size", 30), args.seed)
    print(f"Utdata: {ok_path}\nKasserat: {rej_path}\nGranskning: {out_dir / 'granskning.md'}")


if __name__ == "__main__":
    main()
