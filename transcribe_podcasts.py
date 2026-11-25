"""Transkribera poddar lokalt på Mac (Apple silicon) med Klang Pianissimo.

Läser RSS-flöden (feeds.txt eller --url), laddar ner avsnitt, konverterar med
macOS inbyggda afconvert, transkriberar på GPU:n via MLX och skriver .txt +
_sentenser.txt per avsnitt. Ljudfilerna raderas efter varje transkriberat
avsnitt och körningen är återstartbar via state/progress.jsonl.

Exempel:
    .venv/bin/python transcribe_podcasts.py                 # allt i feeds.txt
    .venv/bin/python transcribe_podcasts.py --url RSS_URL    # en enstaka feed
    .venv/bin/python transcribe_podcasts.py --latest 5       # 5 senaste per feed
    .venv/bin/python transcribe_podcasts.py --limit 2        # röktest, 2 avsnitt

Stoppa mjukt (avslutar efter pågående avsnitt):  touch STOP
"""
import argparse
import feedparser
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request
import wave

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
STATE = BASE / "state" / "progress.jsonl"
AUDIO = BASE / "audio"
TRANSCRIPTS = BASE / "transcripts"
INDEX = TRANSCRIPTS / "index.tsv"
STOP = BASE / "STOP"
UA = {"User-Agent": "Mozilla/5.0 (piano-podcast-transcriber; personal use)"}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def slugify(name):
    return re.sub(r"[^a-z0-9åäö]+", "-", (name or "").lower()).strip("-") or "podcast"


def clean(text, length=200):
    return re.sub(r"[\t\n\r]+", " ", text or "").strip()[:length]


def read_state():
    done, failed = set(), set()
    if STATE.exists():
        for line in STATE.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            (done if rec.get("status") == "done" else failed).add(rec.get("url"))
    return done, failed


def append_state(rec):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_index(row):
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    new = not INDEX.exists()
    with open(INDEX, "a", encoding="utf-8") as f:
        if new:
            f.write("podcast\tepisode_id\ttitle\tpublished\tduration_s\twords\tfile\n")
        f.write("\t".join(str(x) for x in row) + "\n")


def load_model():
    """Pianissimo MLX 8-bit från HF-cache (laddar ner vid första körningen)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download("KlangAI/pianissimo-sv-mlx-8bit")
    sys.path.append(path)
    import pianissimo_mlx

    return pianissimo_mlx.load(path)


def parse_feed(url):
    """RSS/Atom-flöde -> (podcastnamn, lista av avsnitt med ljudlänk)."""
    feed = feedparser.parse(url)
    title = clean(getattr(feed.feed, "title", "") or url)
    episodes = []
    for entry in feed.entries:
        audio_url = None
        for enc in getattr(entry, "enclosures", []):
            if enc.get("url") and (enc.get("type", "").startswith("audio")
                                   or enc["url"].lower().endswith((".mp3", ".m4a", ".mp4", ".aac", ".wav"))):
                audio_url = enc["url"]
                break
        if audio_url is None:
            for link in getattr(entry, "links", []):
                if link.get("rel") == "enclosure" and link.get("href"):
                    audio_url = link["href"]
                    break
        if audio_url is None:
            continue
        pub = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        episodes.append({
            "url": audio_url,
            "id": hashlib.sha1(audio_url.encode()).hexdigest()[:12],
            "title": clean(getattr(entry, "title", "") or audio_url),
            "published": time.strftime("%Y-%m-%d", pub) if pub else "?",
            "published_ts": time.mktime(pub) if pub else 0,
        })
    episodes.sort(key=lambda e: -e["published_ts"])
    return title, episodes


def download(url, dest):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            return True
        except Exception:
            if attempt == 3:
                return False
            time.sleep(5 * (attempt + 1))


def convert(mp3, wav):
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(mp3), str(wav)], capture_output=True)
    return r.returncode == 0


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getframerate(), w.getsampwidth()) != (1, 16000, 2):
            raise ValueError(f"oväntat wav-format: {w.getnchannels()}ch/{w.getframerate()}Hz")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def fmt_ts(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feeds", default="feeds.txt", help="fil med RSS-URL:er, en per rad")
    ap.add_argument("--url", help="transkribera ett enstaka RSS-flöde")
    ap.add_argument("--latest", type=int, default=0, help="bara N senaste avsnitten per feed")
    ap.add_argument("--limit", type=int, default=0, help="max antal avsnitt totalt (röktest)")
    args = ap.parse_args()

    feed_urls = []
    if args.url:
        feed_urls.append(args.url)
    else:
        feeds_path = BASE / args.feeds
        if not feeds_path.exists():
            sys.exit(f"Hittar inte {feeds_path}. Kopiera feeds.txt.example till feeds.txt "
                     f"och klistra in RSS-URL:er, eller använd --url.")
        feed_urls = [line.strip() for line in feeds_path.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.strip().startswith("#")]
    if not feed_urls:
        sys.exit("Inga flöden att bearbeta.")

    log(f"Läser {len(feed_urls)} RSS-flöden...")
    all_episodes = []
    for url in feed_urls:
        try:
            name, eps = parse_feed(url)
        except Exception as exc:
            log(f"KUNDE INTE LÄSA {url}: {exc}")
            continue
        if args.latest:
            eps = eps[: args.latest]
        log(f"  {name}: {len(eps)} avsnitt")
        all_episodes.extend((name, url, ep) for ep in eps)

    done, failed = read_state()
    pending = [(n, u, e) for n, u, e in all_episodes if e["url"] not in done and e["url"] not in failed]
    if args.limit:
        pending = pending[: args.limit]
    log(f"Totalt {len(all_episodes)} avsnitt i flödena | klara {len(done)} | "
        f"misslyckade {len(failed)} | återstår {len(pending)}")

    model = load_model()
    import pianissimo_mlx

    AUDIO.mkdir(parents=True, exist_ok=True)
    n_done = n_failed = 0
    for i, (podcast, feed_url, ep) in enumerate(pending, 1):
        if STOP.exists():
            log("STOP-fil hittad - avslutar korrekt")
            break
        t0 = time.time()
        mp3 = AUDIO / f"tmp{ep['id']}.mp3"
        wav = AUDIO / f"tmp{ep['id']}.wav"
        tdir = TRANSCRIPTS / slugify(podcast)
        tdir.mkdir(parents=True, exist_ok=True)
        status, note, words, dur = "done", "", 0, 0.0
        try:
            if not download(ep["url"], mp3):
                raise RuntimeError("nedladdning misslyckades")
            if not convert(mp3, wav):
                raise RuntimeError("afconvert misslyckades")
            audio = read_wav(wav)
            dur = len(audio) / 16000
            if dur < 2:
                raise RuntimeError(f"för kort ({dur:.1f}s)")
            result = pianissimo_mlx.transcribe(model, audio)
            words = len(result.text.split())
            header = [
                f"Podcast:    {clean(podcast)}",
                f"Avsnitt:    {ep['title']}",
                f"Publicerat: {ep['published']}",
                f"Källa:      {ep['url']}",
                f"Flöde:      {feed_url}",
                f"Längd:      {fmt_ts(dur)}",
                "Modell:     KlangAI/pianissimo-sv-mlx-8bit (Klang Pianissimo, CC BY 4.0)",
            ]
            body = "\n".join(s.text.strip() for s in result.sentences)
            (tdir / f"{ep['id']}.txt").write_text(
                "\n".join(header) + "\n\n" + "=" * 70 + "\n\n" + body + "\n",
                encoding="utf-8")
            with open(tdir / f"{ep['id']}_sentenser.txt", "w", encoding="utf-8") as f:
                for s in result.sentences:
                    f.write(f"[{fmt_ts(s.start)} - {fmt_ts(s.end)}] {s.text.strip()}\n")
            append_index([clean(podcast), ep["id"], ep["title"], ep["published"],
                          int(dur), words, tdir / f"{ep['id']}.txt"])
        except Exception as exc:
            status, note = "failed", str(exc)[:200]
        finally:
            mp3.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)

        append_state({"url": ep["url"], "episode_id": ep["id"], "podcast": clean(podcast),
                      "title": ep["title"], "status": status, "note": note,
                      "duration_s": round(dur), "words": words})
        elapsed = time.time() - t0
        if status == "done":
            n_done += 1
            log(f"[{i}/{len(pending)}] KLAR {clean(podcast)} | {ep['title'][:60]} | "
                f"{dur/60:5.1f} min ljud | {elapsed/60:4.1f} min beräkning "
                f"({dur/elapsed if elapsed else 0:4.0f}x) | {words} ord")
        else:
            n_failed += 1
            log(f"[{i}/{len(pending)}] FEL  {clean(podcast)} | {ep['title'][:60]} | {note}")
        time.sleep(0.5)

    log(f"KÖRNINGEN SLUT | {n_done} klara, {n_failed} fel. "
        f"Resultat i transcripts/ - kör skriptet igen för att fortsätta.")


if __name__ == "__main__":
    main()
