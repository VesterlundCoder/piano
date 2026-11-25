"""Gör piano/SR-transkript klara för diarize.py.

Läser transkript i piano-formatet (metadatarubrik + '======' + text, en mening
per rad) och skriver ren text till <ut>/<källa>/<id>.txt. Källan blir
programmappen (parent_dir), vilket diarize.py använder som källa.

Kör:
    python3 transkript_till_diarize.py --input ~/pianissimo_podcasts/transcripts \
        --output data --program spanarna ekots-l-rdagsintervju --limit 5
"""
import argparse
import pathlib
import re

SEP = re.compile(r"^={10,}\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="transcripts/-katalog (piano-format)")
    ap.add_argument("--output", required=True, help="utdatakatalog, blir data/ till diarize.py")
    ap.add_argument("--program", nargs="*", default=[], help="vissa programmappar (annars alla)")
    ap.add_argument("--limit", type=int, default=0, help="max antal avsnitt per program")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    out = pathlib.Path(args.output)
    progs = [p for p in sorted(src.iterdir()) if p.is_dir()] if not args.program \
        else [src / p for p in args.program]
    n = 0
    for prog in progs:
        if not prog.is_dir():
            print(f"VARNING: hittar inte {prog}")
            continue
        dest = out / prog.name
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in prog.glob("*.txt") if not f.stem.endswith("_sentenser"))
        if args.limit:
            files = files[: args.limit]
        for f in files:
            lines = f.read_text(encoding="utf-8").splitlines()
            body, seen_sep = [], False
            for line in lines:
                if not seen_sep:
                    if SEP.match(line.strip()):
                        seen_sep = True
                    continue
                body.append(line)
            text = "\n".join(body).strip()
            if not text:
                continue
            (dest / f.name).write_text(text + "\n", encoding="utf-8")
            n += 1
        print(f"{prog.name}: {len(files)} avsnitt")
    print(f"\nTotalt {n} avsnitt -> {out} (källa = programmapp)")


if __name__ == "__main__":
    main()
