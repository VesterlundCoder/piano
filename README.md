# piano — transkribera poddar lokalt på Mac med Klang Pianissimo

Laddar ner poddavsnitt från RSS-flöden, transkriberar dem till text och skriver
rent formaterade transkript med meningar och tidsstämplar. Kör helt lokalt på
Mac med Apple silicon (M1 eller senare) — modellen är
[Klang Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv) (svenska,
~4,5 % WER, CC BY 4.0) i MLX-format på GPU:n. Ingen API-nyckel, ingen molntjänst.

Per avsnitt skrivs `transcripts/<podd>/<id>.txt` (en mening per rad, metadatarubrik)
samt `<id>_sentenser.txt` med tidsstämplar per mening. Ljudfilerna raderas efter
transkription. Körningen är återstartbar — redan klara avsnitt hoppas över —
och kan stoppas mjut när som helst.

## Kom igång

```bash
git clone https://github.com/VesterlundCoder/piano.git
cd piano
bash setup.sh          # skapar .venv, installerar beroenden, laddar modellen (759 MB)
cp feeds.txt.example feeds.txt    # klistra in dina RSS-URL:er (en per rad)
.venv/bin/python transcribe_podcasts.py --limit 2   # röktest
.venv/bin/python transcribe_podcasts.py             # kör allt
```

## Användning

```bash
.venv/bin/python transcribe_podcasts.py                  # allt i feeds.txt
.venv/bin/python transcribe_podcasts.py --url RSS_URL    # en enstaka feed
.venv/bin/python transcribe_podcasts.py --latest 5       # bara 5 senaste per feed
.venv/bin/python transcribe_podcasts.py --feeds mina.txt # annan flödesfil
```

Stoppa mjukt (avslutar pågående avsnitt först):

```bash
touch STOP        # kör skriptet igen utan filen för att fortsätta
```

Hitta RSS-URL:er: i de flesta poddappar (»delad länk« / »kopiera flödesadress«)
eller sök `<podnamn> rss feed` — många producenter publicerar flödet öppet.
Sveriges Radios flöden ligger på formen `https://api.sr.se/api/podcast/<programid>.rss`.

## Krav och prestanda

- Mac med Apple silicon (M1/M2/M3/M4/M5). Ingen ffmpeg krävs — ljudkonverteringen
  använder macOS inbyggda `afconvert`.
- Python 3.10–3.13 (setup.sh hittar en lämplig; annars `brew install python@3.13`).
- Hastighet: ~50× realtid på M1 Max, betydligt snabbare på nyare kisel. Ett
  60-minutersavsnitt tar ~1 minut. Minnesbehov ~2–3 GB.
- För andra språk än svenska: Pianissimo är en svensk modell — transkript av
  andra språk blir dåligt.

## Diarisering av transkript (`diarisering/`)

Undermappen `diarisering/` lägger talarbyten på redan transkriberad text med
en lokal LLM: meningsdelning → filtrering av ASR-hallucinationer → LLM märker
varje mening med `Talare 1:`/`Talare 2:` → träningsklar text med rubrik
(monolog/dialog/ej_separerad). Se `diarisering/LÄSMIG.md` för detaljer.

```bash
cd diarisering
ollama pull qwen3:4b-instruct                       # ~3-4B, icke-tänkande
ollama create qwen3-4b-8k -f Modelfile.8k           # baka in 8k-kontext (krävs)
python3 diarize.py --config config.toml --input exempel --output ut_test   # röktest

# SR-transkript från piano-pipelinen (stripar metadatarubriken):
python3 transkript_till_diarize.py --input ../transcripts --output data \
    --program spanarna ekots-lördagsintervju --limit 5
python3 diarize.py --config config.toml --input data --output ut
```

Körningen är återstartbar (hoppas över klara dokument), stoppas mjukt med
Ctrl-C, och `--estimate --gen-tps 30` ger en tidsuppskattning före stor körning.
Vid storskalig körning: peka `base_url` i `config.toml` mot en vLLM-server
och kör med `--workers 16–64` — inget annat behöver ändras.

## Filstruktur

```
feeds.txt            dina RSS-URL:er (ignored av git)
transcripts/         resultat: <podd>/<id>.txt + <id>_sentenser.txt + index.tsv
state/progress.jsonl status per avsnitt (gör körningen återstartbar)
audio/               tillfälliga ljudfiler (raderas per avsnitt)
```

## Bra att veta

- **Licens och upphovsrätt**: transkribering av poddar för eget bruk är
  normalt okej, men viddelning eller vidareanvändning av transkript gäller
  poddens egna licenser. Modellens utdata: CC BY 4.0 (Klang AI).
- Om ett avsnitt misslyckas (död länk, korrupt fil) markeras det `failed` i
  state och hoppas över vid nästa körning.
- Modell och status: modellen cachas i `~/.cache/huggingface`.
