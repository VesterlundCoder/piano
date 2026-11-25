# Diarisering av transkript – instruktioner

Pipelinen tar transkriberad text som saknar talarbyten och ger tillbaka träningsklar text med rubrik och `Talare 1:` / `Talare 2:` per replik. En lokal LLM avgör vem som säger varje mening. Den får bara svara med en etikett per mening-ID och kan alltså aldrig skriva om texten.

```
indata (.txt .srt .vtt .jsonl)
  → meningsdelning
  → filter: Whisper-hallucinationer, loopar, upprepningar
  → redan märkt text ("Anna: ...")?        → används direkt, namn byts mot Talare N
  → känd monologkälla (ljudbok, föreläsning)? → monolog, ingen LLM
  → LLM märker 30 meningar åt gången med 6 meningars sammanhang
  → beslut: monolog / dialog / ej_separerad
  → ut/diariserat.jsonl + ut/kasserat.jsonl + ut/granskning.md
```

## Filer

| Fil | Vad |
|---|---|
| `diarize.py` | Själva pipelinen. Bara Python 3.11+, inga paket. |
| `prompt_sv.txt` | LLM:ens instruktioner för diariseringen. Här justerar du reglerna. |
| `config.toml` | Modell, server, trösklar, hallucinationslista, rubrikformat. |
| `exempel/` | Testfiler i alla format. |

## 1. Starta en lokal modell

Valfri OpenAI-kompatibel server fungerar. Med Ollama:

```bash
# Viktigt: Ollamas standardkontext är för liten. Sätt 8k innan servern startar.
export OLLAMA_CONTEXT_LENGTH=8192
ollama serve
# i en annan terminal:
ollama pull qwen3:14b
```

**Modellval:** ta en instruktionsmodell på 12–32B som är bra på nordiska språk, t.ex. Qwen3 14B eller en Gemma-modell. 7–8B fungerar men missar fler byten. Med en "tänkande" modell rensar skriptet bort `<think>`-blocket, men körningen blir betydligt långsammare. Välj en icke-tänkande variant om du kan.

Skriv modellnamnet i `config.toml` under `[llm] model`. För LM Studio eller llama.cpp ändrar du `base_url` (se kommentarerna i filen).

## 2. Testa på exemplen

```bash
python3 diarize.py --config config.toml --input exempel --output ut_test
```

Öppna `ut_test/granskning.md` och kontrollera att intervjun och nyhetsinslaget får rimliga talarbyten och att föreläsningen blir monolog. Raden `Textning.nu` i podden ska vara bortfiltrerad.

## 3. Organisera indata

Lägg filerna i **en mapp per källa**. Mappnamnet blir källan i rubriken:

```
data/
  podd/         *.txt eller *.srt
  nyheter/      *.srt / *.vtt
  ljudbok/      → hoppar över LLM:en (står i monolog_sources)
  forelasning/  → hoppar över LLM:en
  blandat.jsonl → en rad per dokument
```

JSONL-format, ett dokument per rad (`source` och `lang` är valfria):
```json
{"id": "podd-0042", "source": "podd", "lang": "sv", "text": "hela texten ..."}
{"id": "tv-17", "source": "nyheter", "segments": [{"text": "..."}, {"text": "..."}]}
```

## 4. Uppskatta tiden först

```bash
python3 diarize.py --input data --estimate --gen-tps 30
```

`--gen-tps` är modellens genereringshastighet på din dator. Ollama visar den med `ollama run qwen3:14b --verbose`.

**Räkna med att hela transkriptmängden inte går att köra på en laptop.** Ett anrop tar ungefär 10–20 sekunder, så 100 miljoner tokens transkript blir flera hundra timmar. Därför:

1. **Lokalt:** kör en pilot på några hundra dokument (`--sample 0.01 --limit 300`), granska resultatet och justera prompt och trösklar.
2. **Skarpt:** kör samma skript mot en vLLM-server, t.ex. en nod på LUMI. Det enda som ändras är `base_url` och `--workers 32`. Inget annat behöver ändras.

Transkripten är bara 1–2 % av korpusen. Låt inte diariseringen bli en flaskhals. Hellre `ej_separerad` på det som är svårt än veckor av handpåläggning.

## 5. Kör

```bash
python3 diarize.py --config config.toml --input data --output ut
```

- **Avbryt när du vill** (Ctrl-C). Kör samma kommando igen så fortsätter den. Klara dokument hoppas över.
- `--limit 200` kör de första 200, `--sample 0.05` kör ett slumpvis urval på 5 %.
- `--workers 1` på laptop. Fler parallella anrop hjälper bara om servern klarar det (vLLM, inte Ollama på laptop).

## 6. Utdata

**`ut/diariserat.jsonl`**, en rad per dokument:
```json
{"id": "...", "source": "podd", "type": "dialog", "n_speakers": 2, "language": "sv",
 "speaker_roles": {"Talare 1": "programledare", "Talare 2": "gäst, forskare"},
 "method": "llm", "text": "[källa: podd | typ: dialog | talare: 2 | språk: sv | transkriberat tal]\n\nTalare 1: ...\nTalare 2: ...",
 "stats": {...}}
```

Det är fältet `text` som går in i träningen. `speaker_roles` är LLM:ens gissning och bara metadata. Lägg den inte in i träningstexten.

**Typerna:**
- `monolog`: ingen talaretikett, bara stycken.
- `dialog`: `Talare N:` per replik, numrerade i den ordning de dyker upp.
- `ej_separerad`: LLM:en var för osäker (många "?", misslyckade anrop eller hög osäkerhet). Texten blir oetiketterad och rubriken säger `talare: okänt`. Det är medvetet: hellre ingen etikett än fel etikett.

**`ut/kasserat.jsonl`**: för korta dokument, och dokument där mer än 30 % filtrerats bort som hallucinationer eller loopar. Titta igenom dem stickprovsvis. En hög andel från en viss källa tyder på dålig ASR.

## 7. Kvalitetskontroll (viktigast)

Efter piloten: gå igenom `ut/granskning.md` (30 slumpvisa dokument) och räkna per dokument:

| Feltyp | Exempel | Åtgärd om det är vanligt |
|---|---|---|
| Påhittat byte | Retorisk fråga blir ny talare | Lägg till ett exempel i `prompt_sv.txt` under "svaga signaler" |
| Missat byte | Kort "Ja." slås ihop med föregående talare | Oftast acceptabelt. Annars större modell |
| Talare byter nummer | Gästen är T2 i början och T3 senare | Höj `context_sentences` till 10 |
| Fel typ | Föreläsning blir dialog | Lägg källan i `monolog_sources` |

**Tumregel för godkänt:** högst 1 fel per ~20 talarbyten, och inga påhittade talare i monologer. Klarar en källa inte det, kör den med typen satt till monolog eller låt den hamna som `ej_separerad`.

## 8. Justera

- **Prompten** (`prompt_sv.txt`): reglerna är skrivna för att hellre missa ett byte än hitta på ett. Lägg till konkreta exempel från er egen data när du ser återkommande fel. Ett bra exempel gör mer än en ny regel.
- **Hallucinationslistan** (`config.toml` → `hallucination_patterns`): lägg till fraser du ser återkomma i kasserad eller granskad text. Mönstren matchar bara i korta meningar (högst 12 ord), så riktiga meningar som råkar innehålla frasen överlever.
- **Rubriken** (`header_template`): ska ha samma format som rubrikerna i resten av korpusen.

## Felsökning

| Symptom | Orsak / åtgärd |
|---|---|
| `LLM-servern svarar inte` | Servern är inte startad eller `base_url` är fel. Prova `curl http://localhost:11434/v1/models` |
| Många `ej_separerad` | Kontexten är för liten (sätt `OLLAMA_CONTEXT_LENGTH`), modellen är för liten eller `chunk_sentences` för stort |
| `etiketter saknas för id` om och om igen | Modellen klarar inte formatet. Sänk `chunk_sentences` till 15–20 eller byt modell |
| Mycket långsamt | Tänkande modell, eller modellen ryms inte i GPU/minne. Ta en mindre kvantisering |
| `HTTP 400` direkt | Fel modellnamn i `config.toml` |
