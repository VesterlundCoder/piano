#!/bin/bash
# Engångssetup för piano: Python-miljö + modell (759 MB ner till HF-cachen).
# Kräver macOS med Apple silicon (M1 eller senare). Kör: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# Välj en Python som parakeet-mlx/mlx har hjul för (3.10-3.13 fungerar;
# prova 3.13/3.12 först eftersom mlx ännu saknar hjul för allra senaste).
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
            PY="$cand"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo "Hittade ingen Python 3.10-3.13. Installera en, t.ex:" >&2
    echo "  brew install python@3.13" >&2
    exit 1
fi
echo "Använder $($PY --version) från $PY"

if [ ! -d .venv ]; then
    "$PY" -m venv .venv
fi
.venv/bin/pip install --upgrade pip wheel >/dev/null
.venv/bin/pip install -r requirements.txt

# Ladda ner modellen nu (hamnar i ~/.cache/huggingface) så första körningen
# inte behöver göra det.
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("KlangAI/pianissimo-sv-mlx-8bit")
print("modellen klar i HF-cache:", path)
EOF

cat <<'EOF'

Setup klar! Nästa steg:

  cp feeds.txt.example feeds.txt     # och klistra in RSS-URL:er
  .venv/bin/python transcribe_podcasts.py --limit 2    # röktest
  .venv/bin/python transcribe_podcasts.py              # kör på allvar

Stoppa mjukt när som helst:  touch STOP
EOF
