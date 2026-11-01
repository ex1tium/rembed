#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "torch",
#   "torchvision",
#   "pillow",
#   "sentence-transformers>=6.1",
#   "transformers @ git+https://github.com/huggingface/transformers",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Dev-only: EmbeddingGemma 2 golden + tokenizer fixture generator.

Independent reference for the embedding_gemma2 path: the published
sentence-transformers stack (Transformer -> mean Pooling -> Normalize) run
in fp32 on CPU, text inputs only. Writes
  testdata/golden-embeddinggemma-2.json           (input_ids + embedding)
  tokenizer/gemma/testdata/gemma2_fixture.json    (HF input_ids battery)
The golden set is GOLDEN_TEXTS plus Finnish/English household text and one
input longer than 1025 tokens, so the symmetric sliding window (|i-j| <= 512)
is exercised at its edge.

Usage: uv run models/gemma2_golden.py [model_dir_or_id]
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))
from convert import GOLDEN_TEXTS  # noqa: E402
from gemma_fixture import TEXTS as TOKENIZER_TEXTS  # noqa: E402

EXTRA_TEXTS = [
    # Finnish: inflection and compounds the FTS tokenizer cannot match.
    "missä on vesisulku",
    "Talon päävesisulku on kellarissa pesukoneen takana.",
    "Ostoslistalle maitoa, kaurajuomaa ja ruisleipää.",
    "Lasten neuvola-aika on torstaina kello 14.",
    # Task-prompt framing EmbeddingGemma 2 ships (query / document).
    "task: search result | query: where is the water shutoff valve",
    "title: none | text: The main water shutoff valve is in the basement behind the washer.",
    # > 1025 tokens: pairs beyond the 512 window on sliding layers.
    " ".join([GOLDEN_TEXTS[-1]] * 4),
]


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "models" / "embeddinggemma-2")

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = SentenceTransformer(ref, device="cpu", model_kwargs={"dtype": torch.float32})
    model.max_seq_length = 8192
    tok = model.tokenizer

    golden = []
    for text in GOLDEN_TEXTS + EXTRA_TEXTS:
        ids = tok(text)["input_ids"]
        emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)[0]
        emb = emb.astype(np.float32)
        golden.append({
            "text": text,
            "input_ids": ids,
            "embedding": [float(f"{v:.8g}") for v in emb],
        })
        print(f"{len(ids):5d} tokens  |e|={np.linalg.norm(emb):.6f}  {text[:48]!r}")
    out = REPO_ROOT / "testdata" / "golden-embeddinggemma-2.json"
    out.write_text(json.dumps({"model": "google/embeddinggemma-2", "cases": golden}, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(golden)} cases, dim {len(golden[0]['embedding'])})")

    cases = [{"text": t, "input_ids": tok(t)["input_ids"]} for t in TOKENIZER_TEXTS]
    fix = REPO_ROOT / "tokenizer" / "gemma" / "testdata" / "gemma2_fixture.json"
    fix.write_text(json.dumps({"model": "google/embeddinggemma-2", "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {fix} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
