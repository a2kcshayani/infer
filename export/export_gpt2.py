"""
export_gpt2.py — dump GPT-2 weights + tokenizer + correctness fixtures
to ~/models/ in the flat binary format documented in FORMAT.md.

Run from WSL with python deps available:
    python export/export_gpt2.py

No argparse; paths are literals; output dir is Path.home()/"models".
"""

import os
import sys
import struct
import shutil
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants (locked — match FORMAT.md)
# ---------------------------------------------------------------------------
OUT             = Path.home() / "models"
MAGIC           = 0x32545047   # bytes b"GPT2" packed as little-endian uint32
VERSION         = 1
FIXTURE_PROMPT  = "Hello, I am a language model,"
SEED            = 0


# ---------------------------------------------------------------------------
# write_bin
# ---------------------------------------------------------------------------
def write_bin(path: Path, model: GPT2LMHeadModel, cfg) -> None:
    """Write weights to the flat binary format defined in FORMAT.md.

    Little-endian throughout; all tensors fp32 row-major.
    Header is exactly 64 bytes (16 × LE uint32).
    First tensor starts at byte offset 64; no padding between tensors.

    NOTE on Conv1D weight shape:
        HuggingFace GPT-2 implements Q/K/V projection, output projection,
        and both MLP layers using Conv1D (not nn.Linear).  Conv1D stores
        its weight as (in_features, out_features) — the TRANSPOSE of the
        nn.Linear convention.  We store these four weights AS-IS (no
        transpose) so that the C++ engine can load them without any
        re-layout step.  The four affected keys per layer are:
            attn.c_attn.weight  shape (n_embd, 3*n_embd)
            attn.c_proj.weight  shape (n_embd, n_embd)
            mlp.c_fc.weight     shape (n_embd, 4*n_embd)
            mlp.c_proj.weight   shape (4*n_embd, n_embd)
        The C++ GEMM call must account for this when computing each op.
    """
    assert sys.byteorder == "little", (
        "export_gpt2.py must run on a little-endian host (WSL/Linux x86-64)"
    )

    n_layer    = cfg.n_layer
    n_head     = cfg.n_head
    n_embd     = cfg.n_embd
    n_ctx      = cfg.n_positions
    vocab_size = cfg.vocab_size

    sd = model.state_dict()

    with open(path, "wb") as f:
        # --- 64-byte header: 16 × LE uint32 ---
        # Words 0-6 carry metadata; words 7-15 are reserved (zero).
        header = struct.pack(
            "<16I",
            MAGIC,       # word 0
            VERSION,     # word 1
            n_layer,     # word 2
            n_head,      # word 3
            n_embd,      # word 4
            n_ctx,       # word 5
            vocab_size,  # word 6
            0, 0, 0, 0, 0, 0, 0, 0, 0,  # words 7-15
        )
        assert len(header) == 64
        f.write(header)

        def _write(key: str) -> None:
            t = sd[key].detach().to(torch.float32).contiguous().numpy()
            t.tofile(f)

        # --- Token + position embeddings ---
        _write("transformer.wte.weight")    # (vocab_size, n_embd)
        _write("transformer.wpe.weight")    # (n_ctx,      n_embd)

        # --- Per-layer weights ---
        for i in range(n_layer):
            pfx = f"transformer.h.{i}."
            _write(pfx + "ln_1.weight")          # (n_embd,)
            _write(pfx + "ln_1.bias")             # (n_embd,)
            _write(pfx + "attn.c_attn.weight")   # (n_embd,   3*n_embd)  Conv1D
            _write(pfx + "attn.c_attn.bias")     # (3*n_embd,)
            _write(pfx + "attn.c_proj.weight")   # (n_embd,   n_embd)    Conv1D
            _write(pfx + "attn.c_proj.bias")     # (n_embd,)
            _write(pfx + "ln_2.weight")          # (n_embd,)
            _write(pfx + "ln_2.bias")            # (n_embd,)
            _write(pfx + "mlp.c_fc.weight")      # (n_embd,   4*n_embd)  Conv1D
            _write(pfx + "mlp.c_fc.bias")        # (4*n_embd,)
            _write(pfx + "mlp.c_proj.weight")    # (4*n_embd, n_embd)    Conv1D
            _write(pfx + "mlp.c_proj.bias")      # (n_embd,)

        # --- Final layer norm (no lm_head — tied to wte; see FORMAT.md) ---
        _write("transformer.ln_f.weight")   # (n_embd,)
        _write("transformer.ln_f.bias")     # (n_embd,)

    size = path.stat().st_size
    print(f"  wrote {path}  ({size:,} bytes)")


# ---------------------------------------------------------------------------
# write_fixture
# ---------------------------------------------------------------------------
def write_fixture(path: Path, model: GPT2LMHeadModel, tok: GPT2Tokenizer) -> None:
    """Write a correctness fixture: token ids + full-sequence logits.

    File layout (all LE):
        int32   seq          — sequence length
        int32   vocab        — vocabulary size
        seq × int32          — token ids
        (seq × vocab) × fp32 — logits, row-major [position, vocab_token]

    model.eval() + torch.no_grad() ensure dropout is off so logits are
    deterministic across runs.
    """
    ids = tok(FIXTURE_PROMPT, return_tensors="pt").input_ids   # [1, seq]
    seq   = ids.shape[1]
    vocab = model.config.vocab_size

    # model.eval() already called by caller; guard with no_grad for safety.
    model.eval()
    with torch.no_grad():
        logits = model(ids).logits   # [1, seq, vocab]

    with open(path, "wb") as f:
        f.write(struct.pack("<2i", seq, vocab))
        ids[0].to(torch.int32).numpy().tofile(f)
        logits[0].to(torch.float32).contiguous().numpy().tofile(f)

    size = path.stat().st_size
    print(f"  wrote {path}  ({size:,} bytes)")


# ---------------------------------------------------------------------------
# write_tokenizer
# ---------------------------------------------------------------------------
def write_tokenizer(vocab_path: Path, merges_path: Path, tok: GPT2Tokenizer) -> None:
    """Write vocab.txt (id-ordered, one token per line) and merges.txt
    (rank-ordered BPE merge pairs, "#version:" header line stripped).

    The C++ engine regenerates the standard GPT-2 bytes_to_unicode() table
    itself — it is NOT stored in these files.

    Byte-level encoding guarantees no whitespace or newline characters
    appear inside individual token strings, so a simple line-per-token
    format is unambiguous.
    """
    # --- vocab.txt ---
    vocab      = tok.get_vocab()                    # {token_str: id}
    id_to_tok  = {v: k for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(len(id_to_tok)):
            f.write(id_to_tok[i] + "\n")
    print(f"  wrote {vocab_path}  ({vocab_path.stat().st_size:,} bytes)")

    # --- merges.txt ---
    # save_vocabulary writes merges.txt with a "#version: ..." first line;
    # we drop only that one comment line and copy the rest verbatim.
    src = hf_hub_download(repo_id="gpt2", filename="merges.txt")
    with open(src, "r", encoding="utf-8") as fin, \
         open(merges_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("#version:"):
                continue
            fout.write(line)
    print(f"  wrote {merges_path}  ({merges_path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # --- GPT-2 small (124 M) ---
    print("Loading gpt2-small …")
    small = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    write_bin(OUT / "gpt2-small.bin", small, small.config)
    write_fixture(OUT / "fixture_gpt2.bin", small, tok)

    # --- Tokenizer files (shared by both models) ---
    write_tokenizer(OUT / "vocab.txt", OUT / "merges.txt", tok)

    # --- GPT-2 tiny (random-init, 2-layer/128-embd oracle) ---
    # torch.manual_seed() immediately before construction — any intervening
    # call (e.g. tokenizer ops above) would shift the RNG state, so the seed
    # is set here, not at module level, to keep byte-reproducibility.
    torch.manual_seed(SEED)
    tiny_cfg = GPT2Config(
        n_layer=2,
        n_embd=128,
        n_head=4,
        vocab_size=50257,
        n_positions=1024,
    )
    tiny = GPT2LMHeadModel(tiny_cfg).eval()
    write_bin(OUT / "gpt2-tiny.bin", tiny, tiny_cfg)
    write_fixture(OUT / "fixture_tiny.bin", tiny, tok)

    print("Done.")


if __name__ == "__main__":
    main()
