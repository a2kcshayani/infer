# Binary Format Specification — `infer` project

This document is the authoritative reference for every file that
`export_gpt2.py` writes.  The C++ engine must implement the reading side
to match exactly.  Any deviation produces silent garbage; the correctness
gate will catch it.

---

## 1. Global conventions

| Property | Value |
|---|---|
| Byte order | **Little-endian throughout** (the export script asserts `sys.byteorder == "little"`) |
| Float format | **fp32** (IEEE 754 single precision) for all weight and logit data |
| Tensor layout | **Row-major** (C order) matching HuggingFace's default PyTorch storage |
| Integer format | **int32** (signed, LE) for token ids; **uint32** (LE) for header words |

---

## 2. Weight file — `gpt2-small.bin`, `gpt2-tiny.bin`

### 2.1 Header (64 bytes, exactly)

The header is 16 consecutive LE uint32 words.  Words 7–15 are reserved
and written as zero; the C++ reader must not interpret them.

| Word index | Byte offset | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 0 | uint32 | `magic` | `0x32545047` — bytes `G`, `P`, `T`, `2` in LE order |
| 1 | 4 | uint32 | `version` | Format version; currently `1` |
| 2 | 8 | uint32 | `n_layer` | Number of transformer blocks |
| 3 | 12 | uint32 | `n_head` | Number of attention heads |
| 4 | 16 | uint32 | `n_embd` | Embedding / hidden dimension |
| 5 | 20 | uint32 | `n_ctx` | Maximum context length (`n_positions`) |
| 6 | 24 | uint32 | `vocab_size` | Vocabulary size |
| 7–15 | 28–60 | uint32 | *(reserved)* | Always zero |

Total: 16 × 4 = **64 bytes**.

### 2.2 Tensor stream

The first tensor begins at byte offset **64** (immediately after the
header).  Tensors are written back-to-back with **no padding** between
them.

The byte length of one tensor is `(product of shape dims) × 4`.

#### Running offset formula

```
offset(tensor_k) = 64 + sum_{j < k} (numel_j × 4)
```

where `numel_j` is the total element count of the j-th tensor in the
order below.

#### Tensor order

| # | Key (HF state_dict) | Shape | numel |
|---|---|---|---|
| 0 | `transformer.wte.weight` | `(vocab_size, n_embd)` | `vocab_size × n_embd` |
| 1 | `transformer.wpe.weight` | `(n_ctx, n_embd)` | `n_ctx × n_embd` |
| — | *Repeat for i = 0 … n_layer−1:* | | |
| 2+12i | `transformer.h.{i}.ln_1.weight` | `(n_embd,)` | `n_embd` |
| 3+12i | `transformer.h.{i}.ln_1.bias` | `(n_embd,)` | `n_embd` |
| 4+12i | `transformer.h.{i}.attn.c_attn.weight` | `(n_embd, 3·n_embd)` | `3·n_embd²` |
| 5+12i | `transformer.h.{i}.attn.c_attn.bias` | `(3·n_embd,)` | `3·n_embd` |
| 6+12i | `transformer.h.{i}.attn.c_proj.weight` | `(n_embd, n_embd)` | `n_embd²` |
| 7+12i | `transformer.h.{i}.attn.c_proj.bias` | `(n_embd,)` | `n_embd` |
| 8+12i | `transformer.h.{i}.ln_2.weight` | `(n_embd,)` | `n_embd` |
| 9+12i | `transformer.h.{i}.ln_2.bias` | `(n_embd,)` | `n_embd` |
| 10+12i | `transformer.h.{i}.mlp.c_fc.weight` | `(n_embd, 4·n_embd)` | `4·n_embd²` |
| 11+12i | `transformer.h.{i}.mlp.c_fc.bias` | `(4·n_embd,)` | `4·n_embd` |
| 12+12i | `transformer.h.{i}.mlp.c_proj.weight` | `(4·n_embd, n_embd)` | `4·n_embd²` |
| 13+12i | `transformer.h.{i}.mlp.c_proj.bias` | `(n_embd,)` | `n_embd` |
| — | *After all layers:* | | |
| 2+12·n_layer | `transformer.ln_f.weight` | `(n_embd,)` | `n_embd` |
| 3+12·n_layer | `transformer.ln_f.bias` | `(n_embd,)` | `n_embd` |

### 2.3 Conv1D weight storage (critical — read carefully)

HuggingFace GPT-2 implements the four projection layers using `Conv1D`,
not `nn.Linear`.  `Conv1D` stores its weight tensor as **(in_features,
out_features)** — the **transpose** of the `nn.Linear` convention
`(out_features, in_features)`.

The four affected tensors per layer are:

| Tensor | Stored shape |
|---|---|
| `attn.c_attn.weight` | `(n_embd, 3·n_embd)` — projects x→QKV |
| `attn.c_proj.weight` | `(n_embd, n_embd)` — projects attn output |
| `mlp.c_fc.weight` | `(n_embd, 4·n_embd)` — first MLP projection |
| `mlp.c_proj.weight` | `(4·n_embd, n_embd)` — second MLP projection |

**These shapes are stored AS-IS — no transpose is applied by the export
script.**  The C++ engine must account for this when constructing its
GEMM calls (i.e. it will compute `y = x @ W` where W is already in the
correct memory layout for that operation, not `y = W @ x`).

### 2.4 `lm_head` / weight tying

There is **no `lm_head` tensor** in the file.  GPT-2 ties the language
model head to the token embedding matrix.  The engine computes output
logits as:

```
logits = hidden_state @ wte^T     shape: (seq, vocab_size)
```

where `wte` is the embedding matrix loaded from position 0 in the tensor
stream.

### 2.5 64-byte alignment note

For the standard configurations `n_embd ∈ {128, 768}`, every tensor's
byte length is a multiple of 64:

- `128 × 4 = 512` bytes — multiple of 64 ✓
- `768 × 4 = 3072` bytes — multiple of 64 ✓

Any config that breaks this property (e.g. `n_embd = 100`) would require
explicit inter-tensor padding to maintain alignment.  The current format
spec does **not** insert padding; it is the caller's responsibility to
choose configs whose tensor sizes are multiples of 64, or to handle
unaligned loads explicitly in C++.

---

## 3. Fixture file — `fixture_gpt2.bin`, `fixture_tiny.bin`

These files are used for the correctness gate: the C++ engine runs the
same prompt and asserts `max_abs_diff(engine_logits, fixture_logits) < 1e-3`.

### Layout

All values little-endian.

| Field | Type | Count | Meaning |
|---|---|---|---|
| `seq` | int32 | 1 | Sequence length (number of tokens in FIXTURE_PROMPT) |
| `vocab` | int32 | 1 | Vocabulary size |
| token_ids | int32 | `seq` | Token ids for FIXTURE_PROMPT |
| logits | fp32 | `seq × vocab` | Full-sequence logits, row-major: row i = logit vector at position i |

Total bytes: `8 + seq×4 + seq×vocab×4`.

The logits are the **full-sequence output** (all positions, not just the
last token).  This lets the engine validate intermediate positions too.

**Determinism notes:**
- `model.eval()` is called before generating fixture logits; this
  disables dropout so results are identical across runs.
- `torch.no_grad()` is used during inference.
- For `gpt2-tiny` (random-init), `torch.manual_seed(0)` is called
  **immediately before** model construction to ensure byte-reproducible
  weight initialisation across independent runs of the export script.

---

## 4. Tokenizer files — `vocab.txt`, `merges.txt`

### 4.1 `vocab.txt`

- One token string per line, **id-ordered**: line 0 → token id 0, line 1
  → token id 1, …
- Encoding: UTF-8.
- The GPT-2 byte-level BPE vocabulary uses a `bytes_to_unicode()`
  remapping that avoids control characters; token strings therefore
  contain no literal whitespace or newline characters, making the
  line-per-token format unambiguous.
- **The C++ engine regenerates the `bytes_to_unicode()` table itself**
  from the standard GPT-2 algorithm.  This table is not stored in
  `vocab.txt`.

### 4.2 `merges.txt`

- BPE merge pairs in **rank order** (rank 0 first), one pair per line.
- Format of each line: `tokA tokB` (two space-separated token strings).
- The `#version:` comment line that HuggingFace's `save_vocabulary()`
  prepends is **stripped**; no other lines are removed or reordered.

---

## 5. GELU activation (correctness-critical)

HuggingFace GPT-2 uses the **tanh approximation** of GELU, sometimes
called `gelu_new`:

```
gelu(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

The C++ `ops/gelu.cpp` implementation **must match this formula exactly**.
Using the standard `erf`-based GELU or any other approximation will cause
the correctness gate (`max_abs_diff < 1e-3`) to fail, because the
difference accumulates across 12 MLP blocks.
