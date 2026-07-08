# HANDOFF — CPU GPT-2 Inference Engine (`infer`)

**Repo:** https://github.com/a2kcshayani/infer · **Last session:** 2026-07-06 · **Machine:** Intel Core Ultra 9 288V (Lunar Lake, AVX2, no AVX-512, 8 cores 4P+4E), Windows 11 + WSL2 Ubuntu

---

## Goal

From-scratch GPT-2 (124M) inference engine in C++, CPU-only. Hero artifact: hand-written GEMM taken through naive → cache-tiled → AVX2 SIMD → OpenMP threaded, with a roofline analysis and per-optimization speedup table. Headline claim: **N× over own naive baseline, X% of llama.cpp throughput** — honest measured numbers, never "beat llama.cpp." Full plan lives in `C:\Users\kaise\CLAUDE.md`; approved execution plan in `~\.claude\plans\handoff-cpu-llm-agile-cocoa.md`.

## Current phase: **Phase 0 COMPLETE ✅ → Phase 1 next**

| Gate | Status |
|---|---|
| WSL toolchain (g++ 13.3, CMake 3.28, Py 3.12, venv `~/infer-venv` w/ torch 2.12.1+cpu, transformers 5.13) | ✅ |
| CMake skeleton builds in WSL; `./build/main` prints `infer: ok` | ✅ |
| `export_gpt2.py` produces all 6 outputs in WSL `~/models/` | ✅ |
| Sizes verified byte-exact vs `export/FORMAT.md` (497,759,296 B small / 27,843,136 B tiny; fixtures = 8 tokens × 50,257 fp32 logits, full-sequence) | ✅ |
| Repo public on GitHub, 3 clean commits | ✅ |

**Phase 1 (next session):** tensor (64B-aligned alloc) → mmap model loader → `gemm.h` + naive GEMM → layernorm / gelu_new / softmax / attention → forward pass → BPE tokenizer → **blocking correctness gate**: max-abs-diff vs fixture logits < 1e-3, tiny model first. Nothing proceeds past a red gate.

## Metrics scoreboard — ALL PROJECTIONS, NOTHING MEASURED YET

> ⚠️ These are Claude's order-of-magnitude *expectations* for this hardware, written down so we can check calibration later. Replace every cell with a measured number; delete this table's warning only when real data is in. Decode = batch-1, GPT-2-small fp32.

| Stage | Expected decode tok/s | Expected speedup vs naive |
|---|---|---|
| naive triple-loop | ~1–4 | 1× |
| + cache tiling | ~3–8 | ~2–3× |
| + AVX2 FMA | ~10–20 | ~5–10× |
| + OpenMP (measure 4 vs 8 threads — P/E asymmetry) | ~20–40 | ~10–30× |
| llama.cpp fp32 same machine | ~30–50 | target 60–90% parity |

Reasoning for the ceiling: decode is memory-bound; ~0.5 GB of fp32 weights touched per token against realistic ~50–70 GB/s sustained bandwidth ⇒ hard ceiling around 100 tok/s, so 20–40 is the honest expectation. This is why int8 quantization is the stretch goal — it quarters bytes-per-token.

## What took longer than expected / what bit us

1. **WSL DNS was dead** (auto-generated resolver `10.255.255.254` resolved nothing) — apt was fully broken. Fixed permanently: `generateResolvConf=false` in `/etc/wsl.conf`, static `1.1.1.1`/`8.8.8.8` in `/etc/resolv.conf`. If networking in WSL breaks again, check this first.
2. **transformers 5.x API break:** `save_vocabulary()` now emits a single `tokenizer.model`, not `merges.txt`. Fixed by pulling the original `merges.txt` via `hf_hub_download` (commit `d2dbec7`). Lesson: pin/probe HF APIs before relying on them.
3. **Minor:** a stray `^C` file from a terminal Ctrl-C got swept into a commit by `git add -A`; removed in `c8b8729`. Prefer explicit adds.

## Standing decisions & conventions (don't relitigate)

- **Workflow:** all non-trivial code goes through the `/lazy-dev` skill (`~/.claude/skills/lazy-dev/SKILL.md`): Opus lazy-senior architect (can veto/shrink scope) → Sonnet coder → Opus reviewer, ≤3 review rounds. Main agent orchestrates only. Saved in memory.
- **Layout:** source on Windows at `C:\Users\kaise\infer` (Claude's tools work natively); build/run in WSL against `/mnt/c/...`; model `.bin`s stay in WSL `~/models/` (drvfs mmap is slow — never benchmark against `/mnt/c` files).
- **Root in WSL without password:** `wsl -d Ubuntu -u root -- ...` works; no sudo password needed for installs.
- **Binary format locked** in `export/FORMAT.md`: 64-byte header (all tensors stay 64B-aligned for free), fp32 row-major LE, Conv1D weights stored (in,out) untransposed = ready-made GEMM B operand, wte/lm_head tied. **GELU must be tanh-approx `gelu_new`** or the gate fails.
- `.gitattributes` forces LF (CRLF would break WSL builds).

## Open questions for the user

1. **Benchmark power protocol:** laptop — plugged in + Windows "Best performance" during all measurements? Thermal throttling will otherwise make numbers irreproducible. Need a decided protocol before Phase 2 baselines.
2. **Week 4 fork** (decide start of Week 4 per plan §7): int8 quantization vs write-up + HTTP-server client. Leaning int8 *if* the gate has been green throughout — it's where the memory-bound win is.
3. **GitHub identity:** repo is under `a2kcshayani` but your git email may differ — if this is a resume artifact, confirm commits attribute to the account you want recruiters to see (check `git log --format='%ae'`).

## Things noticed but not asked about

- **Timeline check:** CLAUDE.md says ~4 weeks part-time; Phase 0 was scoped for Days 1–2 and took one session including environment surgery — on pace, but Phase 1 is the real volume. The correctness gate discipline is what protects the schedule.
- **8 logical = 8 physical (no HT)** on this CPU, but 4P+4E asymmetry means OpenMP `num_threads(8)` may *lose* to 4 P-cores on latency-bound decode. The plan already says measure both; expect the answer to be non-obvious and worth a paragraph in the README.
- **WSL as benchmark platform** is a defensibility caveat: numbers are WSL-Ubuntu-on-Windows, not bare Linux. Fine (llama.cpp comparison runs in the same WSL, apples-to-apples), but state it in the README's measurement protocol.
- **`fixture_gpt2.bin` and `fixture_tiny.bin` are byte-identical in size** (1,608,264) — expected (same prompt, same vocab), not a bug. Noting so nobody "fixes" it.
- **Windows Python 3.14 is a trap:** torch doesn't target it; everything Python runs in the WSL venv. Don't let a future session pip-install torch on the Windows side.
- The `~/models/` outputs are **regenerable but not in git** — a fresh clone needs one `export_gpt2.py` run (~2 min with cached HF weights, ~500MB download cold).

## Cold-start commands for next session

```powershell
# build + smoke test
wsl -d Ubuntu -- bash -c "cmake -S /mnt/c/Users/kaise/infer -B /mnt/c/Users/kaise/infer/build && cmake --build /mnt/c/Users/kaise/infer/build && /mnt/c/Users/kaise/infer/build/main"
# regenerate models/fixtures if ~/models/ is missing
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/kaise/infer/export && ~/infer-venv/bin/python export_gpt2.py"
```

First `/lazy-dev` target: **tensor + mmap model loader + naive GEMM + correctness-gate scaffold** as one chunk — it's the minimum slice that can consume a fixture and prove bytes→logits plumbing.
