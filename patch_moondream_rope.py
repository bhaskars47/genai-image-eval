#!/usr/bin/env python3
"""
patch_moondream_rope.py
-----------------------
One-shot patch for the RoPE (rotary positional embedding) IndexError that
occurs when running Moondream2 revision 2024-07-23 with transformers >= 4.42.

Root cause:
  In newer transformers, PhiAttention calls:
      cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
  During token-by-token generation with a KV cache, kv_seq_len == 1, so the
  RoPE table has only 1 row. But position_ids correctly reflects the absolute
  position in the sequence (e.g., 187), causing:
      IndexError: index 187 is out of bounds for dimension 0 with size 1

Fix:
  Change the seq_len argument to use the maximum required position index:
      seq_len=max(kv_seq_len, int(position_ids.max().item()) + 1)
  This ensures the RoPE table covers every position that will be indexed.

Usage:
  python patch_moondream_rope.py
"""

import glob
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the cached modeling_phi.py
# ---------------------------------------------------------------------------

CACHE_GLOB = str(
    Path.home()
    / ".cache/huggingface/modules/transformers_modules/vikhyatk/moondream2"
    / "*"
    / "modeling_phi.py"
)

matches = glob.glob(CACHE_GLOB)
if not matches:
    print("ERROR: Could not find modeling_phi.py in the HuggingFace cache.")
    print(f"Searched: {CACHE_GLOB}")
    print("Make sure you have loaded the moondream2 model at least once.")
    sys.exit(1)

# Take the most recently modified match (should be only one for 2024-07-23)
target = sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)[0]
print(f"Found: {target}")

# ---------------------------------------------------------------------------
# Read and patch
# ---------------------------------------------------------------------------

OLD = "self.rotary_emb(value_states, seq_len=kv_seq_len)"
NEW = "self.rotary_emb(value_states, seq_len=max(kv_seq_len, int(position_ids.max().item()) + 1))"

text = Path(target).read_text(encoding="utf-8")

if NEW in text:
    print("Already patched — nothing to do.")
    sys.exit(0)

if OLD not in text:
    print("ERROR: Expected pattern not found in modeling_phi.py.")
    print(f"Looking for: {OLD!r}")
    print("The file may have changed — inspect it manually.")
    sys.exit(1)

count = text.count(OLD)
patched = text.replace(OLD, NEW)
Path(target).write_text(patched, encoding="utf-8")

print(f"Patched {count} occurrence(s) of rotary_emb seq_len call.")
print("Done. Re-run your evaluation.")
