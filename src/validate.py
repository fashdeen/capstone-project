"""Fail loudly, never silently. After each load/merge, assert the TA set is
EXACTLY the canonical 66 — not merely 66 of something."""
from ta_registry import CANONICAL_KEYS, normalise

class ValidationError(AssertionError):
    pass

def assert_ta_set(names, source_name):
    keys = {normalise(n) for n in names}; keys.discard(None)
    missing = CANONICAL_KEYS - keys     # expected but absent
    extra   = keys - CANONICAL_KEYS     # present but unexpected
    if missing or extra:
        lines = [f"[{source_name}] TA set != canonical {len(CANONICAL_KEYS)}."]
        if missing: lines.append(f"  MISSING ({len(missing)}): {sorted(missing)}")
        if extra:   lines.append(f"  EXTRA   ({len(extra)}): {sorted(extra)}")
        raise ValidationError("\n".join(lines))
    return True

def assert_row_count(df, expected, source_name):
    if len(df) != expected:
        raise ValidationError(f"[{source_name}] expected {expected} rows, got {len(df)}")
    return True
