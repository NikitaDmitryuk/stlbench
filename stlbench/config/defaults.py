"""Defaults used when options are not set in TOML."""

# Orientation mode for scale math if --orientation is omitted (resin: free rotation search).
ORIENTATION_MODE_DEFAULT = "free"
ORIENTATION_SAMPLES_DEFAULT = 4096
ORIENTATION_SEED_DEFAULT = 0

# Part-to-part gap when no config is provided (matches former PackingSection default).
DEFAULT_PACKING_GAP_MM = 2.0
