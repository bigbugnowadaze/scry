"""
kitchen_sink.py — non-interactive trainer. No menu, no prompts. Run it,
walk away, come back to a trained substrate.

USAGE:
    python kitchen_sink.py [light|medium|heavy|max]

    Default: medium

TO RUN IN BACKGROUND ON EC2 (SURVIVES SSH DISCONNECT):
    nohup python kitchen_sink.py heavy > aurexis.log 2>&1 &
    disown

CHECK PROGRESS:
    tail -f aurexis.log

CHECK IF STILL RUNNING:
    ps aux | grep kitchen_sink

KILL IT:
    pkill -f kitchen_sink

This avoids tmux entirely.
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ARCHIVE_DIR = HERE / "aurexis_archive"

# Required modules
from aurexis_v3 import Archive
from sources import SOURCES, has_key, init_url_dedup, flush_url_dedup
from aurexis_train import auto_discover_rules
from aurexis_fast import fast_run_source

# Optional modules
try:
    from aurexis_hf import HF_SOURCES, has_hf_token
    HAS_HF = True
except Exception as e:
    HF_SOURCES = []
    has_hf_token = lambda: False
    HAS_HF = False
    print(f"  [kitchen-sink] HF disabled: {e}")

try:
    from aurexis_batch import bulk_ingest, bulk_ingest_simple
    HAS_BATCH = True
except Exception as e:
    HAS_BATCH = False
    print(f"  [kitchen-sink] batch disabled: {e}")

try:
    from aurexis_aws import AWS_SOURCES, HAS_BOTO3
    HAS_AWS = HAS_BOTO3 and bool(AWS_SOURCES)
except Exception as e:
    AWS_SOURCES = []
    HAS_AWS = False
    print(f"  [kitchen-sink] AWS disabled: {e}")

try:
    from aurexis_eval import (set_global_seed, save_seed,
                               full_evaluation_report, ensure_splits)
    HAS_EVAL = True
except Exception as e:
    HAS_EVAL = False
    print(f"  [kitchen-sink] eval disabled: {e}")


def run_source_safe(fn, n, label, archive):
    """Run one source, catching all errors so one bad source doesn't kill
    the whole job."""
    try:
        if HAS_BATCH:
            return bulk_ingest(archive, fn(n), batch_size=32, max_images=n)
        else:
            return bulk_ingest_simple(archive, fn(n), max_images=n)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  [{label}] failed: {type(e).__name__}: {str(e)[:120]}")
        return None


def main():
    intensity = (sys.argv[1].lower() if len(sys.argv) > 1 else "medium")
    if intensity not in ("light", "medium", "heavy", "max"):
        print(f"unknown intensity '{intensity}'; using medium")
        intensity = "medium"

    presets = {
        "light":  {"hf": 100,  "aws": 200,  "api": 30},
        "medium": {"hf": 300,  "aws": 500,  "api": 60},
        "heavy":  {"hf": 800,  "aws": 2000, "api": 100},
        "max":    {"hf": 2000, "aws": 5000, "api": 200},
    }
    cfg = presets[intensity]

    print("=" * 64)
    print(f"  AUREXIS KITCHEN SINK  [{intensity.upper()}]")
    print(f"  HF per dataset:    {cfg['hf']}")
    print(f"  AWS per bucket:    {cfg['aws']}")
    print(f"  Public per source: {cfg['api']}")
    print(f"  HAS_HF={HAS_HF}  HAS_AWS={HAS_AWS}  "
          f"HAS_BATCH={HAS_BATCH}  HAS_EVAL={HAS_EVAL}")
    print(f"  HF token: {'YES' if has_hf_token() else 'NO'}")
    print("=" * 64)

    archive = Archive(ARCHIVE_DIR)
    init_url_dedup(ARCHIVE_DIR)

    if HAS_EVAL:
        set_global_seed(42)
        save_seed(ARCHIVE_DIR, 42)
        ensure_splits(archive)
        print(f"  seed locked to 42, splits assigned")

    n_start = len(archive._cache)
    t_start = time.time()

    # ============================================================
    # PHASE 1: AWS Open Data (fastest if running on EC2 in us-east-1)
    # ============================================================
    if HAS_AWS:
        print(f"\n{'#'*64}\n# PHASE 1: AWS Open Data ({len(AWS_SOURCES)} buckets)\n{'#'*64}")
        for i, (label, _, fn) in enumerate(AWS_SOURCES):
            print(f"\n  [AWS {i+1}/{len(AWS_SOURCES)}] {label}")
            try:
                run_source_safe(fn, cfg['aws'], label, archive)
            except KeyboardInterrupt:
                print("\n  interrupted; saving progress and continuing")
                archive._save_measurements()
                break
    else:
        print("\n  [PHASE 1 SKIPPED: AWS sources not available]")

    # ============================================================
    # PHASE 2: HuggingFace
    # ============================================================
    if HAS_HF and HF_SOURCES:
        print(f"\n{'#'*64}\n# PHASE 2: HuggingFace ({len(HF_SOURCES)} datasets)\n{'#'*64}")
        for i, (label, key_req, fn) in enumerate(HF_SOURCES):
            if key_req != "no key" and not has_hf_token():
                print(f"\n  [HF {i+1}/{len(HF_SOURCES)}] SKIP {label} "
                      f"(needs token)")
                continue
            print(f"\n  [HF {i+1}/{len(HF_SOURCES)}] {label}")
            try:
                run_source_safe(fn, cfg['hf'], label, archive)
            except KeyboardInterrupt:
                archive._save_measurements()
                break
    else:
        print("\n  [PHASE 2 SKIPPED: HuggingFace not available]")

    # ============================================================
    # PHASE 3: Public APIs (slower, rate-limited, but free and varied)
    # ============================================================
    print(f"\n{'#'*64}\n# PHASE 3: Public APIs\n{'#'*64}")
    api_idx = 0
    for stage_key, sources in SOURCES.items():
        for label, key_req, fn in sources:
            if key_req != "no key" and "DEMO_KEY" not in key_req:
                if not has_key(key_req):
                    continue
            api_idx += 1
            print(f"\n  [API {api_idx}] {label}")
            try:
                fast_run_source(fn, cfg['api'], label, archive,
                                workers=8, online_cluster=False)
            except KeyboardInterrupt:
                archive._save_measurements()
                break
            except Exception as e:
                print(f"  failed: {type(e).__name__}: {str(e)[:120]}")

    flush_url_dedup()

    # ============================================================
    # PHASE 4: Auto-discover rules
    # ============================================================
    print(f"\n{'#'*64}\n# PHASE 4: Auto-discover rules\n{'#'*64}")
    try:
        rules = auto_discover_rules(archive, archive_dir=ARCHIVE_DIR,
                                     verbose=True)
        added = 0
        for name, pred in rules.items():
            if name not in archive.library:
                archive.library[name] = pred
                added += 1
        archive.save_library()
        print(f"\n  added {added} new rules (library total: "
              f"{len(archive.library)})")
    except Exception as e:
        print(f"  rule discovery failed: {type(e).__name__}: {e}")

    # ============================================================
    # PHASE 5: Evaluation report
    # ============================================================
    print(f"\n{'#'*64}\n# PHASE 5: Evaluation\n{'#'*64}")
    if HAS_EVAL:
        try:
            full_evaluation_report(archive, ARCHIVE_DIR)
        except Exception as e:
            print(f"  eval failed: {type(e).__name__}: {e}")

    # Summary
    elapsed = time.time() - t_start
    n_added = len(archive._cache) - n_start
    print(f"\n{'='*64}")
    print(f"  KITCHEN SINK COMPLETE")
    print(f"  Added {n_added} images in {elapsed/60:.1f} minutes")
    print(f"  Archive total: {len(archive._cache)}")
    print(f"  Library total: {len(archive.library)} predicates")
    print(f"  Rate: {n_added/max(elapsed/60, 0.01):.0f} images/minute")
    print(f"{'='*64}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  interrupted by user. archive and library saved.")
