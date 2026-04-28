"""
cli.py — the Aurexis menu-driven runner.

Run with:  python cli.py     (or double-click aurexis.bat on Windows)

Top-level menu:
  [1] Browse sources by stage and pull images
  [2] Run full curriculum (all stages, default counts)
  [3] Show what the substrate has learned
  [4] Teach a concept by labeling examples
  [5] Quit

No raw images are written to disk — only atom values, rich features,
and learned predicates persist in the archive.
"""

import sys
import os
import time
from pathlib import Path

from aurexis_v3 import (
    Archive, ATOMS, threshold,
    cluster_archive, cluster_summary, find_anomalies, pca_axes,
    discover_concepts, learn_predicate_from_labels, explain,
    mean_brightness, contrast, edge_density, dark_fraction,
    bright_fraction, color_entropy, hog_energy, lbp_uniformity, color_diversity,
)
from sources import SOURCES, STAGE_ORDER, STAGE_NAMES, has_key
from aurexis_fast import fast_run_source, fast_run_many
from aurexis_train import (augment_pil, enriched_ingest, auto_discover_rules,
                           analyze_gaps, training_state_report)

# Optional modules — app works without them, but features are limited
try:
    from aurexis_hf import HF_SOURCES, has_hf_token
    HAS_HF = True
except ImportError as e:
    HF_SOURCES = []
    has_hf_token = lambda: False
    HAS_HF = False
    print(f"  [info] HuggingFace streaming disabled: {e}")

try:
    from aurexis_batch import bulk_ingest, bulk_ingest_simple
    HAS_BATCH = True
except ImportError as e:
    HAS_BATCH = False
    print(f"  [info] Batch processing disabled: {e}")

try:
    from aurexis_camera import live_camera, HAS_CV2
    HAS_CAMERA = True
except ImportError as e:
    HAS_CAMERA = False
    HAS_CV2 = False
    print(f"  [info] Live camera disabled: {e}")

# Academic-discipline layer — splits, seeds, learning curves, attribution
try:
    from aurexis_eval import (set_global_seed, get_global_seed,
                               load_seed, save_seed, ensure_splits,
                               split_counts, source_attribution_report,
                               print_learning_curve, full_evaluation_report)
    HAS_EVAL = True
except ImportError as e:
    HAS_EVAL = False
    print(f"  [info] Evaluation layer disabled: {e}")

# Global runtime flag — toggled by menu option [a]
AUGMENT_MODE = {"on": False, "n": 4}

HERE        = Path(__file__).parent.resolve()
ARCHIVE_DIR = HERE / "aurexis_archive"

archive = Archive(ARCHIVE_DIR)

# Initialize URL-level dedup so we don't re-download images we've seen
import sources as _sources
_sources.init_url_dedup(ARCHIVE_DIR)

# Initialize academic-discipline layer
if HAS_EVAL:
    # Restore previously-saved seed if any
    saved_seed = load_seed(ARCHIVE_DIR)
    if saved_seed is not None:
        set_global_seed(saved_seed)
        print(f"  seed: restored {saved_seed} from previous run")
    # Tag any previously-untagged images with train/test split
    n_added = ensure_splits(archive)
    if n_added:
        print(f"  splits: assigned {n_added} previously-untagged images")
    sc = split_counts(archive)
    if sc:
        print(f"  splits: train={sc.get('train',0)}  test={sc.get('test',0)}")


# ============================================================
# UI HELPERS
# ============================================================

def banner():
    n_imgs = len(archive._cache)
    n_lib = len(archive.library)
    sources_seen = set()
    for m in archive._cache.values():
        sources_seen.add(m.get("_source", "?"))
    print()
    print("=" * 64)
    print("  AUREXIS — offline measurement substrate")
    print("=" * 64)
    print(f"  archive: {n_imgs} images   library: {n_lib} learned predicates")
    print(f"  atoms:   {len(ATOMS)}   sources seen: {len(sources_seen)}")
    print("=" * 64)


def prompt(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{msg}{suffix}> ").strip()
    return val if val else (str(default) if default is not None else "")


def prompt_int(msg, default=20, lo=1, hi=200):
    while True:
        v = prompt(msg, default)
        try:
            n = int(v)
            if lo <= n <= hi: return n
            print(f"  please enter {lo}..{hi}")
        except ValueError:
            print("  please enter a number")


# ============================================================
# CORE: stream a source, ingest, recluster
# ============================================================

def run_source(source_fn, n, label, fast=False):
    if fast:
        new, _, _ = fast_run_source(source_fn, n, label, archive,
                                     workers=8, online_cluster=False)
        return new

    aug_label = " (AUG x{})".format(AUGMENT_MODE["n"]+1) if AUGMENT_MODE["on"] else ""
    print(f"\n  fetching {n} from '{label}'{aug_label} ... (Ctrl+C to stop)")
    new_count = 0; dup_count = 0; err_count = 0
    t0 = time.time()
    try:
        for i, item in enumerate(source_fn(n)):
            try:
                pil_img, alias, src = item
            except Exception:
                err_count += 1
                continue
            try:
                if AUGMENT_MODE["on"]:
                    n_new = enriched_ingest(archive, pil_img, alias, src,
                                            n_augment=AUGMENT_MODE["n"])
                    new_count += n_new
                    print(f"    [{i+1:>3}/{n}] {src:>14s}  "
                          f"+{n_new} augmented samples  {alias}", flush=True)
                else:
                    h, is_new = archive.ingest_pil(pil_img, alias, src)
                    if is_new: new_count += 1
                    else: dup_count += 1
                    print(f"    [{i+1:>3}/{n}] {src:>14s}  "
                          f"{'NEW' if is_new else 'dup'}  {alias}", flush=True)
            except Exception as e:
                err_count += 1
                print(f"    [{i+1:>3}/{n}] ingest error: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\n  stopped by user.")

    dt = time.time() - t0
    print(f"\n  done: {new_count} new, {dup_count} duplicates, "
          f"{err_count} errors in {dt:.1f}s")
    return new_count


def maybe_recluster(min_total=4):
    if len(archive._cache) < min_total:
        return
    print("\n  re-clustering with new data ...")
    try:
        best_k, assignments = cluster_archive(archive, k_range=(3, 8))
        print(f"  best K = {best_k}")
        groups = cluster_summary(archive, assignments)
        for cid, members in list(groups.items())[:8]:
            preview = ", ".join(members[:5])
            more = f" ... (+{len(members)-5})" if len(members) > 5 else ""
            print(f"     C{cid} ({len(members):3d}): {preview}{more}")
        # Update the persistent learned-predicate library
        discovered = discover_concepts(archive, assignments)
        for name, pred in discovered.items():
            archive.library[name] = pred
        archive.save_library()
    except Exception as e:
        print(f"  reclustering failed: {type(e).__name__}: {e}")


# ============================================================
# MENU 1 — BROWSE SOURCES
# ============================================================

def menu_browse():
    while True:
        print("\nSTAGES:")
        for i, key in enumerate(STAGE_ORDER):
            print(f"  [{i}] {STAGE_NAMES[key]}")
        print("  [b] back")
        c = prompt("choose stage").lower()
        if c == "b" or c == "": return
        try:
            idx = int(c)
            stage_key = STAGE_ORDER[idx]
        except (ValueError, IndexError):
            print("  invalid choice"); continue

        sources = SOURCES[stage_key]
        while True:
            print(f"\n{STAGE_NAMES[stage_key]} — sources:")
            for i, (label, key_req, _) in enumerate(sources):
                avail = ""
                if key_req == "no key":
                    avail = "[ready]"
                elif "DEMO_KEY" in key_req:
                    avail = "[ready, set NASA_API_KEY for higher rate]"
                else:
                    avail = "[ready]" if has_key(key_req) else f"[needs {key_req}]"
                print(f"  [{i}] {label:38s} {avail}")
            print("  [b] back")
            c = prompt("choose source").lower()
            if c == "b" or c == "": break
            try:
                sidx = int(c)
                label, key_req, fn = sources[sidx]
            except (ValueError, IndexError):
                print("  invalid choice"); continue
            n = prompt_int("how many images?", default=20, lo=1, hi=200)
            run_source(fn, n, label)
            maybe_recluster()


# ============================================================
# MENU 2 — FULL CURRICULUM
# ============================================================

CURRICULUM = [
    ("stage_0_calibrate",       30,  ["Synthetic uniform noise",
                                       "Synthetic gradient",
                                       "Synthetic perlin-like"]),
    ("stage_1_easy_photos",     30,  ["Lorem Picsum"]),
    ("stage_2_diverse_photos",  20,  ["Wikimedia Commons random",
                                       "Openverse 'nature'"]),
    ("stage_3_wildlife",        15,  ["iNaturalist Aves (birds)",
                                       "iNaturalist Plantae",
                                       "GBIF birds"]),
    ("stage_4_art_culture",     15,  ["Met Museum 'painting'",
                                       "Art Institute Chicago",
                                       "Cleveland Museum random"]),
    ("stage_5_astronomy",       10,  ["NASA APOD random history",
                                       "NASA Image Library 'galaxy'",
                                       "JWST via Wikimedia"]),
    ("stage_6_earth_above",     10,  ["NASA GIBS true-color tiles",
                                       "NOAA GOES weather satellite",
                                       "OpenStreetMap tiles z=5"]),
    ("stage_7_synthetic_faces", 5,   ["thispersondoesnotexist"]),
]


def menu_curriculum():
    print("\nFULL CURRICULUM — all stages, sensible defaults.")
    print("This will hit several public APIs courteously.")
    print("Counts are conservative; you can re-run any stage from menu [1].")
    if prompt("proceed? (y/n)", "y").lower() != "y":
        return
    for stage_key, n, labels in CURRICULUM:
        print(f"\n{'#' * 64}")
        print(f"# {STAGE_NAMES[stage_key]}")
        print(f"{'#' * 64}")
        sources = SOURCES[stage_key]
        for label in labels:
            for src_label, key_req, fn in sources:
                if src_label != label: continue
                if key_req != "no key" and "DEMO_KEY" not in key_req:
                    if not has_key(key_req):
                        print(f"  skip '{label}' ({key_req} not set)")
                        break
                run_source(fn, n, src_label)
                break
        maybe_recluster()
    print("\n=" * 1 + " curriculum complete =" * 1)


# ============================================================
# MENU 3 — SHOW STATE
# ============================================================

def menu_curriculum_fast():
    """Like menu_curriculum but runs sources within each stage in parallel."""
    print("\nFAST CURRICULUM — concurrent multi-source per stage.")
    print("Faster than [2] because sources within a stage run in parallel.")
    if prompt("proceed? (y/n)", "y").lower() != "y":
        return
    for stage_key, n, labels in CURRICULUM:
        print(f"\n{'#' * 64}")
        print(f"# {STAGE_NAMES[stage_key]}")
        print(f"{'#' * 64}")
        sources = SOURCES[stage_key]
        # Build a list of primed generators for this stage
        gens = []
        for label in labels:
            for src_label, key_req, fn in sources:
                if src_label != label: continue
                if key_req != "no key" and "DEMO_KEY" not in key_req:
                    if not has_key(key_req):
                        print(f"  skip '{label}' ({key_req} not set)")
                        break
                gens.append((fn(n), src_label))
                break
        if gens:
            fast_run_many(gens, archive, batch_write_every=25)
            maybe_recluster()
    print("\n=" * 1 + " fast curriculum complete =" * 1)



    banner()
    print()
    if not archive._cache:
        print("  archive is empty. run a source first.")
        return
    sources_seen = {}
    for m in archive._cache.values():
        s = m.get("_source", "?")
        sources_seen[s] = sources_seen.get(s, 0) + 1
    print("  by source:")
    for s, c in sorted(sources_seen.items(), key=lambda kv: -kv[1]):
        print(f"     {s:18s} {c:4d}")

    if archive.library:
        print("\n  learned predicates:")
        for name, pred in archive.library.items():
            hits = archive.query(pred)
            print(f"     {name:18s} ({len(hits):4d} matches)  "
                  f"{pred.name[:60]}")

    print("\n  example rule queries:")
    rules = [
        ("dark images",         threshold(mean_brightness, "<", 0.25)),
        ("bright images",       threshold(mean_brightness, ">", 0.65)),
        ("high-contrast",       threshold(contrast, ">", 0.20)),
        ("monochrome-ish",      threshold(color_diversity, "<", 4.0)),
    ]
    for label, p in rules:
        n = len(archive.query(p))
        print(f"     {label:24s} {n:4d}")

    print("\n  anomalies (most unusual 5):")
    try:
        anom = find_anomalies(archive, contamination=0.10)
        for h, alias, score in anom[:5]:
            print(f"     {alias:30s} {score:+.3f}")
    except Exception as e:
        print(f"     anomaly detection failed: {e}")


# ============================================================
# MENU 4 — TEACH
# ============================================================

def menu_teach():
    if not archive._cache:
        print("  archive is empty. run a source first.")
        return
    print("\n  Teach a concept by listing aliases (comma-separated).")
    print("  Hint: type a few characters to filter.")
    flt = prompt("filter aliases by substring (blank = show first 30)").strip()
    aliases = sorted(m["_alias"] for m in archive._cache.values()
                     if flt.lower() in m["_alias"].lower())
    if not aliases:
        print("  no matches"); return
    for a in aliases[:30]:
        print(f"     {a}")
    if len(aliases) > 30:
        print(f"     ... +{len(aliases)-30} more")

    name     = prompt("concept name (e.g. natural_texture)")
    pos_str  = prompt("positive aliases, comma-separated")
    neg_str  = prompt("negative aliases, comma-separated")
    if not (name and pos_str and neg_str):
        print("  missing input — abort"); return

    def resolve(alias):
        for h, m in archive._cache.items():
            if m.get("_alias") == alias.strip():
                return h
        raise KeyError(alias)

    try:
        pos = [resolve(a) for a in pos_str.split(",")]
        neg = [resolve(a) for a in neg_str.split(",")]
    except KeyError as e:
        print(f"  alias not in archive: {e}"); return

    pred = learn_predicate_from_labels(name, archive, pos, neg)
    archive.library[name] = pred
    archive.save_library()
    print(f"\n  learned: {pred.name}")
    hits = archive.query(pred)
    print(f"  matches {len(hits)} of {len(archive._cache)} images:")
    for h, m in sorted(hits, key=lambda hm: hm[1]["_alias"])[:30]:
        print(f"     {m['_alias']}")
    if len(hits) > 30: print(f"     ... +{len(hits)-30}")


# ============================================================
# MAIN LOOP
# ============================================================

def menu_show():
    banner()
    print()
    if not archive._cache:
        print("  archive is empty. run a source first.")
        return
    sources_seen = {}
    for m in archive._cache.values():
        s = m.get("_source", "?")
        sources_seen[s] = sources_seen.get(s, 0) + 1
    print("  by source:")
    for s, c in sorted(sources_seen.items(), key=lambda kv: -kv[1]):
        print(f"     {s:18s} {c:4d}")

    if archive.library:
        print("\n  learned predicates:")
        for name, pred in archive.library.items():
            hits = archive.query(pred)
            print(f"     {name:24s} ({len(hits):4d} matches)  "
                  f"{pred.name[:60]}")

    print("\n  example rule queries:")
    rules = [
        ("dark images",         threshold(mean_brightness, "<", 0.25)),
        ("bright images",       threshold(mean_brightness, ">", 0.65)),
        ("high-contrast",       threshold(contrast, ">", 0.20)),
        ("monochrome-ish",      threshold(color_diversity, "<", 4.0)),
    ]
    for label, p in rules:
        n = len(archive.query(p))
        print(f"     {label:24s} {n:4d}")

    print("\n  anomalies (most unusual 5):")
    try:
        anom = find_anomalies(archive, contamination=0.10)
        for h, alias, score in anom[:5]:
            print(f"     {alias:30s} {score:+.3f}")
    except Exception as e:
        print(f"     anomaly detection failed: {e}")


def menu_toggle_augment():
    AUGMENT_MODE["on"] = not AUGMENT_MODE["on"]
    state = "ON" if AUGMENT_MODE["on"] else "OFF"
    print(f"\n  augmentation now {state}")
    if AUGMENT_MODE["on"]:
        n = prompt_int("how many augmentations per image?",
                       default=4, lo=1, hi=8)
        AUGMENT_MODE["n"] = n
        print(f"  each fetch will produce {n+1} training samples "
              f"(1 original + {n} variants)")


def menu_auto_rules():
    print("\n  Discovering rules from accumulated data ...")
    rules = auto_discover_rules(archive, min_precision=0.75,
                                min_recall=0.30, top_k_per_cluster=3,
                                archive_dir=ARCHIVE_DIR)
    if not rules:
        print("  no rules met the precision/recall thresholds.")
        return
    added = 0
    for name, pred in rules.items():
        if name not in archive.library:
            archive.library[name] = pred
            added += 1
    archive.save_library()
    print(f"\n  added {added} new rules to library "
          f"(total: {len(archive.library)})")


def menu_kitchen_sink():
    """One-button academic protocol: lock seed -> ingest from ALL sources ->
    auto-discover rules -> full evaluation report. Walk away and come back
    to a fully-trained substrate."""
    print("\n" + "="*64)
    print("  KITCHEN-SINK FULL TRAINING + EVAL")
    print("="*64)
    print("  Locks the seed, pulls from every available source")
    print("  (HuggingFace + public APIs), auto-discovers rules,")
    print("  and runs the full evaluation report.")
    print("="*64)

    # Choose intensity
    print("\n  How much do you want?")
    print("  [1] LIGHT     - 100/HF source, 30/API source     (~15 min)")
    print("  [2] MEDIUM    - 300/HF source, 60/API source     (~45 min)")
    print("  [3] HEAVY     - 800/HF source, 100/API source    (~2 hours)")
    print("  [4] CUSTOM    - you pick the numbers")
    print("  [b] back")
    choice = prompt("intensity", default="2").lower()
    if choice == "b": return

    if choice == "1": n_hf, n_api = 100, 30
    elif choice == "2": n_hf, n_api = 300, 60
    elif choice == "3": n_hf, n_api = 800, 100
    elif choice == "4":
        n_hf  = prompt_int("HF images per dataset", default=300, lo=10, hi=10000)
        n_api = prompt_int("API images per source", default=60, lo=10, hi=500)
    else: return

    # Confirm
    n_hf_sources = sum(1 for _, k, _ in HF_SOURCES
                       if k == "no key" or has_hf_token())
    n_api_sources = sum(1 for s in SOURCES.values() for _, k, _ in s
                        if k == "no key" or "DEMO_KEY" in k)
    est = n_hf_sources * n_hf + n_api_sources * n_api
    print(f"\n  Estimated images: ~{est}")
    print(f"  HF sources active: {n_hf_sources}/{len(HF_SOURCES)}")
    print(f"  API sources active: {n_api_sources}")
    if AUGMENT_MODE["on"]:
        print(f"  Augmentation ON: each image becomes {AUGMENT_MODE['n']+1} samples")
        print(f"  -> ~{est * (AUGMENT_MODE['n']+1)} effective training samples")
    if prompt("\n  proceed? (y/n)", "y").lower() != "y": return

    # Lock seed if not already locked
    if HAS_EVAL and get_global_seed() is None:
        set_global_seed(42)
        save_seed(ARCHIVE_DIR, 42)
        print("\n  seed locked to 42 for reproducibility")

    t_start = time.time()
    n_start = len(archive._cache)

    # ============================================================
    # PHASE 1: HuggingFace streaming sources
    # ============================================================
    if HAS_HF and HAS_BATCH:
        print("\n" + "#"*64)
        print(f"# PHASE 1: HuggingFace streaming ({n_hf_sources} datasets)")
        print("#"*64)
        for i, (label, key_req, fn) in enumerate(HF_SOURCES):
            if key_req != "no key" and not has_hf_token():
                print(f"\n  [{i+1}/{len(HF_SOURCES)}] SKIP {label} "
                      f"(needs {key_req})")
                continue
            print(f"\n  [{i+1}/{len(HF_SOURCES)}] {label}")
            try:
                bulk_ingest(archive, fn(n_hf), batch_size=32,
                            max_images=n_hf)
            except KeyboardInterrupt:
                print("\n  user interrupted — moving to next phase")
                break
            except Exception as e:
                print(f"  failed: {type(e).__name__}: {e}")
    else:
        print("\n  [skip] HF or batch module not loaded")

    # ============================================================
    # PHASE 2: All public-API sources (no-key + demo-key)
    # ============================================================
    print("\n" + "#"*64)
    print(f"# PHASE 2: public APIs ({n_api_sources} sources)")
    print("#"*64)
    api_count = 0
    for stage_key, sources in SOURCES.items():
        for label, key_req, fn in sources:
            if key_req != "no key" and "DEMO_KEY" not in key_req:
                if not has_key(key_req):
                    continue
            api_count += 1
            print(f"\n  [{api_count}/{n_api_sources}] {label}")
            try:
                run_source(fn, n_api, label, fast=True)
            except KeyboardInterrupt:
                print("\n  user interrupted — moving to next phase")
                break
            except Exception as e:
                print(f"  failed: {type(e).__name__}: {e}")
        else:
            continue
        break

    # ============================================================
    # PHASE 3: Auto-discover rules
    # ============================================================
    print("\n" + "#"*64)
    print("# PHASE 3: discovering rules")
    print("#"*64)
    try:
        rules = auto_discover_rules(archive, archive_dir=ARCHIVE_DIR,
                                    verbose=True)
        added = 0
        for name, pred in rules.items():
            if name not in archive.library:
                archive.library[name] = pred
                added += 1
        archive.save_library()
        print(f"\n  added {added} new rules (total: {len(archive.library)})")
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")

    # ============================================================
    # PHASE 4: Full evaluation report
    # ============================================================
    print("\n" + "#"*64)
    print("# PHASE 4: evaluation")
    print("#"*64)
    if HAS_EVAL:
        try:
            full_evaluation_report(archive, ARCHIVE_DIR)
        except Exception as e:
            print(f"  evaluation failed: {type(e).__name__}: {e}")
    else:
        menu_show()

    # Summary
    elapsed = time.time() - t_start
    n_added = len(archive._cache) - n_start
    print("\n" + "="*64)
    print(f"  KITCHEN-SINK COMPLETE")
    print(f"  added {n_added} images in {elapsed/60:.1f} minutes")
    print(f"  archive total: {len(archive._cache)}")
    print(f"  library total: {len(archive.library)} predicates")
    print("="*64)


def menu_set_hf_token():
    """Interactively set the HuggingFace token and save it to a file."""
    print("\n  HUGGINGFACE TOKEN SETUP")
    print("  ====================================")
    print("  1. Visit https://huggingface.co/settings/tokens")
    print("  2. Click 'New token', give it any name, role 'Read'")
    print("  3. Copy the token (looks like 'hf_xxxxxxxxxxxxxxx')")
    print("  4. Paste it below.")
    print("  ====================================\n")
    token = prompt("paste HF token (or blank to cancel)").strip()
    if not token: return
    if not token.startswith("hf_"):
        if prompt("doesn't start with 'hf_', save anyway? (y/n)",
                   "n").lower() != "y": return
    token_file = HERE / "_hf_token.txt"
    try:
        token_file.write_text(token, encoding="utf-8")
        print(f"\n  saved to {token_file}")
        print(f"  token will be used automatically on next HF stream.")
    except Exception as e:
        print(f"  failed to save: {e}")


def menu_set_seed():
    """Set or display the global random seed."""
    if not HAS_EVAL:
        print("  evaluation layer not available"); return
    current = get_global_seed()
    print(f"\n  current seed: {current}")
    new = prompt("new seed (integer; blank to keep)", default="").strip()
    if not new: return
    try:
        seed = int(new)
        set_global_seed(seed)
        save_seed(ARCHIVE_DIR, seed)
        print(f"  seed set to {seed} and saved to {ARCHIVE_DIR}/_seed.txt")
    except ValueError:
        print("  invalid integer")



    """Set or display the global random seed."""
    if not HAS_EVAL:
        print("  evaluation layer not available"); return
    current = get_global_seed()
    print(f"\n  current seed: {current}")
    new = prompt("new seed (integer; blank to keep)", default="").strip()
    if not new: return
    try:
        seed = int(new)
        set_global_seed(seed)
        save_seed(ARCHIVE_DIR, seed)
        print(f"  seed set to {seed} and saved to {ARCHIVE_DIR}/_seed.txt")
    except ValueError:
        print("  invalid integer")


def menu_full_evaluation():
    """Run the academic evaluation report — splits, seed, library
    transferability, source attribution, learning curve."""
    if not HAS_EVAL:
        print("  evaluation layer not available"); return
    full_evaluation_report(archive, ARCHIVE_DIR)


def menu_attribution():
    """Show per-source attribution report."""
    if not HAS_EVAL:
        print("  evaluation layer not available"); return
    source_attribution_report(archive)


def menu_learning_curve():
    """Show the learning curve from logged checkpoints."""
    if not HAS_EVAL:
        print("  evaluation layer not available"); return
    print("\n  Learning curves over recorded checkpoints:")
    print_learning_curve(ARCHIVE_DIR, metric="mean_train_f1")
    print()
    print_learning_curve(ARCHIVE_DIR, metric="n_rules_total")


def menu_analyze_gaps():
    print("\n  Analyzing what's underrepresented ...")
    suggestions = analyze_gaps(archive)
    print()
    for s in suggestions:
        print(f"    {s}")
    print()


def menu_state_report():
    training_state_report(archive)


def menu_hf_streaming():
    """Bulk ingest from a HuggingFace dataset using streaming + multiprocessing."""
    print("\nHUGGINGFACE STREAMING SOURCES")
    print("="*64)
    if not has_hf_token():
        print("(no HF token; auth-required datasets will be skipped)")
        print("(set one with menu option [t] -- it's free and takes 30 sec)")
    print()
    for i, (label, key_req, _) in enumerate(HF_SOURCES):
        avail = "[ready]" if key_req == "no key" or has_hf_token() \
                else f"[needs {key_req}]"
        print(f"  [{i:2d}] {label:50s} {avail}")
    print(f"  [A]  ALL datasets in sequence")
    print(f"  [b]  back")
    print()
    c = prompt("choose dataset (or A for all)").lower()
    if c == "b" or c == "": return

    n = prompt_int("how many images per dataset?", default=300, lo=10, hi=10000)

    if c == "a":
        # Run every available HF dataset in sequence
        active = [(l, k, f) for l, k, f in HF_SOURCES
                   if k == "no key" or has_hf_token()]
        print(f"\n  running {len(active)} datasets at {n} each "
              f"(~{len(active)*n} images total)")
        if prompt("proceed? (y/n)", "y").lower() != "y": return
        for i, (label, key_req, fn) in enumerate(active):
            print(f"\n  [{i+1}/{len(active)}] {label}")
            try:
                if HAS_BATCH:
                    bulk_ingest(archive, fn(n), batch_size=32, max_images=n)
                else:
                    bulk_ingest_simple(archive, fn(n), max_images=n)
            except KeyboardInterrupt:
                print("\n  user interrupted; archive kept")
                break
            except Exception as e:
                print(f"  failed: {type(e).__name__}: {e}")
        maybe_recluster()
        return

    try:
        idx = int(c)
        label, key_req, fn = HF_SOURCES[idx]
    except (ValueError, IndexError):
        print("  invalid choice"); return
    if key_req != "no key" and not has_hf_token():
        print(f"  this dataset needs auth. set token via [t]."); return

    use_mp = prompt("use multiprocessing? (y/n)", "y").lower() == "y"
    print(f"\n  starting bulk ingest: {label}, n={n}")
    if use_mp:
        bulk_ingest(archive, fn(n), batch_size=32, max_images=n)
    else:
        bulk_ingest_simple(archive, fn(n), max_images=n)
    maybe_recluster()


def menu_camera():
    """Live webcam inference using the predicate library."""
    if not HAS_CV2:
        print("\n  OpenCV is not installed.")
        print("  Run in cmd:  pip install opencv-python")
        return
    if not archive.library:
        print("\n  Predicate library is empty.")
        print("  Run [r] (auto-discover rules) first to populate it.")
        return
    cam = prompt_int("camera index (0=default)", default=0, lo=0, hi=10)
    print(f"\n  opening camera {cam}. press q in the window to quit.")
    live_camera(archive, camera_index=cam)


def main():
    while True:
        banner()
        aug = "[ON]" if AUGMENT_MODE["on"] else "[off]"
        seed = get_global_seed() if HAS_EVAL else None
        seed_str = f"seed={seed}" if seed is not None else "seed=none"
        hf_str  = "✓" if has_hf_token() else "no token"
        print(f"  --- ONE BUTTON ---")
        print(f"  [F] FULL kitchen-sink: ingest from everything + train + eval")
        print(f"  --- FETCH ---")
        print(f"  [1] Browse public APIs by stage")
        print(f"  [6] FAST curriculum (parallel sources)")
        print(f"  [7] FAST single source")
        if HAS_HF:
            print(f"  [h] HuggingFace streaming  (HF token: {hf_str})")
            print(f"  [t] Set HuggingFace token")
        else:
            print(f"  [h] HuggingFace streaming (UNAVAILABLE)")
        print(f"  --- TRAIN ---")
        print(f"  [a] Toggle augmentation {aug}")
        print(f"  [r] Auto-discover rules from current data")
        print(f"  [g] Analyze gaps + suggest what to fetch next")
        print(f"  [s] Detailed training-state report")
        if HAS_EVAL:
            print(f"  [k] Set / lock random seed  ({seed_str})")
            print(f"  --- EVALUATE ---")
            print(f"  [e] Full academic evaluation report")
            print(f"  [l] Show learning curve")
            print(f"  [d] Per-source attribution")
        print(f"  --- USE ---")
        print(f"  [3] Show what was learned")
        print(f"  [4] Teach a concept by labeling")
        if HAS_CAMERA and HAS_CV2:
            print(f"  [c] LIVE CAMERA inference")
        else:
            print(f"  [c] LIVE CAMERA (UNAVAILABLE)")
        print(f"  [5] Quit")
        c = prompt("choose").lower()
        if   c == "f": menu_kitchen_sink()
        elif c == "1": menu_browse()
        elif c == "3": menu_show()
        elif c == "4": menu_teach()
        elif c == "5" or c in ("q", "quit", "exit"):
            print("\n  bye.\n"); return
        elif c == "6": menu_curriculum_fast()
        elif c == "7": menu_browse_fast()
        elif c == "a": menu_toggle_augment()
        elif c == "r": menu_auto_rules()
        elif c == "g": menu_analyze_gaps()
        elif c == "s": menu_state_report()
        elif c == "t":
            if HAS_HF: menu_set_hf_token()
            else: print("  HF module not loaded")
        elif c == "k":
            if HAS_EVAL: menu_set_seed()
            else: print("  evaluation layer not loaded")
        elif c == "e":
            if HAS_EVAL: menu_full_evaluation()
            else: print("  evaluation layer not loaded")
        elif c == "l":
            if HAS_EVAL: menu_learning_curve()
            else: print("  evaluation layer not loaded")
        elif c == "d":
            if HAS_EVAL: menu_attribution()
            else: print("  evaluation layer not loaded")
        elif c == "h":
            if HAS_HF: menu_hf_streaming()
            else: print("  HuggingFace module not loaded.")
        elif c == "c":
            if HAS_CAMERA and HAS_CV2: menu_camera()
            elif HAS_CAMERA: print("  OpenCV not installed.")
            else: print("  Camera module not loaded.")
        else: print("  invalid choice")


def menu_browse_fast():
    """Same as menu_browse but uses the fast concurrent pipeline."""
    while True:
        print("\n[FAST] STAGES:")
        for i, key in enumerate(STAGE_ORDER):
            print(f"  [{i}] {STAGE_NAMES[key]}")
        print("  [b] back")
        c = prompt("choose stage").lower()
        if c == "b" or c == "": return
        try:
            idx = int(c)
            stage_key = STAGE_ORDER[idx]
        except (ValueError, IndexError):
            print("  invalid choice"); continue
        sources = SOURCES[stage_key]
        while True:
            print(f"\n[FAST] {STAGE_NAMES[stage_key]}:")
            for i, (label, key_req, _) in enumerate(sources):
                avail = "[ready]" if key_req == "no key" or has_key(key_req) \
                        or "DEMO_KEY" in key_req else f"[needs {key_req}]"
                print(f"  [{i}] {label:38s} {avail}")
            print("  [b] back")
            c = prompt("choose source").lower()
            if c == "b" or c == "": break
            try:
                sidx = int(c)
                label, key_req, fn = sources[sidx]
            except (ValueError, IndexError):
                print("  invalid choice"); continue
            n = prompt_int("how many images?", default=30, lo=1, hi=300)
            run_source(fn, n, label, fast=True)
            maybe_recluster()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  interrupted. archive and library are saved.\n")
    finally:
        try:
            _sources.flush_url_dedup()
        except Exception:
            pass
