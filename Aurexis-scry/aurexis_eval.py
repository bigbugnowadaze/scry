"""
aurexis_eval.py — academic-discipline layer.

Four features that move the project from "runs" to "research":

  (1) HELD-OUT TEST SPLIT
      Each image is deterministically tagged as 'train' or 'test' at ingest.
      Auto-discovery uses only train. Rules are then evaluated on test.
      The split assignment is hash-derived, so re-ingesting the same image
      always lands it in the same split.

  (2) REPRODUCIBLE SEEDS
      One global seed controls all stochastic choices: query rotation,
      synthetic image generation, K-means initialization, learning curve
      checkpoint sampling. Saved to disk so re-running gives identical
      cluster assignments and rule sets.

  (3) LEARNING CURVES
      After every recluster, snapshot (n_train, n_rules, train_F1,
      test_F1) into a log. Plot the curve to see when more data stops
      helping. Persists to learning_log.json next to the archive.

  (4) SOURCE ATTRIBUTION
      For each source, compute mean atom values vs. global mean and
      report which atoms that source uniquely stretches. Tells you which
      sources are redundant vs. genuinely new.
"""

import json
import hashlib
import random
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


# ============================================================
# (1) HELD-OUT TEST SPLIT
# ============================================================

DEFAULT_TEST_FRACTION = 0.10  # 10% of data reserved for test


def assign_split(image_hash, test_fraction=DEFAULT_TEST_FRACTION):
    """Deterministically assign 'train' or 'test' to an image hash.
    Same hash always gets the same split, regardless of when ingested."""
    h = hashlib.md5(image_hash.encode()).hexdigest()
    return "test" if int(h[:8], 16) % 1000 < int(test_fraction * 1000) \
                  else "train"


def ensure_splits(archive, test_fraction=DEFAULT_TEST_FRACTION):
    """Tag every image in archive with a split if it doesn't have one.
    Idempotent — existing tags are preserved."""
    n_added = 0
    for h, m in archive._cache.items():
        if "_split" not in m:
            m["_split"] = assign_split(h, test_fraction)
            n_added += 1
    if n_added:
        archive._save_measurements()
    return n_added


def split_counts(archive):
    """Return {'train': N, 'test': M}."""
    counts = Counter()
    for m in archive._cache.values():
        counts[m.get("_split", "train")] += 1
    return dict(counts)


def get_train_hashes(archive):
    return [h for h, m in archive._cache.items()
            if m.get("_split", "train") == "train"]


def get_test_hashes(archive):
    return [h for h, m in archive._cache.items()
            if m.get("_split", "train") == "test"]


def evaluate_rule_on_test(archive, predicate, expected_cluster_test_hashes):
    """Measure precision and recall of a predicate on the held-out test set.

    Args:
        archive                          aurexis Archive
        predicate                        Predicate to evaluate
        expected_cluster_test_hashes    set of test hashes that "should" match

    Returns:
        dict with precision, recall, f1, n_test_in, n_test_out
    """
    test_hashes = set(get_test_hashes(archive))
    expected = set(expected_cluster_test_hashes) & test_hashes
    other = test_hashes - expected
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_test_in": 0, "n_test_out": len(other)}

    # Run predicate on test images
    matched = set()
    for h in test_hashes:
        m = archive._cache[h]
        feat = archive._features.get(h)
        try:
            if predicate.matches(m, features=feat):
                matched.add(h)
        except Exception:
            continue

    tp = len(matched & expected)
    fp = len(matched & other)
    fn = len(expected - matched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_test_in": len(expected), "n_test_out": len(other)}


# ============================================================
# (2) REPRODUCIBLE SEEDS
# ============================================================

_GLOBAL_SEED = None


def set_global_seed(seed):
    """Set every relevant random source to a deterministic state.

    Affects:
        - Python's random module (used in sources.py for query rotation)
        - numpy's global RNG (used in synthetic generators)
        - The seed file is persisted so this is recoverable across runs
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    return _GLOBAL_SEED


def get_global_seed():
    return _GLOBAL_SEED


def save_seed(archive_dir, seed):
    """Persist the seed so the next run can reload it."""
    p = Path(archive_dir) / "_seed.txt"
    try:
        p.write_text(str(int(seed)), encoding="utf-8")
    except Exception:
        pass


def load_seed(archive_dir):
    """Load a previously-saved seed; returns None if not present."""
    p = Path(archive_dir) / "_seed.txt"
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


# ============================================================
# (3) LEARNING CURVES
# ============================================================

def _learning_log_path(archive_dir):
    return Path(archive_dir) / "_learning_log.json"


def log_checkpoint(archive_dir, **fields):
    """Append a checkpoint entry to the learning log.

    Fields typically include:
        n_train, n_test, n_rules_total, n_rules_train_only,
        mean_train_f1, mean_test_f1,
        timestamp, seed
    """
    p = _learning_log_path(archive_dir)
    log = []
    if p.exists():
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            log = []
    fields["timestamp"] = time.time()
    fields["seed"] = _GLOBAL_SEED
    log.append(fields)
    try:
        p.write_text(json.dumps(log, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_learning_log(archive_dir):
    p = _learning_log_path(archive_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def print_learning_curve(archive_dir, metric="mean_test_f1"):
    """Render the learning curve as an ASCII chart."""
    log = get_learning_log(archive_dir)
    if not log:
        print("  no checkpoints logged yet — run [r] auto-discover first.")
        return
    print(f"\n  Learning curve  (metric: {metric})")
    print(f"  {'='*72}")
    n_max = max(c.get("n_train", 0) for c in log) or 1
    val_max = max(c.get(metric, 0) for c in log) or 1
    val_min = min(c.get(metric, 0) for c in log)
    width = 50
    for c in log:
        n = c.get("n_train", 0)
        v = c.get(metric, 0)
        bar_len = int(width * v / max(val_max, 1e-9))
        bar = "█" * bar_len
        n_rules = c.get("n_rules_total", 0)
        print(f"  N={n:>5d} rules={n_rules:>4d}  "
              f"{metric}={v:.3f}  {bar}")
    print(f"  {'='*72}")
    print(f"  range: {val_min:.3f} .. {val_max:.3f} over "
          f"{len(log)} checkpoints")


# ============================================================
# (4) SOURCE ATTRIBUTION
# ============================================================

def source_attribution_report(archive):
    """For each source, report which atoms it stretches most relative
    to the rest of the archive. High z-score = source is providing
    unique structure along that atom; low z-score = source is redundant
    with what we already have."""
    from aurexis_v3 import ATOMS

    if len(archive._cache) < 20:
        print("  need more images for meaningful attribution")
        return {}

    # Group hashes by source (treating "_aug" suffix as same source)
    sources = defaultdict(list)
    for h, m in archive._cache.items():
        s = m.get("_source", "?").replace("_aug", "")
        sources[s].append(h)

    # Compute global mean and std for each atom
    atom_names = list(ATOMS.keys())
    all_vals = {a: [] for a in atom_names}
    for m in archive._cache.values():
        for a in atom_names:
            if a in m:
                all_vals[a].append(m[a])
    global_means = {a: np.mean(v) for a, v in all_vals.items() if v}
    global_stds = {a: np.std(v) + 1e-9 for a, v in all_vals.items() if v}

    print(f"\n  Source attribution  (N={len(archive._cache)} images, "
          f"{len(sources)} sources)")
    print(f"  {'='*72}")
    print(f"  {'source':<20s} {'count':>6s}  distinguishing atoms (z-score)")
    print(f"  {'-'*72}")

    contribution = {}
    for source_name, hashes in sorted(sources.items(),
                                       key=lambda kv: -len(kv[1])):
        if len(hashes) < 3:
            continue
        # Compute mean atoms for this source
        src_means = {}
        for a in atom_names:
            vals = [archive._cache[h][a] for h in hashes
                    if a in archive._cache[h]]
            if vals:
                src_means[a] = np.mean(vals)

        # Z-score per atom
        z_scores = {a: (src_means[a] - global_means[a]) / global_stds[a]
                    for a in atom_names if a in src_means}

        # Top-3 atoms by absolute z-score
        ranked = sorted(z_scores.items(),
                        key=lambda kv: -abs(kv[1]))[:3]
        bits = "  ".join(f"{a}({z:+.1f})" for a, z in ranked)
        print(f"  {source_name:<20s} {len(hashes):>6d}  {bits}")

        contribution[source_name] = {"count": len(hashes),
                                      "z_scores": z_scores,
                                      "top": ranked}

    print(f"  {'-'*72}")
    print(f"  Reading: |z|>1 = source diverges from typical along that atom.")
    print(f"           |z|<0.5 = source is redundant; consider others.")
    print(f"  {'='*72}")
    return contribution


# ============================================================
# UNIFIED EVAL — runs all four diagnostics
# ============================================================

def full_evaluation_report(archive, archive_dir):
    """Run all four academic diagnostics and print a consolidated report."""
    from aurexis_v3 import cluster_archive

    print("\n" + "="*72)
    print("  ACADEMIC EVALUATION REPORT")
    print("="*72)

    # Splits
    n_added = ensure_splits(archive)
    if n_added:
        print(f"  assigned splits to {n_added} previously-untagged images")
    counts = split_counts(archive)
    n_train = counts.get("train", 0)
    n_test = counts.get("test", 0)
    print(f"  splits:  train={n_train}  test={n_test}  "
          f"({100*n_test/max(n_train+n_test,1):.1f}% test)")

    # Seed
    seed = get_global_seed()
    saved = load_seed(archive_dir)
    print(f"  seed:    current={seed}  saved={saved}")

    # Library evaluation
    if archive.library and n_test > 0:
        train_h = set(get_train_hashes(archive))
        test_h = set(get_test_hashes(archive))
        print(f"\n  evaluating {len(archive.library)} predicates on "
              f"{len(test_h)} held-out test images:")
        f1_scores = []
        for name, pred in list(archive.library.items())[:15]:
            try:
                # Apply to all test images, count matches
                test_matches = 0
                for h in test_h:
                    m = archive._cache[h]
                    feat = archive._features.get(h)
                    if pred.matches(m, features=feat):
                        test_matches += 1
                # Apply to train as reference
                train_matches = 0
                for h in train_h:
                    m = archive._cache[h]
                    feat = archive._features.get(h)
                    if pred.matches(m, features=feat):
                        train_matches += 1
                # Crude transferability check: hit-rate ratio
                train_rate = train_matches / max(len(train_h), 1)
                test_rate = test_matches / max(len(test_h), 1)
                # If rates are similar, the rule generalizes
                ratio = test_rate / max(train_rate, 1e-9)
                stable = abs(ratio - 1.0) < 0.3
                marker = "✓" if stable else "✗"
                print(f"    {marker} {name[:50]:<50s}  "
                      f"train={train_rate:.2f} test={test_rate:.2f} "
                      f"ratio={ratio:.2f}")
                f1_scores.append((train_rate, test_rate))
            except Exception:
                continue
        if len(archive.library) > 15:
            print(f"    ... +{len(archive.library)-15} more not shown")
    elif not archive.library:
        print("\n  predicate library is empty — run [r] auto-discover first")
    else:
        print("\n  no test split yet — run more ingest after this")

    print()
    source_attribution_report(archive)

    print()
    print_learning_curve(archive_dir)

    print("\n" + "="*72)
