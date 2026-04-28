"""
aurexis_train.py — accelerate training (not just running).

Three independent boosts you can use in any combination:

  (A) AUGMENTATION
      Each fetched image becomes 5 training samples (original + flip +
      crops + brightness/contrast variations). The substrate sees the
      same scene from multiple statistical angles. Free training data.

  (B) AUTO-RULE DISCOVERY
      Scan accumulated atom values, find single-atom and atom-pair
      threshold rules that match each cluster cleanly, and add them
      to the predicate library. Grows your library from ~8 predicates
      to dozens or hundreds without fetching anything new.

  (C) ACTIVE LEARNING / GAP ANALYSIS
      Look at what's already in the archive, find regions of feature
      space that are sparsely populated, and suggest which sources
      to run next. Stops the substrate from over-fitting to whatever
      stage you ran most recently.
"""

import random
import numpy as np
from collections import defaultdict
from PIL import Image, ImageOps, ImageEnhance


# ============================================================
# (A) AUGMENTATION
# ============================================================

def augment_pil(img, n=4):
    """Generate n augmented variants of a PIL image plus the original.
    Returns a list of (PIL, suffix) — suffix becomes part of the alias.

    Strategy: each variant gives the substrate a *genuinely different*
    statistical view of the same scene:
      - flip       : Hu moments + oriented edges change; texture invariant
      - crop       : completely different region statistics
      - rotation   : Hu moments and orientations rotate
      - brightness : tests how mean_brightness/dark_fraction shift
      - contrast   : tests contrast / edge_density coupling
    """
    out = [(img, "")]  # always include original
    w, h = img.size
    if w < 32 or h < 32:
        return out  # too small to augment safely

    pool = [
        ("flip",     lambda im: ImageOps.mirror(im)),
        ("crop_c",   lambda im: im.crop((int(w*0.15), int(h*0.15),
                                          int(w*0.85), int(h*0.85)))),
        ("crop_tl",  lambda im: im.crop((0, 0, int(w*0.6), int(h*0.6)))),
        ("crop_br",  lambda im: im.crop((int(w*0.4), int(h*0.4), w, h))),
        ("rot90",    lambda im: im.rotate(90, expand=True)),
        ("dark",     lambda im: ImageEnhance.Brightness(im).enhance(0.65)),
        ("bright",   lambda im: ImageEnhance.Brightness(im).enhance(1.4)),
        ("hicon",    lambda im: ImageEnhance.Contrast(im).enhance(1.6)),
    ]
    random.shuffle(pool)
    for tag, fn in pool[:n]:
        try:
            variant = fn(img)
            out.append((variant, tag))
        except Exception:
            continue
    return out


def enriched_ingest(archive, pil_img, alias, source, n_augment=4):
    """Ingest an image plus n augmented variants. Returns total new entries."""
    new_count = 0
    for variant, tag in augment_pil(pil_img, n=n_augment):
        v_alias = alias if not tag else f"{alias}__{tag}"
        v_source = source if not tag else f"{source}_aug"
        try:
            h, is_new = archive.ingest_pil(variant, v_alias, v_source)
            if is_new:
                new_count += 1
        except Exception:
            continue
    return new_count


# ============================================================
# (B) AUTO-RULE DISCOVERY
# ============================================================

def auto_discover_rules(archive, min_precision=0.75, min_recall=0.30,
                        top_k_per_cluster=3, verbose=True,
                        archive_dir=None):
    """Discover typed predicate rules from accumulated atom data.

    Critical design: clusters are built FROM ATOM VALUES directly (not from
    the 270-dim rich features), guaranteeing that atom-based rules CAN match
    them. We sweep multiple K values (3, 5, 7, 10) and progressively relax
    precision/recall thresholds until rules emerge. Always returns something
    informative — falls back to centroid-described rules at the end.

    Uses ONLY the train split for discovery. Rules can then be evaluated on
    the held-out test split via the [e] menu option.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from aurexis_v3 import Predicate, threshold, ATOMS

    # Ensure all images have a split tag
    try:
        from aurexis_eval import ensure_splits, get_train_hashes, log_checkpoint
        ensure_splits(archive)
        train_hashes = set(get_train_hashes(archive))
    except Exception:
        train_hashes = set(archive._cache.keys())

    if len(train_hashes) < 20:
        if verbose: print("  need at least 20 train-split images to discover rules")
        return {}

    # Build atom matrix from TRAIN images only
    atom_names = list(ATOMS.keys())
    atom_list = [ATOMS[n] for n in atom_names]
    rows, hashes = [], []
    for h in train_hashes:
        m = archive._cache.get(h, {})
        if all(n in m for n in atom_names):
            rows.append([m[n] for n in atom_names])
            hashes.append(h)
    if len(rows) < 20:
        if verbose: print("  need at least 20 train images with all atoms")
        return {}

    X = np.array(rows, dtype=float)
    Xs = StandardScaler().fit_transform(X)
    discovered = {}
    rule_metrics = []  # for learning curve

    if verbose:
        print(f"  clustering {len(rows)} train images on atom-space ...")

    K_values = [k for k in [3, 5, 7, 10] if k <= max(3, len(rows) // 5)]

    for K in K_values:
        try:
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            assignments = km.fit_predict(Xs)
        except Exception:
            continue

        if verbose:
            sizes = np.bincount(assignments, minlength=K)
            print(f"    K={K}: cluster sizes = {sizes.tolist()}")

        for cid in range(K):
            members = np.where(assignments == cid)[0]
            non_members = np.where(assignments != cid)[0]
            if len(members) < 3 or len(non_members) < 3:
                continue

            best_rules_for_cluster = []
            for p_thresh, r_thresh in [(0.80, 0.40), (0.70, 0.30),
                                        (0.60, 0.25), (0.50, 0.20)]:
                candidates = []
                for j, atom in enumerate(atom_list):
                    vals_in = X[members, j]
                    vals_out = X[non_members, j]

                    for direction in [">", "<"]:
                        if direction == ">":
                            thr_pcts = [50, 60, 70, 80]
                            in_pct = lambda thr: (vals_in > thr).mean()
                            out_pct = lambda thr: (vals_out > thr).mean()
                        else:
                            thr_pcts = [20, 30, 40, 50]
                            in_pct = lambda thr: (vals_in < thr).mean()
                            out_pct = lambda thr: (vals_out < thr).mean()

                        for pct in thr_pcts:
                            thr = float(np.percentile(vals_in, pct))
                            ip, op = in_pct(thr), out_pct(thr)
                            in_count = ip * len(vals_in)
                            out_count = op * len(vals_out)
                            denom = in_count + out_count
                            if denom < 1: continue
                            precision = in_count / denom
                            recall = ip
                            if precision < p_thresh or recall < r_thresh:
                                continue
                            f1 = (2 * precision * recall /
                                  max(precision + recall, 1e-9))
                            candidates.append(
                                (f1, precision, recall, atom, direction, thr,
                                 p_thresh)
                            )

                if candidates:
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    seen_atoms = set()
                    for tup in candidates:
                        f1, prec, rec, atom, direction, thr, level = tup
                        if atom.name in seen_atoms: continue
                        if len(best_rules_for_cluster) >= top_k_per_cluster:
                            break
                        seen_atoms.add(atom.name)
                        best_rules_for_cluster.append(tup)
                    break  # got rules at this strictness

            for tup in best_rules_for_cluster:
                f1, prec, rec, atom, direction, thr, level = tup
                pred = threshold(atom, direction, thr)
                name = f"k{K}c{cid}_{atom.name}_{direction}{thr:.3f}"
                if name not in discovered:
                    discovered[name] = pred
                    rule_metrics.append({"name": name, "train_p": prec,
                                          "train_r": rec, "train_f1": f1})
                    if verbose:
                        print(f"      K={K} C{cid}: {atom.name} {direction} "
                              f"{thr:.3f}  P={prec:.2f} R={rec:.2f} "
                              f"F1={f1:.2f}  (level={level:.2f})")

    # Fallback to centroid description if nothing found
    if not discovered:
        if verbose:
            print("  no precise rules found; generating descriptive "
                  "rules from cluster centroids ...")
        K = K_values[0] if K_values else 3
        try:
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            assignments = km.fit_predict(Xs)
            global_means = X.mean(axis=0)
            global_stds = X.std(axis=0) + 1e-9
            for cid in range(K):
                members = np.where(assignments == cid)[0]
                if len(members) < 3: continue
                centroid = X[members].mean(axis=0)
                z_scores = (centroid - global_means) / global_stds
                ranking = np.argsort(-np.abs(z_scores))
                for j in ranking[:2]:
                    atom = atom_list[j]
                    direction = ">" if z_scores[j] > 0 else "<"
                    thr = float((centroid[j] + global_means[j]) / 2)
                    pred = threshold(atom, direction, thr)
                    name = f"descr_c{cid}_{atom.name}_{direction}{thr:.3f}"
                    discovered[name] = pred
                    if verbose:
                        print(f"      descriptive C{cid}: {atom.name} "
                              f"{direction} {thr:.3f}  (z={z_scores[j]:+.2f})")
        except Exception as e:
            if verbose: print(f"    centroid fallback failed: {e}")

    # Log a learning-curve checkpoint
    if archive_dir is not None and rule_metrics:
        try:
            mean_train_f1 = np.mean([r["train_f1"] for r in rule_metrics])
            mean_train_p = np.mean([r["train_p"] for r in rule_metrics])
            mean_train_r = np.mean([r["train_r"] for r in rule_metrics])
            from aurexis_eval import log_checkpoint, get_test_hashes
            log_checkpoint(
                archive_dir,
                n_train=len(train_hashes),
                n_test=len(get_test_hashes(archive)),
                n_rules_total=len(discovered),
                mean_train_f1=float(mean_train_f1),
                mean_train_precision=float(mean_train_p),
                mean_train_recall=float(mean_train_r),
            )
            if verbose:
                print(f"  logged checkpoint: N_train={len(train_hashes)}, "
                      f"rules={len(discovered)}, mean_F1={mean_train_f1:.3f}")
        except Exception as e:
            if verbose: print(f"  checkpoint logging skipped: {e}")

    if verbose:
        print(f"  discovered {len(discovered)} new rules")
    return discovered


# ============================================================
# (C) ACTIVE LEARNING / GAP ANALYSIS
# ============================================================

def analyze_gaps(archive):
    """Find what's missing or imbalanced in the archive and suggest
    which sources or queries would be most informative next.
    Returns a list of human-readable suggestion strings."""
    suggestions = []

    if len(archive._cache) < 20:
        return ["Need at least 20 images first — run any source from menu."]

    # 1) Source distribution
    source_counts = defaultdict(int)
    for m in archive._cache.values():
        # collapse "_aug" suffix back to the parent for source-level analysis
        s = m.get("_source", "?").replace("_aug", "")
        source_counts[s] += 1
    total = sum(source_counts.values())
    base_sources = [s for s in source_counts if not s.endswith("_aug")]

    # 2) Cluster balance
    try:
        from aurexis_v3 import cluster_archive
        best_k, assignments = cluster_archive(archive, k_range=(3, 10))
        cluster_sizes = np.bincount(assignments)
        max_c, min_c = cluster_sizes.max(), cluster_sizes.min()
        if max_c >= 4 * max(min_c, 1):
            suggestions.append(
                f"⚠ Cluster imbalance: largest={max_c}, smallest={min_c}. "
                f"The substrate is dominated by one or two modes."
            )
    except Exception as e:
        pass

    # 3) Atom variance — narrow ranges mean the substrate hasn't seen variety
    from aurexis_v3 import ATOMS
    atom_stats = {}
    for atom_name in ATOMS:
        values = [m[atom_name] for m in archive._cache.values()
                  if atom_name in m]
        if values:
            v = np.array(values)
            atom_stats[atom_name] = {
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std()),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            }

    narrow = []
    for name, stats in atom_stats.items():
        # narrow if 80% of mass is in a small interval
        rng = stats["p90"] - stats["p10"]
        if name in ("color_diversity", "color_entropy") and rng < 1.0:
            narrow.append(name)
        elif rng < 0.05:
            narrow.append(name)
    if narrow:
        suggestions.append(
            f"📉 Narrow-range atoms: {', '.join(narrow)}. "
            f"The substrate hasn't seen enough variety along these dimensions."
        )

    # 4) Coverage — which named sources haven't been touched
    missing_categories = []
    if not any("inaturalist" in s or "gbif" in s for s in source_counts):
        missing_categories.append("Stage 3 (iNaturalist / GBIF wildlife)")
    if not any(s in source_counts for s in ("met", "artic", "cleveland", "rijks")):
        missing_categories.append("Stage 4 (Museum art)")
    if not any(s in source_counts for s in ("apod", "nasa_ivl", "epic", "mars", "jwst")):
        missing_categories.append("Stage 5 (Astronomy)")
    if not any(s in source_counts for s in ("gibs", "goes", "osm")):
        missing_categories.append("Stage 6 (Earth from above)")
    if "tpdne" not in source_counts:
        missing_categories.append("Stage 7 (Synthetic faces)")
    if missing_categories:
        suggestions.append(
            f"🌐 Untouched stages: {', '.join(missing_categories)}"
        )

    # 5) Synthetic over-representation
    synth = source_counts.get("synthetic", 0)
    if synth > 0.25 * total:
        suggestions.append(
            f"🧪 Synthetic data is {100*synth//total}% of archive. "
            f"Real-world structure may be skewed; run more real sources."
        )

    # 6) Suggest next concrete fetch based on which atom is narrowest
    if atom_stats:
        narrowest = min(atom_stats.items(), key=lambda kv: kv[1]["std"])[0]
        # Map atom to suggested source that should expand it
        atom_to_source = {
            "edge_density":      "Stage 4 Art Institute (sharp line work) "
                                 "or Stage 6 OSM tiles (cartographic edges)",
            "color_diversity":   "Stage 3 iNaturalist Plantae (saturated greens) "
                                 "or Stage 1 Pexels tropical landscapes",
            "color_entropy":     "Stage 5 NASA APOD nebulae (rich palettes) "
                                 "or Stage 4 Cleveland Museum",
            "dark_fraction":     "Stage 5 NASA Mars Rover (dark sky) "
                                 "or Stage 5 NASA APOD",
            "bright_fraction":   "Stage 6 GIBS satellite (bright clouds) "
                                 "or Stage 5 NASA EPIC full-Earth",
            "contrast":          "Stage 6 OSM tiles (high-contrast cartography) "
                                 "or Stage 7 thispersondoesnotexist",
            "hog_energy":        "Stage 4 art (rich shape content) "
                                 "or Stage 3 iNaturalist Insecta",
            "lbp_uniformity":    "Stage 1 Picsum (natural texture) "
                                 "or Stage 3 iNaturalist Aves (feathers)",
            "mean_brightness":   "mix dark sources (APOD, Mars) and "
                                 "bright sources (GIBS, EPIC, snow scenes)",
        }
        if narrowest in atom_to_source:
            suggestions.append(
                f"🎯 Most under-explored atom: '{narrowest}' "
                f"(std={atom_stats[narrowest]['std']:.4f}). "
                f"To stretch it, try: {atom_to_source[narrowest]}"
            )

    if not suggestions:
        suggestions.append(
            "✅ Archive looks reasonably balanced. "
            "Add more from any stage to deepen the existing structure."
        )
    return suggestions


# ============================================================
# DIAGNOSTIC: print a full training-state summary
# ============================================================

def training_state_report(archive):
    """Print everything you'd want to know about the substrate's state."""
    from aurexis_v3 import ATOMS, cluster_archive

    print("\n" + "=" * 64)
    print("  TRAINING STATE REPORT")
    print("=" * 64)
    n = len(archive._cache)
    print(f"  archive size:     {n} images")
    print(f"  atoms:            {len(ATOMS)}")
    print(f"  library size:     {len(archive.library)} predicates")

    # Source breakdown
    print("\n  by source (base + augmented):")
    src = defaultdict(int)
    for m in archive._cache.values():
        src[m.get("_source", "?")] += 1
    for s, c in sorted(src.items(), key=lambda kv: -kv[1]):
        bar = "█" * min(40, int(40 * c / max(src.values())))
        print(f"     {s:22s} {c:5d}  {bar}")

    # Atom statistics
    print("\n  atom value ranges (p10 .. p90):")
    for name in ATOMS:
        values = [m[name] for m in archive._cache.values() if name in m]
        if not values: continue
        v = np.array(values)
        print(f"     {name:20s} {np.percentile(v, 10):.3f}  ..  "
              f"{np.percentile(v, 90):.3f}   "
              f"(std={v.std():.3f})")

    # Cluster structure
    if n >= 20:
        print("\n  cluster structure:")
        try:
            best_k, assignments = cluster_archive(archive, k_range=(3, 10))
            # assignments may be a dict (hash->cid) or a list of ints
            if isinstance(assignments, dict):
                assignments = list(assignments.values())
            assignments = np.asarray(assignments, dtype=int).ravel()
            if assignments.size:
                sizes = np.bincount(assignments)
                print(f"     K = {best_k}, sizes = {sizes.tolist()}")
            else:
                print("     no assignments produced")
        except Exception as e:
            print(f"     cluster failed: {e}")

    print("=" * 64)
