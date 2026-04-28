"""
aurexis_batch.py — batched feature extraction for high-throughput training.

The feature extractor in aurexis_v3.py runs per-image. When ingesting
thousands of images from a HuggingFace stream, that becomes the bottleneck
(scattering + HOG + LBP each take ~50-100ms on CPU).

This module batches images and uses multiprocessing.Pool to compute
atoms + rich features in parallel across all CPU cores. On a typical
4-core laptop, the speedup is 3-4x; on 8 cores, 6-7x.

Public API:
    bulk_ingest(archive, image_iterable, batch_size=32, workers=None)

The image_iterable should yield (PIL.Image, alias, source) tuples
exactly like the sources.py generators.
"""

import io
import time
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image


# ============================================================
# WORKER FUNCTION — runs in subprocess
# ============================================================

def _process_one(args):
    """Compute atoms + rich features for a single image.
    Runs in a worker process; receives bytes and metadata, returns dict.

    Args is (image_bytes, alias, source). We re-decode from bytes inside
    the worker because PIL images don't pickle cleanly across processes
    on all platforms.
    """
    image_bytes, alias, source = args
    try:
        # Re-decode in worker
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img)

        # Compute hash
        h = hashlib.sha256(image_bytes).hexdigest()[:16]

        # Import atoms inside worker (each process imports once)
        from aurexis_v3 import ATOMS, rich_features

        # Compute atoms
        m = {a.name: float(a(arr)) for a in ATOMS.values()}
        m["_alias"] = alias
        m["_source"] = source
        m["_shape"] = list(arr.shape)

        # Compute rich features
        feat = rich_features(arr).astype(np.float32)

        return (h, m, feat, None)
    except Exception as e:
        return (None, None, None, f"{type(e).__name__}: {e}")


# ============================================================
# BATCH PIPELINE
# ============================================================

def bulk_ingest(archive, image_iterable, batch_size=32, workers=None,
                max_images=None, progress_every=50):
    """
    Stream images through a parallel feature pipeline into the archive.

    Args:
        archive          aurexis_v3.Archive (modified in place)
        image_iterable   yields (PIL.Image, alias, source)
        batch_size       images submitted to workers per round
        workers          worker processes (default: cpu_count - 1)
        max_images       hard cap on total images to ingest
        progress_every   how often to print progress

    Returns:
        (n_new, n_dup, n_err, elapsed_seconds)
    """
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    print(f"\n  bulk-ingest: batch_size={batch_size}, workers={workers}")

    n_new = n_dup = n_err = 0
    t0 = time.time()
    pending_batch = []

    def submit_batch(executor, batch):
        """Convert PIL images to bytes and submit to pool. Returns futures."""
        tasks = []
        for pil_img, alias, source in batch:
            try:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG", optimize=False)
                tasks.append((buf.getvalue(), alias, source))
            except Exception:
                tasks.append(None)
        # filter out failed serializations
        valid_tasks = [t for t in tasks if t is not None]
        return [executor.submit(_process_one, t) for t in valid_tasks]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        items_seen = 0
        for item in image_iterable:
            if max_images and items_seen >= max_images:
                break
            items_seen += 1
            try:
                pil_img, alias, source = item
            except Exception:
                n_err += 1
                continue
            pending_batch.append((pil_img, alias, source))

            # Submit a full batch
            if len(pending_batch) >= batch_size:
                futures.extend(submit_batch(executor, pending_batch))
                pending_batch = []

                # Drain any completed futures so we don't queue infinitely
                while len(futures) >= workers * 4:
                    fut = futures.pop(0)
                    _consume(fut, archive,
                             lambda d, dup, err: None)
                    n_total = n_new + n_dup + n_err
                    if n_total % progress_every == 0 and n_total > 0:
                        rate = n_total / max(time.time() - t0, 0.01)
                        print(f"    progress: {n_total} done "
                              f"({n_new} new, {n_dup} dup, {n_err} err) "
                              f"@ {rate:.1f}/s")
                    # accumulate
                    delta = _last_delta
                    n_new += delta[0]; n_dup += delta[1]; n_err += delta[2]

        # Submit final partial batch
        if pending_batch:
            futures.extend(submit_batch(executor, pending_batch))

        # Drain remaining
        for fut in as_completed(futures):
            _consume(fut, archive, lambda d, dup, err: None)
            delta = _last_delta
            n_new += delta[0]; n_dup += delta[1]; n_err += delta[2]
            n_total = n_new + n_dup + n_err
            if n_total % progress_every == 0:
                rate = n_total / max(time.time() - t0, 0.01)
                print(f"    progress: {n_total} done "
                      f"({n_new} new, {n_dup} dup, {n_err} err) "
                      f"@ {rate:.1f}/s")

    archive._save_measurements()
    elapsed = time.time() - t0
    rate = (n_new + n_dup) / max(elapsed, 0.01)
    print(f"\n  bulk-ingest done: {n_new} new, {n_dup} dup, {n_err} err "
          f"in {elapsed:.1f}s ({rate:.1f}/s)")
    return n_new, n_dup, n_err, elapsed


# Module-level scratch for delta tracking (avoids closure scope issues)
_last_delta = [0, 0, 0]


def _consume(future, archive, _on_done):
    """Apply one worker's result to the archive."""
    global _last_delta
    _last_delta = [0, 0, 0]
    try:
        h, m, feat, err = future.result(timeout=120)
        if err or h is None:
            _last_delta[2] = 1  # error
            return
        if h in archive._cache:
            _last_delta[1] = 1  # dup
            return
        archive._cache[h] = m
        archive._features[h] = feat
        _last_delta[0] = 1  # new
    except Exception:
        _last_delta[2] = 1


# ============================================================
# CONVENIENCE: SIMPLE IN-PROCESS BATCHING (no multiprocessing)
# ============================================================

def bulk_ingest_simple(archive, image_iterable, max_images=None,
                       progress_every=50):
    """Single-process bulk ingestion. Use when multiprocessing has issues
    (e.g., on some Windows configurations) or for small batches."""
    n_new = n_dup = n_err = 0
    t0 = time.time()
    seen = 0
    for item in image_iterable:
        if max_images and seen >= max_images: break
        seen += 1
        try:
            pil_img, alias, source = item
            h, is_new = archive.ingest_pil(pil_img, alias, source)
            if is_new: n_new += 1
            else: n_dup += 1
        except Exception:
            n_err += 1
        total = n_new + n_dup + n_err
        if total % progress_every == 0 and total > 0:
            rate = total / max(time.time() - t0, 0.01)
            print(f"    [simple] {total} done "
                  f"({n_new} new, {n_dup} dup) @ {rate:.1f}/s")

    archive._save_measurements()
    elapsed = time.time() - t0
    print(f"\n  simple bulk-ingest: {n_new} new, {n_dup} dup, {n_err} err "
          f"in {elapsed:.1f}s")
    return n_new, n_dup, n_err, elapsed
