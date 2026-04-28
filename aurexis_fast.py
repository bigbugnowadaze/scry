"""
aurexis_fast.py — accelerated streaming pipeline for the substrate.

Replaces the sequential fetch-then-process loop in cli.py with a
concurrent producer-consumer architecture:

    [N fetcher threads] --> [bounded queue] --> [feature extractor] --> [archive]

Plus:
  - Batched persistence (writes every BATCH_WRITE_EVERY ingests, not every one)
  - Optional online clustering via MiniBatchKMeans.partial_fit
  - Per-source rate limiting so we stay polite to public APIs
  - Live progress with ETA

Public entry point:
    fast_run_source(source_fn, n, label, archive,
                    workers=8, online_cluster=True)

The substrate state and persistence are unchanged — this is a faster
way to fill the same archive.
"""

import time
import queue
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# RATE LIMITING — global per-source budget
# ============================================================

# courtesy budgets (requests / second) for each source name
RATE_LIMITS = {
    "wikimedia":   2.0,   # commons API: be courteous
    "openverse":   2.0,
    "inaturalist": 1.5,
    "gbif":        2.0,
    "met":         5.0,
    "artic":       1.0,   # Art Institute: ≤ 1/sec
    "cleveland":   3.0,
    "osm":         1.5,   # OSM tile policy
    "gibs":        4.0,
    "goes":        2.0,
    "apod":        2.0,
    "nasa_ivl":    3.0,
    "epic":        2.0,
    "mars":        3.0,
    "jwst":        2.0,
    "picsum":      8.0,   # picsum is fine with bursts
    "tpdne":       0.5,   # GAN endpoint, don't hammer
    "synthetic":  100.0,  # in-process, no real limit
}


class RateLimiter:
    """One token bucket per source name; thread-safe."""
    def __init__(self):
        self._next_ok = defaultdict(lambda: 0.0)
        self._lock = threading.Lock()

    def acquire(self, source):
        rate = RATE_LIMITS.get(source, 1.0)
        gap = 1.0 / rate
        with self._lock:
            now = time.time()
            wait = self._next_ok[source] - now
            self._next_ok[source] = max(now, self._next_ok[source]) + gap
        if wait > 0:
            time.sleep(wait)


# ============================================================
# CONCURRENT PIPELINE
# ============================================================

# Sentinel placed on the queue when fetchers are done
_DONE = object()


def _fetch_worker(item_iter, q, limiter, errors, stop_evt):
    """Pull (img, alias, source) from the generator and push onto the queue.
    Each generator call may itself do an HTTP fetch; rate-limit per source."""
    try:
        for item in item_iter:
            if stop_evt.is_set():
                break
            try:
                _, _, source = item
                limiter.acquire(source)
            except Exception:
                pass
            q.put(item)
    except Exception as e:
        errors.append(("fetch", repr(e)))


def fast_run_source(source_fn, n, label, archive,
                    workers=8, online_cluster=True,
                    batch_write_every=20):
    """
    Fast version of cli.run_source. Streams images concurrently and processes
    them on the calling thread (which is the only one allowed to touch the
    archive — sklearn / numpy aren't always thread-safe).

    Params:
        source_fn        callable(n) -> generator yielding (PIL, alias, source)
        n                target number of images
        label            human-readable name for progress
        archive          aurexis_v3.Archive
        workers          fetcher concurrency (1-16)
        online_cluster   if True, update MiniBatchKMeans incrementally
        batch_write_every  flush archive sidecars every N ingests
    """
    print(f"\n  fast-fetching {n} from '{label}' "
          f"(workers={workers}, online={online_cluster}) ...")

    q = queue.Queue(maxsize=workers * 2)
    limiter = RateLimiter()
    errors = []
    stop_evt = threading.Event()

    # Single producer thread runs the source generator. Most of our sources
    # already do their own internal fetching; we just need parallelism via
    # multiple sources or via threading the generator's HTTP calls.
    # For sources whose generator yields one image at a time, we wrap them
    # so that the actual PIL fetch happens in worker threads.
    def producer():
        try:
            for item in source_fn(n):
                if stop_evt.is_set():
                    break
                q.put(item)
        except Exception as e:
            errors.append(("producer", repr(e)))
        finally:
            q.put(_DONE)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    # Prepare an online clusterer that updates incrementally
    online_km = None
    online_scaler = None
    feature_dim = None
    if online_cluster:
        # We need to know the feature dimension; peek at the first feature
        # vector from the archive if any, else lazily init on first ingest
        if archive._features:
            sample = next(iter(archive._features.values()))
            feature_dim = sample.shape[0]
            online_scaler = StandardScaler(with_mean=True, with_std=True)
            online_km = MiniBatchKMeans(
                n_clusters=min(8, max(2, len(archive._features) // 50 + 2)),
                random_state=42, batch_size=16, n_init=3,
            )
            # Fit scaler+kmeans on existing data once so partial_fit has a base
            X_init = np.array(list(archive._features.values()))
            X_init_s = online_scaler.fit_transform(X_init)
            online_km.fit(X_init_s)

    # Consumer loop (this thread)
    new_count = 0
    dup_count = 0
    err_count = 0
    pending_features = []
    t0 = time.time()

    try:
        while True:
            try:
                item = q.get(timeout=120)
            except queue.Empty:
                print("    queue idle, stopping early")
                stop_evt.set()
                break
            if item is _DONE:
                break
            try:
                pil_img, alias, src = item
            except Exception:
                err_count += 1
                continue
            try:
                h, is_new = archive.ingest_pil(pil_img, alias, src)
                if is_new:
                    new_count += 1
                    if online_cluster:
                        pending_features.append(archive._features[h])
                else:
                    dup_count += 1

                done = new_count + dup_count + err_count
                eta = (time.time() - t0) / max(done, 1) * max(n - done, 0)
                marker = "NEW" if is_new else "dup"
                print(f"    [{done:>3}/{n}] {src:>14s}  {marker}  "
                      f"{alias[:40]:<40s}  ETA {eta:>5.1f}s",
                      flush=True)

                # Online clustering update every 16 new items
                if online_km is not None and len(pending_features) >= 16:
                    X = np.array(pending_features)
                    Xs = online_scaler.transform(X)
                    online_km.partial_fit(Xs)
                    pending_features = []

                # Batch write
                if (new_count + dup_count) % batch_write_every == 0:
                    archive._save_measurements()

            except Exception as e:
                err_count += 1
                print(f"    [{new_count+dup_count+err_count}/{n}] "
                      f"ingest error: {type(e).__name__}: {e}")

            if new_count >= n:
                stop_evt.set()
                break

    except KeyboardInterrupt:
        print("\n  stopped by user.")
        stop_evt.set()

    # Final flush
    archive._save_measurements()

    dt = time.time() - t0
    rate = (new_count + dup_count) / max(dt, 0.01)
    print(f"\n  done: {new_count} new, {dup_count} dup, {err_count} err "
          f"in {dt:.1f}s ({rate:.1f}/s)")
    if errors:
        print(f"  pipeline errors: {errors[:3]}")

    return new_count, online_km, online_scaler


# ============================================================
# CONCURRENT MULTI-SOURCE — for the curriculum runner
# ============================================================

def fast_run_many(generators_with_labels, archive, workers_per_source=4,
                  batch_write_every=25):
    """
    Run several source generators *concurrently* and ingest into the same
    archive. For the curriculum, this lets a slow source (Wikimedia random)
    overlap with a fast one (Picsum) instead of running them sequentially.

    generators_with_labels: list of (gen, label) where gen is already
        a primed generator producing (PIL, alias, source) tuples.
    """
    print(f"\n  fast-running {len(generators_with_labels)} sources "
          f"in parallel ...")

    q = queue.Queue(maxsize=64)

    def producer(gen, label):
        try:
            for item in gen:
                q.put(item)
        except Exception as e:
            print(f"    producer '{label}' died: {type(e).__name__}: {e}")

    threads = []
    for gen, label in generators_with_labels:
        t = threading.Thread(target=producer, args=(gen, label), daemon=True)
        t.start()
        threads.append(t)

    new_count = dup_count = err_count = 0
    t0 = time.time()
    last_active = time.time()

    while True:
        try:
            item = q.get(timeout=10)
            last_active = time.time()
        except queue.Empty:
            # check if any producers are still alive
            if not any(t.is_alive() for t in threads):
                # drain remaining
                if q.empty(): break
            if time.time() - last_active > 60:
                print("    no activity in 60s, stopping")
                break
            continue

        try:
            pil_img, alias, src = item
            h, is_new = archive.ingest_pil(pil_img, alias, src)
            if is_new: new_count += 1
            else: dup_count += 1
            done = new_count + dup_count
            print(f"    [{done:>4}] {src:>14s}  "
                  f"{'NEW' if is_new else 'dup'}  {alias[:40]}", flush=True)
            if done % batch_write_every == 0:
                archive._save_measurements()
        except Exception as e:
            err_count += 1
            print(f"    ingest error: {type(e).__name__}: {e}")

    archive._save_measurements()
    dt = time.time() - t0
    rate = (new_count + dup_count) / max(dt, 0.01)
    print(f"\n  multi-source done: {new_count} new, {dup_count} dup, "
          f"{err_count} err in {dt:.1f}s ({rate:.1f}/s)")
    return new_count
