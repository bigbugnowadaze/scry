"""
aurexis_v3.py — extended substrate for offline learning.

What's new versus v2:
  - Richer atoms via skimage:
      * HOG (Histogram of Oriented Gradients): shape signature
      * LBP (Local Binary Patterns): micro-texture
      * Multi-scale color histograms: global appearance
  - "Rich feature vector" = concat(scattering, HOG, LBP, color hist)
    used for similarity search instead of pure scattering.
  - Unsupervised structure discovery:
      * cluster_archive(): K-means with auto-K (silhouette score)
      * find_anomalies(): Isolation Forest on atom values
      * pca_axes(): dominant axes of variation
  - Persistent predicate library: joblib-saved across sessions.
    Each grow run accumulates more learned predicates.
  - explain(): for any archived image, lists matching predicates,
    cluster ID, anomaly score, and nearest neighbors.

All deterministic. All offline. No model weights needed.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json, hashlib, warnings
from scipy import ndimage
from scipy.stats import entropy as shannon_entropy
from kymatio.scattering2d.frontend.numpy_frontend import ScatteringNumPy2D
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

warnings.filterwarnings("ignore")


# -------------------- FEATURE EXTRACTORS --------------------
_SCATTERING_SHAPE = (128, 128)
_SCATTERING = ScatteringNumPy2D(J=2, shape=_SCATTERING_SHAPE, L=8)


def _grayscale(img):
    if img.ndim == 2: return img.astype(np.float32)
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def _resize_gray(img, size=(128, 128)):
    g = _grayscale(img) / 255.0
    pil = Image.fromarray((g * 255).astype(np.uint8)).resize(size)
    return np.asarray(pil).astype(np.float32) / 255.0


def _scattering_features(img):
    g = _resize_gray(img, _SCATTERING_SHAPE)
    coeffs = _SCATTERING(g)
    raw = coeffs.mean(axis=(-1, -2))
    return np.log1p(np.abs(raw[1:]) * 100)  # drop DC, log-norm


def _hog_features(img):
    """Gradient-based shape signature. Reduced HOG for compact features."""
    g = _resize_gray(img, (64, 64))
    return hog(g, orientations=8, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True)


def _lbp_histogram(img):
    """Uniform LBP histogram — 26-bin local-texture signature."""
    g = (_resize_gray(img, (64, 64)) * 255).astype(np.uint8)
    lbp = local_binary_pattern(g, P=24, R=3, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    return hist.astype(np.float32) / (hist.sum() + 1e-9)


def _color_histogram(img):
    """8x8x8 RGB histogram, log-normalized — global color signature."""
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    h, _ = np.histogramdd(
        img.reshape(-1, 3), bins=(8, 8, 8), range=((0,256),(0,256),(0,256))
    )
    h = h.ravel().astype(np.float32) / (h.sum() + 1e-9)
    return np.log1p(h * 1000)


def rich_features(img):
    """Combined feature vector for similarity: scattering + HOG + LBP + color."""
    return np.concatenate([
        _scattering_features(img),
        _hog_features(img),
        _lbp_histogram(img),
        _color_histogram(img),
    ])


# -------------------- ATOMS --------------------

class Atom:
    def __init__(self, name, fn, signal_type, output_type):
        self.name, self.fn = name, fn
        self.signal_type, self.output_type = signal_type, output_type
    def __call__(self, signal): return float(self.fn(signal))
    def __repr__(self): return f"Atom<{self.name}>"


mean_brightness = Atom("mean_brightness",
    lambda i: _grayscale(i).mean() / 255.0, "Image", "ratio[0,1]")
contrast = Atom("contrast",
    lambda i: _grayscale(i).std() / 255.0, "Image", "ratio[0,1]")
edge_density = Atom("edge_density",
    lambda i: (np.abs(ndimage.sobel(_grayscale(i), 0)) +
               np.abs(ndimage.sobel(_grayscale(i), 1))).mean() / 255.0,
    "Image", "ratio[0,1]")
dark_fraction = Atom("dark_fraction",
    lambda i: float((_grayscale(i) < 60).mean()), "Image", "ratio[0,1]")
bright_fraction = Atom("bright_fraction",
    lambda i: float((_grayscale(i) > 200).mean()), "Image", "ratio[0,1]")
color_entropy = Atom("color_entropy",
    lambda i: float(shannon_entropy(np.histogram(i.ravel(), bins=32)[0] + 1e-9)),
    "Image", "nats")
hog_energy = Atom("hog_energy",
    lambda i: float(np.linalg.norm(_hog_features(i))), "Image", "ratio")
lbp_uniformity = Atom("lbp_uniformity",
    lambda i: float(_lbp_histogram(i).max()), "Image", "ratio[0,1]")
color_diversity = Atom("color_diversity",
    lambda i: float(shannon_entropy(_color_histogram(i) + 1e-9)), "Image", "nats")

ATOMS = {a.name: a for a in [
    mean_brightness, contrast, edge_density,
    dark_fraction, bright_fraction, color_entropy,
    hog_energy, lbp_uniformity, color_diversity,
]}


# -------------------- PREDICATES --------------------

class Predicate:
    def __init__(self, name, evaluator, kind="rule", model=None, feature_names=None):
        self.name = name
        self.evaluator = evaluator
        self.kind = kind                      # "rule" | "learned" | "cluster"
        self.model = model                    # for learned: (scaler, clf)
        self.feature_names = feature_names    # for learned: list of atom names
    def __call__(self, m): return bool(self.evaluator(m))
    def __and__(self, o): return Predicate(f"({self.name} AND {o.name})",
                                            lambda m: self(m) and o(m))
    def __or__(self, o):  return Predicate(f"({self.name} OR {o.name})",
                                            lambda m: self(m) or o(m))
    def __invert__(self): return Predicate(f"NOT {self.name}",
                                            lambda m: not self(m))
    def __repr__(self):   return f"Predicate<{self.name}>"


def threshold(atom, op, value):
    ops = {">": lambda a,b: a>b, "<": lambda a,b: a<b,
           ">=": lambda a,b: a>=b, "<=": lambda a,b: a<=b}
    return Predicate(f"{atom.name} {op} {value:.3f}",
                     lambda m: ops[op](m[atom.name], value), kind="rule")


# -------------------- LEARNED PREDICATES --------------------

def learn_predicate_from_labels(name, archive, positive_hashes, negative_hashes):
    """Fit a logistic regression on atom values from labeled examples."""
    feature_names = [a.name for a in ATOMS.values()]
    X_pos = np.array([[archive._cache[h][n] for n in feature_names] for h in positive_hashes])
    X_neg = np.array([[archive._cache[h][n] for n in feature_names] for h in negative_hashes])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*len(X_pos) + [0]*len(X_neg))

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)

    coefs = sorted(zip(feature_names, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    top3 = ", ".join(f"{n}({c:+.2f})" for n, c in coefs[:3])

    def evaluator(m):
        x = np.array([[m[n] for n in feature_names]])
        return bool(clf.predict(scaler.transform(x))[0])

    return Predicate(f"learned[{name}]<{top3}>", evaluator,
                     kind="learned", model=(scaler, clf), feature_names=feature_names)


# -------------------- ARCHIVE --------------------

class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, parents=True)
        self.sidecar = self.root / "_measurements.json"
        self.feat_file = self.root / "_features.npz"
        self.lib_file = self.root / "_predicates.joblib"
        self._cache = json.loads(self.sidecar.read_text()) if self.sidecar.exists() else {}
        self._features = dict(np.load(self.feat_file)) if self.feat_file.exists() else {}
        self.library = self._load_library()
        self._cluster_assignments = {}  # hash -> cluster_id

    def _hash(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

    def ingest(self, path, alias=None):
        path = Path(path)
        h = self._hash(path)
        if h in self._cache: return h
        img = np.asarray(Image.open(path).convert("RGB"))
        m = {a.name: a(img) for a in ATOMS.values()}
        m["_path"] = str(path)
        m["_alias"] = alias or path.stem
        m["_shape"] = list(img.shape)
        m["_source"] = "local_file"
        self._cache[h] = m
        self._features[h] = rich_features(img)
        self._save_measurements()
        return h

    def ingest_pil(self, pil_img, alias, source="streamed"):
        """Ingest a PIL Image directly — never writes the image to disk.
        Only the computed atom values + rich features are persisted."""
        import io as _io
        buf = _io.BytesIO()
        pil_img.save(buf, format="PNG")
        h = hashlib.sha256(buf.getvalue()).hexdigest()[:16]
        if h in self._cache: return h, False  # already known
        img = np.asarray(pil_img.convert("RGB"))
        m = {a.name: a(img) for a in ATOMS.values()}
        m["_alias"] = alias
        m["_source"] = source
        m["_shape"] = list(img.shape)
        # Assign train/test split deterministically by hash
        try:
            from aurexis_eval import assign_split
            m["_split"] = assign_split(h)
        except Exception:
            m["_split"] = "train"
        self._cache[h] = m
        self._features[h] = rich_features(img)
        self._save_measurements()
        return h, True  # newly added

    def _save_measurements(self):
        self.sidecar.write_text(json.dumps(self._cache, indent=2))
        np.savez(self.feat_file, **self._features)

    def _load_library(self):
        """Library is stored as {name: (kind, scaler, clf, feature_names)}.
        Rebuild Predicate objects on load."""
        if not self.lib_file.exists(): return {}
        try:
            stored = joblib.load(self.lib_file)
        except Exception:
            return {}
        rebuilt = {}
        for name, payload in stored.items():
            kind, scaler, clf, feature_names = payload
            def make_evaluator(scaler=scaler, clf=clf, feature_names=feature_names):
                def evaluator(m):
                    x = np.array([[m[n] for n in feature_names]])
                    return bool(clf.predict(scaler.transform(x))[0])
                return evaluator
            coefs = sorted(zip(feature_names, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
            top3 = ", ".join(f"{n}({c:+.2f})" for n, c in coefs[:3])
            rebuilt[name] = Predicate(
                f"learned[{name}]<{top3}>",
                make_evaluator(),
                kind=kind, model=(scaler, clf), feature_names=feature_names,
            )
        return rebuilt

    def save_library(self):
        """Persist only picklable components (scaler, clf, feature_names)."""
        stored = {}
        for name, pred in self.library.items():
            if pred.kind in ("learned", "cluster") and pred.model is not None:
                scaler, clf = pred.model
                stored[name] = (pred.kind, scaler, clf, pred.feature_names)
        joblib.dump(stored, self.lib_file)

    def query(self, predicate):
        return [(h, m) for h, m in self._cache.items() if predicate(m)]

    def all(self): return list(self._cache.items())

    def similar_to(self, hash_or_alias, top_k=5):
        h = hash_or_alias
        if h not in self._features:
            for hh, m in self._cache.items():
                if m.get("_alias") == hash_or_alias:
                    h = hh; break
        query_vec = self._features[h]
        sims = []
        for other_h, vec in self._features.items():
            if other_h == h: continue
            cos = float(np.dot(query_vec, vec) /
                        (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9))
            sims.append((other_h, cos))
        sims.sort(key=lambda kv: -kv[1])
        return sims[:top_k]


# -------------------- UNSUPERVISED LEARNING --------------------

def cluster_archive(archive, k_range=(2, 8)):
    """K-means with auto-K via silhouette score on rich features."""
    hashes = list(archive._features.keys())
    if len(hashes) < k_range[0] + 1:
        return None, {}
    X = np.array([archive._features[h] for h in hashes])
    Xs = StandardScaler().fit_transform(X)

    best_k, best_score, best_labels = None, -1, None
    for k in range(k_range[0], min(k_range[1], len(hashes)) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        if len(set(km.labels_)) < 2: continue
        score = silhouette_score(Xs, km.labels_)
        if score > best_score:
            best_k, best_score, best_labels = k, score, km.labels_

    assignments = dict(zip(hashes, best_labels.tolist()))
    archive._cluster_assignments = assignments
    return best_k, assignments


def cluster_summary(archive, assignments):
    """Group images by cluster and surface representative members."""
    groups = {}
    for h, c in assignments.items():
        groups.setdefault(c, []).append(archive._cache[h]["_alias"])
    return {c: sorted(names) for c, names in sorted(groups.items())}


def find_anomalies(archive, contamination=0.15):
    """Isolation Forest over atom values — detects outliers."""
    hashes = list(archive._cache.keys())
    feature_names = [a.name for a in ATOMS.values()]
    X = np.array([[archive._cache[h][n] for n in feature_names] for h in hashes])
    iso = IsolationForest(contamination=contamination, random_state=42).fit(X)
    scores = iso.score_samples(X)  # lower = more anomalous
    out = sorted(zip(hashes, scores), key=lambda kv: kv[1])
    return [(h, archive._cache[h]["_alias"], s) for h, s in out]


def pca_axes(archive, n_components=2):
    """Return PCA projection of rich features."""
    hashes = list(archive._features.keys())
    X = np.array([archive._features[h] for h in hashes])
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components).fit(Xs)
    proj = pca.transform(Xs)
    return dict(zip(hashes, proj.tolist())), pca.explained_variance_ratio_


# -------------------- CONCEPT DISCOVERY --------------------

def discover_concepts(archive, assignments):
    """For each cluster, fit a learned predicate that distinguishes
    members from non-members. Each becomes a named, persistent predicate."""
    discovered = {}
    by_cluster = {}
    for h, c in assignments.items():
        by_cluster.setdefault(c, []).append(h)

    for c, members in by_cluster.items():
        non_members = [h for h in archive._cache if h not in members]
        if len(members) < 2 or len(non_members) < 2:
            continue
        name = f"cluster_{c}"
        try:
            pred = learn_predicate_from_labels(name, archive, members, non_members)
            discovered[name] = pred
        except Exception:
            continue
    return discovered


# -------------------- EXPLAIN --------------------

def explain(archive, alias_or_hash):
    """Return everything the substrate knows about one image."""
    h = alias_or_hash
    if h not in archive._cache:
        for hh, m in archive._cache.items():
            if m.get("_alias") == alias_or_hash:
                h = hh; break
    if h not in archive._cache:
        return {"error": f"not in archive: {alias_or_hash}"}

    m = archive._cache[h]
    out = {
        "alias": m["_alias"],
        "atoms": {k: v for k, v in m.items() if not k.startswith("_")},
        "cluster": archive._cluster_assignments.get(h, "not yet clustered"),
        "matching_predicates": [
            name for name, pred in archive.library.items() if pred(m)
        ],
        "nearest_neighbors": [
            (archive._cache[oh]["_alias"], round(s, 3))
            for oh, s in archive.similar_to(h, top_k=3)
        ],
    }
    return out
