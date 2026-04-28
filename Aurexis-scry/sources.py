"""
sources.py — live image sources for Aurexis.

Each source yields (PIL.Image, alias, source_name).
Images live in RAM; only measurements + features persist.

URL-level dedup: any URL fetched in a previous run is skipped before
the network call. Persists in `<archive>/_seen_urls.txt`.

Query rotation: text-search sources pick from rotating term lists each
run, so re-running a source pulls genuinely different content.

Optional API keys (env vars):
    NASA_API_KEY, UNSPLASH_ACCESS_KEY, PEXELS_API_KEY, PIXABAY_API_KEY,
    RIJKSMUSEUM_KEY, EUROPEANA_KEY, DATA_GOV_KEY
"""

import io
import os
import time
import random
import datetime
from pathlib import Path
import numpy as np
import requests
from PIL import Image

UA = {"User-Agent": "Aurexis/1.0 (offline-non-LLM-AI; research)"}


# ============================================================
# URL-LEVEL DEDUP
# ============================================================

class SkipURL(Exception):
    """Raised when a URL has already been fetched before."""


SEEN_URLS = set()
_SEEN_FILE = None
_SEEN_DIRTY = 0


def init_url_dedup(archive_dir):
    global SEEN_URLS, _SEEN_FILE
    _SEEN_FILE = Path(archive_dir) / "_seen_urls.txt"
    if _SEEN_FILE.exists():
        try:
            SEEN_URLS = set(_SEEN_FILE.read_text(encoding="utf-8").splitlines())
            print(f"  url-dedup: loaded {len(SEEN_URLS)} previously-seen URLs")
        except Exception:
            SEEN_URLS = set()
    else:
        SEEN_URLS = set()


def _remember_url(url):
    global _SEEN_DIRTY
    SEEN_URLS.add(url)
    _SEEN_DIRTY += 1
    if _SEEN_FILE and _SEEN_DIRTY >= 50:
        flush_url_dedup()


def flush_url_dedup():
    global _SEEN_DIRTY
    if _SEEN_FILE:
        try:
            _SEEN_FILE.write_text("\n".join(sorted(SEEN_URLS)),
                                  encoding="utf-8")
            _SEEN_DIRTY = 0
        except Exception:
            pass


def fetch_image(url, headers=None, params=None, timeout=30):
    if url in SEEN_URLS:
        raise SkipURL(url)
    h = dict(UA)
    if headers: h.update(headers)
    r = requests.get(url, headers=h, params=params, timeout=timeout)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    _remember_url(url)
    return img


def has_key(env_var):
    return bool(os.environ.get(env_var, "").strip())


def _safe_yield(url, alias, source):
    """Try fetching; return tuple or None. Silently skips SkipURL/errors."""
    try:
        img = fetch_image(url)
        return (img, alias, source)
    except SkipURL:
        return None
    except Exception:
        return None


# ============================================================
# QUERY ROTATION
# ============================================================

ART_QUERIES   = ["painting", "sculpture", "drawing", "portrait", "landscape",
                 "still life", "watercolor", "engraving", "tapestry",
                 "manuscript"]
NATURE_QUERIES = ["nature", "forest", "ocean", "mountain", "desert", "river",
                  "wildlife", "flower", "tree", "sky"]
ASTRO_QUERIES = ["galaxy", "nebula", "star cluster", "planet", "moon", "comet",
                 "supernova", "spacecraft", "rocket launch"]
INAT_TAXA     = ["Aves", "Insecta", "Plantae", "Mammalia", "Reptilia",
                 "Amphibia", "Mollusca", "Fungi", "Arachnida", "Crustacea"]


def pick(lst):
    return random.choice(lst)


# ============================================================
# STAGE 0 — SYNTHETIC
# ============================================================

def stream_synthetic(n=20, size=256, kind="perlin"):
    rng = np.random.default_rng()
    for i in range(n):
        if kind == "uniform":
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        elif kind == "gradient":
            x = np.linspace(0, 255, size).astype(np.uint8)
            ch_r = np.tile(x, (size, 1))
            ch_g = np.tile(x[::-1], (size, 1))
            ch_b = np.tile(x.reshape(-1, 1), (1, size))
            arr = np.stack([ch_r, ch_g, ch_b], axis=-1)
            # tiny noise so each gradient image is unique
            arr = (arr.astype(int) + rng.integers(-5, 5, arr.shape)
                  ).clip(0, 255).astype(np.uint8)
        elif kind == "perlin":
            base = rng.normal(128, 40, (size, size, 3))
            for _ in range(3):
                base = (base + np.roll(base, 1, axis=0) +
                        np.roll(base, 1, axis=1)) / 3
            arr = np.clip(base, 0, 255).astype(np.uint8)
        else:
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        yield Image.fromarray(arr), \
              f"synth_{kind}_{rng.integers(0, 1<<30):08x}", "synthetic"


# ============================================================
# STAGE 1
# ============================================================

def stream_picsum(n=20, w=512, h=512):
    fetched = 0; attempts = 0
    while fetched < n and attempts < n * 4:
        attempts += 1
        seed = random.randint(1, 10_000_000)
        item = _safe_yield(f"https://picsum.photos/seed/{seed}/{w}/{h}",
                           f"picsum_{seed}", "picsum")
        if item:
            yield item; fetched += 1


def stream_pexels(query=None, n=20):
    if not has_key("PEXELS_API_KEY"):
        print("    Pexels: PEXELS_API_KEY not set — skipping"); return
    query = query or pick(NATURE_QUERIES)
    headers = {"Authorization": os.environ["PEXELS_API_KEY"]}
    try:
        page = random.randint(1, 30)
        r = requests.get("https://api.pexels.com/v1/search",
                         headers=headers,
                         params={"query": query, "per_page": min(n, 80),
                                 "page": page},
                         timeout=30).json()
        for item in r.get("photos", [])[:n]:
            res = _safe_yield(item["src"]["medium"],
                              f"pexels_{item['id']}", "pexels")
            if res: yield res
    except Exception as e:
        print(f"    Pexels failed: {e}")


# ============================================================
# STAGE 2
# ============================================================

def stream_wikimedia(n=20, thumb=800):
    api = "https://commons.wikimedia.org/w/api.php"
    fetched = 0; attempts = 0
    while fetched < n and attempts < n * 4:
        attempts += 1
        try:
            params = {"action": "query", "format": "json",
                      "generator": "random", "grnnamespace": 6, "grnlimit": 1,
                      "prop": "imageinfo", "iiprop": "url|mime",
                      "iiurlwidth": thumb}
            r = requests.get(api, params=params, headers=UA, timeout=30).json()
            for _, page in r.get("query", {}).get("pages", {}).items():
                info = page.get("imageinfo", [{}])[0]
                mime = info.get("mime", "")
                if not mime.startswith("image/") or "svg" in mime:
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url: continue
                title = page.get("title", f"file_{fetched}").replace("File:", "")
                alias = f"wm_{fetched:03d}_{title[:30].replace(' ', '_')}"
                res = _safe_yield(url, alias, "wikimedia")
                if res:
                    yield res; fetched += 1
            time.sleep(0.5)
        except Exception:
            time.sleep(1)


def stream_openverse(query=None, n=20):
    query = query or pick(NATURE_QUERIES)
    try:
        page = random.randint(1, 20)
        r = requests.get("https://api.openverse.org/v1/images/",
                         params={"q": query, "page_size": min(n, 20),
                                 "page": page,
                                 "license": "cc0,pdm,by"},
                         headers=UA, timeout=30).json()
        for item in r.get("results", [])[:n]:
            url = item.get("thumbnail") or item.get("url")
            if not url: continue
            res = _safe_yield(url,
                              f"openverse_{item.get('id','?')[:8]}",
                              "openverse")
            if res: yield res
    except Exception as e:
        print(f"    Openverse failed: {type(e).__name__}")


def stream_pixabay(query=None, n=20):
    if not has_key("PIXABAY_API_KEY"):
        print("    Pixabay: PIXABAY_API_KEY not set — skipping"); return
    query = query or pick(NATURE_QUERIES)
    try:
        page = random.randint(1, 30)
        r = requests.get("https://pixabay.com/api/",
                         params={"key": os.environ["PIXABAY_API_KEY"],
                                 "q": query, "image_type": "photo",
                                 "page": page,
                                 "per_page": min(max(n, 3), 200)},
                         timeout=30).json()
        for item in r.get("hits", [])[:n]:
            res = _safe_yield(item["webformatURL"],
                              f"pixabay_{item['id']}", "pixabay")
            if res: yield res
    except Exception as e:
        print(f"    Pixabay failed: {e}")


def stream_unsplash(query=None, n=20):
    if not has_key("UNSPLASH_ACCESS_KEY"):
        print("    Unsplash: UNSPLASH_ACCESS_KEY not set — skipping"); return
    headers = {"Authorization": f"Client-ID {os.environ['UNSPLASH_ACCESS_KEY']}"}
    try:
        params = {"count": min(n, 30)}
        if query: params["query"] = query
        r = requests.get("https://api.unsplash.com/photos/random",
                         headers=headers, params=params, timeout=30).json()
        items = r if isinstance(r, list) else [r]
        for item in items[:n]:
            res = _safe_yield(item["urls"]["regular"],
                              f"unsplash_{item['id']}", "unsplash")
            if res: yield res
    except Exception as e:
        print(f"    Unsplash failed: {e}")


# ============================================================
# STAGE 3
# ============================================================

def stream_inaturalist(taxon=None, n=20):
    taxon = taxon or pick(INAT_TAXA)
    try:
        page = random.randint(1, 50)
        r = requests.get("https://api.inaturalist.org/v1/observations",
                         params={"taxon_name": taxon, "photos": "true",
                                 "photo_license": "cc-by,cc-by-sa,cc0",
                                 "per_page": n, "page": page,
                                 "order": "desc",
                                 "order_by": "created_at"},
                         headers=UA, timeout=30).json()
        for obs in r.get("results", [])[:n]:
            for ph in obs.get("photos", []):
                url = ph["url"].replace("square", "medium")
                res = _safe_yield(url, f"inat_{obs['id']}", "inaturalist")
                if res:
                    yield res; break
            time.sleep(0.3)
    except Exception as e:
        print(f"    iNaturalist failed: {type(e).__name__}")


def stream_gbif(taxon_key=212, n=20):
    try:
        offset = random.randint(0, 5000)
        r = requests.get("https://api.gbif.org/v1/occurrence/search",
                         params={"taxonKey": taxon_key,
                                 "mediaType": "StillImage",
                                 "offset": offset,
                                 "limit": n}, timeout=30).json()
        for occ in r.get("results", [])[:n]:
            for m in occ.get("media", []):
                url = m.get("identifier")
                if not url: continue
                res = _safe_yield(url, f"gbif_{occ.get('key','?')}", "gbif")
                if res:
                    yield res; break
            time.sleep(0.3)
    except Exception as e:
        print(f"    GBIF failed: {type(e).__name__}")


# ============================================================
# STAGE 4
# ============================================================

def stream_met(query=None, n=20):
    query = query or pick(ART_QUERIES)
    try:
        ids = requests.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search",
            params={"hasImages": "true", "q": query},
            headers=UA, timeout=30).json().get("objectIDs", []) or []
        random.shuffle(ids)
        fetched = 0
        for oid in ids:
            if fetched >= n: break
            try:
                obj = requests.get(
                    f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}",
                    headers=UA, timeout=30).json()
                if obj.get("isPublicDomain") and obj.get("primaryImageSmall"):
                    res = _safe_yield(obj["primaryImageSmall"],
                                      f"met_{oid}", "met")
                    if res:
                        yield res; fetched += 1
                time.sleep(0.05)
            except Exception:
                continue
    except Exception as e:
        print(f"    Met failed: {e}")


def stream_artic(query=None, n=20):
    query = query or pick(ART_QUERIES)
    try:
        page = random.randint(1, 20)
        r = requests.get("https://api.artic.edu/api/v1/artworks/search",
                         params={"q": query, "limit": n, "page": page,
                                 "fields": "id,image_id,is_public_domain,title"},
                         headers=UA, timeout=30).json()
        for w in r.get("data", []):
            if not (w.get("is_public_domain") and w.get("image_id")):
                continue
            url = f"https://www.artic.edu/iiif/2/{w['image_id']}/full/843,/0/default.jpg"
            res = _safe_yield(url, f"artic_{w['id']}", "artic")
            if res: yield res
            time.sleep(0.5)
    except Exception as e:
        print(f"    Art Institute failed: {e}")


def stream_cleveland(n=20):
    try:
        skip = random.randint(0, 30000)
        r = requests.get("https://openaccess-api.clevelandart.org/api/artworks/",
                         params={"cc0": 1, "has_image": 1,
                                 "limit": n, "skip": skip},
                         headers=UA, timeout=30).json()
        for art in r.get("data", []):
            web = art.get("images", {}).get("web", {}).get("url")
            if not web: continue
            res = _safe_yield(web, f"cma_{art.get('id','?')}", "cleveland")
            if res: yield res
    except Exception as e:
        print(f"    Cleveland failed: {e}")


def stream_rijks(query="Rembrandt", n=20):
    if not has_key("RIJKSMUSEUM_KEY"):
        print("    Rijksmuseum: RIJKSMUSEUM_KEY not set"); return
    try:
        page = random.randint(1, 50)
        r = requests.get("https://www.rijksmuseum.nl/api/en/collection",
                         params={"key": os.environ["RIJKSMUSEUM_KEY"],
                                 "q": query, "ps": n, "p": page,
                                 "imgonly": "True", "format": "json"},
                         timeout=30).json()
        for art in r.get("artObjects", []):
            url = (art.get("webImage") or {}).get("url")
            if not url: continue
            res = _safe_yield(url,
                              f"rijks_{art.get('objectNumber','?')}",
                              "rijksmuseum")
            if res: yield res
    except Exception as e:
        print(f"    Rijksmuseum failed: {e}")


def stream_smithsonian(query=None, n=20):
    key = os.environ.get("DATA_GOV_KEY") or os.environ.get("NASA_API_KEY", "")
    if not key:
        print("    Smithsonian: DATA_GOV_KEY/NASA_API_KEY not set"); return
    query = query or pick(ART_QUERIES)
    try:
        start = random.randint(0, 1000)
        r = requests.get("https://api.si.edu/openaccess/api/v1.0/search",
                         params={"api_key": key,
                                 "q": f"{query} AND online_media_type:Images "
                                      f"AND media_usage:CC0",
                                 "rows": n, "start": start}, timeout=30).json()
        for row in r.get("response", {}).get("rows", []):
            try:
                media = (row["content"]["descriptiveNonRepeating"]
                            ["online_media"]["media"])
                for m in media:
                    if m.get("type") == "Images" and m.get("content"):
                        res = _safe_yield(m["content"],
                                          f"si_{row.get('id','?')[:12]}",
                                          "smithsonian")
                        if res:
                            yield res; break
            except Exception:
                continue
    except Exception as e:
        print(f"    Smithsonian failed: {e}")


def stream_europeana(query=None, n=20):
    if not has_key("EUROPEANA_KEY"):
        print("    Europeana: EUROPEANA_KEY not set"); return
    query = query or pick(ART_QUERIES)
    try:
        start = random.randint(1, 500)
        r = requests.get("https://api.europeana.eu/record/v2/search.json",
                         params={"wskey": os.environ["EUROPEANA_KEY"],
                                 "query": query, "qf": "TYPE:IMAGE",
                                 "reusability": "open",
                                 "rows": n, "start": start,
                                 "media": "true"},
                         timeout=30).json()
        for it in r.get("items", []):
            url = (it.get("edmPreview") or [None])[0] or \
                  (it.get("edmIsShownBy") or [None])[0]
            if not url: continue
            res = _safe_yield(url,
                              f"euro_{it.get('id','?')[:12].replace('/','_')}",
                              "europeana")
            if res: yield res
    except Exception as e:
        print(f"    Europeana failed: {e}")


# ============================================================
# STAGE 5
# ============================================================

def _nasa_key():
    return os.environ.get("NASA_API_KEY", "DEMO_KEY")


def stream_apod(n=20):
    start = datetime.date(1995, 6, 16)
    end = datetime.date.today()
    span = (end - start).days
    fetched = 0; attempts = 0
    while fetched < n and attempts < n * 3:
        attempts += 1
        d = (start + datetime.timedelta(days=random.randint(0, span))).isoformat()
        try:
            r = requests.get("https://api.nasa.gov/planetary/apod",
                             params={"api_key": _nasa_key(), "date": d},
                             timeout=30).json()
            if r.get("media_type") != "image": continue
            url = r.get("hdurl") or r.get("url")
            if not url: continue
            res = _safe_yield(url, f"apod_{d}", "apod")
            if res:
                yield res; fetched += 1
        except Exception:
            pass


def stream_nasa_ivl(query=None, n=20):
    query = query or pick(ASTRO_QUERIES)
    try:
        page = random.randint(1, 20)
        s = requests.get("https://images-api.nasa.gov/search",
                         params={"q": query, "media_type": "image",
                                 "page": page},
                         timeout=30).json()
        items = s.get("collection", {}).get("items", [])[:n*2]
        random.shuffle(items)
        fetched = 0
        for it in items:
            if fetched >= n: break
            href = it.get("href")
            if not href: continue
            try:
                manifest = requests.get(href, timeout=30).json()
                jpegs = [u for u in manifest if u.endswith(".jpg") and "~medium" in u]
                if not jpegs:
                    jpegs = [u for u in manifest if u.endswith(".jpg")]
                if jpegs:
                    nid = it.get("data", [{}])[0].get("nasa_id", f"{fetched}")
                    res = _safe_yield(jpegs[0], f"nasa_{nid[:20]}", "nasa_ivl")
                    if res:
                        yield res; fetched += 1
            except Exception:
                continue
    except Exception as e:
        print(f"    NASA IVL failed: {e}")


def stream_epic(n=10):
    try:
        meta = requests.get("https://api.nasa.gov/EPIC/api/natural",
                            params={"api_key": _nasa_key()}, timeout=30).json()
        random.shuffle(meta)
        for item in meta[:n]:
            d = item["date"][:10].replace("-", "/")
            name = item["image"]
            url = (f"https://api.nasa.gov/EPIC/archive/natural/{d}/png/"
                   f"{name}.png?api_key={_nasa_key()}")
            res = _safe_yield(url, f"epic_{name}", "epic")
            if res: yield res
    except Exception as e:
        print(f"    EPIC failed: {e}")


def stream_mars(rover="curiosity", sol=None, n=20):
    sol = sol or random.randint(100, 3000)
    try:
        r = requests.get(
            f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos",
            params={"sol": sol, "api_key": _nasa_key()}, timeout=30).json()
        for p in r.get("photos", [])[:n]:
            res = _safe_yield(p["img_src"], f"mars_{p['id']}", "mars")
            if res: yield res
    except Exception as e:
        print(f"    Mars failed: {e}")


def stream_jwst(n=20):
    try:
        api = "https://commons.wikimedia.org/w/api.php"
        offset = random.randint(0, 200)
        params = {"action": "query", "format": "json",
                  "generator": "categorymembers",
                  "gcmtitle": "Category:James Webb Space Telescope images",
                  "gcmtype": "file", "gcmlimit": n,
                  "gcmstartsortkeyprefix": chr(ord('a') + offset % 26),
                  "prop": "imageinfo", "iiprop": "url|mime",
                  "iiurlwidth": 1024}
        r = requests.get(api, params=params, headers=UA, timeout=30).json()
        for _, page in r.get("query", {}).get("pages", {}).items():
            info = page.get("imageinfo", [{}])[0]
            if not info.get("mime", "").startswith("image/"): continue
            url = info.get("thumburl") or info.get("url")
            title = page.get("title", "").replace("File:", "")[:30]
            res = _safe_yield(url, f"jwst_{title.replace(' ','_')}", "jwst")
            if res: yield res
    except Exception as e:
        print(f"    JWST failed: {e}")


# ============================================================
# STAGE 6
# ============================================================

def stream_gibs(n=12, layer="MODIS_Terra_CorrectedReflectance_TrueColor",
                date=None, zoom=3):
    date = date or (datetime.date.today() -
                    datetime.timedelta(days=random.randint(2, 30))).isoformat()
    tm = "250m"
    max_x = 2 ** (zoom + 1); max_y = 2 ** zoom
    seen = set(); fetched = 0; attempts = 0
    while fetched < n and attempts < n * 4:
        attempts += 1
        x, y = random.randint(0, max_x - 1), random.randint(0, max_y - 1)
        if (x, y) in seen: continue
        seen.add((x, y))
        url = (f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/"
               f"default/{date}/{tm}/{zoom}/{y}/{x}.jpg")
        res = _safe_yield(url, f"gibs_{date}_{zoom}_{x}_{y}", "gibs")
        if res:
            yield res; fetched += 1


def stream_goes(n=4):
    feeds = [
        ("https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/1808x1808.jpg",
         "goes16_fd"),
        ("https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/1808x1808.jpg",
         "goes19_fd"),
        ("https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/GEOCOLOR/2500x1500.jpg",
         "goes16_conus"),
        ("https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/GEOCOLOR/2500x1500.jpg",
         "goes19_conus"),
    ][:n]
    for url, alias in feeds:
        res = _safe_yield(url, alias, "goes")
        if res: yield res


def stream_osm(n=20, zoom=5):
    max_t = 2 ** zoom
    fetched = 0; attempts = 0
    while fetched < n and attempts < n * 3:
        attempts += 1
        x, y = random.randint(0, max_t - 1), random.randint(0, max_t - 1)
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        res = _safe_yield(url, f"osm_{zoom}_{x}_{y}", "osm")
        if res:
            yield res; fetched += 1
            time.sleep(0.5)


def stream_radar(n=1):
    res = _safe_yield("https://radar.weather.gov/ridge/standard/CONUS_0.gif",
                      "noaa_radar_conus", "noaa_radar")
    if res: yield res


# ============================================================
# STAGE 7
# ============================================================

def stream_tpdne(n=10, sleep=1.0):
    # tpdne returns a fresh face per call; use cache-busting query param
    for i in range(n):
        url = f"https://thispersondoesnotexist.com/image?_t={time.time()}"
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            yield img, f"tpdne_{int(time.time()*1000)}", "tpdne"
        except Exception:
            pass
        time.sleep(sleep)


# ============================================================
# REGISTRY
# ============================================================

SOURCES = {
    "stage_0_calibrate": [
        ("Synthetic uniform noise",  "no key",
         lambda n: stream_synthetic(n, kind="uniform")),
        ("Synthetic gradient",       "no key",
         lambda n: stream_synthetic(n, kind="gradient")),
        ("Synthetic perlin-like",    "no key",
         lambda n: stream_synthetic(n, kind="perlin")),
    ],
    "stage_1_easy_photos": [
        ("Lorem Picsum",             "no key",       stream_picsum),
        ("Pexels (rotating queries)", "PEXELS_API_KEY",
         lambda n: stream_pexels(None, n)),
    ],
    "stage_2_diverse_photos": [
        ("Wikimedia Commons random", "no key",       stream_wikimedia),
        ("Openverse (rotating)",     "no key",
         lambda n: stream_openverse(None, n)),
        ("Pixabay (rotating)",       "PIXABAY_API_KEY",
         lambda n: stream_pixabay(None, n)),
        ("Unsplash random",          "UNSPLASH_ACCESS_KEY",
         lambda n: stream_unsplash(None, n)),
    ],
    "stage_3_wildlife": [
        ("iNaturalist (rotating taxa)", "no key",
         lambda n: stream_inaturalist(None, n)),
        ("iNaturalist Aves",         "no key",
         lambda n: stream_inaturalist("Aves", n)),
        ("iNaturalist Insecta",      "no key",
         lambda n: stream_inaturalist("Insecta", n)),
        ("iNaturalist Plantae",      "no key",
         lambda n: stream_inaturalist("Plantae", n)),
        ("GBIF birds",               "no key",
         lambda n: stream_gbif(212, n)),
    ],
    "stage_4_art_culture": [
        ("Met Museum (rotating)",    "no key",
         lambda n: stream_met(None, n)),
        ("Art Institute Chicago (rotating)", "no key",
         lambda n: stream_artic(None, n)),
        ("Cleveland Museum random",  "no key",       stream_cleveland),
        ("Rijksmuseum 'Rembrandt'",  "RIJKSMUSEUM_KEY",
         lambda n: stream_rijks("Rembrandt", n)),
        ("Smithsonian (rotating)",   "DATA_GOV_KEY",
         lambda n: stream_smithsonian(None, n)),
        ("Europeana (rotating)",     "EUROPEANA_KEY",
         lambda n: stream_europeana(None, n)),
    ],
    "stage_5_astronomy": [
        ("NASA APOD random history", "DEMO_KEY ok",  stream_apod),
        ("NASA Image Library (rotating)", "no key",
         lambda n: stream_nasa_ivl(None, n)),
        ("NASA EPIC (full Earth)",   "DEMO_KEY ok",  stream_epic),
        ("NASA Mars rover Curiosity", "DEMO_KEY ok",
         lambda n: stream_mars("curiosity", None, n)),
        ("JWST via Wikimedia",       "no key",       stream_jwst),
    ],
    "stage_6_earth_above": [
        ("NASA GIBS true-color tiles", "no key",     stream_gibs),
        ("NOAA GOES weather satellite", "no key",    stream_goes),
        ("OpenStreetMap tiles z=5",  "no key",       stream_osm),
        ("NOAA US weather radar",    "no key",       stream_radar),
    ],
    "stage_7_synthetic_faces": [
        ("thispersondoesnotexist",   "no key",       stream_tpdne),
    ],
}

STAGE_ORDER = list(SOURCES.keys())
STAGE_NAMES = {
    "stage_0_calibrate":         "Stage 0 — Calibrate (synthetic, no network)",
    "stage_1_easy_photos":       "Stage 1 — Easy photographs",
    "stage_2_diverse_photos":    "Stage 2 — Diverse photographs",
    "stage_3_wildlife":          "Stage 3 — Wildlife textures",
    "stage_4_art_culture":       "Stage 4 — Art & culture",
    "stage_5_astronomy":         "Stage 5 — Astronomy & science",
    "stage_6_earth_above":       "Stage 6 — Earth from above",
    "stage_7_synthetic_faces":   "Stage 7 — Synthetic faces",
}
