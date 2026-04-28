# Aurexis — live image substrate

A measurement substrate that learns from images streamed live from public APIs.
**No raw images are stored on your disk** — only computed measurements,
features, and learned predicates are persisted.

## Quick start (Windows)

1. Drop these files into a folder:
   - `aurexis_v3.py`
   - `sources.py`
   - `cli.py`
   - `aurexis.bat`
2. Double-click `aurexis.bat`. On first run it will install Python libraries
   automatically (~3 minutes), then open the menu.

## Quick start (any OS)

```
pip install requests numpy scipy pillow scikit-image scikit-learn kymatio joblib
python cli.py
```

## What the menu does

```
[1] Browse sources by stage     -> pick a stage, pick a source, set N
[2] Run full curriculum         -> runs all stages 0-7 with sensible defaults
[3] Show what was learned       -> sources, predicates, clusters, anomalies
[4] Teach a concept             -> label positive/negative aliases, learn it
[5] Quit
```

## Live sources wired in

### Works without any signup or API key

| Source                       | Stage | Content |
|------------------------------|-------|---------|
| Synthetic uniform/gradient/perlin | 0 | Locally generated; calibrates atoms |
| Lorem Picsum                 | 1 | Random curated photographs |
| Wikimedia Commons random     | 2 | ~110M files: photos, scans, art, science |
| Openverse                    | 2 | 800M+ CC-licensed aggregated images |
| iNaturalist                  | 3 | Wildlife photos by species (Aves, Insecta, Plantae) |
| GBIF                         | 3 | Biodiversity occurrence images |
| Met Museum                   | 4 | ~400K CC0 public-domain artworks |
| Art Institute of Chicago     | 4 | ~50K CC0 artworks |
| Cleveland Museum             | 4 | ~37K CC0 artworks |
| NASA APOD                    | 5 | Astronomy Picture of the Day archive (1995-now) |
| NASA Image & Video Library   | 5 | 140K+ NASA images (galaxies, nebulae, missions) |
| NASA EPIC                    | 5 | Full-disc Earth from L1 |
| NASA Mars Rover              | 5 | Curiosity, Opportunity, etc. |
| JWST via Wikimedia           | 5 | James Webb Space Telescope releases |
| NASA GIBS satellite tiles    | 6 | True-color Earth from MODIS |
| NOAA GOES weather satellite  | 6 | Live full-disc Earth, weather |
| OpenStreetMap tiles          | 6 | Cartographic renderings |
| NOAA US weather radar        | 6 | Live mosaic |
| thispersondoesnotexist       | 7 | GAN-generated faces |

NASA endpoints work with `DEMO_KEY` at low rate. Set `NASA_API_KEY` for higher rate.

### Optional — set environment variable to enable

| Source       | Env var               | Free key from |
|--------------|-----------------------|---------------|
| Pexels       | `PEXELS_API_KEY`      | pexels.com/api/ |
| Pixabay      | `PIXABAY_API_KEY`     | pixabay.com/api/docs/ |
| Unsplash     | `UNSPLASH_ACCESS_KEY` | unsplash.com/developers |
| Rijksmuseum  | `RIJKSMUSEUM_KEY`     | rijksmuseum.nl/en/rijksstudio |
| Smithsonian  | `DATA_GOV_KEY`        | api.data.gov (NASA key works too) |
| Europeana    | `EUROPEANA_KEY`       | api.europeana.eu |

To set in Windows persistently:
```
setx PEXELS_API_KEY your_key_here
```
Then close and reopen Command Prompt for the change to take effect.

## Files

- `aurexis_v3.py` — the substrate core. Atoms, predicates, archive,
  unsupervised methods, label-based learning, persistent library.
- `sources.py` — generators for every live image source.
- `cli.py` — menu-driven runner. The single entry point you interact with.
- `aurexis.bat` — Windows launcher. Installs deps if needed, then runs cli.

## Persistence

Everything goes into `aurexis_archive/` next to the script:

- `_measurements.json` — atom values per image (human-readable)
- `_features.npz` — rich feature vectors (numpy)
- `_predicates.joblib` — learned predicate library

Delete the folder to start over. Typical size: 1-2 KB of measurements + 1-2 KB
of features per image, regardless of the original image's resolution.

## What "learning" means here

When you run a source, the substrate:
1. Streams images one at a time, in memory
2. Computes 9 deterministic measurement atoms per image
3. Computes a rich feature vector (scattering + HOG + LBP + color hist)
4. Stores both, discards the image
5. Re-runs K-means clustering with auto-K selection
6. Turns each cluster into a named, learned predicate
7. Saves the library

Each new batch of images can shift the cluster structure and refine the
learned predicates.

## Honest limitations

- Atoms here are *low-level perceptual statistics* — brightness, edges,
  texture distributions. No learned semantic encoder (no CLIP), so similarity
  is geometric/textural, not conceptual. "Find me cats" does not work;
  "find me images that look like that brick texture" does.
- Public APIs sometimes go down or rate-limit. Each source is wrapped in
  try/except; if one fails, the pipeline keeps running.
- OpenStreetMap and GBIF ask for courteous use (≤1-2 req/sec). The defaults
  respect that. Don't crank counts to thousands without thinking.
