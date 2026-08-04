# 3D Model Segmentation AI

Local tool to split 3D models into **printable parts** with **contoured cuts**, constrained by printer build volume. Mechanical parts first; organic/figurine rules later (separate track).

## Design principles

1. **Build plate is law** — every proposed part must fit the active printer profile (default: Bambu Lab P1S).
2. **Your successful cuts are ground truth** — manually optimized splits become the training set.
3. **Geometry first, learning second** — Phase 0 ships a deterministic contoured-cut engine; ML learns *where* to cut after the engine is solid.
4. **Mechanical vs organic diverge** — mechanical (flat registers, pins, wall thickness) and organic (aesthetic seams) stay separate rule sets / models.

## Default printer profile

| Setting | Default |
|---------|---------|
| Printer | Bambu Lab P1S |
| Build volume (mm) | 256 × 256 × 256 |
| Margin (mm) | 2.0 (keeps parts off bed edges) |
| Max part AABB | 252 × 252 × 252 |

Override via `config/printers/*.yaml` or CLI flags. Profiles for other beds can be added without changing code.

## Phase roadmap

| Phase | Focus |
|-------|--------|
| **0** | Contoured cut engine + build-plate checks + basic synthetic shapes |
| **1** | Ingest your cut examples (3MF/STL pairs) into a dataset schema |
| **2** | Heuristic seam proposals for mechanical parts |
| **3** | Train a small model from your corrections |
| **4** | Organic / aesthetic track (separate) |

### Image-to-3D (Mac Studio, optional)

[TRELLIS.2 via trellis-mac](docs/trellis-mac-integration.md) is a **sibling upstream generator** (photo → GLB/STL), not merged into the cut engine. On Mac Studio, with `../trellis-mac` set up:

```bash
source .venv/bin/activate
segmentation-ai generate-from-image path/to.png --slug my_figure --target-mm 120
```

Stages `data/raw/organic/<slug>/` then cut for print:

```bash
# Quality-first split (recommended)
segmentation-ai process-organic data/raw/organic/my_figure --force-split --repair-mode basic
# FROZEN mating pins: male concatenate + female cut-cap holes / recessed sleeves
segmentation-ai process-organic data/raw/organic/my_figure \
  --force-split --repair-mode basic --with-pins --with-pin-holes
```

Pipeline plan (frozen pin method + remaining tasks): [docs/organic-pipeline.md](docs/organic-pipeline.md). Official TRELLIS.2 requires NVIDIA; the Mac port uses MPS/Metal.

## Quick start (Mac Studio preferred)

```bash
cd ~/Desktop/Projects/3D-Model-Segmentation-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m segmentation_ai.cli demo-basic-shapes
```

## Project layout

```
config/printers/     # Build volume profiles
data/
  raw/               # Your original + cut examples (not committed)
  synthetic/         # Generated basic shapes
docs/                # Design / integration plans
src/segmentation_ai/ # Engine + CLI
```

## Docs

- [TRELLIS.2 / trellis-mac integration plan](docs/trellis-mac-integration.md)
- [Dataset layout](data/README.md)

## Status

Phase 0 scaffold — MacBook workspace for now; move to Mac Studio for heavier training later.
