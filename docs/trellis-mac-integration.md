# TRELLIS.2 (trellis-mac) Integration Plan

Upstream **image → 3D** generation for Mac Studio, feeding this repo’s **cut → print** pipeline.
Official Microsoft TRELLIS.2 is CUDA/Linux-only; we use the Apple Silicon port
[shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac).

## Goals

1. Generate organic / figurine meshes on Mac Studio without NVIDIA.
2. Hand those meshes into this project for build-plate validation and contoured cuts.
3. Keep heavy generative deps **out of** the Phase 0 geometry package.

## Non-goals

- Merging PyTorch / MPS / Metal kernels into `cut.py` or `requirements.txt`.
- Training TRELLIS.2 (port is inference-only).
- Replacing mechanical 3MF ingest (Bambu Production) — that path stays primary for Phase 1–3.

## Architecture

```text
┌─────────────────────────────────────┐
│  Sibling: trellis-mac (own venv)    │
│  photo → TRELLIS.2-4B (MPS)         │
│  → GLB (+ optional OBJ, high-res)   │
└─────────────────┬───────────────────┘
                  │ copy / symlink into data/
                  ▼
┌─────────────────────────────────────┐
│  This repo (lightweight geometry)   │
│  ingest → validate AABB vs P1S      │
│  → plane / contour cut → part STLs  │
└─────────────────────────────────────┘
```

**Principle:** trellis-mac is an **upstream mesh factory**. This repo remains the **printability engine**.

| Concern | Owns it |
|---------|---------|
| Image-to-3D, PBR bake, HF weights (~15GB) | `trellis-mac` sibling checkout |
| Build volume, cuts, 3MF ingest, seam ML | `3D-Model-Segmentation-AI` |
| Organic aesthetic seam rules | This repo, Phase 4 |

## Unique functions (what trellis-mac adds)

| Function | Detail |
|----------|--------|
| Image → mesh | TRELLIS.2-4B on PyTorch MPS; ~400K-vert class meshes |
| PBR GLB export | Base color / metallic / roughness textures |
| Mac-native backends | Metal sparse conv (`mtlgemm`), texture bake (`mtldiffrast`), SDPA attention |
| Subject prep | RMBG-2.0 background removal + DINOv3 features (gated HF models) |
| Resolution knobs | `--pipeline-type` `512` / `1024` / `1024_cascade`; `--texture-size` |

### Port limitations (plan around these)

- Decode-time hole fill disabled → may need mesh repair before cutting.
- Texture bake often pre-simplifies (~800K → ~200K faces); use OBJ if full geo is required.
- Inference only; peaks ~18GB unified memory; cool machine recommended (thermal throttle is severe).
- RMBG-2.0 is CC BY-NC — non-commercial unless separately licensed.

## Placement on disk

Recommended sibling layout (not inside this package’s `.venv`):

```text
~/Desktop/Projects/
  3D-Model-Segmentation-AI/     # this repo
  trellis-mac/                  # git clone of the port
```

Generated assets land here (gitignored under `data/raw/`):

```text
data/raw/organic/
  <slug>/
    source_image.png            # prompt image
    source.glb                  # trellis-mac PBR output
    source.obj                  # optional high-res geo (if exported)
    source.stl                  # watertight mesh used for cutting (derived)
    meta.yaml
    parts/                      # after this repo cuts
      part_01.stl
      ...
```

### `meta.yaml` (organic / generated)

```yaml
domain: organic
source: trellis-mac
generator: microsoft/TRELLIS.2-4B
pipeline_type: "512"
texture_size: 1024
seed: 42
printer: bambu_p1s
source_image: source_image.png
notes: "Figurine from photo; cut for P1S after repair"
```

## Phased delivery

### Phase A — Sibling setup (Mac Studio)

1. Clone trellis-mac next to this repo.
2. Install Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`) when available.
3. Hugging Face login + access to DINOv3 and RMBG-2.0.
4. `bash setup.sh` (or `SKIP_METAL=1` fallback).
5. Smoke test: `python generate.py <image.png> --pipeline-type 512`.

**Exit criteria:** One GLB produced; machine has ≥24GB unified memory headroom in practice (~18GB peak).

### Phase B — Manual bridge (no code in this repo yet)

1. Copy GLB/OBJ + source image into `data/raw/organic/<slug>/`.
2. Convert to STL with trimesh (`Scene.to_geometry()`, `repair.fill_holes`). Expect `watertight=False` from the Mac port.
3. **Scale from TRELLIS unit-cube to mm** before build-plate checks. Raw extents are ~1 unit; a typical figurine target is longest axis ≈ 100–150 mm. Example smoke test: scale ≈ 120 → **42.7 × 55.7 × 120 mm**, fits P1S.
4. Write `meta.yaml` including `print_stl`, `scale_to_mm`, and `watertight`.
5. Repair for watertightness before boolean / contoured cuts; display GLB can stay as-generated.

**Exit criteria:** At least one organic project folder follows the convention above.

**Done (2026-08-02 smoke test on Mac Studio):** `data/raw/organic/smoke_test/` with `source.glb`, `source.stl`, `source_120mm.stl`, `source_image.png`, `meta.yaml`. trellis-mac sibling setup + first `generate.py` run succeeded.

### Phase C — Thin CLI glue in this repo

Implemented: `segmentation-ai generate-from-image` shells out to trellis-mac (no PyTorch in this venv).

```bash
cd ~/Desktop/Projects/3D-Model-Segmentation-AI
source .venv/bin/activate
pip install -e .

segmentation-ai generate-from-image path/to.png \
  --slug my_figure \
  --trellis-root ../trellis-mac \
  --pipeline-type 512 \
  --target-mm 120
```

Re-stage an existing GLB without regenerating:

```bash
segmentation-ai generate-from-image path/to.png \
  --slug my_figure \
  --skip-generate \
  --glb /path/to/model.glb \
  --target-mm 120
```

Behavior:

1. Subprocess: `trellis-mac/.venv/bin/python generate.py …`
2. Stage under `data/raw/organic/<slug>/` (`source_image`, `source.glb`, `source.stl`, `source_<N>mm.stl`)
3. Write `meta.yaml` with scale + P1S fit
4. Stop before organic cutting (Phase D / roadmap Phase 4)

Config: `config/defaults.yaml` → `trellis.*`

**Exit criteria:** One command produces a staged organic project ready for human cut or later Phase 4.

**Done (2026-08-02 Mac Studio):**
- Dry-run: `smoke_test_cli` via `--skip-generate` → Fits P1S at 120 mm
- Full run: `shoe_cli_full` via subprocess to trellis-mac (~71s gen+bake, weights cached) → `source_120mm.stl`, Fits P1S: True, Watertight: False

### Phase D — Organic cut track (aligns with roadmap Phase 4)

1. Repair pass (fill small holes from skipped `cumesh`).
2. Scale / orient to mm and printer frame.
3. Aesthetic seam proposals (separate from mechanical pins/registers).
4. Contoured cut + P1S validation.
5. Dataset examples feed future organic seam model (parallel to mechanical Phase 3).

**Exit criteria:** Photo → printable multi-part organic kit without leaving the two-repo workflow.

## Dependency boundary

| Package | In this repo’s default venv? |
|---------|------------------------------|
| numpy, trimesh, manifold3d, shapely, pyyaml | Yes |
| torch, trellis2, mtlgemm, nvdiffrast/Metal ports, flash-attn substitutes | **No** — trellis-mac venv only |

Do not add TRELLIS to `pyproject.toml` / `requirements.txt` unless we later decide on a monorepo optional extra (`pip install segmentation-ai[trellis]` that still shells out or documents the sibling).

## Risk register

| Risk | Mitigation |
|------|------------|
| Thermal throttle (minutes → tens of minutes) | Cool Mac Studio; avoid stacking heavy jobs |
| Non-watertight meshes | Repair before boolean cuts |
| License (RMBG NC) | Personal/research OK; commercial needs BRIA license or alternate matting |
| Huge meshes crash cut/boolean | Decimate for cut path; keep hi-res GLB for display |
| HF gated model access | Document login + URL approvals in setup checklist |
| TRELLIS outputs ~1-unit normalized | Always scale to mm before P1S checks / cutting |
| MacBook vs Studio split | Push via GitHub (`gene7313-jpg/3D-Model-Segmentation-AI`); Studio clones; keep `data/raw` local |

## Decision log

| Decision | Choice | Why |
|----------|--------|-----|
| Merge strategy | Sibling + thin CLI | Keeps geometry stack installable and fast |
| When to automate | After Phase 0–1 solid | Generation is useless if cut/validate isn’t trusted |
| Primary domain for TRELLIS | `organic/` | Matches roadmap Phase 4; mechanical stays 3MF-led |
| Default pipeline | `512` | Faster iteration; raise to `1024` when quality needs it |

## References

- Upstream model: [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (NVIDIA, ≥24GB VRAM, Linux)
- Mac port: [shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac)
- Weights: [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
- This repo roadmap: [README.md](../README.md) Phases 0–4
- Dataset convention: [data/README.md](../data/README.md)
