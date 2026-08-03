# Dataset layout for your manually optimized cuts (Phase 1)

Do not commit large mesh binaries. Keep them under `data/raw/` (gitignored).

## Suggested folder convention

```text
data/raw/
  mechanical/
    <project_name>/
      source.stl|source.3mf      # original / uncut (if you still have it)
      parts/
        part_01.stl
        part_02.stl
        ...
      meta.yaml                  # optional notes (see below)
  organic/                       # Phase 4 + trellis-mac generated assets
    <slug>/
      source_image.png           # optional; required if from trellis-mac
      source.glb                 # PBR from trellis-mac (display)
      source.stl                 # watertight mesh used for cutting
      parts/
        part_01.stl
      meta.yaml
```

## meta.yaml example (mechanical)

```yaml
domain: mechanical
printer: bambu_p1s
success: true
notes: "Cut at mid wall for bed fit; 3mm pins added in Studio"
part_count: 2
source_of_cut: bambu_studio | blender | meshmixer | other
```

## meta.yaml example (organic / trellis-mac)

```yaml
domain: organic
source: trellis-mac
generator: microsoft/TRELLIS.2-4B
pipeline_type: "512"
texture_size: 1024
seed: 42
printer: bambu_p1s
source_image: source_image.png
notes: "Figurine from photo; repair holes before cut"
```

See [docs/trellis-mac-integration.md](../docs/trellis-mac-integration.md) for the full sibling-repo plan.

## What we will learn from these

- Seam placement that already worked on P1S
- Part AABB sizes that fit after your orientation
- Mechanical preferences (pins, flat registers) vs organic aesthetic seams

Export from Bambu Studio / whatever you used as individual STLs per part when possible.
