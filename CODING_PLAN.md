# Coding Plan

## Tech Stack

| Component | Choice        |
|-----------|---------------|
| Physics   | PyBullet      |
| Rendering | PyBullet built-in or Pillow post-process |
| Language  | Python 3.14   |
| Platform  | Mesocosm (bench_common SDK) |

## Repository Layout

```
jenga-bench/
├── env.py              # BaseEnv subclass — reset() and step()
├── adapter.py          # HTTP wrapper (provided by bench_common)
├── benchanything.json   # manifest
├── requirements.txt
├── jenga/
│   ├── sim.py          # PyBullet tower setup, physics stepping, settling
│   ├── tower.py        # tower/block/layer definitions, block ID scheme
│   ├── camera.py       # orbital camera state, positioning
│   ├── render.py       # RGB image capture, lighting setup
│   ├── actions.py      # action parsing, validation, force application
│   ├── scoring.py      # removal detection, collapse detection, reward
│   └── replay.py       # step-by-step trace export
├── tests/
└── showcase/
```

## Physics Setup (sim.py)

| Parameter          | Value       |
|--------------------|-------------|
| Mode               | DIRECT (headless) |
| Gravity            | (0, 0, -9.81) |
| Timestep           | 1/240 s     |
| Solver iterations  | 100         |
| Lateral friction   | 0.01        |
| Rolling friction   | 0.02        |
| Spinning friction  | 0.02        |

Deterministic under (seed + action sequence). All randomness via `np.random.default_rng(seed)`. No global RNG, no timestamps, no nondeterministic iteration order.

## Block Constants (tower.py)

```python
BLOCK_L = 0.075   # 7.5 cm
BLOCK_W = 0.025   # 2.5 cm
BLOCK_H = 0.015   # 1.5 cm
BLOCK_MASS = 0.120  # 120 g
```

## Tower Construction (tower.py)

- 18 layers × 3 blocks = 54 blocks
- Bottom layer North-South, alternating
- Each block gets a unique ID: (layer, color)
- Block colors assigned per DESIGN.md color table
- Base: black box, ~4.5cm tall (3 layers), 2.5×2.5m
- Floor: beneath base, for blocks to land on

## Camera (camera.py)

Orbital camera state: target_block, azimuth, pitch, distance. Render from current position after every action. See DESIGN.md Camera State for behavior.

## Rendering (render.py)

- 512×512 RGB PNG
- White background, black base
- Studio-style diffuse lighting: high ambient, moderate diffuse, low specular (wood-like sheen for edge definition)
- Soft shadows for depth cues
- Blocks color-coded per oklch table (convert to RGB for rendering)

## Action Processing (actions.py)

### ChangeViewpoint
Update camera state. No physics step.

### Push
1. Validate: block exists, face matches block orientation
2. Compute force vector from cardinal face direction
3. Compute contact point from 3×3 grid position on face
4. Apply bell-curve-ramped force over fixed duration (quasi-static)
5. Run settle loop
6. Check removal / collapse

### PlaceBack
1. Determine next top layer orientation (alternation rule)
2. Position block at chosen slot (Left/Middle/Right)
3. Drop from 0.5 cm above top
4. Run settle loop
5. Check collapse

## Settling (sim.py)

```python
SETTLE_TIMEOUT = 3.0
VELOCITY_THRESHOLD = 1e-3

def settle(self):
    while sim_time < self.SETTLE_TIMEOUT:
        step_simulation()
        if all_velocities_below(self.VELOCITY_THRESHOLD):
            return SETTLED
    return COLLAPSE
```

## Removal Detection (scoring.py)

Block has zero contact points with any other body after settling = removed.

## Collapse Detection (scoring.py)

After settling, check all non-pushed blocks. If any lost contact with a body it was previously in contact with = collapse.

Implementation: snapshot contact pairs before action, compare after settle.

## Force Mapping (actions.py)

| Intensity | Force (approx) |
|-----------|----------------|
| Gentle    | ~0.15 N        |
| Firm      | ~0.6 N         |
| Hard      | ~1.2 N         |

Bell-curve ramp, same duration for all intensities (only magnitude varies). Duration tuned empirically.

## Replay Export (replay.py)

Every step exports to info (as JSON strings per Mesocosm requirement):

| Key          | Content                                         |
|--------------|-------------------------------------------------|
| tower_state  | block transforms: id, position, rotation, removed |
| camera_state | current camera pose                              |
| events       | removal, collapse, rejection events             |

## benchanything.json

Update from scaffold to match Jenga:

| Field                  | Value                                |
|------------------------|--------------------------------------|
| id                     | jenga-bench                          |
| observation_space.type | json                                 |
| action_space.type      | json                                 |
| reward.range           | { low: -1.0, high: 1.0 }            |
| episode.max_steps      | 100 (tunable)                        |
| episode.deterministic  | true                                 |
| scoring.primary_metric | blocks_removed                       |

## Build Order

0. define dataclasses, enums, TypedDicts in each module before writing logic (Block/Layer in tower.py, CameraState in camera.py, action types in actions.py, etc.)
1. sim.py + tower.py — get a tower standing in PyBullet
2. camera.py + render.py — render an image of the tower
3. env.py reset() — wire up to BaseEnv, return first observation
4. actions.py ChangeViewpoint — camera movement works
5. actions.py Push — force application, settling
6. scoring.py — removal + collapse detection
7. actions.py PlaceBack — drop block on top
8. replay.py — export traces
9. benchanything.json — update manifest
10. tests — determinism, replay consistency, action validation

## Resources

| Resource | Link |
|----------|------|
| PyBullet quickstart | https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA |
| PyBullet GitHub | https://github.com/bulletphysics/bullet3 |
| Mesocosm platform | https://mesocosm.swecc.org |
| Mesocosm wiki | https://wiki.swecc.org/Sweccathon/START_HERE |
| bench_common SDK | https://pypi.org/project/swecc-mesocosm |
| Jenga block dimensions | https://en.wikipedia.org/wiki/Jenga |
