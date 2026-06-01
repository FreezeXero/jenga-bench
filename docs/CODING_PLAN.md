# Coding Plan

## Tech Stack

| Component | Choice        |
|-----------|---------------|
| Physics   | PyBullet      |
| Rendering | PyBullet TinyRenderer |
| Language  | Dockerized Python 3.11 |
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
│   ├── settings.py     # adjustable generation, geometry, physics, render presets
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

Deterministic under the restart seed and action sequence. Missing seeds use
seed `0`. Adjustable presets are centralized in `jenga/settings.py`.

## Adjustable Presets (settings.py)

```python
BLOCK_L = 0.075   # 7.5 cm
BLOCK_W = 0.025   # 2.5 cm
BLOCK_H = 0.015   # 1.5 cm
BLOCK_MASS = 0.120  # 120 g
```

## Tower Construction (tower.py)

- 18 layers × 3 blocks = 54 blocks
- Bottom layer North-South, alternating
- Each block gets an immutable internal ID independent of its visible color
- Assign display colors dynamically from the current orientation and slot:
  North-South layers use Red, Lime, Blue from East to West; East-West layers
  use Wintergreen, Purple, Green from South to North
- Recolor extracted blocks when placed into a new top-layer slot
- Base: black box, ~4.5cm tall (3 layers), 25×25cm
- Floor: beneath base, for blocks to land on

Use a 25×25cm base footprint. Build deterministic tower variants from the reset
seed. Variants apply bounded per-block longitudinal offsets, per-layer shared
spacing, a tiny cumulative x/y layer-center walk, and small per-layer yaw around
each block's own center. Tune these ranges in `jenga/settings.py`.

## Camera (camera.py)

Orbital camera state: target_block, azimuth, pitch, distance. Render from current position after every action. See DESIGN.md Camera State for behavior.

## Rendering (render.py)

- 512×512 RGB PNG from PyBullet TinyRenderer
- White background, black base
- Studio-style diffuse lighting: high ambient, moderate diffuse, low specular (wood-like sheen for edge definition)
- Soft shadows for depth cues
- Blocks color-coded with the authoritative RGB palette in DESIGN.md

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
2. Center a new row over the current highest occupied row and keep that anchor until all three slots fill
3. Map Left/Middle/Right to the orientation-specific positional slots and apply a validated -5 to +5 degree yaw offset
4. Recolor the block for its new orientation and slot while preserving its internal ID
5. Drop from 0.5 cm above top
6. Run settle loop
7. Check collapse

Placed rows extend the logical tower height. Push validation accepts layers from
1 through one below the current logical top layer so recycled blocks remain
targetable after another row is started without allowing removal from the top.

## Settling (sim.py)

```python
SETTLE_TIMEOUT = 3.0
LINEAR_VELOCITY_THRESHOLD = 1e-3
ANGULAR_VELOCITY_THRESHOLD = 1e-3
SETTLE_STABLE_STEPS = 30

def settle(self):
    while sim_time < self.SETTLE_TIMEOUT:
        step_simulation()
        if all_velocities_below_for_consecutive_steps(
            self.LINEAR_VELOCITY_THRESHOLD,
            self.ANGULAR_VELOCITY_THRESHOLD,
            self.SETTLE_STABLE_STEPS,
        ):
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

## Scoring

- Successful extraction: +1 raw point
- Invalid action: -0.5 raw points
- Collapse: terminate and preserve accumulated raw points
- 10 consecutive ChangeViewpoint actions without Push or PlaceBack: terminate
  and subtract 10 raw points
- Reset the viewpoint counter after Push or PlaceBack
- Perfect completion: 98 extractions
- Leaderboard score: `round(raw_points / 98 * 100, 2)`, including negatives

## benchanything.json

Update from scaffold to match Jenga:

| Field                  | Value                                |
|------------------------|--------------------------------------|
| id                     | jenga-bench                          |
| observation_space.type | image                                |
| action_space.type      | json                                 |
| reward.range           | { low: -10.0, high: 1.0 }           |
| episode.max_steps      | 1000                                  |
| episode.deterministic  | true                                 |
| scoring.primary_metric | normalized_score                     |

## Incremental Build Order

1. Contract slice — deterministic placeholder PNG observations, ChangeViewpoint,
   manifest, prompts, adapter compatibility, and focused tests
2. Static tower — PyBullet setup, exact prebuilt snapshot geometry, and rendering
3. Camera controls — orbital targeting and viewpoint-timeout verification
4. Push slice — validation, force ramp, settling, and probe behavior
5. Extraction and placement — held blocks, dynamic recoloring, top-layer slots,
   and 98-extraction completion
6. Collapse detection — support snapshots and deterministic scenario tuning
7. Replay export — state frames and events in string-valued info fields
8. Showcase viewer — Three.js playback, inspection, reasoning, and sound

Complete and test each milestone before starting the next.

## Resources

| Resource | Link |
|----------|------|
| PyBullet quickstart | https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA |
| PyBullet GitHub | https://github.com/bulletphysics/bullet3 |
| Mesocosm platform | https://mesocosm.swecc.org |
| Mesocosm wiki | https://wiki.swecc.org/Sweccathon/START_HERE |
| bench_common SDK | https://pypi.org/project/swecc-mesocosm |
| Jenga block dimensions | https://en.wikipedia.org/wiki/Jenga |
