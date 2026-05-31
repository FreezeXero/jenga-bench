# Design Specifications of Jenga Bench

## Terminology

### Coordinates

There are (x, y, z) where z represents vertical space, upwards being higher.

### Cardinal Directions

| Direction | Definition                              |
|-----------|-----------------------------------------|
| North     | y in the positive direction              |
| South     | y in the negative direction              |
| West      | x in the positive direction              |
| East      | x in the negative direction              |

### Block

Each block is 1.5 cm tall, 2.5 cm wide, 7.5 cm long. It has the weight of 120 grams.

Each block has their own color. In the following table:

| Color       | oklch                 | RGB Hex   |
|-------------|-----------------------|-----------|
| Red         | oklch(0.2, 0.05, 0)   | `#A04848` |
| Brown       | oklch(0.2, 0.05, 60)  | `#7C5B3F` |
| Lime        | oklch(0.2, 0.05, 120) | `#789146` |
| Wintergreen | oklch(0.2, 0.05, 180) | `#40947F` |
| Blue        | oklch(0.2, 0.05, 240) | `#4664A5` |
| Purple      | oklch(0.2, 0.05, 300) | `#7B5498` |

#### Long Axis of a Block

The dimension where block is long. 

### Layer

Each layer contains three blocks. There are two type of Layer. North-South layers, and East-West layers.

North-South layers are when the long axis of each block is aligned North to South. 
East-West layers are when the long axis is aligned East to West. 

| Layer Type   | Position Order       | Colors                     |
|--------------|----------------------|----------------------------|
| North-South  | East, Middle, West   | Red, Lime, Blue            |
| East-West    | South, Middle, North | Wintergreen, Purple, Brown |

Note that these colors is to maximize purely contrast.

### Tower

The Jenga tower. It consists of 54 blocks at the start, with 18 layers where the bottom most layer is North-South, and then it alternates from then. 

### Base

The base where the Jenga tower stands on. It is black, about 3 layers tall
(~4.5 cm), and 25 x 25 cm in area.

### Camera State

The camera is persistent and stateful. It stays where it was last placed until a Change Viewpoint action moves it. The camera always aims at its target (a block, or layer-center by default), at a fixed distance, orbiting by azimuth + pitch.

| Action           | Camera Effect                                              |
|------------------|------------------------------------------------------------|
| Change Viewpoint | repositions the camera; tower unchanged                    |
| Push             | camera stays in place; renders post-push result from current position |

The camera state contains:

| Field        | Description                                                                                                      |
|--------------|------------------------------------------------------------------------------------------------------------------|
| target_block | what the camera aims at. A specific block's center. Resolved live from current sim state. If the target block is removed, resets to the default target from reset. |
| azimuth      | angle around the tower. Continuous, 0–360 degrees.                                                               |
| pitch        | vertical angle, looking up or down. Continuous, -90 to 90 degrees.                                               |
| distance     | distance from target in cm. Range: minimum close-up to maximum that fits entire tower from center.               |

Default camera on reset: three-quarter view (diagonal azimuth, e.g. SW), mid-height (~layer 9–10), slightly looking down, whole tower in frame with base visible. Diagonal azimuths (NE, SE, SW, NW) show two faces at once and give better depth reading than flat cardinal views.

### Physics State

The tower and base in a 3D physics simulation.

In terms of global coordinates, the base surface is at z = 0. The tower sits on top of the base.

### Settling

After every Push and PlaceBack, the simulation runs until settled or timed out.

| Parameter       | Value                                  |
|-----------------|----------------------------------------|
| Settle check    | all blocks below 1e-3 m/s linear and 1e-3 rad/s angular velocity for 30 consecutive steps |
| Settle timeout  | 3 seconds sim-time                     |
| Timeout exceeded | treated as collapse                   |

### Deterministic Tower Generation

`reset(seed)` constructs a prebuilt tower variant deterministically from the
Mesocosm restart seed. A missing seed is treated as seed `0`. Each variant uses
small configurable perturbations: individual blocks slide forward or backward,
same-layer blocks receive a shared extra gap, layer centers make a tiny
cumulative x/y walk so the tower can begin with a coherent lean, and each layer
receives a small yaw applied around each block's own center. This yaw creates a
stair-like silhouette rather than symmetrically rotating the row of blocks.

All adjustable generation, geometry, physics, and rendering presets live in
`jenga/settings.py`. Exact same-seed repeatability is guaranteed within the
pinned Docker runtime.

## Player Loop (LLM mode)

### Reset

`reset(seed, **params)` — loads the seeded prebuilt tower snapshot and resets the camera. Returns the first observation.

### Observation

The agent-facing observation payload is the rendered PNG only. The environment
reports non-privileged metadata through the observation's `system_prompt`:
camera pose, removal count, available placement slots, current phase, and the
last five action outcomes. Replay-only state stays in string-valued `info`
fields and is not shown to the model.

#### Render

White background, black base, blocks color-coded per the color table.
Soft diffuse lighting (studio-style, no direct sun) — high ambient, moderate
diffuse, low specular for edge definition. Shadows are soft for depth cues
without harsh contrast or blown-out highlights that would obscure block colors
for the LLM.

### Action Space
Exactly one JSON action is submitted per `step` call. The three full benchmark
action types are ChangeViewpoint, Push, and PlaceBack. Mesocosm records the
model's response text separately for replay, so actions do not duplicate
reasoning annotations.

#### Change Viewpoint

Sets a new camera state. Tower state unchanged. Reward = 0.

#### Push

| Field     | Description                                                                                              |
|-----------|----------------------------------------------------------------------------------------------------------|
| layer     | 1 through one below the current logical top layer (bottom to top), identifies the target layer. The top layer cannot be pushed. |
| color     | identifies the target block within the layer (color = slot)                                              |
| face      | cardinal direction of the face to push from. North-South blocks: North or South. East-West blocks: East or West. Push from North = force applied southward through the block. |
| contact   | where on the face to apply force. Discrete 3x3 grid: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right. Off-center contact generates torque. |
| intensity | Gentle, Firm, or Hard. Gentle doubles as a probe — may not fully extract, letting the player read looseness from the result. |

| Rule      | Description                                                                                              |
|-----------|----------------------------------------------------------------------------------------------------------|
| Effect    | applies a bell-curve-ramped axial force over a fixed duration (same duration for all intensities, only magnitude varies), then settles. |

#### Place Back

After a successful extraction, the agent may look (ChangeViewpoint) before
placing, but must PlaceBack before any Push. Viewpoints during this phase count
toward the consecutive-viewpoint timeout.

| Field    | Description                                                                                    |
|----------|------------------------------------------------------------------------------------------------|
| position | Left, Middle, or Right slot on the next top layer. Observation tells the agent which slots are available. Picking an occupied slot is an invalid action. |
| rotation_degrees | Yaw offset from the required layer orientation, from -5 to +5 degrees. Values outside this range are invalid. |

| Rule     | Description                                                                                    |
|----------|------------------------------------------------------------------------------------------------|
| Orientation | follows layer alternation (next layer rotates 90° from current top)                         |
| Slot mapping | Left/Middle/Right maps to East/Middle/West for North-South rows and South/Middle/North for East-West rows |
| Row anchor | a new top row is centered over the average x/y center of the current highest occupied row; that center is reused until all three slots fill |
| Drop     | block is released 0.5 cm above the top of the tower, then settles                             |

### Env Response

`step` returns (observation, reward, done, info).

#### Reward

| Outcome               | Reward   |
|------------------------|----------|
| Successful extraction  | 1 raw point |
| Change Viewpoint       | 0        |
| PlaceBack              | 0        |
| Invalid action         | -0.5     |
| Collapse               | 0 (episode terminates) |
| 10 consecutive viewpoints | -10 raw points (episode terminates) |

Theoretical max: 98 extractions = 100% score. Report the leaderboard score as
`round(raw_points / 98 * 100, 2)`. Penalties can produce a score below 0.00.

Invalid actions: pushing a removed block, pushing when PlaceBack is required, PlaceBack when no extraction happened, placing in an occupied slot, values outside allowed ranges.

### Removal

A block is considered removed when it has zero contacts with any other body (blocks, base) after settling.

### Collapse

A non-pushed block loses contact with something it should be in contact with after settling. Something fell that the agent didn't intend to move.

### Episode Termination (done = true)

| Condition    | Description                                                              |
|--------------|--------------------------------------------------------------------------|
| Collapse     | a non-pushed block loses a required contact                              |
| Viewpoint timeout | 10 consecutive ChangeViewpoint actions without a Push or PlaceBack. The counter resets after Push or PlaceBack. |
| Perfect completion | 98 successful extractions |

No retries; score is locked at termination.

## Non-Goals (v1)

| Excluded                     | Rationale                                        |
|------------------------------|--------------------------------------------------|
| Robotic arms                 | benchmark tests reasoning, not control           |
| Arbitrary force vectors      | constrained push keeps action space LLM-friendly |
| Continuous trajectory control | same as above                                   |
| RL training APIs             | benchmark is for evaluation, not training        |
| Deformable materials         | blocks are rigid                                 |
| Photorealism                 | studio lighting is sufficient for LLM perception |

## Showcase / Replay Viewer

The showcase is a Three.js replay viewer. Not authoritative — driven entirely by exported replay traces.

| Feature        | Description                                                  |
|----------------|--------------------------------------------------------------|
| Floor           | visible floor beneath the base for blocks to land on        |
| Block physics   | high restitution (bouncy) for fallen/extracted blocks       |
| Sound effects   | impact sounds on block collisions, extraction, collapse     |
| 3D playback     | full tower replay with timeline scrubbing                   |
| Camera replay   | replay the agent's camera movements                         |
| AI reasoning    | display Mesocosm-exported model reasoning per step          |
