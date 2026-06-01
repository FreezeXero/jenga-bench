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

| Color       | oklch                  | RGB Hex   |
|-------------|------------------------|-----------|
| Red         | oklch(0.7, 0.1525, 30) | `#EE7563` |
| Green       | oklch(0.73, 0.135, 149)| `#4AB078` |
| Blue        | oklch(0.7, 0.1275, 270)| `#8099EE` |

#### Long Axis of a Block

The dimension where block is long. 

### Layer

Each layer contains three blocks. There are two type of Layer. North-South layers, and East-West layers.

North-South layers are when the long axis of each block is aligned North to South. 
East-West layers are when the long axis is aligned East to West. 

| Layer Type   | Position Order       | Colors                     |
|--------------|----------------------|----------------------------|
| North-South  | East, Middle, West   | Blue, Green, Red           |
| East-West    | South, Middle, North | Blue, Green, Red           |

All layers use the same three colors. No alternation.

### Tower

The Jenga tower. It consists of 54 blocks at the start, with 18 layers where the bottom most layer is North-South, and then it alternates from then. 

### Base

The base where the Jenga tower stands on. It is black, about 3 layers tall
(~4.5 cm), and 25 x 25 cm in area.

### Camera State

The camera is persistent and stateful. It stays where it was last placed until a Change Viewpoint action moves it. The camera always aims at its target block (or layer-center by default), orbiting at the height of the elevation layer.

| Action           | Camera Effect                                              |
|------------------|------------------------------------------------------------|
| Change Viewpoint | repositions the camera; tower unchanged                    |
| Push             | camera stays in place; renders post-push result from current position |

The camera state contains:

| Field           | Description                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| direction       | compass direction the camera looks from: N, NE, E, SE, S, SW, W, NW                                            |
| elevation_layer | which layer height the camera orbits at (integer, 1–18)                                                          |
| distance        | Close (15 cm), Medium (30 cm), or Full (45 cm)                                                                   |
| target_block    | what the camera aims at, identified by layer + color (e.g. layer 5, Blue). Resolved live from current sim state. If the target block is removed, resets to the default target. Optional — omit to aim at the elevation layer's center. |

Default camera on reset: SW direction, mid-height (layer 9), distance 45 cm. Diagonal directions (NE, SE, SW, NW) show two faces at once and give better depth reading than flat cardinal views.

### Physics State

The tower and base in a 3D physics simulation.

In terms of global coordinates, the base surface is at z = 0. The tower sits on top of the base.

### Settling

After every Push and PlaceBack, the simulation runs until settled or timed out.

| Parameter       | Value                                  |
|-----------------|----------------------------------------|
| Settle check    | all blocks below 5e-3 m/s linear and 5e-2 rad/s angular velocity for 30 consecutive steps |
| Settle timeout  | 10 seconds sim-time                    |
| Timeout exceeded | check final state — collapse only if collapse conditions are met, otherwise settled |

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
camera pose, removal count, available placement slots, current phase, moves
remaining before the next required extraction, and the last five model
`context` strings. Replay-only state stays in string-valued `info`
fields and is not shown to the model.

#### Render

White background, black base, blocks color-coded per the color table.
Soft diffuse lighting (studio-style, no direct sun) — high ambient, moderate
diffuse, low specular for edge definition. Shadows are soft for depth cues
without harsh contrast or blown-out highlights that would obscure block colors
for the LLM.

### Action Space
Exactly one JSON object is submitted per `step` call:

```json
{
  "context": "Brief rationale for this turn.",
  "action": { "type": "ChangeViewpoint" | "Push" | "PlaceBack", "...": "..." }
}
```

`context` is a short rationale string that is fed back to the model in the next
turn's prompt history. `action` is the executable env action. The three full
benchmark action types remain ChangeViewpoint, Push, and PlaceBack.

#### Change Viewpoint

Sets a new camera state. Tower state unchanged. Reward = 0.

| Field           | Description                                                                                    |
|-----------------|------------------------------------------------------------------------------------------------|
| direction       | compass direction: N, NE, E, SE, S, SW, W, NW                                                 |
| elevation_layer | camera height as layer number (1–18)                                                           |
| distance        | Close (15 cm), Medium (30 cm), or Full (45 cm)                                                 |
| target_block    | optional — layer + color of block to aim at (e.g. `{"layer": 5, "color": "Blue"}`). Omit for layer center. |

#### Push

| Field     | Description                                                                                              |
|-----------|----------------------------------------------------------------------------------------------------------|
| layer     | 1 through one below the current logical top layer (bottom to top), identifies the target layer. The top layer cannot be pushed. |
| color     | identifies the target block within the layer (color = slot)                                              |
| face      | cardinal direction of the face to push from. North-South blocks: North or South. East-West blocks: East or West. Push from North = force applied southward through the block. |
| contact   | where on the face to apply force: left, center, or right. Off-center contact generates torque. |
| intensity | Gentle, Firm, or Hard. Gentle doubles as a probe — may not fully extract, letting the player read looseness from the result. |

| Rule      | Description                                                                                              |
|-----------|----------------------------------------------------------------------------------------------------------|
| Effect    | applies a bell-curve-ramped force scaled to the block's load (weight from above), capped at a velocity threshold per intensity (Gentle = slow probe, Firm = deliberate push, Hard = aggressive shove), then settles. |
| Extraction | a block is considered extracted when it has no contact with any other block in the tower. |

#### Place Back

After a successful extraction, the agent may look (ChangeViewpoint) before
placing, but must PlaceBack before any Push. These turns still consume the
extraction countdown until the next successful extraction happens.

| Field    | Description                                                                                    |
|----------|------------------------------------------------------------------------------------------------|
| position | slot on the next top layer, using directional names: East/Middle/West for North-South rows, South/Middle/North for East-West rows. Observation tells the agent which slots are available. Picking an occupied slot is an invalid action. |

| Rule     | Description                                                                                    |
|----------|------------------------------------------------------------------------------------------------|
| Orientation | follows layer alternation (next layer rotates 90° from current top)                         |
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
| 10 turns without a successful extraction | -10 raw points (episode terminates) |

Theoretical max: 98 extractions = 100% score. Report the leaderboard score as
`round(raw_points / 98 * 100, 2)`. Penalties can produce a score below 0.00.

Invalid actions: pushing a removed block, pushing when PlaceBack is required, PlaceBack when no extraction happened, placing in an occupied slot, values outside allowed ranges.

### Removal

A block is considered removed when it has zero contacts with any other body (blocks, base) after settling.

### Collapse

Collapse is checked continuously during both the push ramp and settling phases. Two conditions:

| Condition | Detection |
|-----------|-----------|
| Ground contact | any non-layer-1 block touches the base or floor — instant collapse |
| Lost vertical contact | a non-target block loses a vertical contact (contact normal mostly up/down) it had before the push, for 30 consecutive simulation steps |

Before each push, the simulation snapshots which blocks are vertically in contact with which (excluding the target block). During the push and settle, if any non-target block loses one of those vertical contacts for 30 consecutive steps, it means something fell that the agent didn't intend to move.

Only a fully extracted block resets the countdown to 10. Viewpoint changes,
placements, invalid actions, failed pushes, and collapse-causing pushes all
consume one move from the countdown.

### Episode Termination (done = true)

| Condition    | Description                                                              |
|--------------|--------------------------------------------------------------------------|
| Collapse     | ground contact or lost vertical contact (see above)                      |
| Extraction timeout | 10 turns without a successful extraction. The countdown resets only after a fully extracted block. |
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
| AI reasoning    | display annotation (saw/did/why) per step                   |
