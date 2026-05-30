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

| Color       | oklch               |
|-------------|---------------------|
| Red         | oklch(0.2, 0.05, 0)   |
| Brown       | oklch(0.2, 0.05, 60)  |
| Lime        | oklch(0.2, 0.05, 120) |
| Wintergreen | oklch(0.2, 0.05, 180) |
| Blue        | oklch(0.2, 0.05, 240) |
| Purple      | oklch(0.2, 0.05, 300) |

#### Long Axis of a Block

The dimension where block is long. 

### Layer

Each layer contains three blocks. There are two type of Layer. North-South layers, and East-West layers.

North-South layers are when the long axis of each block is aligned North to South. 
East-West layers are when the long axis is aligned East to West. 

| Layer Type   | Starting From | Colors                      |
|--------------|---------------|-----------------------------|
| North-South  | east          | Red, Lime, Blue             |
| East-West    | south         | Wintergreen, Purple, Brown  |

Note that these colors is to maximize purely contrast.

### Tower

The Jenga tower. It consists of 54 blocks at the start, with 18 layers where the bottom most layer is North-South, and then it alternates from then. 

### Base

The base where the Jenga tower stands on. It's black, about one meter tall, and 2.5 x 2.5 meter squared in terms of area.

### Camera State

The camera is persistent and stateful. It stays where it was last placed until a Change Viewpoint action moves it. The camera always aims at its target (a block, or layer-center by default), at a fixed distance, orbiting by azimuth + pitch.

| Action           | Camera Effect                                              |
|------------------|------------------------------------------------------------|
| Change Viewpoint | repositions the camera; tower unchanged                    |
| Push             | camera stays in place; renders post-push result from current position |

The camera state contains:

| Field        | Description                                                                                                      |
|--------------|------------------------------------------------------------------------------------------------------------------|
| target_block | what the camera aims at. A specific block's center. Resolved live from current sim state. If the target block is gone, aim where it would have been. |
| azimuth      | angle around the tower. Continuous (degrees).                                                                    |
| pitch    | vertical pitch/X angle on the orbit (looking up / level / down). Continuous.                                             |
| distance     | distance from target.                                                                                            |

Begins SW in the middle layers.

### Physics State

The tower and base in a 3D physics simulation.

In terms of global coordinates, the base is located at y = 0.

### Settling

After every Push and PlaceBack, the simulation runs until settled or timed out.

| Parameter       | Value                                  |
|-----------------|----------------------------------------|
| Settle check    | all block velocities below threshold   |
| Settle timeout  | 3 seconds sim-time                     |
| Timeout exceeded | treated as collapse                   |

## Player Loop (LLM mode)

### Reset

`reset(seed, **params)` — builds the default tower, settles it to rest, and resets the camera. Returns the first observation.

### Observation

Every observation (from `reset` and every `step`) is a dictionary with:

| Key            | Description                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------------|
| image          | a render from the camera's current position, taken after the action has fully completed and the tower has settled.   |
| camera         | the camera's current pose, reported back so the player can calibrate intent against result: { azimuth, pitch, distance, target_block }. |
| blocks_removed | count of successfully extracted blocks so far.                                                                       |
| log            | array of the last 5 action+annotation pairs (most recent first).                                                     |

#### Render

White background, black base, blocks color-coded per the color table.
Soft diffuse lighting (studio-style, no direct sun) — high ambient, moderate
diffuse, no specular. Shadows are soft for depth cues without harsh contrast
or blown-out highlights that would obscure block colors for the LLM.

### Action Space
Exactly one action is submitted per `step` call. Three action types: ChangeViewpoint, Push, PlaceBack. Every action includes an annotation where the player self-reports reasoning:

| Field           | Description                        |
|-----------------|------------------------------------|
| action          | the action dict (ChangeViewpoint, Push, or PlaceBack) |
| annotation.saw  | what the player observed           |
| annotation.did  | what action it chose               |
| annotation.why  | reasoning behind the choice        |

#### Change Viewpoint

Sets a new camera state. Tower state unchanged. Reward = 0.

#### Push

| Field     | Description                                                                                              |
|-----------|----------------------------------------------------------------------------------------------------------|
| layer     | 0–17, identifies the target layer                                                                        |
| color     | identifies the target block within the layer (color = slot)                                              |
| face      | cardinal direction of the face to push from. North-South blocks: North or South. East-West blocks: East or West. Push from North = force applied southward through the block. |
| contact   | where on the face to apply force. Discrete 3x3 grid: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right. Off-center contact generates torque. |
| intensity | Gentle, Firm, or Hard. Gentle doubles as a probe — may not fully extract, letting the player read looseness from the result. |

| Rule      | Description                                                                                              |
|-----------|----------------------------------------------------------------------------------------------------------|
| Effect    | applies a bell-curve-ramped axial force over a fixed slow duration (quasi-static), then settles.         |

#### Place Back

Triggered automatically after a successful extraction. The agent chooses where to place the removed block on top of the tower. Does not count toward the action budget.

| Field    | Description                                                                                    |
|----------|------------------------------------------------------------------------------------------------|
| position | Left, Middle, or Right slot on the next top layer                                              |

| Rule     | Description                                                                                    |
|----------|------------------------------------------------------------------------------------------------|
| Orientation | follows layer alternation (next layer rotates 90° from current top)                         |
| Drop     | block is released 0.5 cm above the top of the tower, then settles                             |

### Env Response

`step` returns (observation, reward, done, info).

#### Reward

| Outcome               | Reward |
|------------------------|--------|
| Successful extraction  | +1     |
| Change Viewpoint       | 0      |
| Invalid action         | -0.05  |
| Collapse               | -1     |

### Removal

A block is considered removed when it has zero contacts with any other body (blocks, base) after settling.

### Collapse

A non-pushed block loses contact with something it should be in contact with after settling. Something fell that the agent didn't intend to move.

### Episode Termination (done = true)

| Condition    | Description                                                              |
|--------------|--------------------------------------------------------------------------|
| Collapse     | a non-pushed block loses a required contact                              |
| Action budget | 10 total actions without a successful extraction terminates the episode |

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
