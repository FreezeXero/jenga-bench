# Design Specifications of Jenga Bench

## Terminology

### Coordinates

There are (x, y, z) where z represents vertical space, upwards being higher.

### Cardinal Directions

North is defined by y in the positive direction.

South is defined by y in the negative direction.

West is defined by x in the positive direction, etc.

### Block

Each block is 1.5 cm tall, 2.5 cm wide, 7.5 cm long. It has the weight of 120 grams.

Each block has their own color. In the following table:

Red:

Brown:

Lime:

Wintergreen:

Blue:

Purple:

TODO decided oklab colors

#### Long Axis of a Block

The dimension where block is long. 

### Layer

Each layer contains three blocks. There are two type of Layer. North-South layers, and East-West layers.

North-South layers are when the long axis of each block is aligned North to South. 
East-West layers are when the long axis is aligned East to West. 

North-South layers, starting from east, have the following colors: Red, Lime, Blue.

East-West layers, starting from south, have the following colors: Wintergreen, Purple, Brown.

Note that these colors is to maximize purely contrast.

### Tower

The Jenga tower. It consists of 54 blocks at the start, with 18 layers where the bottom most layer is North-South, and then it alternates from then. 

### Base

The base where the Jenga tower stands on. It's black, about one meter tall, and 2.5 x 2.5 meter squared in terms of area.

### Camera State

Camera always have a TODO

### Physics State

The tower and base in a 3D physics simulation.

In terms of global coordinates, the base is located at y = 0.

## Player Loop (LLM mode)

### Reset
`reset(seed, **params)` — builds the default tower with seeded jitter (x/y offset
per block, no yaw), settles it to rest, and places the camera at a default vantage
(three-quarter, mid-height, whole tower in frame). Returns the first observation.

### Observation
Every observation (from `reset` and every `step`) is a dictionary with:
- image          — a render from the camera's CURRENT position, taken AFTER the
                   action has fully completed and the tower has settled.
- camera         — the camera's current pose, reported back so the player can
                   calibrate intent against result:
                   { azimuth, elevation, distance, target_block }.
- blocks_removed — count of successfully extracted blocks so far.
- TODO: text log / running history exposed to the player? (windowed, ~last 5)

#### Render
TODO — white background, black base, no shadows; blocks color-coded per the
color table. Camera intrinsics (FOV ~45°), distance presets, resolution: TODO.

### Camera Model
The camera is PERSISTENT and STATEFUL. It stays where it was last placed until a
Change Viewpoint action moves it. Both action types render from the camera's
current position:
- Change Viewpoint repositions the camera; the tower is unchanged.
- Push leaves the camera in place; the render shows the post-push result from
  wherever the camera currently sits.
The camera always aims at its target (a block, or layer-center by default), at a
fixed distance, orbiting by azimuth + elevation.

### Action Space
Exactly one action is submitted per `step` call.

#### Change Viewpoint
(formerly "Look" — renamed: the player always sees an image; this action only
moves the eye, it does not grant or withhold sight.)

ChangeViewpoint { target_block: (layer, color) | layer_center, azimuth, elevation, zoom }
- target_block — what the camera aims at. A specific block (to inspect a protruding
                 one), or the center of a layer for an overview. Target is resolved
                 LIVE from current sim state (a pushed-out block is tracked as it moves).
- azimuth      — angle around the tower. CONTINUOUS (degrees). The player is told the
                 resulting azimuth in the observation so it can calibrate.
- elevation    — vertical angle on the orbit (looking up / level / down). CONTINUOUS.
- zoom         — Wide | Medium | Close.
                   Wide   = whole tower in frame (overrides target to tower-center).
                   Medium = target layer ± a few neighbors.
                   Close  = tight on the target block.
- Effect       — repositions the camera, renders the new view. Tower state unchanged.
                 Reward = 0.
- Budget       — max 3 consecutive viewpoint changes before a Push is required.
                 A 4th consecutive one ends the episode (forced reset).
                 TODO: confirm counter resets on Push, and that this hard-reset
                 (vs. soft penalty) is the desired behavior.

#### Push
Push { layer: 0–17, color: Color, face: Face, intensity: Intensity }
- layer, color — identify the target block (color = slot
