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

For ```reset(seed, **paraams)```, we create a default tower. 

### Observation

The following is included in the observation is a dictionary with:
- image - a render from the relevant camera after the action has fully completed 
- blocks removed  

#### Render

The render is TODO 

### Action Space
Exactly one action is submitted per `/step` call.

#### Look

TODO - add pitch/tilt

Look { layer: 0–17, azimuth: Azimuth, elevation: Elevation }
- layer        — the layer the camera vertically centers on (whole tower stays in frame).
- azimuth      — discrete, 8 points: N, NE, E, SE, S, SW, W, NW.
- elevation    — discrete: Low | Level | High.
- Effect       — renders a view. Tower state is unchanged. Reward = 0.
- Budget       — max 3 Looks before a Push is required. A 4th consecutive Look
                 ends the episode (forced reset).

#### Push

TODO - add push opposite

Push { layer: 0–17, color: Color, face: Face, intensity: Intensity }
- layer, color — identify the target block (color = slot within the layer).
- face         — Near | Far, RELATIVE TO THE LAST LOOK's CAMERA. Near = the block
                 end facing that camera; the block is driven inward, away from it.
- intensity    — discrete: Gentle | Firm | Hard.  (TODO: map → force, empirical.)
- Validity     — the block's long axis must be sufficiently aligned with the last
                 camera direction (|axis · view| > τ). If not, the Push is REJECTED
                 and the player must Look from an aligned angle first. τ TODO.
- Effect       — applies a bell-curve-ramped axial force over a fixed slow duration,
                 then settles. Resets the Look counter.


### Env Response

TODO

#### Reward
- Successful extraction (block fully out AND tower still standing after settle):
      +1 / N   where N = practical max removable.  (TODO: N empirical.)
- Look: 0.
- A Push that topples the tower: 0 for that block; episode ends.

### Episode Termination (done = true)

- Collapse — any still-stacked block leaves its position (e.g. ends up on the base).
- Forced reset — a 4th consecutive Look.
- (Optional) step cap — TODO.
No retries; score is locked at termination.
