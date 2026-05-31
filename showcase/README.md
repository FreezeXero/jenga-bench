# Repo showcase (your frontend)

Build a marketing or demo UI **in this repository** that replays a real bench run from Mesocosm.

## Workflow

1. Submit this repo: `mesocosm env submit --name "..." --github-url https://github.com/you/your-repo`
2. Wait for `ready`, then bench a model:
   ```bash
   mesocosm run create --domain YOUR_DOMAIN_ID --vow-version 1.0.0 --model gemini/gemini-3.1-flash-lite --episodes 1 --visibility gallery_public
   ```
3. Export the run (after it completes):
   ```bash
   mesocosm run export RUN_ID -o showcase/data/replay.json
   ```
4. Point your frontend at `replay.json`. Each turn includes:
   - `observation` — env state for your UI
   - `reasoning` — model text (what the agent said before acting)
   - `action` — parsed action sent to the env
   - `reward`, `terminated`, etc.

## `replay.json` shape

Treat a real `mesocosm run export RUN_ID -o showcase/replay.json` export as the
source of truth for the replay shape. Use each exported turn's `reasoning` field
for showcase-style prose.
