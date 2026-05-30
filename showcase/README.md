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

Public runs (`gallery_public` + completed) can also be viewed at:

`https://mesocosm…/runs/RUN_ID` (no login).

## `replay.json` shape

See `replay.example.json`. Use `replay[episode_id][i].reasoning` for showcase-style prose, similar to Mesocosm’s curated trading demo.
