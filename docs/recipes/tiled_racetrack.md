# Draw maps with Tiled

## Overview

Install [Tiled](https://www.mapeditor.org/), then open either starter in
`Experiments/Scripts/RL/Python/maps/tiled/`:

- `harbor_bend.tmj`: a right-turn racetrack.
- `your_obstacle_map.tmj`: the grid extracted from your uploaded diagram.

Keep `rl_tiles.tsj` and `rl_tiles.png` beside these maps. Tiled is a separate
application and is not installed by this Python integration.

## Usage

1. Open a starter .tmj file in Tiled; use Save As to make your own .tmj copy in
   the same folder, preserving the relative tileset reference.
2. Select the tile layer named **Racetrack**.
3. Select a tile: dark W = wall, light dot = floor, red S = start, green G =
   finish. Paint with Stamp Brush or fill with Bucket Fill. Erasing means wall.
4. Keep at least one start and finish, and a reachable route from every start.
5. Save and run the command below. No screenshot or image export is needed.

Map custom properties configure `movement_mode` (string: textbook or
bidirectional), `max_velocity` (int, exclusive upper speed bound), `fail_prob`
(float 0 to 1), and `max_episode_steps` (positive int). Starter maps already have
these configured. Omitted movement_mode defaults to bidirectional. CLI options
have priority over map properties.

## Examples

From the repository root:

```powershell
py -3.12 Experiments/Scripts/RL/Python/exp_multi_map_racetrack.py --maps Experiments/Scripts/RL/Python/maps/tiled/harbor_bend.tmj --episodes 1000 --seeds 1 --plot --show
```

Replace the path with your saved .tmj file. Each map needs its own trained
policy; connectivity does not guarantee quick learning.

## Supported format

- Finite orthogonal maps, exactly one top-level tile layer named Racetrack,
  matching map dimensions and with zero tile/pixel offsets.
- Native JSON-array tile data (CSV format in Tiled), without compression.
- Inline tilesets or external JSON .tsj tilesets. Each used tile declares an
  integer custom property cell_type: 0 wall, 1 floor, 2 start, 3 finish.
- Erased tiles (GID zero) mean wall. Tile flipping preserves semantic meaning.
  Other layers are decorative and ignored, even when visible.
- .tmj and direct Tiled .json exports work. XML .tmx/.tsx, infinite maps,
  group-layer geometry and object-based obstacles are unsupported.

The supplied tileset already declares the properties. Tile artwork is only for
editing; its 32-pixel display size does not enlarge the RL grid.

## Verification

Both starters import to grids identical to their originals. A 100-episode run
through Harbor Bend's .tmj completed, with 2 of 4 evaluation starts finishing
within 100 steps. This is a smoke test, not convergence.

## See Also

- [Official Tiled JSON format](https://doc.mapeditor.org/en/stable/reference/json-map-format/)
- [Environment guide](../guides/racetrack_maps.md)
