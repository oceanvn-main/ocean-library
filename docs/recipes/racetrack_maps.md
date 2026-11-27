# Import and train a racetrack

From the Python project directory, train with the geometric distance objective:

```sh
python -m ocean_rl --maps maps/tiled/n_shape.tmj --objective geometric-distance --gamma 1 --max-velocity 2 --episodes 100000 --output-dir results/n_shape_distance --plot
```

Use `--objective minimum-time` for the original objective. Retrain for a new
objective. Map JSON environment objects and Tiled string properties also accept
`objective`. Reports preserve its name and measured evaluation distances.

Install from `Experiments/Scripts/RL/Python`:

```sh
python -m pip install -e ".[images,plots]"
ocean-rl --system racetrack --maps maps/tiled/harbor_bend.tmj --episodes 1000 --plot
python -m unittest discover -s tests -v
```

For a regular black-dot diagram with a complete dot border, `import_dot_map.py`
can extract a grid using an explicitly selected pixel crop and S/E coordinates.
Example for the supplied screenshot (replace the source path):

```powershell
py -3.12 Experiments/Scripts/RL/Python/import_dot_map.py source.png --output-stem Experiments/Scripts/RL/Python/maps/my_image --grid-box 30 15 291 275 --start-pixel 56 247 --finish-pixel 264 210
```

Crop bounds are left/top/right/bottom, with right/bottom exclusive. Endpoint
annotations are x/y pixels and snap to the nearest detected cell. Inspect the
generated preview before use. The importer saves an exact-palette PNG, environment
JSON and extraction diagnostics. It detects regularly spaced border dots, then
classifies dark cell centers; it does not interpret arbitrary photographs.
The diagnostics' cardinal BFS path checks connectivity only, not learned control.

The supplied image extracted to 30 x 30, start (row 26, column 3), finish
(row 22, column 27), with 300 obstacle cells. PNG round-trip equality passed.
The subsequent MC smoke run timed out at 5,000 steps on its first episode;
no successful learned policy is claimed for that map.

For a regular black-dot diagram with a complete dot border, `import_dot_map.py`
can extract a grid using an explicitly selected pixel crop and S/E coordinates.
Example for the supplied screenshot (replace the source path):

```powershell
py -3.12 Experiments/Scripts/RL/Python/import_dot_map.py source.png --output-stem Experiments/Scripts/RL/Python/maps/my_image --grid-box 30 15 291 275 --start-pixel 56 247 --finish-pixel 264 210
```

Crop bounds are left/top/right/bottom, with right/bottom exclusive. Endpoint
annotations are x/y pixels and snap to the nearest detected cell. Inspect the
generated preview before use. The importer saves an exact-palette PNG, environment
JSON and extraction diagnostics. It detects regularly spaced border dots, then
classifies dark cell centers; it does not interpret arbitrary photographs.
The diagnostics' cardinal BFS path checks connectivity only, not learned control.

The supplied image extracted to 30 x 30, start (row 26, column 3), finish
(row 22, column 27), with 300 obstacle cells. PNG round-trip equality passed.
The subsequent MC smoke run timed out at 5,000 steps on its first episode;
no successful learned policy is claimed for that map.

Create `my_track.map` using rectangular rows. Use `#` for walls, spaces for
track, `S` for starts and `G` for finishes. Preserve spaces in each row. Use a
plain text editor: a screenshot with printed text/grid lines is not a labeled map.

Create `my_track.json` alongside it:

```json
{
  "map": "my_track.map",
  "name": "My track",
  "environment": {
    "movement_mode": "bidirectional",
    "max_velocity": 5,
    "fail_prob": 0.1,
    "max_episode_steps": 100000
  }
}
```

Run from the repository root (replace the map path):

```powershell
py -3.12 Experiments/Scripts/RL/Python/exp_multi_map_racetrack.py --maps my_track.json --episodes 1000 --seeds 0 1 2 --output-dir Experiments/Scripts/RL/Python/results/my_track --plot
```

Compare the bundled maps:

```powershell
py -3.12 Experiments/Scripts/RL/Python/exp_multi_map_racetrack.py --episodes 100 --max-steps 100000 --seeds 0 --plot
```

Run regression tests:

```powershell
py -3.12 -m unittest discover -s Experiments/Scripts/RL/Python/tests -v
```

For array inputs, `RacetrackEnv.from_numpy(grid)` accepts a 2D integer grid:
0=wall, 1=track, 2=start, 3=finish. Save a grid as `.npy` to use it directly with
`--maps`. Never infer start/finish from the first or last white pixel.

For image inputs, see the `from_image` docstring for explicit palette and
annotation parameters. Use labeled images at one pixel per cell. There is no automatic resizing or
grid extraction; prepare the occupancy grid explicitly so obstacles are preserved.

See [the guide](../guides/racetrack_maps.md) for MC timeout behavior and movement
conventions. A new map requires training a new policy.

