# AGENTS.md — Ocean Library

This file is the entry point for any AI agent working on this codebase.

## Read This First

**Full architecture reference**: [`.agents/architecture.md`](.agents/architecture.md)

This document contains the complete package map, interface contracts, design patterns, dependency rules, naming conventions, and instrucitons for extending each part of the library. Read it before generating or modifying any code.

## Agent Rules

Additional behavioral rules are in `.agents/rules/`:
- [`mathexpression.md`](.agents/rules/mathexpression.md) — How to format math expressions
- [`no_hardcoding.md`](.agents/rules/no_hardcoding.md) — Absolute prohibition of hardcoded mock data or simulated shortcuts
- [`useoceanvizengineforplottingifpossible.md`](.agents/rules/useoceanvizengineforplottingifpossible.md) — Always use `OceanViz.Core` for plotting
- [`research_philosophy.md`](.agents/rules/research_philosophy.md) — Prioritize research legibility and understanding over production black-boxes
- [`mathworks_best_practices.md`](.agents/rules/mathworks_best_practices.md) — MathWorks official conventions, arguments blocks, vectorization, & unit testing

## Workflows

Process documents for how to carry out multi-step agentic work (not architecture, not
behavioral rules — how to structure the work itself):
- [`multi-agent-build-critic-loop.md`](.agents/workflows/multi-agent-build-critic-loop.md) —
  when and how to use parallel builder agents + an independent critic + a bounded (max 3
  round) revision loop, gated at a numeric score threshold. Read this before starting any
  multi-file feature build with real correctness risk (a published algorithm/formula, numeric
  code) — for a small, well-scoped, single-file change, just do it directly instead (see the
  document's §1 for the explicit line between the two).

## Quick Reference

| Task                | Where to look                                                             |
| ------------------- | ------------------------------------------------------------------------- |
| Add a new ML model  | `+OceanStats/+Supervised/` + implement `IModelStrategy`                   |
| Add a new metric    | `+OceanMath/+Information/+Strategies/` + implement `IMetricStrategy`      |
| Add a new smoother  | `+OceanStats/+NonParametric/+Strategies/` + implement `ISplineStrategy`   |
| Add a new plot type | `+OceanViz/+Strategies/` + implement `IVizStrategy`                       |
| Run a demo          | `Experiments/Scripts/exp_*.m`                                             |
| Run tests           | `+OceanTest/TestRunner.m`                                                 |
| Load a dataset      | `OceanStats.Data.DataFactory.load('name')`                                |
| Create a plot       | `viz = OceanViz.Core('Theme','Dark'); viz.plot(ax, 'StrategyName', data)` |
| Record / replay a computation | `OceanReplay` (implement `IRecordable`) + `OceanViz.App.Replay.ReplayStudio(file)`; Python twin `Python/ocean_replay`; recipes `docs/recipes/replay_recipes.md` |
