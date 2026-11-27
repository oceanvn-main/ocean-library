# Transit Layer Architecture

> How the 4-layer design connects Graph, Geometry, Domain, and Simulation.

## The 4 Layers

```
┌───────────────────────────────────────────────────────────────┐
│  Layer 4: SIMULATION        TransitSimulation, TransitAgent  │
│           "Who moves, how fast"                              │
├───────────────────────────────────────────────────────────────┤
│  Layer 3: DOMAIN            TransitNetwork, TransitLine      │
│           "Transit vocabulary"                               │
├───────────────────────────────────────────────────────────────┤
│  Layer 2: GEOMETRY          BezierCurve, CatmullRomSpline    │
│           "Shape of each track segment"                      │
├───────────────────────────────────────────────────────────────┤
│  Layer 1: GRAPH ENGINE      DigraphCus                       │
│           "Raw nodes + weighted edges + algorithms"          │
└───────────────────────────────────────────────────────────────┘
```

Each layer only talks to the one **directly below it**.

## Layer 1: DigraphCus — The Graph Engine

A domain-blind, general-purpose weighted graph. Knows nothing about trains.

**Stores:** Nodes (id, name, position), Edges (source, target, weight), W_ matrix (V x V).

**Provides:** `addNode()`, `addEdge()`, `neighbors()`, `hasEdge()`, `getAdjacencyMatrix()`.

**Key design:** Edge weight = spline arc length. Graph algorithms (Dijkstra) automatically find physically shortest routes.

## Layer 2: Geometry — The Curves

`IParametricCurve` implementations that define the physical shape of each track segment.

**Provides:** `evaluate(t)`, `tangent(t)`, `secondDerivative(t)`, `totalLength()`.

**Key design:** Maps `t ∈ [0,1]` to a point in space. At t=0 you're at station A, at t=1 you're at station B.

See [Parametric Curves Guide](../guides/parametric-curves.md).

## Layer 3: TransitNetwork + TransitLine — The Domain

Wraps the graph and geometry into transit vocabulary.

**TransitNetwork** owns:
- `Graph_` → DigraphCus (the generic backend)
- `Splines_` → cell array (one curve per edge)
- `Lines_` → containers.Map (lineCode → TransitLine)

**TransitLine** owns:
- Code, Name, Color (identity)
- StationIds_ (ordered route: `[8, 2, 3, 5]`)
- Policy_ (speed limits, frequency, etc.)

**Key translation table:**

| You say | TransitNetwork does | DigraphCus receives |
|---------|--------------------|--------------------|
| `addStation('Orchard', [5,7])` | Creates a transit station | `addNode('Orchard', [5,7])` |
| `addRouteToLine('NSL', 1, 2, spline)` | Stores spline, registers on line | `addEdge(1, 2, arcLength)` |
| `getInterchanges()` | Finds multi-line stations | Scans node degrees |

## Layer 4: Simulation — The Agents

Manages train movement across multi-segment routes.

**TransitAgent:** One train traversing one line's route. For each segment pair (A→B), it creates a `CurveSimulator` on that segment's spline, then chains to the next when t >= 1. Supports dwell time at stations.

**TransitSimulation:** Orchestrates multiple agents on a shared clock. Provides `step()`, `getPositions()`, `isAllArrived()`.

## How Data Flows: Building a Route

```
tn.addRouteToLine('NSL', 8, 2, bezierCurve)
│
├── 1. bezierCurve.totalLength()     → compute arc length
├── 2. Graph_.addEdge(8, 2, L)      → weighted edge in DigraphCus
├── 3. Splines_{idx} = bezierCurve  → store geometry
├── 4. Lines_('NSL').registerStation(8)
└── 5. Lines_('NSL').registerStation(2)
```

## How Data Flows: Running a Simulation

```
sim.addAgent('DTL', 'Speed', 0.9)
│
├── Gets TransitLine('DTL') → route = [7, 3, 6, 10]
├── For each pair: findEdgeIndex → get spline from Splines_{}
├── Creates TransitAgent with those splines
│
└── Each step():
    ├── CurveSimulator.step()  → advances parameter on current spline
    ├── curve.evaluate(t)      → position
    ├── curve.tangent(t)       → direction
    └── At t >= 1: dwell → switch to next segment
```

## Component Summary

| Component | One-line job | Knows transit? |
|-----------|-------------|---------------|
| **DigraphCus** | Raw graph: nodes, edges, weights | No |
| **BezierCurve** | Shape of one track segment | No |
| **TransitLine** | Identity + ordered station list | Yes (one line) |
| **TransitNetwork** | Facade: glues graph + splines + lines | Yes (whole system) |
| **TransitAgent** | One train traversing a line | Yes (one train) |
| **TransitSimulation** | Orchestrates all agents | Yes (all trains) |

## See Also

- [Parametric Curves Guide](../guides/parametric-curves.md)
- [Transit Network Guide](../guides/transit-network.md) *(coming soon)*
- [Simulation Guide](../guides/simulation.md) *(coming soon)*
