# Parametric Curves

> How to create, evaluate, swap, and integrate curves in the Ocean Library.

## Overview

All curves implement `IParametricCurve` — a shared interface that maps a parameter `t ∈ [0, 1]` to a point in 2D or 3D space. Because the interface is shared, **any curve can be swapped for another without changing downstream code**.

## Available Curves

| Class | Key Trait | Passes Through All Points? | Segments |
|-------|-----------|---------------------------|----------|
| `BezierCurve` | Simple, one piece | Endpoints only | 1 |
| `CatmullRomSpline` | Smooth through waypoints | Yes (all points) | K-3 |
| `BSplineCurve` | High continuity (C2+) | No | Varies |
| `NURBSCurve` | Exact conics, weighted | No | Varies |

## Choosing the Right Curve

**Use `BezierCurve` when:**
- You have 2 endpoints + a few control handles
- Simple segment between two stations
- You want direct shape control

**Use `CatmullRomSpline` when:**
- You have waypoints the curve MUST pass through
- You want automatic smooth joining
- Good for complex multi-waypoint paths

**Use `BSplineCurve` when:**
- You need C2+ continuity (smoother than Catmull-Rom)
- You need explicit knot vector control
- Approximating, not interpolating

**Use `NURBSCurve` when:**
- You need exact circles, ellipses, or conics
- Importing from CAD format
- You need weighted control points

## Usage

### Creating Curves

```matlab
% 2D Bezier (cubic — 4 control points)
curve = OceanMath.Geometry.BezierCurve([0 0; 1 3; 4 2; 6 0]);

% 3D Bezier
curve = OceanMath.Geometry.BezierCurve([0 0 0; 1 2 1; 3 1 2; 5 0 3]);

% CatmullRom through waypoints
curve = OceanMath.Geometry.CatmullRomSpline([0 0; 2 3; 5 1; 8 4; 10 2]);

% B-Spline (cubic, uniform knots)
pts = [0 0; 1 3; 3 4; 5 2; 7 1];
knots = OceanMath.Geometry.KnotVector.uniform(size(pts,1), 3);
curve = OceanMath.Geometry.BSplineCurve(pts, 3, knots);
```

### Evaluating

```matlab
t = linspace(0, 1, 200)';      % 200 sample parameters

P  = curve.evaluate(t);          % [200 x d] points
T  = curve.tangent(t);           % [200 x d] first derivative
D2 = curve.secondDerivative(t);  % [200 x d] second derivative
L  = curve.totalLength();        % scalar arc length
```

### Evenly-Spaced Sampling (by arc length)

```matlab
tEven = curve.parameterize(50);  % 50 values, evenly distributed BY DISTANCE
P = curve.evaluate(tEven);       % points are equally spaced along the curve
```

This is important because `linspace(0,1,N)` gives even spacing in **parameter space**, but the curve may move faster or slower at different t values. `parameterize(N)` corrects for this.

### Querying

```matlab
d  = curve.getDimension();       % 2 or 3
n  = curve.getNumSegments();     % number of polynomial pieces
cp = curve.getControlPoints();   % [K x d] control point matrix
curve.describe();                % prints human-readable summary
```

## Swapping Curve Types

Because all curves share `IParametricCurve`, swapping is **one line**:

```matlab
% Before: Bezier
curve = OceanMath.Geometry.BezierCurve([0 0; 2 3; 5 1; 8 0]);

% After: CatmullRom (just change the constructor)
curve = OceanMath.Geometry.CatmullRomSpline([0 0; 2 3; 5 1; 8 0]);
```

Everything downstream — `FrenetFrame`, `CurveSimulator`, `TransitNetwork` — works unchanged.

**In a transit network:**

```matlab
% Just pass a different curve type to addRouteToLine
tn.addRouteToLine('NSL', 1, 2, OceanMath.Geometry.BezierCurve(pts));
% OR
tn.addRouteToLine('NSL', 1, 2, OceanMath.Geometry.CatmullRomSpline(pts));
```

## Integration with Other Systems

### Frenet Frame (orientation + curvature)

```matlab
frenet = OceanMath.Geometry.FrenetFrame(curve);
[T, N, B] = frenet.frame(0.5);     % unit tangent, normal, binormal
kappa     = frenet.curvature(0.5);  % curvature scalar
q         = frenet.quaternion(0.5); % orientation as quaternion [w,x,y,z]
```

### Curve Simulator (physics)

```matlab
sim = OceanMath.Simulation.CurveSimulator(curve, ...
    'InitialVelocity', 2, 'Acceleration', 0, 'Dt', 0.01);
sim.step();
s = sim.getState();  % KinematicState with Position, Velocity, Parameter, etc.
```

### Transit Network (route building)

```matlab
tn.addRouteToLine('NSL', srcId, tgtId, curve);
% Internally calls curve.totalLength() and uses it as the edge weight
```

### Visualization (FrenetVizStrategy)

```matlab
OceanViz.Strategies.FrenetVizStrategy.render(ax, curve, viz.Colors, ...
    'Mode', 'all', 'ArrowScale', 0.8);
```

## See Also

- [Transit Network Guide](transit-network.md)
- [Simulation Guide](simulation.md)
- [Recipes: Curves](../recipes/curves.md)
- [Concept: Layer Architecture](../concepts/layer-architecture.md)
