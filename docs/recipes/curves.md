# Recipes: Curves

> Copy-paste code blocks for common curve operations.

## Draw a Curve with Tangent Arrows

```matlab
curve = OceanMath.Geometry.BezierCurve([0 0; 1 3; 4 2; 6 0]);
t = linspace(0, 1, 200)';
P = curve.evaluate(t);
T = curve.tangent(t);

figure; hold on; axis equal;
plot(P(:,1), P(:,2), 'b-', 'LineWidth', 2);
idx = 1:20:200;  % every 20th point
quiver(P(idx,1), P(idx,2), T(idx,1), T(idx,2), 0.3);
title('Curve + Tangent Arrows');
```

## Animate an Object on a Curve

```matlab
curve = OceanMath.Geometry.CatmullRomSpline([0 0; 3 4; 6 1; 9 3]);
sim = OceanMath.Simulation.CurveSimulator(curve, ...
    'InitialVelocity', 1, 'Dt', 0.005);

figure; hold on; axis equal;
P = curve.evaluate(linspace(0,1,200)');
plot(P(:,1), P(:,2), '-', 'Color', [0.5 0.5 0.5]);
dot = plot(NaN, NaN, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

while sim.getState().Parameter < 1
    sim.step();
    p = sim.getState().Position;
    set(dot, 'XData', p(1), 'YData', p(2));
    drawnow limitrate;
end
```

## Compare Two Curve Types

```matlab
pts = [0 0; 2 4; 5 3; 8 1; 10 4];
c1 = OceanMath.Geometry.BezierCurve(pts);
c2 = OceanMath.Geometry.CatmullRomSpline(pts);

t = linspace(0, 1, 200)';
P1 = c1.evaluate(t);
P2 = c2.evaluate(t);

figure; hold on;
plot(P1(:,1), P1(:,2), 'r-', 'LineWidth', 2, 'DisplayName', 'Bezier');
plot(P2(:,1), P2(:,2), 'b-', 'LineWidth', 2, 'DisplayName', 'CatmullRom');
plot(pts(:,1), pts(:,2), 'ko', 'MarkerSize', 8, 'DisplayName', 'Control Points');
legend; title('Bezier vs CatmullRom on Same Points');
```

## Curvature Profile

```matlab
curve = OceanMath.Geometry.CatmullRomSpline([0 0; 2 3; 5 1; 8 4]);
frenet = OceanMath.Geometry.FrenetFrame(curve);
t = linspace(0.01, 0.99, 100)';
kappa = frenet.curvature(t);

figure;
plot(t, kappa, 'r-', 'LineWidth', 2);
xlabel('t'); ylabel('kappa');
title('Curvature Along the Curve');
```

## 3D Helix with Frenet Frame

```matlab
n = 30; th = linspace(0, 4*pi, n)';
pts3D = [2*cos(th), 2*sin(th), 0.5*th];
curve = OceanMath.Geometry.BezierCurve(pts3D);
frenet = OceanMath.Geometry.FrenetFrame(curve);

[T, N, B] = frenet.frame(0.5);
fprintf('T = [%.3f, %.3f, %.3f]\n', T);
fprintf('N = [%.3f, %.3f, %.3f]\n', N);
fprintf('B = [%.3f, %.3f, %.3f]\n', B);
```

## Frenet Visualization (TNB Arrows + Comb + Osculating Circle)

```matlab
curve = OceanMath.Geometry.CatmullRomSpline([0 0; 1 3; 3 4; 5 2; 7 4; 9 1; 10 3]);
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Color', viz.Colors.Background);
ax = axes('Parent', fig); hold(ax, 'on'); axis(ax, 'equal');

OceanViz.Strategies.FrenetVizStrategy.render(ax, curve, viz.Colors, ...
    'Mode', 'all', 'CombN', 50, 'CombScale', 1.5, 'ArrowScale', 0.8);
```

## See Also

- [Parametric Curves Guide](../guides/parametric-curves.md)
- [Simulation Guide](../guides/simulation.md) *(coming soon)*
