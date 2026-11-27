# QuadTree Recipes

## Find Nearest Station to Mouse Click

```matlab
% Build index from transit network
positions = transit.getStationPositions();
ids = (1:size(positions, 1))';
qt = OceanMath.Geometry.QuadTree(positions(:,1:2), ids);

% Query: find station within 5 units of click
clickPos = [25.3, 14.7];
nearId = qt.findNearest(clickPos, 5.0);
if nearId > 0
    fprintf('Nearest station: %s\n', transit.getStation(nearId).Name);
end
```

## Find All Stations in a Region

```matlab
% All stations within 10 units of a center point
ids = qt.rangeQuery([50, 50], 10);
fprintf('Found %d stations in region\n', numel(ids));
```

## Find Stations Inside a Rectangle

```matlab
% All stations inside a bounding box
box = [20, 30, 60, 70];  % [xMin, yMin, xMax, yMax]
ids = qt.queryBox(box);
```

## Rebuild After Station Changes

```matlab
% After adding or removing stations, rebuild the tree
td.rebuildSpatialIndex_();
```

## Inspect Tree Structure

```matlab
qt.describe();
```
