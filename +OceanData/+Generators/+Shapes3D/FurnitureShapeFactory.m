classdef FurnitureShapeFactory < handle
    % FURNITURESHAPEFACTORY  Procedural 3-D furniture as unions of boxes
    %   (BoxAssembly): the offline stand-in for ShapeNet / Pix3D categories
    %   used by single-image shape-reconstruction experiments (Mesh R-CNN).
    %
    %   Every shape is produced by the same random process a CAD modeller
    %   would parameterise: seat/top size, thickness, height, leg thickness,
    %   inset, optional stretchers and back styles. Class identity is carried
    %   by STRUCTURE (a chair has a back, a table a wide top, a stool is tall
    %   and narrow), not by colour -- so a classifier must look at shape.
    %
    %   Topology varies on purpose:
    %     chair, "frame" back  -> two posts + top rail enclose a slot (genus >= 1
    %                              for the back alone; the full chair's genus also
    %                              counts the loops through the legs)
    %     chair, "solid" back  -> slab back
    %   A method that deforms a sphere cannot produce the slot; one that starts
    %   from voxels (Cubify) can -- this is the phenomenon Mesh R-CNN targets.
    %
    % OBJECT FRAME: y is UP, the floor is y = 0, the chair back is at -z.
    %   Returned assemblies are normalized(): bounding box centred at the
    %   origin, largest half-extent = 1.
    %
    % USAGE:
    %   fac = OceanData.Generators.Shapes3D.FurnitureShapeFactory('Seed', 3);
    %   [asm, info] = fac.create("chair");           % info: class, style, parameters
    %   [asm, info] = fac.createRandom();            % uniform over Classes
    %
    % See also: OceanMath.Vision.Geometry.BoxAssembly, OceanMath.Vision.Geometry.BoxGeometry

    properties (Constant)
        Classes = ["chair", "table", "stool"]
    end

    properties (SetAccess = private)
        Stream      % RandStream driving every random choice (reproducible)
    end

    methods
        function obj = FurnitureShapeFactory(opts)
            arguments
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            obj.Stream = RandStream('mt19937ar', 'Seed', opts.Seed);
        end

        function [asm, info] = createRandom(obj)
            k = randi(obj.Stream, numel(obj.Classes));
            [asm, info] = obj.create(obj.Classes(k));
        end

        function [asm, info] = create(obj, className)
            arguments
                obj
                className (1,1) string {mustBeMember(className, ["chair", "table", "stool"])}
            end
            switch className
                case "chair"
                    [boxes, info] = obj.makeChair();
                case "table"
                    [boxes, info] = obj.makeTable();
                case "stool"
                    [boxes, info] = obj.makeStool();
            end
            info.Class = className;
            info.ClassIndex = find(obj.Classes == className);
            asm = OceanMath.Vision.Geometry.BoxAssembly(boxes).normalized();
        end
    end

    methods (Access = private)
        function x = U(obj, lo, hi)
            x = lo + (hi - lo) * rand(obj.Stream);
        end

        function [boxes, info] = makeChair(obj)
            sw = obj.U(0.80, 1.10); sd = obj.U(0.75, 1.00);     % seat width / depth
            st = obj.U(0.08, 0.14);                             % seat thickness
            seatTop = obj.U(0.75, 1.00);
            lt = obj.U(0.08, 0.13);                             % leg thickness
            legH = seatTop - st;
            boxes = {box3([0, seatTop - st/2, 0], [sw, st, sd])};
            boxes = [boxes, legs4(sw, sd, lt, legH, 0)];

            backH = obj.U(0.70, 1.10); bt = obj.U(0.07, 0.12);
            zb = -(sd - bt) / 2;
            yb = seatTop + backH / 2;
            if rand(obj.Stream) < 0.5
                style = "solid";
                boxes{end+1} = box3([0, yb, zb], [sw, backH, bt]);
            else
                style = "frame";
                rail = obj.U(0.15, 0.25);
                for sx = [-1 1]
                    boxes{end+1} = box3([sx*(sw - lt)/2, yb, zb], [lt, backH, bt]); %#ok<AGROW>
                end
                boxes{end+1} = box3([0, seatTop + backH - rail/2, zb], [sw - 2*lt, rail, bt]);
            end
            info = struct('Style', style, 'Params', struct('SeatWidth', sw, 'SeatDepth', sd, ...
                'SeatThickness', st, 'SeatHeight', seatTop, 'LegThickness', lt, ...
                'BackHeight', backH, 'BackThickness', bt));
        end

        function [boxes, info] = makeTable(obj)
            tw = obj.U(1.40, 2.00); td = obj.U(0.80, 1.20);
            tt = obj.U(0.07, 0.12); th = obj.U(0.75, 1.05);
            lt = obj.U(0.08, 0.14); inset = obj.U(0.00, 0.12);
            legH = th - tt;
            boxes = {box3([0, th - tt/2, 0], [tw, tt, td])};
            boxes = [boxes, legs4(tw - 2*inset, td - 2*inset, lt, legH, 0)];
            style = "plain";
            if rand(obj.Stream) < 0.5
                style = "stretchers";
                sh = obj.U(0.15, 0.30);
                barLen = tw - 2*inset - 2*lt;
                for sz = [-1 1]
                    boxes{end+1} = box3([0, sh, sz*(td - 2*inset - lt)/2], [barLen, 0.8*lt, 0.8*lt]); %#ok<AGROW>
                end
            end
            info = struct('Style', style, 'Params', struct('TopWidth', tw, 'TopDepth', td, ...
                'TopThickness', tt, 'Height', th, 'LegThickness', lt, 'Inset', inset));
        end

        function [boxes, info] = makeStool(obj)
            s = obj.U(0.45, 0.70); tt = obj.U(0.07, 0.12);
            h = obj.U(1.00, 1.40); lt = obj.U(0.07, 0.11);
            legH = h - tt;
            boxes = {box3([0, h - tt/2, 0], [s, tt, s])};
            boxes = [boxes, legs4(s, s, lt, legH, 0)];
            style = "plain";
            if rand(obj.Stream) < 0.6
                style = "ring";
                ry = obj.U(0.30, 0.50) * h;
                barLen = s - 2*lt;
                bt = 0.7 * lt;
                for sgn = [-1 1]
                    boxes{end+1} = box3([0, ry, sgn*(s - lt)/2], [barLen, bt, bt]); %#ok<AGROW>
                    boxes{end+1} = box3([sgn*(s - lt)/2, ry, 0], [bt, bt, barLen]); %#ok<AGROW>
                end
            end
            info = struct('Style', style, 'Params', struct('TopSize', s, 'TopThickness', tt, ...
                'Height', h, 'LegThickness', lt));
        end
    end
end

% ------------------------------------------------------------------------ %
function b = box3(center, fullSize)
% Axis-aligned box from centre and FULL extents (half-extents stored).
b = OceanMath.Vision.Geometry.BoxGeometry('Center', center(:), 'HalfSize', fullSize(:) / 2);
end

function legs = legs4(width, depth, thickness, height, yBottom)
% Four square legs at the corners of a width x depth footprint.
legs = cell(1, 4);
k = 0;
for sx = [-1 1]
    for sz = [-1 1]
        k = k + 1;
        legs{k} = box3([sx*(width - thickness)/2, yBottom + height/2, sz*(depth - thickness)/2], ...
                       [thickness, height, thickness]);
    end
end
end
