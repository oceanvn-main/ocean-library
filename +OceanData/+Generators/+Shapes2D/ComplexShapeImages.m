classdef ComplexShapeImages
    % COMPLEXSHAPEIMAGES  Procedural grey-scale images of hard-to-model 2-D
    %   shapes -- a controlled CHALLENGE set for generative models, with the
    %   class and every generative factor known exactly.
    %
    %   Classes (why each is hard for a generative model):
    %     1 "spiral"    thin Archimedean stroke, 1.5-2.75 turns   (thin, long-range)
    %     2 "star"      5-7 sharp points, random depth            (sharp corners)
    %     3 "ringgap"   annulus with a 40-110 deg gap            (topology: hole + cut)
    %     4 "crescent"  disk minus an offset disk                 (thin tips, concavity)
    %     5 "cross"     plus sign, random arm width               (right angles)
    %     6 "heart"     parametric heart curve, filled            (cusp + smooth lobes)
    %   Factors per image: rotation U(0, 2 pi), scale, translation, and a
    %   class-specific parameter (turns, points, gap, offset, arm width).
    %   Rendering: supersampled occupancy (default 4 x 4 per pixel), then box
    %   averaged -> anti-aliased intensities in [0, 1] (background 0).
    %
    %   [X, y, F] = OceanData.Generators.Shapes2D.ComplexShapeImages.generate(n, 'Size', 32, 'Seed', 1)
    %     X  (Size x Size x 1 x n) "SSCB" double in [0, 1]
    %     y  (n x 1) class index 1..6 (balanced: round-robin over 'Classes')
    %     F  table of the generative factors

    properties (Constant)
        ClassNames = ["spiral" "star" "ringgap" "crescent" "cross" "heart"]
    end

    methods (Static)
        function [X, y, F] = generate(n, opts)
            arguments
                n (1,1) double {mustBePositive, mustBeInteger}
                opts.Size (1,1) double {mustBePositive, mustBeInteger} = 32
                opts.Classes (1,:) double = 1:6
                opts.Supersample (1,1) double {mustBePositive, mustBeInteger} = 4
                opts.Seed (1,1) double = 0
            end
            S = OceanData.Generators.Shapes2D.ComplexShapeImages;
            rs = RandStream('mt19937ar', 'Seed', opts.Seed);
            H = opts.Size; ss = opts.Supersample; G = H * ss;
            g = ((1:G) - 0.5) / G * 2 - 1;                             % supersampled grid in [-1, 1]
            [U, V] = meshgrid(g, -g);                                   % V up
            y = opts.Classes(mod(0:n - 1, numel(opts.Classes)) + 1).';
            y = y(randperm(rs, n));
            rot = 2 * pi * rand(rs, n, 1);
            scale = 0.55 + 0.3 * rand(rs, n, 1);
            tx = 0.12 * (2 * rand(rs, n, 1) - 1); ty = 0.12 * (2 * rand(rs, n, 1) - 1);
            param = rand(rs, n, 1);
            width = rand(rs, n, 1);
            X = zeros(H, H, 1, n);
            for i = 1:n
                % inverse transform: image coords -> shape frame (unit size)
                c = cos(rot(i)); s = sin(rot(i));
                u = ( c * (U - tx(i)) + s * (V - ty(i))) / scale(i);
                v = (-s * (U - tx(i)) + c * (V - ty(i))) / scale(i);
                occ = S.occupancy(y(i), u, v, param(i), width(i), scale(i), G);
                X(:, :, 1, i) = squeeze(mean(mean(reshape(double(occ), ss, H, ss, H), 1), 3));
            end
            F = table(y, rot, scale, tx, ty, param, width, 'VariableNames', ...
                {'Class', 'Rotation', 'Scale', 'TX', 'TY', 'Param', 'Width'});
        end

        function occ = occupancy(cls, u, v, p, w, scale, G)
            % Inside test in the shape frame; p, w in [0, 1] set the class parameter.
            r = hypot(u, v); th = atan2(v, u);
            px = 2 / G / scale;                                         % one supersample in shape units
            switch cls
                case 1                                                  % spiral: distance to a dense polyline
                    turns = 1.5 + 1.25 * p;
                    t = linspace(0, 2 * pi * turns, 600);
                    rr = 0.08 + 0.9 * t / t(end);
                    P = [rr .* cos(t); rr .* sin(t)];
                    half = 0.05 + 0.04 * w;                             % stroke half-width
                    d2 = inf(size(u));
                    for k = 1:60:numel(t)                               % chunked min distance (memory friendly)
                        kk = k:min(numel(t), k + 59);
                        dx = u(:) - P(1, kk); dy = v(:) - P(2, kk);
                        d2 = min(d2, reshape(min(dx .^ 2 + dy .^ 2, [], 2), size(u)));
                    end
                    occ = d2 <= max(half, px) ^ 2;
                case 2                                                  % star polygon
                    k = 5 + floor(3 * p); inner = 0.35 + 0.25 * w;
                    a = (0:2 * k - 1) * pi / k;
                    rad = repmat([1 inner], 1, k);
                    occ = inpolygon(u, v, rad .* cos(a), rad .* sin(a));
                case 3                                                  % annulus with a gap
                    gap = deg2rad(40 + 70 * p);
                    thick = 0.18 + 0.14 * w;
                    occ = r <= 1 & r >= 1 - thick & abs(th) >= gap / 2;
                case 4                                                  % crescent: disk minus offset disk
                    off = 0.3 + 0.35 * p; r2 = 0.85 + 0.1 * w;
                    occ = r <= 1 & hypot(u - off, v) > r2;
                case 5                                                  % cross / plus
                    arm = 0.18 + 0.2 * w;
                    occ = (abs(u) <= arm & abs(v) <= 1) | (abs(v) <= arm & abs(u) <= 1);
                case 6                                                  % heart (parametric boundary)
                    t = linspace(0, 2 * pi, 400);
                    hx = 16 * sin(t) .^ 3; hy = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t);
                    m = max(abs([hx hy]));
                    occ = inpolygon(u, v, hx / m, (hy + 2) / m);
                otherwise
                    error('OceanData:Generators:ComplexShapeImages:Class', 'Unknown class %d.', cls);
            end
        end
    end
end
