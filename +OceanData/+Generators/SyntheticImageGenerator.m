classdef SyntheticImageGenerator
    % SYNTHETICIMAGEBUILDER  Procedurally generates clean test images for compression experiments.
    %
    %   Produces images containing a mixture of:
    %     - Smooth gradient regions (low-frequency content, easy for DCT)
    %     - Sharp geometric shapes with step edges (high-frequency, reveals blocking)
    %     - Textured zones (mid-frequency sinusoidal patterns)
    %
    %   This combination is designed to make JPEG blocking artifacts clearly visible
    %   while also testing edge-preservation quality of deblocking algorithms.
    %
    %   Modes:
    %     'gradient'    — Smooth bilinear gradient only
    %     'checkerboard'— High-contrast black/white blocks (worst case for JPEG)
    %     'mixed'       — Gradient background + geometric shapes + texture  (default)
    %
    %   Usage:
    %     gen = OceanData.Generators.SyntheticImageGenerator( ...
    %               'Height', 256, 'Width', 256, 'Mode', 'mixed', 'Seed', 42);
    %     img = gen.generate();   % Returns (H x W x 3) double [0, 1]
    %
    %   Assumptions:
    %     - Output is always in double [0, 1] RGB.
    %     - Seed ensures full reproducibility via rng().
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19

    properties
        Height  (1,1) double {mustBePositive, mustBeInteger} = 256
        Width   (1,1) double {mustBePositive, mustBeInteger} = 256
        Mode    (1,1) string {mustBeMember(Mode, ...
                    ["gradient","checkerboard","mixed"])} = "mixed"
        Seed    (1,1) double = 42
    end

    methods
        function obj = SyntheticImageGenerator(opts)
            % SYNTHETICIMAGEBUILDER Constructor.
            arguments
                opts.Height  (1,1) double = 256
                opts.Width   (1,1) double = 256
                opts.Mode    (1,1) string = "mixed"
                opts.Seed    (1,1) double = 42
            end
            obj.Height = opts.Height;
            obj.Width  = opts.Width;
            obj.Mode   = opts.Mode;
            obj.Seed   = opts.Seed;
        end

        function img = generate(obj)
            % GENERATE  Produce the synthetic test image.
            %
            %   Output:
            %     img — (Height x Width x 3) double RGB image in [0, 1]
            rng(obj.Seed, 'twister');

            H = obj.Height;
            W = obj.Width;

            switch obj.Mode
                case "gradient"
                    img = obj.makeGradient(H, W);
                case "checkerboard"
                    img = obj.makeCheckerboard(H, W);
                case "mixed"
                    img = obj.makeMixed(H, W);
            end

            % Ensure strict [0, 1] clamping
            img = max(0.0, min(1.0, img));
        end
    end

    methods (Access = private)
        function img = makeGradient(~, H, W)
            % MAKEGRADIENT  Smooth bilinear colour gradient.
            [X, Y] = meshgrid(linspace(0, 1, W), linspace(0, 1, H));
            R = X;
            G = Y;
            B = 1 - (X + Y) / 2;
            img = cat(3, R, G, B);
        end

        function img = makeCheckerboard(~, H, W)
            % MAKECHECKERBOARD  High-contrast 8x8 black/white checkerboard.
            block = 8;
            [X, Y] = meshgrid(1:W, 1:H);
            mask = double(mod(floor((X-1)/block) + floor((Y-1)/block), 2) == 0);
            img = cat(3, mask, mask, mask);
        end

        function img = makeMixed(obj, H, W)
            % MAKEMIXED  Multi-layer composition:
            %   Layer 1: Smooth bilinear gradient (background)
            %   Layer 2: Geometric shapes (circle, rectangle, triangle)
            %   Layer 3: Sinusoidal texture patch (mid-frequency zone)
            %   Layer 4: Sharp step edge band

            % --- Layer 1: Gradient background ---
            base = obj.makeGradient(H, W);

            % --- Layer 2: Geometric shapes ---
            [X, Y] = meshgrid(linspace(0, 1, W), linspace(0, 1, H));

            % Circle (centre 0.3, 0.3; radius 0.15)
            circle = ((X - 0.30).^2 + (Y - 0.28).^2) < 0.15^2;

            % Filled rectangle (0.55..0.80 x, 0.15..0.45 y)
            rect = (X > 0.55) & (X < 0.80) & (Y > 0.15) & (Y < 0.45);

            % Triangle — approximate with a wedge mask
            tri = (Y > 0.55) & (X > 0.3 - (Y - 0.55)) & (X < 0.7 + (Y - 0.55)) & (Y < 0.85);

            % Compose shapes with distinct saturated colours
            shape_r = 0.9 * double(circle) + 0.2 * double(rect) + 0.1 * double(tri);
            shape_g = 0.1 * double(circle) + 0.85 * double(rect) + 0.2 * double(tri);
            shape_b = 0.1 * double(circle) + 0.1 * double(rect) + 0.9 * double(tri);

            shape_mask = double(circle | rect | tri);

            % Blend shapes over base
            alpha = shape_mask * 0.9;
            img_R = (1 - alpha) .* base(:,:,1) + alpha .* shape_r;
            img_G = (1 - alpha) .* base(:,:,2) + alpha .* shape_g;
            img_B = (1 - alpha) .* base(:,:,3) + alpha .* shape_b;

            % --- Layer 3: Sinusoidal texture patch (top-right quadrant) ---
            tex_mask = (X > 0.55) & (Y > 0.55);
            freq = 12;  % cycles per image width
            tex = 0.5 + 0.4 * sin(2 * pi * freq * X) .* cos(2 * pi * freq * Y);

            tex_alpha = double(tex_mask) * 0.6;
            img_R = (1 - tex_alpha) .* img_R + tex_alpha .* tex;
            img_G = (1 - tex_alpha) .* img_G + tex_alpha .* tex * 0.8;
            img_B = (1 - tex_alpha) .* img_B + tex_alpha .* tex * 0.6;

            % --- Layer 4: Sharp horizontal step edge band ---
            step_band = (Y > 0.48) & (Y < 0.52) & (X < 0.5);
            img_R(step_band) = 1.0;
            img_G(step_band) = 1.0;
            img_B(step_band) = 0.0;

            img = cat(3, img_R, img_G, img_B);
        end
    end
end
