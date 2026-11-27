classdef OcclusionSensitivity
    % OCCLUSIONSENSITIVITY  "What happens if this region is removed?"
    %   (Zeiler & Fergus, 2013.) Pure forward-only: slides an occluding
    %   patch over a single input image, reruns predict() at every
    %   position, and records how much the target class's score DROPS.
    %   Needs no gradient at all.
    %
    %   USAGE:
    %     r = OcclusionSensitivity.run(engine, X, 3, 'PatchSize', [8 8], 'Stride', [4 4]);
    %     r.Heatmap   % score-drop per occluded position (larger = more important region)

    methods (Static)
        function result = run(engine, X, targetClass, opts)
            arguments
                engine
                X (:,:,:,:) double
                targetClass (1,1) double {mustBePositive, mustBeInteger}
                opts.PatchSize (1,2) double {mustBePositive, mustBeInteger} = [8 8]
                opts.Stride (1,2) double {mustBePositive, mustBeInteger} = [4 4]
                opts.OccludeValue (1,1) double = 0
                opts.OutputLayer (1,:) char = ''
            end
            [H, W, ~, N] = size(X);
            if N ~= 1
                error('OceanDL:Interpretability:OcclusionSensitivity:SingleImageOnly', ...
                    'run() expects a single image (N=1); got N=%d.', N);
            end

            Ybase = engine.predict(X);
            baseScore = Ybase(targetClass, 1);

            ph = opts.PatchSize(1); pw = opts.PatchSize(2);
            sh = opts.Stride(1); sw = opts.Stride(2);
            if ph > H || pw > W
                error('OceanDL:Interpretability:OcclusionSensitivity:PatchTooLarge', ...
                    'PatchSize [%d %d] does not fit input [%d %d].', ph, pw, H, W);
            end
            rows = 1:sh:(H - ph + 1);
            cols = 1:sw:(W - pw + 1);

            heatmap = zeros(numel(rows), numel(cols));
            for ri = 1:numel(rows)
                for ci = 1:numel(cols)
                    r0 = rows(ri); c0 = cols(ci);
                    Xocc = X;
                    Xocc(r0:r0+ph-1, c0:c0+pw-1, :, :) = opts.OccludeValue;
                    Yocc = engine.predict(Xocc);
                    heatmap(ri, ci) = baseScore - Yocc(targetClass, 1);
                end
            end

            result = struct('Heatmap', heatmap, 'Rows', rows, 'Cols', cols, ...
                'PatchSize', opts.PatchSize, 'BaseScore', baseScore);
        end
    end
end
