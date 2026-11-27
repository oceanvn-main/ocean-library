classdef ChromaSubsampler
    % CHROMASUBSAMPLER  YCbCr chrominance subsampling and guided/JBF upsampling.
    %
    %   Implements standard chroma subsampling schemes (4:2:0, 4:2:2, 4:4:4)
    %   on YCbCr images. The high-resolution luminance Y channel is preserved
    %   at full resolution. Cb and Cr are downsampled spatially. Upsampling
    %   uses either:
    %     - 'guided'  : Guided Image Filter (He et al. 2013) — default
    %     - 'jbf'     : Joint Bilateral Filter (guided by Y channel)
    %     - 'nearest' : Naive nearest-neighbour (baseline for comparison)
    %
    %   Usage:
    %     sub = OceanMath.Image.Compression.ChromaSubsampler('Scheme','4:2:0');
    %     [Y, Cb_sub, Cr_sub] = sub.subsample(YCbCr_img);
    %     YCbCr_up = sub.upsample(Y, Cb_sub, Cr_sub);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19
    %   Ref:    Szeliski §4.3, MPEG/JPEG chroma subsampling standard

    properties
        % Scheme: '4:4:4' (no subsampling) | '4:2:2' (2x horizontal) | '4:2:0' (2x both)
        Scheme          (1,1) string {mustBeMember(Scheme, ...
                            ["4:4:4", "4:2:2", "4:2:0"])} = "4:2:0"

        % UpsamplingMethod: Method used for chroma reconstruction upsampling
        %   'guided'  — Guided Image Filter (default, high quality)
        %   'jbf'     — Joint Bilateral Filter (alternative)
        %   'nearest' — Nearest-neighbour (baseline)
        UpsamplingMethod (1,1) string {mustBeMember(UpsamplingMethod, ...
                            ["guided", "jbf", "nearest"])} = "guided"

        % GuidedRadius: Local window radius for GuidedFilter (pixels)
        GuidedRadius    (1,1) double {mustBePositive} = 2

        % GuidedEpsilon: Regularization for GuidedFilter (1e-4 prevents over-smoothing)
        GuidedEpsilon   (1,1) double {mustBePositive} = 1e-4

        % JBF spatial/range sigma (only used when UpsamplingMethod='jbf')
        JBFSpatialSigma (1,1) double {mustBePositive} = 5.0
        JBFRangeSigma   (1,1) double {mustBePositive} = 0.15
    end

    methods
        function obj = ChromaSubsampler(opts)
            % CHROMASUBSAMPLER Constructor.
            arguments
                opts.Scheme             (1,1) string = "4:2:0"
                opts.UpsamplingMethod   (1,1) string = "guided"
                opts.GuidedRadius       (1,1) double = 8
                opts.GuidedEpsilon      (1,1) double = 0.01
                opts.JBFSpatialSigma    (1,1) double = 5.0
                opts.JBFRangeSigma      (1,1) double = 0.15
            end
            obj.Scheme           = opts.Scheme;
            obj.UpsamplingMethod = opts.UpsamplingMethod;
            obj.GuidedRadius     = opts.GuidedRadius;
            obj.GuidedEpsilon    = opts.GuidedEpsilon;
            obj.JBFSpatialSigma  = opts.JBFSpatialSigma;
            obj.JBFRangeSigma    = opts.JBFRangeSigma;
        end

        function [Y, Cb_sub, Cr_sub, sub_info] = subsample(obj, YCbCr_img)
            % SUBSAMPLE  Downsample Cb and Cr channels from a YCbCr image.
            %
            %   Inputs:
            %     YCbCr_img — (H x W x 3) double image in YCbCr color space,
            %                  channels in [0,1] range
            %
            %   Outputs:
            %     Y         — (H x W) luminance channel (full resolution)
            %     Cb_sub    — Downsampled Cb chrominance
            %     Cr_sub    — Downsampled Cr chrominance
            %     sub_info  — struct with .Scheme, .OrigSize, .SubFactor fields
            arguments
                obj
                YCbCr_img (:,:,3) double
            end

            Y  = YCbCr_img(:,:,1);
            Cb = YCbCr_img(:,:,2);
            Cr = YCbCr_img(:,:,3);

            [H, W] = size(Y);
            [sf_v, sf_h] = obj.getSubsampleFactor();

            % Downsample Cb and Cr by block-averaging
            Cb_sub = obj.blockAverage(Cb, sf_v, sf_h);
            Cr_sub = obj.blockAverage(Cr, sf_v, sf_h);

            sub_info.Scheme    = obj.Scheme;
            sub_info.OrigSize  = [H, W];
            sub_info.SubFactor = [sf_v, sf_h];
        end

        function YCbCr_out = upsample(obj, Y, Cb_sub, Cr_sub)
            % UPSAMPLE  Reconstruct full-resolution Cb and Cr from subsampled arrays.
            %
            %   The full-resolution luminance Y is used as guidance for the
            %   GuidedFilter or JointBilateralFilter upsampling.
            %
            %   Inputs:
            %     Y       — (H x W) full-resolution luminance channel
            %     Cb_sub  — Subsampled Cb (H/sf_v x W/sf_h)
            %     Cr_sub  — Subsampled Cr
            %
            %   Outputs:
            %     YCbCr_out — (H x W x 3) reconstructed YCbCr image
            arguments
                obj
                Y       (:,:) double
                Cb_sub  (:,:) double
                Cr_sub  (:,:) double
            end

            [H, W] = size(Y);

            switch lower(obj.UpsamplingMethod)
                case 'guided'
                    Cb_full = obj.guidedUpsample(Y, Cb_sub, [H, W]);
                    Cr_full = obj.guidedUpsample(Y, Cr_sub, [H, W]);

                case 'jbf'
                    Cb_full = obj.jbfUpsample(Y, Cb_sub, [H, W]);
                    Cr_full = obj.jbfUpsample(Y, Cr_sub, [H, W]);

                case 'nearest'
                    Cb_full = imresize(Cb_sub, [H, W], 'nearest');
                    Cr_full = imresize(Cr_sub, [H, W], 'nearest');

                otherwise
                    error('ChromaSubsampler:UnknownMethod', ...
                        'Unknown UpsamplingMethod: %s', obj.UpsamplingMethod);
            end

            YCbCr_out = cat(3, Y, Cb_full, Cr_full);
        end
    end

    methods (Access = private)
        function [sf_v, sf_h] = getSubsampleFactor(obj)
            % Returns vertical and horizontal subsampling factors.
            switch obj.Scheme
                case "4:4:4"
                    sf_v = 1; sf_h = 1;
                case "4:2:2"
                    sf_v = 1; sf_h = 2;
                case "4:2:0"
                    sf_v = 2; sf_h = 2;
            end
        end

        function out = blockAverage(~, img, sf_v, sf_h)
            % BLOCKAVERAGE  Downsample by averaging sf_v x sf_h blocks.
            [H, W] = size(img);
            H2 = floor(H / sf_v);
            W2 = floor(W / sf_h);
            out = zeros(H2, W2);
            for r = 1:H2
                for c = 1:W2
                    ri = (r-1)*sf_v + 1;
                    ci = (c-1)*sf_h + 1;
                    out(r, c) = mean(mean(img(ri:min(ri+sf_v-1, H), ci:min(ci+sf_h-1, W))));
                end
            end
        end

        function out = guidedUpsample(obj, Y, p_sub, target_size)
            % GUIDEDUPSAMPLE  Guided filter upsampling.
            % Nearest-neighbour upscale first, then guided filter with Y as guidance.
            p_nn = imresize(p_sub, target_size, 'nearest');
            gf = OceanMath.Image.Filters.GuidedFilter(...
                'Radius',  obj.GuidedRadius, ...
                'Epsilon', obj.GuidedEpsilon);
            out = gf.applyGuided(p_nn, Y);
        end

        function out = jbfUpsample(obj, Y, p_sub, target_size)
            % JBFUPSAMPLE  Joint Bilateral Filter upsampling.
            p_nn = imresize(p_sub, target_size, 'nearest');
            jbf = OceanMath.Image.Filters.JointBilateralFilter(...
                'SpatialSigma', obj.JBFSpatialSigma, ...
                'RangeSigma',   obj.JBFRangeSigma);
            out = jbf.applyGuided(p_nn, Y);
        end
    end
end
