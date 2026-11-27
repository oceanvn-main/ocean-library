classdef DeconvVisualizer
    % DECONVVISUALIZER  "What input-space structure corresponds to this
    %   activation?" (Zeiler & Fergus, 2013, "deconvnet".)
    %
    %   NOT a true gradient (see Autograd.Tape for that -- SaliencyMap/
    %   GradCAM use it). The deconvnet reconstruction rule differs from
    %   backprop at exactly one place: going through a ReLU, it masks by
    %   the sign of the RECONSTRUCTED signal itself (max(dY,0)), not by
    %   the sign of the forward pre-activation (dY .* (Z>0)) that true
    %   backprop uses. Every OTHER op in the stack (Conv2D, Pool, Dense,
    %   Flatten) is LINEAR, so its backward() IS the correct "deconvolution"
    %   regardless of which rule you call it by -- this lets
    %   reconstruct() reuse CNNEngine's persistent layer objects'
    %   backward() directly for those, and only special-case ReLU.
    %
    %   Only defined for ReLU networks (as in the original paper); errors
    %   clearly if the path back to the input crosses a sigmoid/tanh/
    %   softmax layer.
    %
    %   USAGE:
    %     r = DeconvVisualizer.reconstruct(engine, X, 'conv2', 5);
    %     r.Reconstruction   % (H, W, Cin, 1), input-space reconstruction

    methods (Static)
        function result = reconstruct(engine, X, targetLayerName, channelIdx, opts)
            arguments
                engine
                X (:,:,:,:) double
                targetLayerName (1,:) char
                channelIdx (1,1) double {mustBePositive, mustBeInteger}
                opts.SpatialIndex (1,:) double = []
            end
            engine.forward(X);
            A = engine.getActivations(targetLayerName);
            [H, W, C, N] = size(A);
            if N ~= 1
                error('OceanDL:Interpretability:DeconvVisualizer:SingleImageOnly', ...
                    'reconstruct() expects a single image (N=1); got N=%d.', N);
            end
            if channelIdx > C
                error('OceanDL:Interpretability:DeconvVisualizer:BadChannel', ...
                    'channelIdx=%d but layer "%s" only has %d channels.', channelIdx, targetLayerName, C);
            end

            if isempty(opts.SpatialIndex)
                chan = A(:, :, channelIdx, 1);
                [~, linIdx] = max(chan(:));
                [ri, ci] = ind2sub([H, W], linIdx);
            else
                ri = opts.SpatialIndex(1); ci = opts.SpatialIndex(2);
            end

            dY = zeros(size(A));
            dY(ri, ci, channelIdx, 1) = A(ri, ci, channelIdx, 1);

            layerIdx = find(strcmp(engine.Spec.LayerNames, targetLayerName), 1);
            if isempty(layerIdx)
                error('OceanDL:Interpretability:DeconvVisualizer:UnknownLayer', ...
                    'Layer "%s" not found.', targetLayerName);
            end

            for i = layerIdx : -1 : 1
                ls = engine.Spec.LayerSpecs{i};
                lname = engine.Spec.LayerNames{i};
                switch ls.Type
                    case {'conv2d', 'maxpool2d', 'avgpool2d', 'flatten', 'dense'}
                        layer = engine.getLayerObject(lname);
                        dY = layer.backward(dY);
                    case 'relu'
                        dY = max(dY, 0);
                    case {'sigmoid', 'tanh'}
                        error('OceanDL:Interpretability:DeconvVisualizer:UnsupportedActivation', ...
                            'Deconvnet reconstruction (Zeiler & Fergus, 2013) is only defined for ReLU networks; layer "%s" is %s.', ...
                            lname, ls.Type);
                    case 'softmax'
                        error('OceanDL:Interpretability:DeconvVisualizer:UnsupportedLayer', ...
                            'Cannot reconstruct through a softmax layer -- choose an earlier targetLayerName.');
                    otherwise
                        error('OceanDL:Interpretability:DeconvVisualizer:UnknownLayerType', ...
                            'Unknown layer type "%s".', ls.Type);
                end
            end

            result = struct('Reconstruction', dY, 'SeedRow', ri, 'SeedCol', ci, 'SeedChannel', channelIdx);
        end
    end
end
