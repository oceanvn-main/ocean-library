classdef SaliencyMap
    % SALIENCYMAP  "Which pixels locally affect the class score most?"
    %   Vanilla gradient saliency (Simonyan, Vedaldi & Zisserman, 2013):
    %   d(classScore)/d(input), max-magnitude-reduced across channels.
    %   A thin, direct use of Tape.gradient via CNNEngine.gradient().
    %
    %   BEST PRACTICE: pass OutputLayer as the PRE-softmax logits layer,
    %   not the softmax output -- softmax saturates near 0/1 and flattens
    %   the gradient there, which is well known to make saliency maps
    %   noisier/less informative than using raw logits (this is a
    %   documented recommendation, not an automatic default, since
    %   CNNTopologySpec does not tag which layer is "the logits layer").
    %
    %   USAGE:
    %     r = SaliencyMap.compute(engine, X, 3, 'OutputLayer', 'dense1');
    %     r.Saliency   % (H, W, N)

    methods (Static)
        function result = compute(engine, X, targetClass, opts)
            arguments
                engine
                X (:,:,:,:) double
                targetClass (1,1) double {mustBePositive, mustBeInteger}
                opts.OutputLayer (1,:) char = ''
            end
            outputLayer = opts.OutputLayer;
            if isempty(outputLayer)
                outputLayer = engine.Spec.LayerNames{end};
            end

            Y = engine.forward(X);
            seed = zeros(size(Y));
            seed(targetClass, :) = 1;
            dX = engine.gradient(outputLayer, 'input', seed);

            saliency = max(abs(dX), [], 3);   % (H, W, 1, N)
            saliency = reshape(saliency, size(saliency, 1), size(saliency, 2), size(saliency, 4));

            result = struct('Gradient', dX, 'Saliency', saliency, 'Score', Y(targetClass, :));
        end
    end
end
