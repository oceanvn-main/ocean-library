classdef CNNEngine < handle
    % CNNENGINE  Builds a CNNTopologySpec's layer stack into stateful
    %   layer objects and drives them through an
    %   OceanDL.Autograd.Tape on every forward() call.
    %
    %   Each layer type maps to a Tape op:
    %     conv2d/maxpool2d/avgpool2d/flatten/dense — a persistent
    %       buffer-on-self layer object (Conv2D/MaxPool2D/AvgPool2D/
    %       Flatten/Dense); the Tape op just calls its forward()/backward().
    %     relu/sigmoid/tanh — a stateless OceanDL.Activation.*
    %       object (IActivation); dX = dY .* act.grad(Z), Z captured by
    %       closure (no buffer needed, matches IActivation's documented
    %       "pure strategy object" contract).
    %     softmax — forward reuses SoftmaxActivation.forward(Z) for the
    %       numeric value, but backward is the FULL per-sample
    %       vector-Jacobian product dX = Y.*(dY - sum(dY.*Y)), NOT
    %       SoftmaxActivation.grad()'s diagonal-only term (that
    %       simplification is only valid for the fused softmax+CE shortcut
    %       BackpropEngine uses -- a general-purpose Tape must support
    %       gradient queries through softmax alone, e.g. "how does this
    %       predicted probability change with the input").
    %
    %   CHANNEL/FEATURE-COUNT INFERENCE: Conv2D's InChannels is computed
    %   structurally while walking the spec (conv2d sets it, pooling/
    %   activation leave it unchanged) -- no data needed. The Dense layer
    %   immediately after Flatten is the one exception: its fan-in is
    %   H*W*C of the actual input image, which CNNTopologySpec deliberately
    %   does not declare, so that Dense layer defers weight initialization
    %   to its first forward() call (see Dense.m).
    %
    %   USAGE:
    %     eng = OceanDL.Conv.CNNEngine(spec);              % fresh random weights
    %     eng = OceanDL.Conv.CNNEngine(spec, weights);     % weights from Checkpoint.load
    %     Y  = eng.predict(X);                                         % X: (H x W x Cin x N)
    %     fmap = eng.getActivations('L4_conv2d');                      % any layer's cached output
    %     dIn  = eng.gradient('L9_softmax', 'input');                  % or 'L4_conv2d' as source (Grad-CAM-shaped)
    %
    % See also: OceanDL.Conv.CNNTopologySpec
    %           OceanDL.Autograd.Tape
    %           OceanDL.Conv.Checkpoint

    properties (SetAccess = private)
        Spec
    end

    properties (Access = private)
        LayerObjs_    = {}   % cell, one per Spec.LayerSpecs entry (layer object, or [] for relu/sigmoid/tanh/softmax)
        LastTape_           % OceanDL.Autograd.Tape from the most recent forward()
        LastNodeIds_  = {}   % cell of node ids, parallel to Spec.LayerSpecs
        InputNodeId_  = ''
    end

    methods
        function obj = CNNEngine(spec, weights)
            arguments
                spec (1,1) OceanDL.Conv.CNNTopologySpec
                weights = struct()
            end
            obj.Spec = spec;
            obj.buildLayers(weights);
        end

        function [Y, outputId] = forward(obj, X)
            % FORWARD  Run the full stack once, recording a fresh Tape.
            %   Y = eng.forward(X)   % X: (H x W x Cin x N)
            import OceanDL.Autograd.Tape
            tape = Tape();
            xId = tape.watch(X, 'input');

            curId = xId;
            curVal = X;
            nL = numel(obj.Spec.LayerSpecs);
            nodeIds = cell(1, nL);

            for i = 1:nL
                ls = obj.Spec.LayerSpecs{i};
                lname = obj.Spec.LayerNames{i};
                switch ls.Type
                    case {'conv2d', 'maxpool2d', 'avgpool2d', 'flatten', 'dense'}
                        layer = obj.LayerObjs_{i};
                        Yi = layer.forward(curVal);
                        id = tape.record(ls.Type, {curId}, Yi, @(dY) {layer.backward(dY)}, lname);
                    case {'relu', 'sigmoid', 'tanh'}
                        act = OceanDL.Activation.ActivationFactory.create(ls.Type);
                        Z = curVal;
                        Yi = act.forward(Z);
                        id = tape.record(ls.Type, {curId}, Yi, @(dY) {dY .* act.grad(Z)}, lname);
                    case 'softmax'
                        Z = curVal;
                        Yi = OceanDL.Activation.SoftmaxActivation().forward(Z);
                        id = tape.record('softmax', {curId}, Yi, ...
                            @(dY) {OceanDL.Conv.CNNEngine.softmaxVJP(Yi, dY)}, lname);
                    otherwise
                        error('OceanDL:Conv:CNNEngine:UnknownLayerType', ...
                            'Unknown layer type "%s".', ls.Type);
                end
                nodeIds{i} = id;
                curId = id;
                curVal = Yi;
            end

            Y = curVal;
            outputId = curId;
            obj.LastTape_ = tape;
            obj.LastNodeIds_ = nodeIds;
            obj.InputNodeId_ = xId;
        end

        function Y = predict(obj, X)
            % PREDICT  Convenience wrapper around forward() returning only Y.
            Y = obj.forward(X);
        end

        function A = getActivations(obj, layerName)
            % GETACTIVATIONS  Cached forward-pass output of any layer (by
            %   LayerNames entry, or 'input') from the most recent forward().
            obj.requireForwardCalled();
            A = obj.LastTape_.getValue(obj.resolveNodeId(layerName));
        end

        function grads = gradient(obj, targetLayerName, sourceLayerName, seed)
            % GRADIENT  d(targetLayer)/d(sourceLayer), from the most recent
            %   forward() pass. Either name may be 'input' or any entry of
            %   Spec.LayerNames -- an intermediate layer is a perfectly
            %   valid target OR source (e.g. Grad-CAM: target='L9_softmax'
            %   or a pre-softmax logits layer, source='L4_conv2d').
            arguments
                obj
                targetLayerName (1,:) char
                sourceLayerName
                seed = []
            end
            obj.requireForwardCalled();
            targetId = obj.resolveNodeId(targetLayerName);
            if iscell(sourceLayerName)
                sourceId = cellfun(@(n) obj.resolveNodeId(n), sourceLayerName, 'UniformOutput', false);
            else
                sourceId = obj.resolveNodeId(sourceLayerName);
            end
            grads = obj.LastTape_.gradient(targetId, sourceId, seed);
        end

        function tape = getTape(obj)
            % GETTAPE  The Tape from the most recent forward() (advanced use:
            %   custom gradient queries not covered by gradient()).
            obj.requireForwardCalled();
            tape = obj.LastTape_;
        end

        function layer = getLayerObject(obj, layerName)
            % GETLAYEROBJECT  The persistent Conv2D/Dense/Pooling/Flatten
            %   instance backing a layer (for weight visualization, or
            %   reading getWeightGrad() after a training backward()). []
            %   for stateless relu/sigmoid/tanh/softmax layers.
            idx = obj.layerIndex(layerName);
            layer = obj.LayerObjs_{idx};
        end
    end

    methods (Access = private)
        function buildLayers(obj, weights)
            nL = numel(obj.Spec.LayerSpecs);
            obj.LayerObjs_ = cell(1, nL);
            channels = obj.Spec.InputChannels;

            for i = 1:nL
                ls = obj.Spec.LayerSpecs{i};
                lname = obj.Spec.LayerNames{i};
                w = obj.lookupWeights(weights, lname);

                switch ls.Type
                    case 'conv2d'
                        if ~isempty(w) && isfield(w, 'Weight') && ~isempty(w.Weight)
                            obj.LayerObjs_{i} = OceanDL.Conv.Conv2D( ...
                                ls.KernelSize, ls.OutChannels, 'Stride', ls.Stride, ...
                                'Padding', ls.Padding, 'HasBias', ls.HasBias, ...
                                'Weight', w.Weight, 'Bias', obj.getFieldOr(w, 'Bias', []));
                        else
                            obj.LayerObjs_{i} = OceanDL.Conv.Conv2D( ...
                                ls.KernelSize, ls.OutChannels, 'InChannels', channels, ...
                                'Stride', ls.Stride, 'Padding', ls.Padding, 'HasBias', ls.HasBias);
                        end
                        channels = ls.OutChannels;
                    case 'maxpool2d'
                        obj.LayerObjs_{i} = OceanDL.Pooling.MaxPool2D( ...
                            ls.PoolSize, 'Stride', ls.Stride, 'Padding', ls.Padding);
                    case 'avgpool2d'
                        obj.LayerObjs_{i} = OceanDL.Pooling.AvgPool2D( ...
                            ls.PoolSize, 'Stride', ls.Stride, 'Padding', ls.Padding);
                    case 'flatten'
                        obj.LayerObjs_{i} = OceanDL.Conv.Flatten();
                    case 'dense'
                        if ~isempty(w) && isfield(w, 'Weight') && ~isempty(w.Weight)
                            obj.LayerObjs_{i} = OceanDL.Conv.Dense( ...
                                size(w.Weight, 2), ls.OutFeatures, 'HasBias', ls.HasBias, ...
                                'Weight', w.Weight, 'Bias', obj.getFieldOr(w, 'Bias', []));
                        else
                            obj.LayerObjs_{i} = OceanDL.Conv.Dense( ...
                                0, ls.OutFeatures, 'HasBias', ls.HasBias); % 0 = defer InFeatures to first forward()
                        end
                    case {'relu', 'sigmoid', 'tanh', 'softmax'}
                        obj.LayerObjs_{i} = []; % stateless, constructed fresh per forward() where needed
                    otherwise
                        error('OceanDL:Conv:CNNEngine:UnknownLayerType', ...
                            'Unknown layer type "%s".', ls.Type);
                end
            end
        end

        function w = lookupWeights(~, weights, layerName)
            if isstruct(weights) && isfield(weights, layerName)
                w = weights.(layerName);
            elseif isa(weights, 'containers.Map') && isKey(weights, layerName)
                w = weights(layerName);
            else
                w = [];
            end
        end

        function v = getFieldOr(~, s, field, default)
            if isstruct(s) && isfield(s, field) && ~isempty(s.(field))
                v = s.(field);
            else
                v = default;
            end
        end

        function requireForwardCalled(obj)
            if isempty(obj.LastTape_)
                error('OceanDL:Conv:CNNEngine:NoForwardCache', ...
                    'Call forward()/predict() before reading activations or gradients.');
            end
        end

        function idx = layerIndex(obj, layerName)
            idx = find(strcmp(obj.Spec.LayerNames, layerName), 1);
            if isempty(idx)
                error('OceanDL:Conv:CNNEngine:UnknownLayer', ...
                    'Layer "%s" not found. Valid names: %s (or ''input'').', ...
                    layerName, strjoin(obj.Spec.LayerNames, ', '));
            end
        end

        function id = resolveNodeId(obj, layerName)
            if strcmp(layerName, 'input')
                id = obj.InputNodeId_;
                return
            end
            idx = obj.layerIndex(layerName);
            id = obj.LastNodeIds_{idx};
        end
    end

    methods (Static, Access = private)
        function dX = softmaxVJP(Y, dY)
            % SOFTMAXVJP  Full per-sample softmax vector-Jacobian product:
            %   dX(:,n) = Y(:,n) .* (dY(:,n) - sum(dY(:,n) .* Y(:,n)))
            %   O(d) per sample, no explicit d x d Jacobian -- the same
            %   identity PyTorch's softmax backward uses internally.
            dot = sum(dY .* Y, 1);       % (1 x N)
            dX = Y .* (dY - dot);
        end
    end
end
