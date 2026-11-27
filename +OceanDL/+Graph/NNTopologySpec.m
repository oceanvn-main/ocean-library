classdef NNTopologySpec
    % NNTOPOLOGYSPEC  Immutable blueprint describing a neural network's shape.
    %   Blueprint of the neural network architecture.
    %   VALUE CLASS (not handle) — copies are fully independent.
    %
    %   DESIGN PRINCIPLE (JAX/Flax NNX):
    %     Topology (this class) is SEPARATE from Parameters (stored in NNGraph).
    %     Two NNGraph instances can share the same spec but carry different weights.
    %     The spec itself is hashable, serializable, and comparable.
    %
    %   USAGE:
    %     spec = OceanDL.Graph.NNTopologySpec.make( ...
    %                [4 8 8 3], {'linear','relu','relu','softmax'});
    %
    %     graph = OceanDL.Graph.NNGraph.fromSpec(spec);
    %
    %   VALID ACTIVATIONS:
    %     'linear'  — identity (used for input layer)
    %     'relu'    — max(0, z)
    %     'sigmoid' — 1/(1+exp(-z))
    %     'tanh'    — tanh(z)
    %     'softmax' — exp(z)/sum(exp(z))  [applied row-wise over a layer]
    %     'gelu'    — z * normcdf(z)     (Hendrycks & Gimpel, 2016)
    %     'swish'   — z * sigmoid(z)     (Ramachandran et al., 2017)
    %     'sine'    — sin(omega_0 * z)   (SIREN, Sitzmann et al., 2020)
    %
    %   NORMALIZATION MODES ('NormalizationMode'):
    %     'none'   — no normalization (default)
    %     'batch'  — BatchNormLayer per hidden layer
    %     'layer'  — LayerNormLayer per hidden layer
    %     'rms'    — RMSNormLayer per hidden layer
    %   Applied AFTER Z = W*A+b, BEFORE activation. Input/output layers are never normed.
    %
    %   BLOCK TYPE ('BlockType'):
    %     'plain'    — standard feedforward (default)
    %     'residual' — every pair of same-width hidden layers forms a residual block.
    %                   If widths differ, a learned projection shortcut is used.
    %   Applied in VectorizedEngine; BackpropEngine accumulates gradient through both paths.
    %
    %   See also: OceanDL.Graph.NNGraph
    %             OceanDL.ResidualBlock

    % ------------------------------------------------------------------ %
    properties
        LayerSizes        (1,:) double {mustBePositive}  % [d1, d2, ..., dL] node counts
        Activations       (1,:) cell                      % {'act1', 'act2', ...}
        HasBias           (1,1) logical = true             % add bias to each non-input node
        ConnectMode       (1,1) string  = "dense"          % 'dense' (Phase 1 only)
        LayerNames        (1,:) cell                       % human-readable layer labels
        NormalizationMode (1,1) string  = "none"           % 'none'/'batch'/'layer'/'rms'
        NormLayers        (1,:) cell                       % INormalization per hidden layer (or [])
        BlockType         (1,1) string  = "plain"          % 'plain' or 'residual'
        ResidualStart     (1,:) logical                    % true at L means: add A_{L-2} skip at L
        SkipProjections   (1,:) cell                       % W_skip{L} if d_{L-2} ~= d_{L} (or [])
    end

    % ================================================================== %
    methods
        function obj = NNTopologySpec(layerSizes, activations, options)
            % NNTOPOLOGYSPEC  Construct a new topology specification.
            %
            %   spec = NNTopologySpec(layerSizes, activations)
            %   spec = NNTopologySpec(layerSizes, activations, 'HasBias', false)
            %   spec = NNTopologySpec(layerSizes, activations, 'NormalizationMode', 'layer')
            arguments
                layerSizes  (1,:) double {mustBePositive}
                activations (1,:) cell
                options.HasBias             (1,1) logical = true
                options.ConnectMode         (1,1) string  = "dense"
                options.LayerNames          (1,:) cell    = {}
                options.NormalizationMode   (1,1) string  = "none"
                options.BlockType           (1,1) string  = "plain"
            end

            nL = numel(layerSizes);

            % --- Guards ---
            if nL < 2
                error('OceanDL:Graph:NNTopologySpec:TooFewLayers', ...
                    'A network needs at least 2 layers (input + output). Got %d.', nL);
            end
            if numel(activations) ~= nL
                error('OceanDL:Graph:NNTopologySpec:SizeMismatch', ...
                    'LayerSizes has %d entries but Activations has %d.', ...
                    nL, numel(activations));
            end

            validActs = {'linear','relu','sigmoid','tanh','softmax','gelu','swish','sine'};
            for i = 1:nL
                if ~ismember(lower(activations{i}), validActs)
                    error('OceanDL:Graph:NNTopologySpec:InvalidActivation', ...
                        'Layer %d: unknown activation "%s". Valid: %s.', ...
                        i, activations{i}, strjoin(validActs, ', '));
                end
            end

            validNorm = {'none','batch','layer','rms'};
            if ~ismember(lower(options.NormalizationMode), validNorm)
                error('OceanDL:Graph:NNTopologySpec:InvalidNormMode', ...
                    'NormalizationMode "%s" unknown. Valid: %s.', ...
                    options.NormalizationMode, strjoin(validNorm, ', '));
            end

            validBlock = {'plain','residual'};
            if ~ismember(lower(options.BlockType), validBlock)
                error('OceanDL:Graph:NNTopologySpec:InvalidBlockType', ...
                    'BlockType "%s" unknown. Valid: plain, residual.', options.BlockType);
            end

            % --- Assign ---
            obj.LayerSizes        = layerSizes;
            obj.Activations       = cellfun(@lower, activations, 'UniformOutput', false);
            obj.HasBias           = options.HasBias;
            obj.ConnectMode       = options.ConnectMode;
            obj.NormalizationMode = lower(options.NormalizationMode);
            obj.BlockType         = lower(options.BlockType);

            % Auto-generate layer names if not provided
            if isempty(options.LayerNames)
                names = cell(1, nL);
                for i = 1:nL
                    if     i == 1,  names{i} = 'Input';
                    elseif i == nL, names{i} = 'Output';
                    else,           names{i} = sprintf('Hidden_%d', i-1);
                    end
                end
                obj.LayerNames = names;
            else
                if numel(options.LayerNames) ~= nL
                    error('OceanDL:Graph:NNTopologySpec:SizeMismatch', ...
                        'LayerNames must have %d entries (one per layer).', nL);
                end
                obj.LayerNames = options.LayerNames;
            end

            % --- Build NormLayers (hidden layers only: L=2..nL-1) --- %
            import OceanDL.Norm.*
            normMode = obj.NormalizationMode;
            obj.NormLayers = cell(1, nL);
            for L = 2:nL-1
                d = layerSizes(L);
                switch normMode
                    case 'batch'
                        obj.NormLayers{L} = BatchNormLayer(d);
                    case 'layer'
                        obj.NormLayers{L} = LayerNormLayer(d);
                    case 'rms'
                        obj.NormLayers{L} = RMSNormLayer(d);
                    otherwise
                        obj.NormLayers{L} = [];
                end
            end
            % --- Build ResidualStart + SkipProjections --- %
            obj.ResidualStart   = false(1, nL);
            obj.SkipProjections = cell(1, nL);
            if strcmp(obj.BlockType, 'residual')
                % Hidden layers only: L = 3..nL-1, blocks ending at even offsets
                % Block: layers [L-1, L] where both are hidden (not input/output)
                for L = 3 : nL-1
                    obj.ResidualStart(L) = true;   % skip from L-2 to L
                    d_skip = layerSizes(L - 2);     % source dim
                    d_here = layerSizes(L);         % target dim
                    if d_skip ~= d_here
                        obj.SkipProjections{L} = ...
                            OceanDL.ResidualBlock.makeProjection(d_skip, d_here);
                    end
                end
            end
        end

        % ---- Display ------------------------------------------------- %

        function disp(obj)
            fprintf('NNTopologySpec\n');
            fprintf('  Layers : %d\n', numel(obj.LayerSizes));
            fprintf('  Shape  : [%s]\n', num2str(obj.LayerSizes, '%d '));
            for i = 1:numel(obj.LayerSizes)
                fprintf('    L%d  %s  (%s)\n', i, obj.LayerNames{i}, obj.Activations{i});
            end
            fprintf('  HasBias     : %s\n', mat2str(obj.HasBias));
            fprintf('  ConnectMode : %s\n', obj.ConnectMode);
        end
    end

    % ================================================================== %
    methods (Static)
        function spec = make(layerSizes, activations, varargin)
            % MAKE  Convenience factory — shorter than calling constructor directly.
            %
            %   spec = NNTopologySpec.make([4 8 3], {'linear','relu','softmax'})
            %   spec = NNTopologySpec.make([4 8 3], {'linear','relu','softmax'}, 'HasBias', false)
            spec = OceanDL.Graph.NNTopologySpec( ...
                layerSizes, activations, varargin{:});
        end

        function tf = isEqual(specA, specB)
            % ISEQUAL  True iff two specs describe the same architecture.
            %   Does NOT compare LayerNames or ConnectMode — only the mathematical shape.
            tf = isequal(specA.LayerSizes,  specB.LayerSizes)  && ...
                 isequal(specA.Activations, specB.Activations) && ...
                 (specA.HasBias == specB.HasBias);
        end
    end
end
