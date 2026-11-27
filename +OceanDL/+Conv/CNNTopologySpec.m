classdef CNNTopologySpec
    % CNNTOPOLOGYSPEC  Immutable blueprint describing a CNN's layer stack.
    %   VALUE CLASS (not handle) -- mirrors
    %   OceanDL.Graph.NNTopologySpec's "shape is a value,
    %   separate from parameters" split, applied to a heterogeneous
    %   Conv2D/Pool/Activation/Flatten/Dense stack instead of a uniform
    %   dense layer-size list.
    %
    %   SCOPE (v1): a LINEAR stack only (no residual/skip declarations --
    %   the underlying OceanDL.Autograd.Tape supports arbitrary
    %   DAGs, so a future CNNTopologySpec extension can add skip syntax
    %   without changing Tape or CNNEngine's core wiring).
    %
    %   USAGE:
    %     import OceanDL.Conv.CNNTopologySpec
    %     spec = CNNTopologySpec.make({ ...
    %         CNNTopologySpec.conv2d([3 3], 8, 'Padding', [1 1]), ...
    %         CNNTopologySpec.relu(), ...
    %         CNNTopologySpec.maxpool2d([2 2]), ...
    %         CNNTopologySpec.conv2d([3 3], 16, 'Padding', [1 1]), ...
    %         CNNTopologySpec.relu(), ...
    %         CNNTopologySpec.maxpool2d([2 2]), ...
    %         CNNTopologySpec.flatten(), ...
    %         CNNTopologySpec.dense(10), ...
    %         CNNTopologySpec.softmax() ...
    %     }, 'InputChannels', 1);
    %
    %   See also: OceanDL.Conv.CNNEngine
    %             OceanDL.Graph.NNTopologySpec

    properties
        LayerSpecs    (1,:) cell     % struct per layer, see Static factories below
        InputChannels (1,1) double {mustBePositive, mustBeInteger} = 1
        LayerNames    (1,:) cell     % human-readable, auto "L<k>_<type>" if not given
    end

    properties (Constant, Access = private)
        ValidTypes = {'conv2d', 'relu', 'sigmoid', 'tanh', 'maxpool2d', 'avgpool2d', ...
                      'flatten', 'dense', 'softmax'}
    end

    methods
        function obj = CNNTopologySpec(layerSpecs, options)
            arguments
                layerSpecs (1,:) cell
                options.InputChannels (1,1) double {mustBePositive, mustBeInteger} = 1
                options.LayerNames (1,:) cell = {}
            end
            nL = numel(layerSpecs);
            if nL < 1
                error('OceanDL:Conv:CNNTopologySpec:Empty', ...
                    'A CNN needs at least one layer.');
            end
            for i = 1:nL
                s = layerSpecs{i};
                if ~isstruct(s) || ~isfield(s, 'Type') || ~ismember(s.Type, obj.ValidTypes)
                    error('OceanDL:Conv:CNNTopologySpec:InvalidLayer', ...
                        'Layer %d: expected a struct from a CNNTopologySpec.* factory (conv2d/relu/.../softmax).', i);
                end
            end
            if ~strcmp(layerSpecs{1}.Type, 'conv2d')
                error('OceanDL:Conv:CNNTopologySpec:MustStartWithConv', ...
                    'The first layer must be conv2d (got "%s").', layerSpecs{1}.Type);
            end

            obj.LayerSpecs = layerSpecs;
            obj.InputChannels = options.InputChannels;

            if isempty(options.LayerNames)
                names = cell(1, nL);
                for i = 1:nL
                    names{i} = sprintf('L%d_%s', i, layerSpecs{i}.Type);
                end
                obj.LayerNames = names;
            else
                if numel(options.LayerNames) ~= nL
                    error('OceanDL:Conv:CNNTopologySpec:SizeMismatch', ...
                        'LayerNames must have %d entries (one per layer).', nL);
                end
                obj.LayerNames = options.LayerNames;
            end
        end

        function disp(obj)
            fprintf('CNNTopologySpec (%d layers, InputChannels=%d)\n', ...
                numel(obj.LayerSpecs), obj.InputChannels);
            for i = 1:numel(obj.LayerSpecs)
                fprintf('  %-16s %s\n', obj.LayerNames{i}, obj.LayerSpecs{i}.Type);
            end
        end
    end

    methods (Static)
        function spec = make(layerSpecs, varargin)
            spec = OceanDL.Conv.CNNTopologySpec(layerSpecs, varargin{:});
        end

        function s = conv2d(kernelSize, outChannels, opts)
            arguments
                kernelSize (1,:) double {mustBePositive, mustBeInteger}
                outChannels (1,1) double {mustBePositive, mustBeInteger}
                opts.Stride (1,:) double {mustBePositive, mustBeInteger} = [1 1]
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0]
                opts.HasBias (1,1) logical = true
            end
            s = struct('Type', 'conv2d', 'KernelSize', kernelSize, 'OutChannels', outChannels, ...
                'Stride', opts.Stride, 'Padding', opts.Padding, 'HasBias', opts.HasBias);
        end

        function s = maxpool2d(poolSize, opts)
            arguments
                poolSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride (1,:) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0]
            end
            s = struct('Type', 'maxpool2d', 'PoolSize', poolSize, 'Stride', opts.Stride, 'Padding', opts.Padding);
        end

        function s = avgpool2d(poolSize, opts)
            arguments
                poolSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Stride (1,:) double {mustBePositive, mustBeInteger} = poolSize
                opts.Padding (1,:) double {mustBeNonnegative, mustBeInteger} = [0 0]
            end
            s = struct('Type', 'avgpool2d', 'PoolSize', poolSize, 'Stride', opts.Stride, 'Padding', opts.Padding);
        end

        function s = relu()
            s = struct('Type', 'relu');
        end

        function s = sigmoid()
            s = struct('Type', 'sigmoid');
        end

        function s = tanhLayer()
            % TANHLAYER  Named to avoid shadowing the built-in tanh().
            s = struct('Type', 'tanh');
        end

        function s = flatten()
            s = struct('Type', 'flatten');
        end

        function s = dense(outFeatures, opts)
            arguments
                outFeatures (1,1) double {mustBePositive, mustBeInteger}
                opts.HasBias (1,1) logical = true
            end
            s = struct('Type', 'dense', 'OutFeatures', outFeatures, 'HasBias', opts.HasBias);
        end

        function s = softmax()
            s = struct('Type', 'softmax');
        end
    end
end
