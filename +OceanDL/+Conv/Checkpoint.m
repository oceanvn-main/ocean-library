classdef Checkpoint
    % CHECKPOINT  Save/load a CNNTopologySpec + its weights to a
    %   documented, MATLAB-native two-file format:
    %     <name>.json — architecture ONLY (InputChannels, LayerNames, and
    %                   each LayerSpecs{i} independently JSON-encoded --
    %                   see save() for why each layer is encoded as its
    %                   own JSON string rather than one heterogeneous
    %                   struct array).
    %     <name>.mat  — a `weights` struct keyed by layer name (matching
    %                   Spec.LayerNames), each entry a struct with
    %                   `.Weight` (and `.Bias` when HasBias) for conv2d/
    %                   dense layers. Layers with no parameters
    %                   (pooling/activation/flatten/softmax) have no entry.
    %
    %   This mirrors PyTorch's split between model code (here:
    %   CNNTopologySpec, reconstructed from the .json) and a state_dict
    %   (here: the `weights` struct, loaded from the .mat) -- "pretrained"
    %   just means "load a weights struct produced elsewhere into this
    %   format." No ONNX/protobuf parsing is implemented; this is this
    %   library's own checkpoint format, not an interchange format.
    %
    %   USAGE:
    %     OceanDL.Conv.Checkpoint.save('models/mininet', spec, weights);
    %     [spec, weights] = OceanDL.Conv.Checkpoint.load('models/mininet');
    %     eng = OceanDL.Conv.CNNEngine(spec, weights);
    %
    %     % Dev/test convenience -- a small valid random-weight checkpoint:
    %     [spec, weights] = OceanDL.Conv.Checkpoint.makeDummyCheckpoint('models/dummy');
    %
    % See also: OceanDL.Conv.CNNTopologySpec
    %           OceanDL.Conv.CNNEngine

    methods (Static)
        function save(path, spec, weights, imageSize)
            % SAVE  imageSize (optional [H W]) records the input spatial
            %   size the SAVED WEIGHTS actually commit to. It matters
            %   whenever any 'dense' layer sits right after 'flatten': that
            %   Dense's InFeatures gets fixed to H*W*C at whatever image
            %   size produced these weights (see CNNEngine's doc on the
            %   flatten-adjacent Dense's deferred-InFeatures inference) --
            %   a caller MUST reuse this exact size, or CNNEngine.forward()
            %   will throw InputSizeMismatch. Omit it only for checkpoints
            %   with no such layer (e.g. fully convolutional).
            arguments
                path (1,:) char
                spec (1,1) OceanDL.Conv.CNNTopologySpec
                weights = struct()
                imageSize (1,:) double {mustBePositive, mustBeInteger} = []
            end
            [folder, name, ~] = fileparts(path);
            if isempty(folder)
                folder = pwd;
            end
            if ~isfolder(folder)
                mkdir(folder);
            end
            jsonPath = fullfile(folder, [name '.json']);
            matPath  = fullfile(folder, [name '.mat']);

            arch = struct();
            arch.InputChannels = spec.InputChannels;
            arch.LayerNames = spec.LayerNames;
            if ~isempty(imageSize)
                arch.ReferenceImageSize = imageSize;
            end
            % Each layer encoded independently (heterogeneous fields across
            % conv2d/relu/.../dense would otherwise trip up jsondecode's
            % struct-array field-union inference on load).
            arch.LayerSpecsJSON = cellfun(@jsonencode, spec.LayerSpecs, 'UniformOutput', false);

            fid = fopen(jsonPath, 'w');
            if fid == -1
                error('OceanDL:Conv:Checkpoint:CannotWrite', ...
                    'Could not open %s for writing.', jsonPath);
            end
            cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
            fwrite(fid, jsonencode(arch, 'PrettyPrint', true), 'char');
            clear cleanupObj; % close before returning (mat save is separate)

            save(matPath, 'weights');
        end

        function [spec, weights, imageSize] = load(path)
            arguments
                path (1,:) char
            end
            [folder, name, ~] = fileparts(path);
            if isempty(folder)
                folder = pwd;
            end
            jsonPath = fullfile(folder, [name '.json']);
            matPath  = fullfile(folder, [name '.mat']);
            if ~isfile(jsonPath) || ~isfile(matPath)
                error('OceanDL:Conv:Checkpoint:NotFound', ...
                    'Expected both %s and %s to exist.', jsonPath, matPath);
            end

            arch = jsondecode(fileread(jsonPath));
            layerSpecsJSON = arch.LayerSpecsJSON;
            if ~iscell(layerSpecsJSON)
                layerSpecsJSON = cellstr(layerSpecsJSON);
            end
            layerSpecs = cellfun(@jsondecode, layerSpecsJSON, 'UniformOutput', false);
            % jsondecode returns JSON arrays as COLUMN vectors (no row/col
            % concept in JSON) -- CNNTopologySpec/Conv2D/Pooling all
            % require (1,:) row vectors for KernelSize/Stride/Padding, so
            % every multi-element numeric field must be reshaped back to a
            % row on the way in.
            layerSpecs = cellfun(@OceanDL.Conv.Checkpoint.rowifyFields, ...
                layerSpecs, 'UniformOutput', false);

            layerNames = arch.LayerNames;
            if ~iscell(layerNames)
                layerNames = cellstr(layerNames);
            end

            spec = OceanDL.Conv.CNNTopologySpec.make(layerSpecs, ...
                'InputChannels', arch.InputChannels, 'LayerNames', layerNames);

            S = load(matPath, 'weights');
            weights = S.weights;

            imageSize = [];
            if isfield(arch, 'ReferenceImageSize')
                imageSize = reshape(arch.ReferenceImageSize, 1, []);
            end
        end

        function [spec, weights, imageSize] = makeDummyCheckpoint(path)
            % MAKEDUMMYCHECKPOINT  A small Conv-ReLU-Pool-Conv-ReLU-Pool-
            %   Flatten-Dense-ReLU-Dense-Softmax classifier with random
            %   He-normal weights -- for exercising the loader/CNNEngine
            %   before real pretrained weights are available.
            %
            %   [spec, weights] = Checkpoint.makeDummyCheckpoint()          % in-memory only
            %   [spec, weights] = Checkpoint.makeDummyCheckpoint(path)      % also saved to disk
            arguments
                path (1,:) char = ''
            end
            import OceanDL.Conv.CNNTopologySpec
            import OceanDL.Conv.CNNEngine

            spec = CNNTopologySpec.make({ ...
                CNNTopologySpec.conv2d([3 3], 4, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.conv2d([3 3], 8, 'Padding', [1 1]), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.maxpool2d([2 2]), ...
                CNNTopologySpec.flatten(), ...
                CNNTopologySpec.dense(16), ...
                CNNTopologySpec.relu(), ...
                CNNTopologySpec.dense(10), ...
                CNNTopologySpec.softmax() ...
            }, 'InputChannels', 1);

            imageSize = [8 8];
            eng = CNNEngine(spec);                     % random He-normal init
            eng.forward(zeros(imageSize(1), imageSize(2), 1, 1));  % forces the flatten-adjacent Dense to lazily init

            weights = struct();
            for i = 1:numel(spec.LayerSpecs)
                t = spec.LayerSpecs{i}.Type;
                if strcmp(t, 'conv2d') || strcmp(t, 'dense')
                    layer = eng.getLayerObject(spec.LayerNames{i});
                    w = struct('Weight', layer.Weight);
                    if layer.HasBias
                        w.Bias = layer.Bias;
                    end
                    weights.(spec.LayerNames{i}) = w;
                end
            end

            if ~isempty(path)
                OceanDL.Conv.Checkpoint.save(path, spec, weights, imageSize);
            end
        end
    end

    methods (Static, Access = private)
        function s = rowifyFields(s)
            fn = fieldnames(s);
            for k = 1:numel(fn)
                v = s.(fn{k});
                if isnumeric(v) && isvector(v) && numel(v) > 1
                    s.(fn{k}) = reshape(v, 1, []);
                end
            end
        end
    end
end
