classdef NNCheckpointRecordable < OceanReplay.Record.IRecordable
    % NNCHECKPOINTRECORDABLE  Converts a legacy NNReplayEngine training
    %   checkpoint (e.g. the Mandelbrot Fourier-feature MLP) into an OceanReplay
    %   recording -- the bridge between the old replay and Replay Studio.
    %
    %   Axis "epoch" = 1..numEpochs. Channels
    %     cost, rSquared   costHistory / accHistory (R^2 for the Mandelbrot run)   scalars
    %     prediction       network output on the display grid at that epoch        image
    %                      (computed ONCE here with the renderer's own code:
    %                      OceanViz.Strategies.DecisionBoundaryRenderer.predictMap)
    %     weights          the flat weight vector [b1; W1; ...; bL; WL]             vector
    %     groundTruth      the target map (no axis)                                 image
    %   Weights are read lazily from the sidecar (<name>_weights.mat) or the main
    %   file, ONE epoch at a time (matfile), and only for the epochs the spec wants.
    %   Image row 1 = the first grid row (Im(c) minimum for the Mandelbrot grid).
    %
    %   Example:
    %     d = OceanViz.Visualizers.NNCheckpointRecordable("mandelbrot_fourier_checkpoint.mat");
    %     S = OceanReplay.Core.Sampling;
    %     spec = OceanReplay.Core.Spec().addGroup("curves", 0, ["cost" "rSquared"]) ...
    %            .addGroup("maps", 1, "prediction", struct('epoch', S.every(20)));
    %     OceanReplay.Record.Recorder.run(d, spec, "mandelbrot_replay.h5");

    properties (SetAccess = private)
        File (1,1) string
        WeightsFile (1,1) string
        Ck struct               % checkpoint fields except weightHistory
    end

    properties (Access = private)
        Weights                 % matfile of the file that holds weightHistory
        CacheEpoch = NaN
        CacheW = []
    end

    properties (Constant)
        Required = ["layerSizes" "numEpochs" "costHistory" "accHistory" "gridNorm" "DISPLAY_RES"]
    end

    methods
        function obj = NNCheckpointRecordable(file)
            arguments
                file (1,1) string {mustBeFile}
            end
            obj.File = file;
            names = string({whos('-file', file).name});
            missing = setdiff(obj.Required, names);
            if ~isempty(missing)
                error('OceanViz:NNCheckpointRecordable:fields', 'Checkpoint "%s" lacks: %s.', file, strjoin(missing, ', '));
            end
            obj.Ck = load(file, names{names ~= "weightHistory"});
            [d, n] = fileparts(file);
            side = fullfile(d, n + "_weights.mat");
            if any(names == "weightHistory"), obj.WeightsFile = file;
            elseif isfile(side), obj.WeightsFile = side;
            else
                error('OceanViz:NNCheckpointRecordable:weights', 'No weightHistory in "%s" and no sidecar "%s".', file, side);
            end
            obj.Weights = matfile(obj.WeightsFile);
        end

        function axes = axesInfo(obj)
            axes = OceanReplay.Core.Axis("epoch", 1:obj.Ck.numEpochs, 'Symbol', "epoch", 'Description', "training epoch");
        end

        function [c, v] = catalog(obj)
            C = @OceanReplay.Core.ChannelInfo; G = obj.Ck.DISPLAY_RES;
            nW = sum(obj.Ck.layerSizes(2:end) .* (obj.Ck.layerSizes(1:end-1) + 1));
            c = [C("cost", "scalar", 'Axes', "epoch", 'Symbol', "cost", 'Description', "training cost (costHistory)"), ...
                 C("rSquared", "scalar", 'Axes', "epoch", 'Symbol', "R^2", 'Description', "accuracy history (R^2 for regression runs)"), ...
                 C("prediction", "image", 'Axes', "epoch", 'ValueShape', [G G 1], 'DisplayRange', [0 1], ...
                   'Symbol', "f(c; W_epoch)", 'Description', "network output (sigmoid) on the display grid with this epoch's weights"), ...
                 C("weights", "vector", 'Axes', "epoch", 'ValueShape', nW, 'Symbol', "W", ...
                   'Description', "flat weight vector [b1; W1; ...; bL; WL] of this epoch")];
            if isfield(obj.Ck, 'groundTruth')
                gt = obj.Ck.groundTruth;
                c(end+1) = C("groundTruth", "image", 'ValueShape', [size(gt) 1], ...
                    'DisplayRange', [min(gt(:)) max(gt(:))], 'Symbol', "target", 'Description', "the target map the network approximates");
            end
            v = OceanReplay.Core.ViewInfo.empty;
        end

        function out = run(obj, rec)
            ck = obj.Ck;
            for ep = 1:ck.numEpochs
                e = ep;
                state = struct('cost', @() ck.costHistory(e), 'rSquared', @() ck.accHistory(e), ...
                    'prediction', @() OceanViz.Strategies.DecisionBoundaryRenderer.predictMap(ck, obj.weightsAt(e)), ...
                    'weights', @() obj.weightsAt(e));
                if isfield(ck, 'groundTruth'), state.groundTruth = @() ck.groundTruth; end
                rec.point(struct('epoch', ep), state);
            end
            out = struct('numEpochs', ck.numEpochs);
        end

        function d = describe(obj)
            ck = obj.Ck;
            m = struct('name', "legacy NN checkpoint", 'file', obj.File, 'layerSizes', ck.layerSizes);
            if isfield(ck, 'activationFns'), m.activationFns = string(ck.activationFns); end
            d = struct('model', m, ...
                'references', {{'Converted from an OceanViz.Visualizers.NNReplayEngine checkpoint (exp_mandelbrot_train.m)'}}, ...
                'notation', struct(), 'layout', "default");
        end

        function w = weightsAt(obj, ep)
            % One epoch's flat weight vector (partial load; last epoch cached).
            if ep ~= obj.CacheEpoch
                if size(obj.Weights, 'weightHistory', 1) == 1           % row or column cell array
                    c = obj.Weights.weightHistory(1, ep);
                else
                    c = obj.Weights.weightHistory(ep, 1);
                end
                obj.CacheW = c{1}; obj.CacheEpoch = ep;
            end
            w = obj.CacheW;
        end
    end
end
