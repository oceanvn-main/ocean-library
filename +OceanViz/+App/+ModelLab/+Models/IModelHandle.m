classdef (Abstract) IModelHandle < handle
    % IMODELHANDLE  What a loaded model can do, for the Model Lab. Panels ask
    %   for CAPABILITIES (has("sample")), never for concrete classes, so a new
    %   model family only needs a new handle, not new panels.
    %
    %   Capabilities in use: "cnn-legacy", "classify", "diffusion",
    %   "noise-schedule", "conditional", "sample", "vae", "encode", "decode",
    %   "hierarchical".
    %   Images: panels work in DISPLAY range [0, 1]; toModel / toDisplay map to
    %   and from the model's DataRange (e.g. [-1, 1] for DDPM).

    properties (SetAccess = protected)
        Name (1,1) string = ""
        Capabilities (1,:) string = strings(1, 0)
        ImageSize (1,3) double = [16 16 1]
        DataRange (1,2) double = [0 1]
        ClassNames (1,:) string = strings(1, 0)
        Meta struct = struct()
    end

    methods
        function tf = has(obj, capability)
            tf = any(obj.Capabilities == capability);
        end

        function name = datasetName(obj)
            name = "";
            if isfield(obj.Meta, 'Dataset'), name = string(obj.Meta.Dataset); end
        end

        function x = toModel(obj, img01)
            r = obj.DataRange;
            x = r(1) + (r(2) - r(1)) * img01;
        end

        function img = toDisplay(obj, x)
            r = obj.DataRange;
            img = min(1, max(0, (double(gather(OceanDL.Autograd.Tensor.raw(x))) - r(1)) / (r(2) - r(1))));
        end

        function img = fitImage(obj, img)
            % Resize (nearest) and match channels to ImageSize; values in [0, 1].
            H = obj.ImageSize(1); W = obj.ImageSize(2); C = obj.ImageSize(3);
            img = double(img(:, :, :, 1));
            if size(img, 3) == 3 && C == 1, img = mean(img, 3); end
            if size(img, 3) == 1 && C == 3, img = repmat(img, 1, 1, 3); end
            if size(img, 1) ~= H || size(img, 2) ~= W
                img = OceanViz.Strategies.CNN.CNNVizUtils.resizeNearest(img, H, W);
            end
            img = min(1, max(0, reshape(img, H, W, C)));
        end
    end
end
