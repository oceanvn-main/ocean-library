classdef Context < handle
    % CONTEXT  State shared by every Model Lab panel: the current model (an
    %   IModelHandle), the current input image (display range [0, 1], shaped
    %   for the model), the model's dataset, the theme and a status sink.
    %   Panels listen to its events instead of talking to each other.
    %
    %   events: ModelChanged, ImageChanged

    properties (SetAccess = private)
        Model = []              % OceanViz.App.ModelLab.Models.IModelHandle
        Image = []              % H x W x C, double in [0, 1]
        Dataset = []            % struct X ([0,1] SSCB), y, ClassNames, Name  (or [])
        Theme                   % struct of RGB colours (OceanViz theme + Panel / Accent)
    end

    properties
        StatusFcn = []          % @(msg) -- set by the Shell
        Figure = []             % the app figure (for dialogs / progress)
    end

    events
        ModelChanged
        ImageChanged
    end

    methods
        function obj = Context(themeName)
            if nargin < 1, themeName = 'Dark'; end
            c = OceanViz.Core('Theme', themeName).Colors;
            c.Panel = min(1, c.Background + 0.05);
            c.Accent = c.Primary;
            obj.Theme = c;
        end

        function setModel(obj, handle)
            obj.Model = handle;
            obj.Dataset = [];
            name = handle.datasetName();
            if strlength(name) > 0
                try
                    obj.Dataset = OceanViz.App.ModelLab.Datasets.get(name);
                catch err
                    obj.status("Dataset '" + name + "' unavailable: " + err.message);
                end
            end
            if ~isempty(obj.Dataset)
                obj.Image = handle.fitImage(obj.Dataset.Xte(:, :, :, 1));
            else
                obj.Image = handle.fitImage(rand(handle.ImageSize));
            end
            notify(obj, 'ModelChanged');
        end

        function setImage(obj, img)
            % Any image (H x W x C, [0, 1] or uint8) -> the model's size and channels.
            if isempty(obj.Model), return; end
            if isinteger(img), img = double(img) / double(intmax(class(img))); end
            obj.Image = obj.Model.fitImage(double(img));
            notify(obj, 'ImageChanged');
        end

        function useDatasetImage(obj, k)
            % Test-split image k of the model's dataset (random if k omitted).
            if isempty(obj.Dataset), obj.status("This model has no dataset."); return; end
            n = size(obj.Dataset.Xte, 4);
            if nargin < 2, k = randi(n); end
            obj.setImage(obj.Dataset.Xte(:, :, :, k));
        end

        function status(obj, msg)
            if ~isempty(obj.StatusFcn), obj.StatusFcn(char(msg)); end
        end
    end
end
