classdef ExperimentResult < handle
    % EXPERIMENTRESULT  Standardized self-describing experiment output container.
    %
    %   Part of the Model-Driven Dynamic Visualization Architecture.
    %   Encapsulates experiment outputs (images, 1D signals, metrics, parameters)
    %   and automatically inspects shape, type, and recommended OceanViz strategies.
    %
    %   Usage:
    %       res = OceanData.ExperimentResult('Total Variation Denoising', ...
    %                 'Description', 'Szeliski §4.2 TV denoising comparison');
    %       
    %       % Add parameters & scalar metrics
    %       res.addParam('Lambda', 0.08);
    %       res.addParam('Solver', 'PrimalDual');
    %       res.addMetric('PSNR_dB', 28.45);
    %       
    %       % Add data channels (auto-inspected)
    %       res.add('NoisyInput', g_noisy, 'Description', 'Input image with AWGN');
    %       res.add('TVDenoised', f_tv,    'Description', 'Reconstructed TV image');
    %       res.add('EnergyHistory', h.Energy, 'Description', 'Objective energy per iter');
    %
    %       % Launch generic studio UI
    %       OceanViz.App.ExperimentStudio.inspect(res);
    %
    %   Author: Ocean Library  |  2026-08-05

    properties
        Title       (1,1) string = "Experiment Result"
        Description (1,1) string = ""
        Timestamp   (1,1) string = ""
        Parameters  struct = struct()
        Metrics     struct = struct()
        Channels    struct = struct()   % Named channel structs with .Data, .Type, .Strategy, .Description
    end

    methods
        function obj = ExperimentResult(title, options)
            % EXPERIMENTRESULT Construct a new experiment result container.
            arguments
                title (1,1) string = "Experiment Result"
                options.Description (1,1) string = ""
            end
            obj.Title       = title;
            obj.Description = options.Description;
            obj.Timestamp   = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
        end

        function addParam(obj, name, val)
            % ADDPARAM Add a hyperparameter or configuration scalar.
            obj.Parameters.(validName(name)) = val;
        end

        function addMetric(obj, name, val)
            % ADDMETRIC Add an evaluation score or timing metric.
            obj.Metrics.(validName(name)) = val;
        end

        function add(obj, name, data, options)
            % ADD Add a data channel (image, signal, matrix, table).
            %   Automatically infers data type and recommended OceanViz strategy.
            arguments
                obj
                name (1,1) string
                data
                options.Type (1,1) string = ""
                options.Strategy (1,1) string = ""
                options.Description (1,1) string = ""
                options.XLabel (1,1) string = "X"
                options.YLabel (1,1) string = "Y"
            end

            field_name = validName(name);

            % Auto-infer type if not explicitly provided
            type = options.Type;
            if type == ""
                type = obj.inferDataType(data);
            end

            % Auto-infer strategy if not explicitly provided
            strategy = options.Strategy;
            if strategy == ""
                strategy = obj.inferStrategy(type, data);
            end

            channel_info = struct( ...
                'Name',         name, ...
                'Data',         data, ...
                'Type',         type, ...
                'Strategy',     strategy, ...
                'Description',  options.Description, ...
                'XLabel',       options.XLabel, ...
                'YLabel',       options.YLabel, ...
                'Shape',        size(data), ...
                'Class',        class(data) ...
            );

            obj.Channels.(field_name) = channel_info;
        end

        function channel = getChannel(obj, name)
            % GETCHANNEL Retrieve channel struct by name.
            field_name = validName(name);
            if isfield(obj.Channels, field_name)
                channel = obj.Channels.(field_name);
            else
                error('ExperimentResult:ChannelNotFound', 'Channel "%s" not found.', name);
            end
        end

        function names = listChannelNames(obj)
            % LISTCHANNELNAMES Returns string array of channel names.
            fields = fieldnames(obj.Channels);
            names = string(fields);
        end

        function save(obj, filepath)
            % SAVE Save experiment result container to .mat file.
            data_struct = struct( ...
                'Title',       obj.Title, ...
                'Description', obj.Description, ...
                'Timestamp',   obj.Timestamp, ...
                'Parameters',  obj.Parameters, ...
                'Metrics',     obj.Metrics, ...
                'Channels',    obj.Channels ...
            );
            save(filepath, 'data_struct', '-v7.3');
        end
    end

    methods (Static)
        function obj = load(filepath)
            % LOAD Load experiment result container from .mat file.
            loaded = load(filepath);
            if isfield(loaded, 'data_struct')
                s = loaded.data_struct;
            elseif isfield(loaded, 'obj')
                s = loaded.obj;
            else
                error('ExperimentResult:InvalidFile', 'Invalid experiment result file.');
            end

            obj = OceanData.ExperimentResult(s.Title, 'Description', s.Description);
            obj.Timestamp  = s.Timestamp;
            obj.Parameters = s.Parameters;
            obj.Metrics    = s.Metrics;
            obj.Channels   = s.Channels;
        end
    end

    methods (Access = private)
        function type = inferDataType(~, data)
            % INFERDATATYPE Auto-detect data type based on dimensions and properties.
            if isnumeric(data) || islogical(data)
                nd = ndims(data);
                sz = size(data);
                if numel(data) == 1
                    type = "scalar";
                elseif isvector(data)
                    type = "vector_1d";
                elseif nd == 2
                    type = "image_gray";
                elseif nd == 3 && (sz(3) == 3 || sz(3) == 4)
                    type = "image_rgb";
                else
                    type = "matrix_nd";
                end
            elseif istable(data)
                type = "table";
            elseif isstruct(data)
                type = "struct";
            else
                type = "unknown";
            end
        end

        function strategy = inferStrategy(~, type, data)
            % INFERSTRATEGY Map data type to default OceanViz strategy.
            switch type
                case "vector_1d"
                    strategy = "TimeSeries";
                case {"image_gray", "image_rgb"}
                    strategy = "Image";
                case "matrix_nd"
                    strategy = "PartialDependence2D";
                case "table"
                    strategy = "BarGroup";
                otherwise
                    strategy = "TimeSeries";
            end
        end
    end
end

function str = validName(name)
    % Helper: Convert string to valid MATLAB struct field name
    str = char(regexprep(name, '[^a-zA-Z0-9_]', '_'));
    if isempty(str) || ~isletter(str(1))
        str = ['ch_' str];
    end
end
