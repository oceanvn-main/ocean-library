classdef Axis
    % AXIS  A named dimension a computation moves along (Format spec Sec. 1).
    %   Values are the REAL axis values in run order (e.g. diffusion step
    %   1000, 999, ..., 0). An axis with empty Values is OPEN: its values are
    %   appended as the run visits them (e.g. k-means iterations, unknown in
    %   advance). The name must be a valid MATLAB identifier (used as a field
    %   of coordinate structs and of the spec's "at" objects).
    %
    %   Example:
    %     ax = OceanReplay.Core.Axis("step", 1000:-1:0, 'Symbol', "i", ...
    %          'Description', "reverse-process step (Lai et al. Sec. 2.2)");

    properties
        Name (1,1) string
        Values (1,:) double = []
        Symbol (1,1) string = ""
        Units (1,1) string = ""
        Description (1,1) string = ""
    end

    methods
        function obj = Axis(name, values, opts)
            arguments
                name (1,1) string
                values (1,:) double = []
                opts.Symbol (1,1) string = ""
                opts.Units (1,1) string = ""
                opts.Description (1,1) string = ""
            end
            if ~isvarname(name)
                error('OceanReplay:Axis:name', 'Axis name "%s" must be a valid MATLAB identifier.', name);
            end
            if numel(unique(values)) ~= numel(values)
                error('OceanReplay:Axis:duplicate', 'Axis "%s" has repeated values; values must be unique.', name);
            end
            obj.Name = name; obj.Values = values;
            obj.Symbol = opts.Symbol; obj.Units = opts.Units; obj.Description = opts.Description;
        end

        function tf = isOpen(obj)
            tf = isempty(obj.Values);
        end
    end
end
