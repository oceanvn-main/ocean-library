classdef (Abstract) BaseStrategy < handle
    properties
        Parent % Reference to the OceanViz.Core engine for theme and sequence
    end
    
    methods (Abstract)
        %Every stragegy must implement this
        plot(obj, ax, data, options)
    end
end