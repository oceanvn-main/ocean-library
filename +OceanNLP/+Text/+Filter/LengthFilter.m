classdef LengthFilter < OceanNLP.Text.ITokenFilter
    % LENGTHFILTER  Keeps tokens whose character length is within [Min, Max]
    % (Lucene LengthFilter). Defaults keep everything.
    properties (SetAccess = private)
        Min (1,1) double = 0
        Max (1,1) double = Inf
    end
    properties (Constant)
        Reference = "Lucene LengthFilter"
        InDomain = "token-sequence"
        OutDomain = "token-sequence"
    end
    methods
        function obj = LengthFilter(options)
            arguments
                options.Min (1,1) double {mustBeNonnegative} = 0
                options.Max (1,1) double {mustBeNonnegative} = Inf
            end
            if options.Min > options.Max
                error('OceanNLP:Text:LengthFilter:BadRange', 'Min (%g) must be <= Max (%g).', options.Min, options.Max);
            end
            obj.Min = options.Min;
            obj.Max = options.Max;
        end
        function out = forward(obj, tokens)
            arguments
                obj
                tokens (1,:) string
            end
            n = strlength(tokens);
            out = tokens(n >= obj.Min & n <= obj.Max);
        end
        function s = name(~), s = "length"; end
    end
end
