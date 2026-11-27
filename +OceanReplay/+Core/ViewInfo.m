classdef ViewInfo < OceanReplay.Core.ChannelInfo
    % VIEWINFO  A channel produced by an explanation METHOD at a point
    %   (saliency, occlusion, ...), not by the computation itself.
    %
    %   In a domain's catalog a ViewInfo describes a method the domain offers:
    %     Name      the method id the spec refers to ("saliency")
    %     Compute   @(state, params, stream) -> value; must not modify the state;
    %               random numbers only from `stream` (the Recorder also guards rng)
    %     HowMade   the "HOW IT WAS MADE" text shown under the view
    %     Reference citation; InTexts = false -> the GUI shows "NOT IN THE TEXTS"
    %   The Recorder copies it per spec entry, setting Name (output name),
    %   MethodId and Params.
    %
    %   Example:
    %     v = OceanReplay.Core.ViewInfo("saliency", "image", 'Axes', "step", ...
    %         'Compute', @(s, p, r) abs(s.grad()), 'HowMade', "|d objective / d x|", ...
    %         'Reference', "Simonyan et al. 2014", 'InTexts', false);

    properties
        HowMade (1,1) string = ""
        Reference (1,1) string = ""
        InTexts (1,1) logical = false
        Params (1,1) struct = struct()
        MethodId (1,1) string = ""
        Compute = []                        % function handle (catalog only; never stored)
    end

    methods
        function obj = ViewInfo(name, kind, opts)
            arguments
                name (1,1) string
                kind (1,1) string
                opts.?OceanReplay.Core.ViewInfo
            end
            obj@OceanReplay.Core.ChannelInfo(name, kind);
            obj = obj.setup(name, kind, opts);
            if obj.MethodId == "", obj.MethodId = name; end
        end

        function s = attrs(obj)
            s = attrs@OceanReplay.Core.ChannelInfo(obj);
            s.method = obj.HowMade;
            s.reference = obj.Reference;
            s.inTexts = uint8(obj.InTexts);                % HDF5 has no logical attribute type (G0)
            s.params = string(jsonencode(obj.Params));
            s.methodId = obj.MethodId;
        end
    end

    methods (Static)
        function obj = fromAttrs(name, a)
            opts = struct('Axes', a.axes, 'ValueShape', a.valueShape, 'Dtype', a.dtype, ...
                'Symbol', a.symbol, 'Units', a.units, 'Description', a.description, 'Level', a.level, ...
                'HowMade', a.method, 'Reference', a.reference, 'InTexts', logical(a.inTexts));
            if isfield(a, 'methodId'), opts.MethodId = a.methodId; end
            if isfield(a, 'params') && strlength(a.params) > 0
                p = jsondecode(char(a.params));
                if isempty(p), p = struct(); end
                opts.Params = p;
            end
            opts = OceanReplay.Core.ChannelInfo.optionals(opts, a);
            c = namedargs2cell(opts);
            obj = OceanReplay.Core.ViewInfo(name, a.kind, c{:});
        end
    end
end
