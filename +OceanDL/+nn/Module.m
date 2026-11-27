classdef (Abstract) Module < handle
    % MODULE  Base class of every trainable building block (PyTorch torch.nn.Module).
    %
    %   A module owns Parameters and child Modules as ordinary PUBLIC
    %   properties (SetAccess may be protected). parameters() discovers them
    %   by walking the properties -- including cell arrays of modules (as in
    %   Sequential) -- so a subclass only declares them and writes forward().
    %
    %   CONTRACT (repo PyTorch-style convention):
    %     y = m.forward(x)             the computation, on Tensors
    %     P = m.parameters()           unique Parameters (shared ones once)
    %     m.namedParameters()          struct array  Name / Param  ("fc1.Weight")
    %     m.train() / m.eval()         Training flag, recursive (dropout, ...)
    %     m.zeroGrad()                 clear .Grad of every parameter
    %     sd = m.stateDict()           containers.Map  name -> array (a copy)
    %     m.loadStateDict(sd)          shape-checked restore (cast to each parameter's type/device)
    %     m.to('DataType', "single", 'Device', "gpu")   cast/move every parameter and buffer
    %
    %   BUFFERS (non-trained state, e.g. an EMA codebook): a subclass that owns
    %   numeric state outside Parameters overrides applyBuffers(obj, fn) and
    %   assigns obj.X = fn(obj.X) for each buffer, so to() moves it too.
    %
    % EXAMPLE:
    %   classdef TwoLayer < OceanDL.nn.Module
    %     properties (SetAccess = protected)
    %       fc1; fc2
    %     end
    %     methods
    %       function obj = TwoLayer(), obj.fc1 = OceanDL.nn.Linear(4, 8);
    %                                   obj.fc2 = OceanDL.nn.Linear(8, 2); end
    %       function y = forward(obj, x), y = obj.fc2.forward(relu(obj.fc1.forward(x))); end
    %     end
    %   end

    properties (SetAccess = protected)
        Training (1,1) logical = true
    end

    methods (Abstract)
        y = forward(obj, varargin)
    end

    methods
        function P = parameters(obj)
            named = obj.namedParameters();
            P = {named.Param};
        end

        function named = namedParameters(obj)
            named = struct('Name', {}, 'Param', {});
            seen = [];
            [named, ~] = obj.collect('', named, seen);
        end

        function kids = children(obj)
            % Direct child modules, including those inside (nested) cell arrays.
            kids = {};
            for name = obj.memberNames()
                kids = [kids, OceanDL.nn.Module.modulesIn(obj.(name{1}))]; %#ok<AGROW>
            end
        end

        function train(obj, tf)
            if nargin < 2
                tf = true;
            end
            obj.Training = tf;
            kids = obj.children();
            for k = 1:numel(kids)
                kids{k}.train(tf);
            end
        end

        function eval(obj)
            obj.train(false);
        end

        function zeroGrad(obj)
            P = obj.parameters();
            for k = 1:numel(P)
                P{k}.zeroGrad();
            end
        end

        function obj = to(obj, opts)
            % Cast / move all parameters (and buffers) of this module tree in
            % place; returns the module for chaining. Gradients are cleared.
            arguments
                obj
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["", "single", "double"])} = ""
                opts.Device (1,1) string {mustBeMember(opts.Device, ["", "cpu", "gpu"])} = ""
            end
            fn = @(x) OceanDL.Device.cast(x, 'DataType', opts.DataType, 'Device', opts.Device);
            P = obj.parameters();
            for k = 1:numel(P)
                P{k}.Data = fn(P{k}.Data);
                P{k}.zeroGrad();
            end
            obj.visitModules(@(m) m.applyBuffers(fn));
        end

        function t = dataType(obj)
            P = obj.parameters();
            if isempty(P), t = ""; else, t = OceanDL.Device.dataType(P{1}.Data); end
        end

        function d = device(obj)
            P = obj.parameters();
            if isempty(P), d = "cpu"; else, d = OceanDL.Device.name(P{1}.Data); end
        end

        function visitModules(obj, fn)
            % fn(m) on this module and every descendant module (depth first).
            fn(obj);
            kids = obj.children();
            for k = 1:numel(kids)
                kids{k}.visitModules(fn);
            end
        end

        function n = numParameters(obj)
            P = obj.parameters();
            n = sum(cellfun(@(p) numel(p.Data), P));
        end

        function sd = stateDict(obj)
            sd = containers.Map('KeyType', 'char', 'ValueType', 'any');
            named = obj.namedParameters();
            for k = 1:numel(named)
                sd(named(k).Name) = named(k).Param.Data;
            end
        end

        function loadStateDict(obj, sd, opts)
            arguments
                obj
                sd containers.Map
                opts.Strict (1,1) logical = true
            end
            named = obj.namedParameters();
            for k = 1:numel(named)
                key = named(k).Name;
                if ~isKey(sd, key)
                    if opts.Strict
                        error('OceanDL:nn:Module:MissingKey', 'State dict has no entry "%s".', key);
                    end
                    continue;
                end
                value = sd(key);
                if ~isequal(size(value), size(named(k).Param.Data))
                    error('OceanDL:nn:Module:ShapeMismatch', ...
                        '"%s": expected [%s], got [%s].', key, num2str(size(named(k).Param.Data)), num2str(size(value)));
                end
                named(k).Param.Data = cast(value, 'like', named(k).Param.Data);   % keep the module's precision/device
            end
            if opts.Strict
                extra = setdiff(keys(sd), {named.Name});
                if ~isempty(extra)
                    error('OceanDL:nn:Module:UnexpectedKey', 'Unexpected state dict entries: %s.', strjoin(extra, ', '));
                end
            end
        end

        function s = describe(obj)
            named = obj.namedParameters();
            lines = strings(numel(named) + 1, 1);
            lines(1) = sprintf('%s: %d parameters in %d tensors', class(obj), obj.numParameters(), numel(named));
            for k = 1:numel(named)
                lines(k + 1) = sprintf('  %-28s [%s]', named(k).Name, strjoin(string(size(named(k).Param.Data)), 'x'));
            end
            s = strjoin(lines, newline);
            if nargout == 0
                disp(s);
                clear s;
            end
        end
    end

    methods (Access = protected)
        function applyBuffers(obj, fn) %#ok<INUSD>
            % Override in modules that own non-Parameter numeric state.
        end

        function names = memberNames(obj)
            % Public, non-dependent properties (where modules keep their state).
            mc = metaclass(obj);
            props = mc.PropertyList;
            keep = arrayfun(@(p) ischar(p.GetAccess) && strcmp(p.GetAccess, 'public') && ~p.Dependent && ~p.Constant ...
                && ~strcmp(p.Name, 'Training'), props);
            names = {props(keep).Name};
        end
    end

    methods (Access = private)
        function [named, seen] = collect(obj, prefix, named, seen)
            for name = obj.memberNames()
                v = obj.(name{1});
                [named, seen] = OceanDL.nn.Module.visit(v, [prefix name{1}], named, seen);
            end
        end
    end

    methods (Static, Access = private)
        function m = modulesIn(v)
            m = {};
            if isa(v, 'OceanDL.nn.Module')
                m = {v};
            elseif iscell(v)
                for i = 1:numel(v)
                    m = [m, OceanDL.nn.Module.modulesIn(v{i})]; %#ok<AGROW>
                end
            end
        end

        function [named, seen] = visit(v, path, named, seen)
            if isa(v, 'OceanDL.nn.Parameter')
                if ~any(seen == v.Uid)                           % shared parameter: count once
                    named(end+1) = struct('Name', path, 'Param', v);
                    seen(end+1) = v.Uid;
                end
            elseif isa(v, 'OceanDL.nn.Module')
                [named, seen] = v.collect([path '.'], named, seen);
            elseif iscell(v)
                for i = 1:numel(v)
                    [named, seen] = OceanDL.nn.Module.visit(v{i}, sprintf('%s{%d}', path, i), named, seen);
                end
            end
        end
    end
end
