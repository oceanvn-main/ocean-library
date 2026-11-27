classdef FrameCoordinator < handle
    % FRAMECOORDINATOR  Pull-based frame assembly for GUI, plots, video and inspectors (parent section 5). Views register
    % their channel demand ONCE; each display pulse captures one coherent time context per domain, unions the visible
    % demands, evaluates each shared channel once, and returns one bundle. Nothing is pushed to agents and no clock
    % message travels per entity.
    %
    %   fc = OceanAgents.Time.FrameCoordinator();
    %   fc.addSource("arm", log);                              % a CommitLog; channels are fields of the stored state
    %   fc.addFormulaChannel("arm", "charge", formula);        % pure formula channel, evaluated at the resolved time
    %   fc.addView("scene", "arm", ["pos" "charge"]);          % demand registered once
    %   fc.addView("plot",  "arm", "pos");
    %   b = fc.frame(cursor);                                  % b.Values.arm.pos, b.Views.plot.pos, b.Contexts
    %   fc.frame(cursor).Changed                               % false: paused and unchanged, nothing re-evaluated
    %   fc.accept(b)                                           % false when a newer seek superseded this bundle
    %
    % A bundle carries one context per domain with its ACTUAL resolved time and staleness (time asked minus time shown);
    % domains are never merged into one implied time: CommonBoundary is the shared time only when every resolved time
    % is equal, [] otherwise. Missing coverage is reported in Unavailable and the channel is absent, not invented.
    properties (SetAccess = private)
        Evaluations double = 0      % channel evaluations performed (cache hits not counted)
        Frames double = 0           % frames that required work
        Latest                      % last accepted bundle
        Cache OceanAgents.Time.QueryCache
    end
    properties (Access = private)
        Sources_ struct = struct('Domain', {}, 'Log', {})
        Formulas_ struct = struct('Domain', {}, 'Name', {}, 'Formula', {})
        Views_ struct = struct('Name', {}, 'Domain', {}, 'Channels', {}, 'Visible', {})
        LastSig_ string = ""
        LastBundle_ struct = struct()
    end
    methods
        function obj = FrameCoordinator(opts)
            arguments
                opts.CacheCapacity (1,1) double = 512
            end
            obj.Cache = OceanAgents.Time.QueryCache('Capacity', opts.CacheCapacity);
        end

        function addSource(obj, domainID, log)
            obj.Sources_(end+1) = struct('Domain', string(domainID), 'Log', log);
        end

        function addFormulaChannel(obj, domainID, name, formula)
            obj.Formulas_(end+1) = struct('Domain', string(domainID), 'Name', string(name), 'Formula', formula);
        end

        function addView(obj, name, domainID, channels, opts)
            arguments
                obj
                name (1,1) string
                domainID (1,1) string
                channels string
                opts.Visible (1,1) logical = true
            end
            obj.Views_(end+1) = struct('Name', name, 'Domain', domainID, 'Channels', reshape(channels, 1, []), ...
                'Visible', opts.Visible);
        end

        function setVisible(obj, name, tf)
            k = find([obj.Views_.Name] == string(name), 1);
            if isempty(k), error('OceanAgents:Time:BadValue', 'Unknown view "%s".', name); end
            obj.Views_(k).Visible = tf;
        end

        function setChannels(obj, name, channels)
            k = find([obj.Views_.Name] == string(name), 1);
            if isempty(k), error('OceanAgents:Time:BadValue', 'Unknown view "%s".', name); end
            obj.Views_(k).Channels = reshape(string(channels), 1, []);
        end

        function b = frame(obj, cursors)
            % FRAME  One coherent bundle for the given cursors (one per domain).
            arguments
                obj
                cursors (1,:) OceanAgents.Time.ViewCursor
            end
            sig = obj.signature_(cursors);
            if sig == obj.LastSig_ && ~isempty(fieldnames(obj.LastBundle_))
                b = obj.LastBundle_; b.Changed = false;
                return
            end
            obj.Frames = obj.Frames + 1;
            b = struct('Contexts', {cell(1, numel(cursors))}, 'Values', struct(), 'Views', struct(), ...
                'Unavailable', strings(1, 0), 'Staleness', struct(), 'Generations', [cursors.Generation], ...
                'Domains', [cursors.DomainID], 'CommonBoundary', [], 'Changed', true);
            times = zeros(1, numel(cursors));
            ok = true(1, numel(cursors));
            for i = 1:numel(cursors)
                c = cursors(i); d = c.DomainID;
                src = obj.Sources_([obj.Sources_.Domain] == d);
                if isempty(src)
                    b.Unavailable(end+1) = d; ok(i) = false; continue
                end
                r = src(1).Log.at(c.Target, c.Boundary);
                if r.Status ~= "ok"
                    b.Unavailable(end+1) = d; ok(i) = false; continue
                end
                ctx = r.Context.withGeneration(c.Generation);
                b.Contexts{i} = ctx; times(i) = ctx.Time;
                b.Staleness.(d) = r.Staleness;
                b.Values.(d) = struct();
                want = obj.visibleChannels_(d);
                for ch = want
                    key = OceanAgents.Time.QueryCache.key(ctx, ch, obj.formulaStamp_(d, ch));
                    [hit, v] = obj.Cache.get(key);
                    if ~hit
                        [v, found] = obj.evaluate_(d, ch, r, ctx);
                        if ~found
                            b.Unavailable(end+1) = d + "/" + ch; continue
                        end
                        obj.Cache.put(key, v);
                        obj.Evaluations = obj.Evaluations + 1;
                    end
                    b.Values.(d).(ch) = v;
                end
            end
            for v = obj.Views_
                if ~v.Visible, continue, end
                b.Views.(v.Name) = struct();
                if isfield(b.Values, v.Domain)
                    for ch = v.Channels
                        if isfield(b.Values.(v.Domain), ch), b.Views.(v.Name).(ch) = b.Values.(v.Domain).(ch); end
                    end
                end
            end
            if all(ok) && all(times == times(1)), b.CommonBoundary = times(1); end
            obj.LastSig_ = sig; obj.LastBundle_ = b;
        end

        function tf = accept(obj, bundle, cursors)
            % ACCEPT  A late (asynchronous) bundle may become Latest only if no newer seek superseded it.
            arguments
                obj
                bundle struct
                cursors (1,:) OceanAgents.Time.ViewCursor
            end
            tf = isequal([cursors.Generation], bundle.Generations) && isequal([cursors.DomainID], bundle.Domains);
            if tf, obj.Latest = bundle; end
        end
    end
    methods (Access = private)
        function want = visibleChannels_(obj, d)
            want = strings(1, 0);
            for v = obj.Views_
                if v.Visible && v.Domain == d, want = [want v.Channels]; end %#ok<AGROW>
            end
            want = unique(want, 'stable');
        end

        function s = formulaStamp_(obj, d, ch)
            s = "";
            for f = obj.Formulas_
                if f.Domain == d && f.Name == ch, s = "seg" + f.Formula.segmentCount(); end
            end
        end

        function [v, found] = evaluate_(obj, d, ch, r, ctx)
            found = true;
            for f = obj.Formulas_
                if f.Domain == d && f.Name == ch
                    try
                        v = f.Formula.valueAt(ctx.Time);
                    catch err
                        if err.identifier == "OceanAgents:Time:NoCoverage", v = []; found = false; return, end
                        rethrow(err)
                    end
                    return
                end
            end
            if isstruct(r.State) && isfield(r.State, ch)
                v = r.State.(ch);
            else
                v = []; found = false;
            end
        end

        function sig = signature_(obj, cursors)
            parts = strings(1, 0);
            for c = cursors
                src = obj.Sources_([obj.Sources_.Domain] == c.DomainID);
                ver = 0;
                if ~isempty(src), ver = src(1).Log.Version; end
                parts(end+1) = sprintf('%s:%d:%d', c.DomainID, c.Generation, ver); %#ok<AGROW>
            end
            vis = strings(1, 0);
            for v = obj.Views_
                vis(end+1) = v.Name + string(v.Visible) + strjoin(v.Channels, ","); %#ok<AGROW>
            end
            fs = strings(1, 0);
            for f = obj.Formulas_, fs(end+1) = f.Domain + f.Name + f.Formula.segmentCount(); end %#ok<AGROW>
            sig = strjoin([parts vis fs], ";");
        end
    end
end
