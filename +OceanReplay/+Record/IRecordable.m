classdef (Abstract) IRecordable < handle
    % IRECORDABLE  The adapter contract: the ONLY thing a new domain implements
    %   to become recordable (Planning/System/Replay/Replay_Master_Plan.md Sec. 3.2).
    %
    %   axes = axesInfo()   OceanReplay.Core.Axis array -- what the slider(s) move
    %                       along; an axis with empty Values is open (filled as visited)
    %   [channels, views] = catalog()
    %                       ChannelInfo array: everything this domain CAN record;
    %                       ViewInfo array: the explanation methods it offers
    %   out = run(recorder) runs the REAL computation; at each point it calls
    %                         recorder.point(coords, state)
    %                       coords: struct, one field per axis (the point's values);
    %                       state : struct of function handles, one per channel,
    %                               evaluated only when that channel is wanted.
    %                       Returns the computation's normal output (verifyInvariance
    %                       compares it with and without recording).
    %   d = describe()      optional: struct(model, references, notation[, layout]) for /meta;
    %                       layout = name of a replay layout that suits this domain
    %
    %   Example (a loop in a domain):
    %     for i = L:-1:1
    %         x = step(x, i);
    %         recorder.point(struct('step', i), struct('x', @() x, 'norm', @() norm(x(:))));
    %     end

    methods (Abstract)
        axes = axesInfo(obj)
        [channels, views] = catalog(obj)
        out = run(obj, recorder)
    end

    methods
        function d = describe(obj)
            d = struct('model', struct('name', string(class(obj))), 'references', {{}}, 'notation', struct());
        end
    end
end
