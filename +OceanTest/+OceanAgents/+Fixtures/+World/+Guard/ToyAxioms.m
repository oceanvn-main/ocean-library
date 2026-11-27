classdef ToyAxioms
    % TOYAXIOMS  Module-scope predicates of the "modules" test door. Each returns [ok, detail]; probe predicates draw
    % only from the probe stream they are handed.
    %   TM-1 capacityPositive(m)          capacity is a positive integer                    (static)
    %   TM-2 applyWellFormed(m, s)        apply(x) is finite and the same size as x           (probe)
    %   TM-3 sameSeedSameSamples(m, s)    reset(seed) twice gives the same sample sequence    (probe)
    %   TM-4 globalStreamUntouched(m, s)  reset + sample leave the global stream state intact (probe)
    methods (Static)
        function [ok, detail] = capacityPositive(m)
            c = m.capacity();
            ok = isnumeric(c) && isscalar(c) && c >= 1 && c == fix(c);
            detail = sprintf("capacity = %s", mat2str(c));
        end

        function [ok, detail] = applyWellFormed(m, s)
            x = rand(s, 1, 5);
            y = m.apply(x);
            ok = isequal(size(y), size(x)) && all(isfinite(y));
            detail = sprintf("apply(%s) = %s", mat2str(x, 3), mat2str(y, 3));
        end

        function [ok, detail] = sameSeedSameSamples(m, s)
            seed = floor(rand(s) * 1e6);
            m.reset(seed);
            a = [m.sample() m.sample()];
            m.reset(seed);
            b = [m.sample() m.sample()];
            ok = isequal(a, b);
            detail = sprintf("seed %d", seed);
        end

        function [ok, detail] = globalStreamUntouched(m, s)
            g = RandStream.getGlobalStream();
            before = g.State;
            m.reset(floor(rand(s) * 1e6));
            m.sample();
            ok = isequal(g.State, before);
            detail = "global stream state changed";
            if ok, detail = ""; end
        end

        function [ok, detail] = logFirstDraw(m, s)
            % Probe-mechanics spy: records the first draw of each probe stream on the module.
            m.DrawLog(end + 1) = rand(s);
            ok = true;
            detail = "";
        end
    end
end
