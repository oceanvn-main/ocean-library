classdef RLDoor
    % RLDOOR  The RL door (W2 contract section 6; design section 9A.5): it certifies environment MODULES that implement
    % OceanStats.RL.Env.IEnvironment by probing their behaviour, never by their class name. A new environment (a chess
    % board, a new grid) is admitted by passing the probes, not by being on a list.
    %
    %   door = OceanAgents.World.Doors.RLDoor.spec();                        % DoorSpec (Admits IEnvironment)
    %   cert = OceanAgents.World.Guard.GuardEngine.admitModule(env, door, 'ModuleVersion', "1", 'Seed', 0);
    %   cert.Admitted, cert.failed()                                       % e.g. "RL-5" for a module that fails RL-5
    %
    % Axioms (module scope). RL-1..RL-8 are probed; each probe draws its episode seed and every action choice from the
    % probe stream it is handed (never from the global stream), resets the module with reset('Seed', seed) and runs at
    % most MaxSteps steps of uniformly random LEGAL actions (labels whose info.action_mask entry is true):
    %   RL-1  postcondition  reset(seed) returns an observation in ObservationSpace and an action mask of length |A|
    %   RL-2  postcondition  step returns an observation in the space, a finite real scalar reward, logical scalar
    %                        terminated and truncated, and an action mask of length |A|
    %   RL-3  invariant      never both terminated and truncated; terminated => all-false mask; otherwise (live, or
    %                        truncated with its successor mask) >= 1 legal action
    %   RL-4  trajectory     the same seed and the same actions give the same trajectory (two runs per probe)
    %   RL-5  precondition   a step with an action outside the ActionSpace, or after termination, is refused (throws)
    %                        and changes nothing: the episode continues exactly as a run without the refused call.
    %                        Coverage: the post-termination half needs a terminated episode. With an exact model the
    %                        probe is directed to a terminal; otherwise it depends on the random episode, and if no
    %                        probe terminates the certificate reports RL-5 as "partial", not "probed"
    %   RL-6  postcondition  a truncation keeps the pre-reset final observation: probed with a TimeLimit wrapper, and
    %                        on the module's own truncations against its unwrapped() dynamics
    %   RL-7  invariant      seeding and running the module leave MATLAB's global random stream untouched
    %   RL-8  postcondition  declared capabilities are real: IEnumerableEnvironment => every probed non-terminal
    %                        observation has a table index that maps back to it; IModelEnvironment => every sampled
    %                        transition has positive probability in get_transitions. Nothing declared => nothing to check.
    %   RL-9  assumed        the observation is a sufficient statistic for the task (never verified)
    % Interpretation notes (A0 conventions): "ended => all-false mask" applies to TERMINATION; a truncated step carries
    % the successor's mask (A0 section 5.1, TimeLimit). "Illegal action" in RL-5 means a label outside the ActionSpace:
    % the native Racetrack accepts in-space labels that its mask excludes, and must be admitted. RL-5 and RL-6 compare
    % observations, masks and flags (the state), not rewards; reward reproducibility is RL-4's job.
    % 3 pillars: getMetadata / getPipelineSpec (IsSymbolic = false: a container of axioms, no formula).

    properties (Constant)
        Name = "rl"
        Version = "1"
        Admits = "OceanStats.RL.Env.IEnvironment"
        MaxSteps = 50
        Probes = 16
    end

    methods (Static)
        function door = spec(options)
            % SPEC  The RL door as a DoorSpec: no components or rules, module axioms RL-1..RL-9.
            arguments
                options.Probes (1,1) double {mustBeInteger, mustBePositive} = OceanAgents.World.Doors.RLDoor.Probes
                options.Version = OceanAgents.World.Doors.RLDoor.Version
            end
            C = 'OceanAgents.World.Doors.RLDoor.';
            A = @(id, kind, fn, desc) OceanAgents.World.Guard.AxiomSpec('ID', id, 'Scope', "module", 'Kind', kind, ...
                'Mode', "probe", 'Probes', options.Probes, 'Predicate', str2func([C fn]), 'Description', desc);
            ax = [A("RL-1", "postcondition", 'rl1', "reset returns an observation in the space and a |A| mask"), ...
                  A("RL-2", "postcondition", 'rl2', "step returns well-formed observation, reward, flags and mask"), ...
                  A("RL-3", "invariant", 'rl3', "never both ended flags; terminated => no legal action; live => one"), ...
                  A("RL-4", "trajectory", 'rl4', "same seed and actions => same trajectory"), ...
                  A("RL-5", "precondition", 'rl5', "illegal or post-termination steps are refused and change nothing"), ...
                  A("RL-6", "postcondition", 'rl6', "a truncation keeps the pre-reset final observation"), ...
                  A("RL-7", "invariant", 'rl7', "seeding and running never touch the global random stream"), ...
                  A("RL-8", "postcondition", 'rl8', "declared capabilities (Enumerable, ExactModel) are real"), ...
                  OceanAgents.World.Guard.AxiomSpec('ID', "RL-9", 'Scope', "module", 'Kind', "invariant", ...
                      'Mode', "assumed", 'Description', "the observation is a sufficient statistic for the task")];
            door = OceanAgents.World.DoorSpec('Name', OceanAgents.World.Doors.RLDoor.Name, 'Version', options.Version, ...
                'Admits', OceanAgents.World.Doors.RLDoor.Admits, 'Axioms', ax, 'Drive', "never");
        end

        % ------------------------------------------------------------------ axioms: [ok, detail] = P(env, stream)
        function [ok, detail] = rl1(env, stream)
            [n, detail] = OceanAgents.World.Doors.RLDoor.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            seed = OceanAgents.World.Doors.RLDoor.seed_(stream);
            [obs, info] = env.reset('Seed', seed);
            [ok, detail] = OceanAgents.World.Doors.RLDoor.checkObsMask_(env, obs, info, n, sprintf('reset(seed %d)', seed));
        end

        function [ok, detail] = rl2(env, stream)
            R = OceanAgents.World.Doors.RLDoor;
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            T = R.rollout_(env, R.seed_(stream), stream, []);
            for t = 1:numel(T.Steps)
                s = T.Steps(t);
                where = sprintf('step %d (seed %d)', t, T.Seed);
                [ok, detail] = R.checkObsMask_(env, s.Obs, s.Info, n, where);
                if ~ok, return; end
                if ~(isnumeric(s.Reward) && isreal(s.Reward) && isscalar(s.Reward) && isfinite(s.Reward))
                    ok = false;  detail = sprintf('%s: reward %s is not a finite real scalar', where, mat2str(s.Reward));
                    return
                end
                if ~(islogical(s.Terminated) && isscalar(s.Terminated) && islogical(s.Truncated) && isscalar(s.Truncated))
                    ok = false;
                    detail = sprintf('%s: terminated (%s) and truncated (%s) must be logical scalars', where, ...
                        class(s.Terminated), class(s.Truncated));
                    return
                end
            end
            detail = sprintf('%d steps well formed (seed %d)', numel(T.Steps), T.Seed);
        end

        function [ok, detail] = rl3(env, stream)
            R = OceanAgents.World.Doors.RLDoor;
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            T = R.rollout_(env, R.seed_(stream), stream, []);
            if ~any(T.Mask0)
                ok = false;  detail = sprintf('reset (seed %d): a live start has no legal action', T.Seed);
                return
            end
            for t = 1:numel(T.Steps)
                s = T.Steps(t);
                te = logical(s.Terminated);  tr = logical(s.Truncated);
                where = sprintf('step %d (seed %d)', t, T.Seed);
                if te && tr
                    ok = false;  detail = sprintf('%s: both terminated and truncated', where);  return
                end
                if te && any(s.Mask)
                    ok = false;  detail = sprintf('%s: terminated but the mask %s is not all false', where, mat2str(s.Mask));
                    return
                end
                if ~te && ~any(s.Mask)
                    ok = false;  detail = sprintf('%s: not terminated but no legal action', where);  return
                end
            end
            detail = "";
        end

        function [ok, detail] = rl4(env, stream)
            R = OceanAgents.World.Doors.RLDoor;
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            A = R.rollout_(env, R.seed_(stream), stream, []);
            B = R.rollout_(env, A.Seed, [], [A.Steps.Action]);
            [ok, detail] = R.sameTrajectory_(A, B, true);
            if ~ok, detail = sprintf('seed %d: %s', A.Seed, detail); end
        end

        function [ok, detail, reached] = rl5(env, stream)
            % reached (coverage): outsideAction = the out-of-space calls were tried; afterTermination = a terminal
            % state was reached, so the post-termination half was exercised. When the random episode does not
            % terminate and the module declares an exact model, the probe is DIRECTED to a terminal with the model.
            R = OceanAgents.World.Doors.RLDoor;
            reached = struct('outsideAction', false, 'afterTermination', false);
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            start = env.ActionSpace.Start;
            outside = [start - 1, start + n];
            A = R.rollout_(env, R.seed_(stream), stream, []);
            actions = [A.Steps.Action];
            j = 1 + floor(rand(stream) * max(numel(actions), 1));    % before which step the illegal calls are tried
            [B, refused] = R.rollout_(env, A.Seed, [], actions, j, outside);
            reached.outsideAction = true;
            if ~refused.ok
                ok = false;  detail = sprintf('seed %d, before step %d: %s', A.Seed, j, refused.detail);
                return
            end
            [ok, detail] = R.sameTrajectory_(A, B, false);
            if ~ok
                detail = sprintf('seed %d: the refused call changed the episode: %s', A.Seed, detail);
                return
            end
            obs0 = A.Obs0;
            atTerminal = ~isempty(A.Steps) && logical(A.Steps(end).Terminated);
            if ~atTerminal
                [atTerminal, obs0] = R.driveToTerminal_(env, A.Seed);
            end
            if atTerminal
                % the episode has just terminated: every further step must be refused as well
                reached.afterTermination = true;
                for label = [start outside]
                    try
                        env.step(label);
                        ok = false;
                        detail = sprintf('seed %d: step(%g) after termination was accepted', A.Seed, label);
                        return
                    catch
                    end
                end
                obs = env.reset('Seed', A.Seed);
                if ~isequaln(obs, obs0)
                    ok = false;  detail = sprintf('seed %d: a refused post-termination step changed the next reset', A.Seed);
                    return
                end
            end
            detail = "";
        end

        function [ok, detail] = rl6(env, stream)
            R = OceanAgents.World.Doors.RLDoor;
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            A = R.rollout_(env, R.seed_(stream), stream, []);
            actions = [A.Steps.Action];
            inner = env.unwrapped();
            % (a) the module's own truncations must keep the final observation of its unwrapped dynamics
            if ~isempty(A.Steps) && logical(A.Steps(end).Truncated) && ~isequal(inner, env)
                U = R.rollout_(inner, A.Seed, [], actions);
                k = numel(actions);
                if numel(U.Steps) < k || ~isequaln(U.Steps(k).Obs, A.Steps(k).Obs)
                    ok = false;
                    detail = sprintf(['seed %d: the module truncated at step %d with observation %s, but its ' ...
                        'unwrapped dynamics are at %s'], A.Seed, k, mat2str(A.Steps(k).Obs), ...
                        mat2str(R.obsAt_(U, k)));
                    return
                end
            end
            % (b) a TimeLimit wrapper truncating at step k must return the module's step-k observation and mask
            live = find(~arrayfun(@(s) logical(s.Terminated) || logical(s.Truncated), A.Steps), 1, 'last');
            if isempty(live), detail = "no live step to truncate at"; return; end
            k = 1 + floor(rand(stream) * min(live, 5));
            W = OceanStats.RL.Env.Wrappers.TimeLimit(env, k);
            B = R.rollout_(W, A.Seed, [], actions(1:k));
            if numel(B.Steps) < k
                ok = false;  detail = sprintf('seed %d: the TimeLimit(%d) run ended early', A.Seed, k);  return
            end
            b = B.Steps(k);  a = A.Steps(k);
            if ~logical(b.Truncated) || ~isequaln(b.Obs, a.Obs) || ~isequal(b.Mask, a.Mask)
                ok = false;
                detail = sprintf(['seed %d: TimeLimit(%d) truncation returned observation %s (truncated %d), expected ' ...
                    'the final observation %s'], A.Seed, k, mat2str(b.Obs), logical(b.Truncated), mat2str(a.Obs));
                return
            end
            detail = "";
        end

        function [ok, detail] = rl7(env, stream)
            R = OceanAgents.World.Doors.RLDoor;
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            g = RandStream.getGlobalStream();
            before = g.State;
            T = R.rollout_(env, R.seed_(stream), stream, []);
            ok = isequal(g.State, before);
            detail = "";
            if ~ok
                detail = sprintf('seed %d: the global random stream changed during reset/%d steps', T.Seed, numel(T.Steps));
            end
        end

        function [ok, detail] = rl8(env, stream)
            R = OceanAgents.World.Doors.RLDoor;
            [n, detail] = R.numActions_(env);
            ok = n > 0;
            if ~ok, return; end
            enumerable = isa(env, 'OceanStats.RL.Env.IEnumerableEnvironment');
            model = isa(env, 'OceanStats.RL.Env.IModelEnvironment');
            if ~enumerable && ~model
                ok = true;  detail = "no capability declared";
                return
            end
            T = R.rollout_(env, R.seed_(stream), stream, []);
            obs = [{T.Obs0}, {T.Steps.Obs}];
            ended = [false, arrayfun(@(s) logical(s.Terminated), T.Steps)];
            if enumerable
                N = env.get_num_states();
                for k = find(~ended)
                    idx = env.state_to_index(obs{k});
                    if ~(isscalar(idx) && idx == fix(idx) && idx >= 1 && idx <= N) || ...
                            ~isequal(reshape(env.index_to_state(idx), 1, []), reshape(double(obs{k}), 1, []))
                        ok = false;
                        detail = sprintf('seed %d: probed state %s (step %d) is not in the declared state table (index %s)', ...
                            T.Seed, mat2str(obs{k}), k - 1, mat2str(idx));
                        return
                    end
                end
            end
            if model
                for t = 1:numel(T.Steps)
                    s = obs{t};  s2 = reshape(obs{t + 1}, 1, []);  a = T.Steps(t).Action;
                    [nextStates, prob] = env.get_transitions(s, a);
                    hit = all(nextStates == s2, 2) & prob(:) > 0;
                    if ~any(hit)
                        ok = false;
                        detail = sprintf('seed %d: sampled %s -(%g)-> %s has no positive probability in the model', ...
                            T.Seed, mat2str(s), a, mat2str(s2));
                        return
                    end
                end
            end
            detail = "";
        end

        % ------------------------------------------------------------------ pillars
        function meta = getMetadata()
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {'admits IEnvironment modules by probing RL-1..RL-8 (RL-9 assumed)', ...
                'probing gives evidence, not proof'};
            meta.IsSymbolic = false;
            meta.Reference = "RL door axiom suite (World_Kernel_Design.md section 9A.5; W2 contract section 6)";
        end

        function spec = getPipelineSpec()
            spec = struct();
            spec.ComponentName = "RLDoor";
            spec.Inputs = struct('Name', "env", 'Domain', "RL-Environment", 'Dtype', "OceanStats.RL.Env.IEnvironment", ...
                'Shape', "1x1");
            spec.Outputs = struct('Name', "door", 'Domain', "World-DoorSpec", 'Dtype', "OceanAgents.World.DoorSpec", ...
                'Shape', "1x1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "rldoor");
            spec.ComposesFrom = {};
        end
    end

    methods (Static, Access = private)
        function [got, obs0] = driveToTerminal_(env, seed)
            % Directed probe: reset(seed), then follow the declared model towards the nearest terminal state (fewest
            % steps along positive-probability transitions). The model is the module's own, or that of its
            % unwrapped() dynamics when the module is a wrapper (a time limit); the steps go through the module.
            % got = the episode terminated; the module is then AT that terminal. No model, or any failure of the
            % model methods, means "not reached" (RL-8 judges the model itself).
            got = false;  obs0 = [];
            try
                stepped = env;
                if ~isa(env, 'OceanStats.RL.Env.IModelEnvironment'), env = env.unwrapped(); end
                if ~isa(env, 'OceanStats.RL.Env.IModelEnvironment'), return; end
                states = env.get_all_states();
                N = size(states, 1);
                term = false(N, 1);
                for i = 1:N, term(i) = logical(env.is_terminal(states(i, :))); end
                acts = cell(N, 1);  succ = cell(N, 1);
                for i = find(~term)'
                    acts{i} = reshape(env.get_valid_actions(states(i, :)), 1, []);
                    succ{i} = cell(1, numel(acts{i}));
                    for k = 1:numel(acts{i})
                        [nx, p] = env.get_transitions(states(i, :), acts{i}(k));
                        nx = nx(p(:) > 0, :);
                        idx = zeros(1, size(nx, 1));
                        for r = 1:size(nx, 1), idx(r) = env.state_to_index(nx(r, :)); end
                        succ{i}{k} = idx;                     % 0 = a terminal outside the table
                    end
                end
                dist = inf(N, 1);  dist(term) = 0;
                best = zeros(N, 1);
                changed = true;
                while changed
                    changed = false;
                    for i = find(~term)'
                        for k = 1:numel(acts{i})
                            j = succ{i}{k};
                            d = 1 + min([dist(j(j >= 1)); zeros(nnz(j < 1), 1)]);
                            if d < dist(i), dist(i) = d;  best(i) = k;  changed = true; end
                        end
                    end
                end
                obs0 = stepped.reset('Seed', seed);
                obs = obs0;
                for t = 1:min(4 * N + 50, 2000)
                    i = env.state_to_index(obs);
                    if i < 1 || ~isfinite(dist(i)) || best(i) == 0, return; end
                    [obs, ~, te, tr] = stepped.step(acts{i}(best(i)));
                    if logical(te), got = true;  return; end
                    if logical(tr), return; end
                end
            catch
                got = false;
            end
        end

        function s = seed_(stream)
            s = floor(rand(stream) * 2^31);
        end

        function [n, detail] = numActions_(env)
            detail = "";
            sp = env.ActionSpace;
            if isa(sp, 'OceanStats.RL.Env.Spaces.Discrete')
                n = sp.N;
            else
                n = 0;
                detail = sprintf('the action space %s is not Discrete', sp.describe());
            end
        end

        function m = mask_(info, n)
            if isstruct(info) && isfield(info, 'action_mask')
                m = reshape(logical(info.action_mask), 1, []);
            else
                m = true(1, n);
            end
        end

        function [ok, detail] = checkObsMask_(env, obs, info, n, where)
            ok = false;
            if ~env.ObservationSpace.contains(obs)
                detail = sprintf('%s: observation %s is not in %s', where, mat2str(obs), env.ObservationSpace.describe());
                return
            end
            if ~(isstruct(info) && isfield(info, 'action_mask'))
                detail = sprintf('%s: info has no action_mask', where);
                return
            end
            m = info.action_mask;
            if ~((islogical(m) || (isnumeric(m) && all(m(:) == 0 | m(:) == 1))) && numel(m) == n && isvector(m))
                detail = sprintf('%s: action_mask %s is not a length-%d 0/1 vector', where, mat2str(m), n);
                return
            end
            ok = true;
            detail = "";
        end

        function [T, refused] = rollout_(env, seed, stream, actions, refuseAt, outside)
            % One episode: random legal actions from stream (actions empty) or a replay of actions. With refuseAt,
            % every label in outside is tried (and must throw) just before step refuseAt.
            if nargin < 5, refuseAt = 0; outside = []; end
            refused = struct('ok', true, 'detail', "");
            maxSteps = OceanAgents.World.Doors.RLDoor.MaxSteps;
            n = env.ActionSpace.N;
            start = env.ActionSpace.Start;
            [obs, info] = env.reset('Seed', seed);
            T = struct('Seed', seed, 'Obs0', obs, 'Mask0', OceanAgents.World.Doors.RLDoor.mask_(info, n), ...
                'Steps', struct('Action', {}, 'Obs', {}, 'Reward', {}, 'Terminated', {}, 'Truncated', {}, ...
                'Info', {}, 'Mask', {}));
            mask = T.Mask0;
            replay = isempty(stream);
            for t = 1:maxSteps
                if t == refuseAt
                    for label = outside
                        try
                            env.step(label);
                            refused = struct('ok', false, 'detail', sprintf('step(%g) outside the action space was accepted', label));
                            return
                        catch
                        end
                    end
                end
                if replay && t > numel(actions), break; end
                if replay
                    a = actions(t);
                else
                    legal = find(mask(1:min(end, n)));
                    if isempty(legal), break; end
                    a = start - 1 + legal(1 + floor(rand(stream) * numel(legal)));
                end
                [obs, r, te, tr, info] = env.step(a);
                mask = OceanAgents.World.Doors.RLDoor.mask_(info, n);
                T.Steps(end + 1) = struct('Action', a, 'Obs', obs, 'Reward', r, 'Terminated', te, 'Truncated', tr, ...
                    'Info', info, 'Mask', mask);
                if logical(te) || logical(tr), break; end
            end
        end

        function [ok, detail] = sameTrajectory_(A, B, withRewards)
            ok = false;
            if ~isequaln(A.Obs0, B.Obs0) || ~isequal(A.Mask0, B.Mask0)
                detail = "different first observation or mask";
                return
            end
            if numel(A.Steps) ~= numel(B.Steps)
                detail = sprintf('episode lengths differ (%d vs %d)', numel(A.Steps), numel(B.Steps));
                return
            end
            for t = 1:numel(A.Steps)
                a = A.Steps(t);  b = B.Steps(t);
                same = isequaln(a.Obs, b.Obs) && isequal(a.Mask, b.Mask) && ...
                    isequal(logical(a.Terminated), logical(b.Terminated)) && isequal(logical(a.Truncated), logical(b.Truncated));
                if withRewards, same = same && isequaln(a.Reward, b.Reward); end
                if ~same
                    detail = sprintf('step %d differs: obs %s vs %s, reward %s vs %s', t, mat2str(a.Obs), ...
                        mat2str(b.Obs), mat2str(a.Reward), mat2str(b.Reward));
                    return
                end
            end
            ok = true;
            detail = "";
        end

        function o = obsAt_(U, k)
            if numel(U.Steps) >= k, o = U.Steps(k).Obs; else, o = "(ended earlier)"; end
        end
    end
end
