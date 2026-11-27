classdef ItemBatch
    % ITEMBATCH  Test fixture: a reproducible batch of calendar items that all fall due at ONE (time, phase), with
    % deliberately colliding key fields, and a way to schedule them in any insertion order - for the tie-break
    % determinism test (the released order must not depend on insertion order).
    %
    %   b = ItemBatch.random(7, 40);        % 40 items: predictions and commands, seed 7 (own RandStream)
    %   ItemBatch.schedule(cal, b, order);  % one schedule call per item, in the given permutation
    properties (Constant)
        Time = 5
        Phase = 2
    end
    methods (Static)
        function b = random(seed, n)
            rs = RandStream('mrg32k3a', 'Seed', seed);
            isCmd = rand(rs, n, 1) < 0.5;
            b = struct('IsCmd', num2cell(isCmd), ...
                'DoorRank', num2cell(randi(rs, 3, n, 1)), ...
                'RuleRank', num2cell(randi(rs, 2, n, 1)), ...
                'Entity', num2cell((1:n)'), ...              % unique: predictions are unique per (key, entity)
                'Issuer', num2cell(randi(rs, [0 3], n, 1)), ...
                'Seq', num2cell((1:n)'), ...                 % commands: World-stamped unique sequence numbers
                'Key', num2cell("door" + randi(rs, 3, n, 1) + "/pred" + randi(rs, 2, n, 1)));
        end

        function schedule(cal, b, order)
            T = OceanTest.OceanAgents.Fixtures.World.Time.ItemBatch;
            for i = reshape(order, 1, [])
                it = b(i);
                if it.IsCmd
                    cmd = OceanAgents.World.Command.remove(it.Entity).stamp("door" + it.DoorRank, it.Issuer, ...
                        it.Seq, T.Time, "p" + T.Phase);
                    cal.scheduleCommand(cmd, T.Time, T.Phase, it.DoorRank, 'Rank', it.RuleRank);
                else
                    cal.schedulePrediction(it.Key, it.Entity, T.Time, T.Phase, it.DoorRank, "Due", ...
                        'Rank', it.RuleRank);
                end
            end
        end
    end
end
