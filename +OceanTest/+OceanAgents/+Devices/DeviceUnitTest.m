classdef DeviceUnitTest < OceanTest.BaseTestCase
    % DEVICEUNITTEST  DV1 devices one by one (critic DV1 findings 1, 2, 6, 7): the terrain refuses a forbidden move by
    % itself, the reward device refuses malformed requests, bad configurations are refused, and a body's position
    % moves only on the terrain of the ONE world it stands in (a spoofed "terrain" on another router cannot move it).

    methods (Static)
        function [r, world, terrain, position] = wire(varargin)
            terrain = OceanAgents.Devices.Terrain(4, 12, 'Cliff', 38:47, varargin{:});
            position = OceanAgents.Components.GridPosition(48);
            r = OceanAgents.Minds.Signals.Router();
            world = r.attach(OceanTest.OceanAgents.Fixtures.Signals.RecorderComponent("world"), "world");
            tp = r.attach(terrain, "terrain");
            r.attach(position, "position");
            T = @(kind, role) OceanAgents.Minds.Signals.SignalType('ValueKind', kind, 'Role', role);
            r.addRoute("move", "terrain", "command");
            r.negotiate("move", OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', string(1:4), 'Role', "command"));
            r.addRoute("place", "terrain", "command");  r.negotiate("place", T("record", "command"));
            r.addRoute("where", "position", "query");   r.negotiate("where", T("record", "query"));
            r.addRoute("moveTo", "position", "command"); r.negotiate("moveTo", T("record", "command"));
            r.seal();
            terrain.connect(tp);
        end
    end

    methods (Test)
        function theTerrainRefusesAForbiddenMoveItself(tc)
            [~, world, ~, position] = tc.wire('Forbidden', 26);
            world.send("place", OceanAgents.Minds.Signals.Signal.make(OceanAgents.Devices.IDevice.commandRecord(), ...
                struct('Op', "start")));
            tc.verifyEqual(position.Cell, 37);
            M = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', string(1:4), 'Role', "command");
            world.send("move", OceanAgents.Minds.Signals.Signal.make(M, "1"));          % 37 -> 25
            reply = world.send("move", OceanAgents.Minds.Signals.Signal.make(M, "2"));  % 25 -> 26 is forbidden
            tc.verifyTrue(reply(1).isRefusal());
            tc.verifySubstring(char(reply(1).Value.Reason), 'Laws forbid');
            tc.verifyEqual(position.Cell, 25, 'the refused move changed nothing');
        end

        function aSpoofedTerrainCannotMoveTheBody(tc)
            env = OceanAgents.Devices.DeviceWorld.cliffWalking();
            env.reset('Seed', 1);
            env.step(1);
            p = env.Position;
            rogue = OceanAgents.Minds.Signals.Router();
            tc.verifyError(@() rogue.attach(p, "position"), 'OceanAgents:Devices:BadConfig');   % one world per body
            tc.verifyError(@() OceanAgents.Devices.DeviceWorld.cliffWalking('Position', p), 'OceanAgents:Devices:BadConfig');
            tc.verifyEqual(p.Cell, 25);
        end

        function badRequestsAndConfigurationsAreRefused(tc)
            rw = OceanAgents.Devices.Reward('Goal', 48);
            out = rw.receive(OceanAgents.Minds.Signals.Signal.make(OceanAgents.Devices.IDevice.commandRecord(), struct('Op', "pay")));
            tc.verifyTrue(out.isRefusal());
            ev = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "sense");
            tc.verifyTrue(rw.receive(OceanAgents.Minds.Signals.Signal.make(ev, struct('To', 3))).isRefusal());   % no Fell
            tc.verifyEqual(rw.Steps, 0, 'refusals changed nothing');
            tc.verifyError(@() OceanAgents.Devices.Terrain(4, 12, 'Cliff', 37), 'OceanAgents:Devices:BadConfig');
            tc.verifyError(@() OceanAgents.Devices.DeviceWorld.cliffWalking('Cols', 2), 'OceanAgents:Devices:BadConfig');
            tc.verifyError(@() OceanAgents.Devices.DeviceWorld.cliffWalking('Rows', 1), 'OceanAgents:Devices:BadConfig');
            tc.verifyError(@() OceanAgents.Devices.DeviceWorld.cliffWalking('Forbidden', 48), 'OceanAgents:Devices:BadConfig');
        end
        function windMovesAndCellRewardsFollowTheNativeRules(tc)
            % S1.1: wind pushes up from the column moved FROM; stochastic wind w-1, w, w+1 (1/3 each, native order)
            t = OceanAgents.Devices.Terrain(7, 10, 'Start', [4 1], 'Wind', [0 0 0 1 1 1 2 2 1 0], 'StochasticWind', true);
            [cells, probs, fell] = t.outcomes(37, 2);          % row 4, col 7 (wind 2), move right
            tc.verifyEqual(cells', [28 18 8]);                 % pushed up 1, 2, 3 rows into col 8
            tc.verifyEqual(probs', [1 1 1] / 3, 'AbsTol', 1e-15);
            tc.verifyFalse(any(fell));
            tc.verifyEqual(t.resolve(37, 2, 0.5), 18);         % native rule k = find(u * c(end) <= c, 1)
            tc.verifyError(@() t.resolve(37, 2), 'OceanAgents:Devices:BadConfig');   % a random move needs a draw
            k = OceanAgents.Devices.Terrain(3, 3, 'Start', [2 2], 'Moves', [-1 1; 0 0]);   % one diagonal, stay
            tc.verifyEqual([k.resolve(5, 1) k.resolve(5, 2) k.numMoves()], [3 5 2]);
            tc.verifyError(@() OceanAgents.Devices.Terrain(3, 3, 'Wind', [1 0 0], 'Forbidden', 1, 'Start', [3 3]), ...
                'OceanAgents:Devices:BadConfig');                   % a wall with wind is not defined
            r = OceanAgents.Devices.Reward('Goal', [1 7], 'StepReward', 0, 'CellReward', [-1 NaN NaN NaN NaN NaN 1]);
            tc.verifyEqual([r.assess(false, 7) r.assess(false, 1) r.assess(false, 4)], [1 -1 0]);
            [~, term] = r.assess(false, 7);
            tc.verifyTrue(term);
        end
    end
end
