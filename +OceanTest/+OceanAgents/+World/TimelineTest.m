classdef TimelineTest < OceanTest.BaseTestCase
    % TIMELINETEST  OceanAgents.World.Timeline (W1_Kernel_Contract.md section 5.4; design U3/U4; S1 step S1.2b-1): GOD
    % holds the clock and the phase order only - no participants, triggers, rules, commands or events; monotone Now;
    % assembly errors; pillars. Who is due when is tested in ParticipantsTest.

    properties (Constant)
        Phases = ["resolve" "act" "settle" "project"]
    end

    methods (Test)
        function clockOnly(tc)
            % U3 and S1.2b-1: the Timeline holds the clock and the phase order - nothing about who takes part.
            mc = ?OceanAgents.World.Timeline;
            props = string({mc.PropertyList(~[mc.PropertyList.Constant]).Name});
            tc.verifyEqual(sort(props), sort(["Now" "Phases"]));
            meths = string({mc.MethodList.Name});
            tc.verifyFalse(any(ismember(["driven" "nextDriven"], meths)), 'GOD does not know who is due');
            tl = OceanAgents.World.Timeline(tc.Phases);
            tc.verifyEqual(tl.Now, 0);
            tl.setNow(4);
            tl.setNow(4);
            tc.verifyEqual(tl.Now, 4);
            tc.verifyError(@() tl.setNow(3), 'OceanAgents:World:Retroactive');
            tl.reset();
            tc.verifyEqual(tl.Now, 0);
            tc.verifyEqual(tl.phaseIndex("settle"), 3);
            tc.verifyError(@() tl.phaseIndex("noon"), 'OceanAgents:World:BadValue');
            tc.verifyError(@() setfield(tl, 'Now', 9), 'MATLAB:class:SetProhibited');
        end

        function badAssembly(tc)
            ID = 'OceanAgents:World:BadAssembly';
            T = @OceanAgents.World.Timeline;
            tc.verifyError(@() T(strings(1, 0)), ID);
            tc.verifyError(@() T(["a" "a"]), ID);
            tc.verifyError(@() T(["a" ""]), ID);
        end

        function pillars(tc)
            tl = OceanAgents.World.Timeline(tc.Phases);
            tc.verifyFalse(tl.getMetadata().IsSymbolic);
            tc.verifyEqual(tl.getPipelineSpec().ComponentName, "Timeline");
        end
    end
end
