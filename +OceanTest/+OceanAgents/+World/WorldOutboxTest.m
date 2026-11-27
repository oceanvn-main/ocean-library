classdef WorldOutboxTest < OceanTest.BaseTestCase
    % WORLDOUTBOXTEST  W2 kernel addition 'Outbox' / takeOutbox (W2_Grid_Doors_Contract.md section 2; K-W2): events of
    % the outbox types emitted during an advance are recorded for the host, returned in tie-break order and cleared by
    % takeOutbox; reset clears them; an outbox type counts as consumed for V4 (and may even be never emitted); the
    % record is invisible to rules and projections; a failed phase records nothing (fail-stop).

    methods (Static, Access = private)
        function d = emitter(order)
            % Two rules in "act" emit Score (payload who) with SourceEntity 9 (rule A) and 3 (rule B), every tick.
            R = @OceanAgents.World.RuleSpec;
            a = R('Name', "A", 'Phase', "act", 'Emits', "Score", 'Fn', @(c) struct('Events', ...
                OceanAgents.World.Event.make("Score", struct('who', "A", 't', c.Now), 'SourceEntity', 9)));
            b = R('Name', "B", 'Phase', "act", 'Emits', "Score", 'Fn', @(c) struct('Events', ...
                OceanAgents.World.Event.make("Score", struct('who', "B", 't', c.Now), 'SourceEntity', 3)));
            rules = [a b];
            d = OceanAgents.World.DoorSpec('Name', "em", 'Rules', rules(order), 'Drive', "every-tick");
        end
    end

    methods (Test)
        function outboxTypeCountsAsConsumed(tc)
            tc.verifyError(@() OceanAgents.World.World(tc.emitter([1 2])), 'OceanAgents:World:UnconsumedEvent');
            w = OceanAgents.World.World(tc.emitter([1 2]), 'Outbox', "Score");
            tc.verifyEqual(w.Outbox, "Score");
            % an outbox type nobody emits is allowed (e.g. Truncated in a world without a time limit)
            w2 = OceanAgents.World.World(tc.emitter([1 2]), 'Outbox', ["Score" "Truncated"]);
            tc.verifyEqual(w2.Outbox, ["Score" "Truncated"]);
        end

        function recordedInTieBreakOrderAndCleared(tc)
            w = OceanAgents.World.World(tc.emitter([1 2]), 'Outbox', "Score", 'SkipIdle', false);
            w.reset(0);
            w.run(2);
            ev = w.takeOutbox();
            tc.verifyEqual([ev.Time], [0 0 1 1 2 2]);
            tc.verifyEqual(arrayfun(@(e) e.Payload.who, ev), ["A" "B" "A" "B" "A" "B"], ...
                'rule rank before source entity (9 vs 3)');
            tc.verifyEqual([ev.Phase], repmat("act", 1, 6));
            tc.verifyEmpty(w.takeOutbox(), 'taking clears the record');
            w.advance();
            tc.verifyEqual(numel(w.takeOutbox()), 2);
            w.run(5);
            w.reset(0);
            tc.verifyEmpty(w.takeOutbox(), 'reset clears the record');
        end

        function invisibleToRulesAndProjections(tc)
            % A consumer rule sees the events as usual and its ctx has the usual fields only; a projection's ctx too.
            R = @OceanAgents.World.RuleSpec;
            log = R('Name', "Log", 'Phase', "settle", 'Consumes', "Score", 'Reads', "Fields", 'Writes', "Fields", ...
                'Fn', @(c) struct('Writes', struct('Resource', "Fields", 'Values', {string(fieldnames(c))'})));
            see = R('Name', "See", 'Phase', "project", 'Kind', "projection", 'Fn', @(c) struct('View', ...
                struct('Fields', {string(fieldnames(c))'})));
            sink = OceanAgents.World.DoorSpec('Name', "sink", 'Resources', struct('Fields', {strings(1, 0)}), ...
                'Rules', [log see], 'Drive', "on-event");
            w = OceanAgents.World.World([tc.emitter([1 2]) sink], 'Outbox', "Score");
            w.reset(0);
            w.run(1);
            std = sort(["Now" "Phase" "Door" "Rows" "IDs" "Read" "Events" "Stream"]);
            tc.verifyEqual(sort(w.Resources.Fields), std);
            tc.verifyEqual(sort(w.project(0).sink_See.Fields), std);
            tc.verifyEqual(numel(w.takeOutbox()), 4, 'the host record is independent of consumption');
        end

        function failedInstantRecordsNothing(tc)
            % An error in the emitting phase: that synchronisation never completes, so its events are not recorded
            % (the outbox is restored with the rest of the state; fail-stop).
            boom = OceanAgents.World.RuleSpec('Name', "Boom", 'Phase', "act", 'Fn', @(c) boomFn(c));
            bad = OceanAgents.World.DoorSpec('Name', "bad", 'Rules', boom, 'Drive', "every-tick");
            w = OceanAgents.World.World([tc.emitter([1 2]) bad], 'Outbox', "Score");
            w.reset(0);
            w.run(1);
            tc.verifyError(@() w.advance(), 'Test:Boom');
            ev = w.takeOutbox();
            tc.verifyEqual([ev.Time], [0 0 1 1], 'nothing of the failed instant 2');

            function out = boomFn(c)
                out = struct();
                if c.Now == 2, error('Test:Boom', 'boom'); end
            end
        end
    end
end
