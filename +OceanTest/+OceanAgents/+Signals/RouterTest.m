classdef RouterTest < OceanTest.BaseTestCase
    % ROUTERTEST  AB1 sections 3.4-3.5 and 4: trusted sender, negotiation (direct / converter / refused), sealing,
    % chatter limit, verify-then-act components whose refusals change nothing, and time-binding configuration.

    methods (Static)
        function [r, mindPort, counter, T] = setup()
            T = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', ["left" "right" "forbidden"], ...
                'Role', "command");
            r = OceanAgents.Minds.Signals.Router('MaxEventsPerInstant', 5);
            mindPort = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("mind"), "mind");
            counter = OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("motor");
            r.attach(counter, "motor");
            r.addRoute("motor", "motor", "command");
        end
    end

    methods (Test)
        function deliveryIsStampedByTheRouter(tc)
            [r, port, counter, T] = tc.setup();
            r.advanceTo(3);
            replies = port.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "left"));
            tc.verifyEqual(counter.Count, 1);
            tc.verifyEqual(counter.Last, "left");
            tc.verifyEqual(numel(replies), 1);
            tc.verifyEqual([replies.Sender, replies.Route], ["motor" "motor/reply"]);
            tc.verifyEqual(replies.Time, 3);
            tc.verifyFalse(replies.isRefusal());
            tc.verifyEqual(replies.Value.Count, 1);
            tc.verifyError(@() port.send("motor", replies), 'OceanAgents:Signals:ForgedSender');   % a stamped signal
            tc.verifyError(@() port.send("nowhere", OceanAgents.Minds.Signals.Signal.make(T, "left")), ...
                'OceanAgents:Signals:NoRoute');
            tc.verifyError(@() r.advanceTo(2), 'OceanAgents:Signals:BadValue');
        end

        function refusalChangesNothing(tc)
            [~, port, counter, T] = tc.setup();
            before = counter.exportState();
            replies = port.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "forbidden"));
            tc.verifyTrue(replies.isRefusal());
            tc.verifySubstring(char(replies.Value.Reason), 'forbidden');
            tc.verifyEqual(counter.exportState(), before);
            other = OceanAgents.Minds.Signals.SignalType('ValueKind', "number", 'Role', "command");
            out = counter.receive(OceanAgents.Minds.Signals.Signal.make(other, 2));                 % wrong type
            tc.verifyTrue(out.isRefusal());
            tc.verifyEqual(counter.exportState(), before);
        end

        function negotiationIsByTypeAndFixedAtSeal(tc)
            [r, port, ~, T] = tc.setup();
            sub = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', "left", 'Role', "command");
            tc.verifyEqual(r.negotiate("motor", sub), "direct");
            wrongRole = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', "left", 'Role', "sense");
            tc.verifyError(@() r.negotiate("motor", wrongRole), 'OceanAgents:Signals:TypeMismatch');
            num = OceanAgents.Minds.Signals.SignalType('ValueKind', "number", 'Role', "command");
            tc.verifyError(@() r.negotiate("motor", num), 'OceanAgents:Signals:TypeMismatch');
            r.seal();
            tc.verifyError(@() r.addRoute("x", "motor", "command"), 'OceanAgents:Signals:Sealed');
            tc.verifyError(@() r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("y"), "y"), ...
                'OceanAgents:Signals:Sealed');
            tc.verifyError(@() r.negotiate("motor", sub), 'OceanAgents:Signals:Sealed');
            port.send("motor", OceanAgents.Minds.Signals.Signal.make(sub, "left"));                 % negotiated type
            tc.verifyError(@() port.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "right")), ...
                'OceanAgents:Signals:TypeMismatch');                                                % wider than negotiated
        end

        function aConverterIsInsertedByType(tc)
            r = OceanAgents.Minds.Signals.Router();
            src = r.attach(OceanTest.OceanAgents.Fixtures.Signals.CounterComponent("sensor"), "sensor");
            flags = OceanTest.OceanAgents.Fixtures.Signals.FlagSink();
            r.attach(flags, "brain");
            r.registerConverter(OceanAgents.Components.Converters.ThresholdConverter(2, 1), "threshold");
            r.addRoute("level", "brain", "sense");
            reading = OceanAgents.Minds.Signals.SignalType('Shape', "stream", 'ValueKind', "number", 'Role', "sense", ...
                'Rate', 1);
            tc.verifyEqual(r.negotiate("level", reading), "converter");
            r.seal();
            S = @(v) OceanAgents.Minds.Signals.Signal.make(reading, v);
            for v = [0 1.5 2.2 2.5 1.2 0.9 3]
                r.advanceTo(r.Now + 1);
                src.send("level", S(v));
            end
            tc.verifyEqual(flags.Values, [true false true]);
            tc.verifyEqual(flags.Times, [3 6 7]);
            tc.verifyEqual(flags.Senders, ["threshold" "threshold" "threshold"]);
            tc.verifyEqual(r.modeOf("level"), "converter");
        end

        function chatterStopsTheRun(tc)
            [r, port, ~, T] = tc.setup();
            r.advanceTo(1);
            for k = 1:5, port.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "left")); end
            tc.verifyError(@() port.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "left")), ...
                'OceanAgents:Signals:Chatter');
            r.advanceTo(2);                                                       % a new instant resets the count
            port.send("motor", OceanAgents.Minds.Signals.Signal.make(T, "left"));
        end

        function timeBindingIsConfiguration(tc)
            c = OceanTest.OceanAgents.Fixtures.Signals.CounterComponent();
            tc.verifyEqual(c.Binding, "calendar");                                 % first supported = default
            c.useBinding("formula");
            tc.verifyEqual(c.Binding, "formula");
            tc.verifyError(@() c.useBinding("array"), 'OceanAgents:Components:BadBinding');
            tc.verifyEqual(c.nextChange(0), Inf);
        end
    end
end
