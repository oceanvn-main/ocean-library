classdef ProcessExplorerTest < OceanTest.BaseTestCase
    methods (Test)
        function testPagedNavigation(tc)
            count = 205;
            template = struct('Id', "", 'Name', "", 'Stage', "Tokens", ...
                'InputIds', strings(0, 1), 'OutputIds', strings(0, 1), ...
                'Summary', "", 'Status', "recorded", 'Sequence', 0);
            steps = repmat(template, 1, count);
            for k = 1:count
                steps(k).Id = "step-" + string(k);
                steps(k).Name = "Token " + string(k);
                steps(k).Sequence = k;
            end
            reader = OceanNLP.Inspect.MemoryReader(steps);
            app = OceanViz.App.NLP.ProcessExplorer(reader, 'Visible', false);
            cleanup = onCleanup(@() app.close()); %#ok<NASGU>
            tc.verifyEqual(numel(app.Steps), 200);
            app.selectStep("step-200");
            app.move(1);
            tc.verifyEqual(app.SelectedId, "step-201");
            tc.verifyEqual(numel(app.Steps), 5);
            app.move(-1);
            tc.verifyEqual(app.SelectedId, "step-200");
            app.search("Token 205");
            tc.verifyEqual(app.SelectedId, "step-205");
        end

        function testSelectSearchAndInspect(tc)
            source = "The sea is blue";
            tokens = split(lower(source));
            artifacts = struct('Id', {"source", "tokens"}, ...
                'Kind', {"text", "tokens"}, ...
                'Name', {"Source text", "Normalized tokens"}, ...
                'Value', {source, tokens}, ...
                'Summary', {"Original text", string(numel(tokens)) + " tokens"});
            steps(1) = struct('Id', "tokenize", 'Name', "Tokenize text", ...
                'Stage', "Preprocess", 'InputIds', "source", ...
                'OutputIds', "tokens", 'Summary', "Lowercase and split on whitespace", ...
                'Status', "recorded", 'Sequence', 1);
            steps(2) = struct('Id', "count", 'Name', "Count tokens", ...
                'Stage', "Count", 'InputIds', "tokens", ...
                'OutputIds', strings(0, 1), 'Summary', "Count normalized tokens", ...
                'Status', "recorded", 'Sequence', 2);
            reader = OceanNLP.Inspect.MemoryReader(steps, artifacts);
            app = OceanViz.App.NLP.ProcessExplorer(reader, 'Visible', false);
            cleanup = onCleanup(@() app.close()); %#ok<NASGU>
            tc.verifyEqual(app.SelectedId, "tokenize");
            app.search("Count");
            tc.verifyEqual(app.SelectedId, "count");
            app.move(1);
            tc.verifyEqual(app.SelectedId, "count");
            app.search("no matching step");
            tc.verifyEqual(app.SelectedId, "");
            app.search("Tokenize");
            tc.verifyEqual(app.SelectedId, "tokenize");
            app.selectStep("tokenize");
            tc.verifyTrue(isgraphics(app.Figure));
        end
    end
end
