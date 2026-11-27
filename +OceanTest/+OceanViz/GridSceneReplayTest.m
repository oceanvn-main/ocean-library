classdef GridSceneReplayTest < OceanTest.BaseTestCase
    % GRIDSCENEREPLAYTEST  GFX1-GFX2 of Graphics_Rendering_and_Replay_Architecture.md: committed-frame recording of a
    % grid AgentWorld, the retained GridScene, the GridWorld replay layout, causal time-paced reads and the clock.

    methods (Static, Access = private)
        function [aw, terrain] = world()
            gw = OceanAgents.World.Doors.GridWorlds.cliffWalking();
            aw = OceanAgents.World.AgentWorld(gw.Doors, gw.Agency, 'Name', "cliff");
            terrain = zeros(4, 12); terrain(4, 2:11) = 1; terrain(4, 12) = 2;
        end

        function out = drive(aw, commit)
            % Fixed action script (up, right x3, down, right): a real episode; commit() at every committed boundary.
            f = aw.reset(7); commit(); out = zeros(1, 0);
            for a = [1 2 2 2 3 2]
                if f.EpisodeEnded, break; end
                d = OceanAgents.Core.Decision('AgentID', f.Roster(1), 'EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, 'Action', a);
                f = aw.step(OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, 'Decisions', d));
                commit(); out(end + 1) = f.Rewards(1); %#ok<AGROW>
            end
        end

        function rec = record(file)
            [aw, terrain] = OceanTest.OceanViz.GridSceneReplayTest.world();
            r = OceanAgents.Observe.GridSceneRecordable(aw, @OceanTest.OceanViz.GridSceneReplayTest.drive, 'Terrain', terrain);
            OceanReplay.Record.Recorder.run(r, r.defaultSpec(), file, 'Overwrite', true);
            rec = OceanReplay.Core.Recording(file);
        end

        function spec = toySpec()
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec().addGroup("all", 0, ["tick" "episode" "position"], struct('commit', S.all()));
        end

        function rec = recordToy(tc, toy)
            f = string(tempname) + ".h5"; tc.addTeardown(@() delete(f));
            OceanReplay.Record.Recorder.run(toy, OceanTest.OceanViz.GridSceneReplayTest.toySpec(), f, 'Overwrite', true);
            rec = OceanReplay.Core.Recording(f);
        end
    end

    methods (Test)
        function recordsEveryCommittedFrame(tc)
            f = string(tempname) + ".h5"; tc.addTeardown(@() delete(f));
            rec = tc.record(f);
            [x, t] = rec.series("tick");
            tc.verifyEqual(x(:).', 1:7, 'reset + six steps = seven commits');
            tc.verifyEqual(t(:).', [0 0:5], 'reset and the first step share world time 0 (Tick is the time of the last advance)');
            p0 = rec.get("position", struct('commit', 1), "exact");
            tc.verifyEqual(double(p0), [4 1], 'start cell of cliff walking');
            p1 = rec.get("position", struct('commit', 2), "exact");
            tc.verifyEqual(double(p1), [3 1], 'after action up');
            tc.verifyEqual(string(rec.get("roster")), "agent1");
            tc.verifyEqual(size(rec.get("terrain")), [4 12]);
        end

        function recordingDoesNotChangeTheRun(tc)
            factory = @() OceanAgents.Observe.GridSceneRecordable(OceanTest.OceanViz.GridSceneReplayTest.world(), ...
                @OceanTest.OceanViz.GridSceneReplayTest.drive, 'Terrain', zeros(4, 12));
            r = factory();
            [ok, rep] = OceanReplay.Record.verifyInvariance(factory, r.defaultSpec());
            tc.addTeardown(@() delete(rep.File));
            tc.verifyTrue(ok, 'recorded and unrecorded runs return identical rewards');
        end

        function sceneKeepsObjectCountAndSelectsByIdentity(tc)
            fig = uifigure('Visible', 'off'); tc.addTeardown(@() delete(fig));
            ax = uiaxes(fig);
            ids = "a" + (1:50);
            s = OceanViz.Scene.GridScene(ax, zeros(5, 5), ids);
            n0 = numel(ax.Children);
            s.update(rand(50, 2) * 4 + 1); s.update(rand(50, 2) * 4 + 1);
            tc.verifyEqual(numel(ax.Children), n0, 'updates create no graphics objects');
            tc.verifyEqual(n0, s.NumGraphicsObjects);
            P = repmat([1 1], 50, 1); P(7, :) = [3 4]; s.update(P);
            tc.verifyEqual(s.idAt([3 4]), "a7");
            s.select("a7"); s.update(flipud(P));                           % reordered rows: identity stays a7
            tc.verifyEqual(s.Selected, "a7");
            tc.verifyError(@() s.update(ones(3, 2)), 'OceanViz:Scene:badPositions');
            tc.verifyError(@() s.select("zz"), 'OceanViz:Scene:unknownId');
        end

        function layoutDrawsRecordingInPlace(tc)
            f = string(tempname) + ".h5"; tc.addTeardown(@() delete(f));
            rec = tc.record(f);
            L = OceanViz.App.Replay.Layouts.GridWorldLayout();
            tc.verifyTrue(L.appliesTo(rec));
            fig = uifigure('Visible', 'off'); tc.addTeardown(@() delete(fig));
            ax = uiaxes(fig);
            studio = struct('Recording', rec, 'Theme', OceanViz.Core('Theme', 'Dark').Colors, 'Coords', struct('commit', 3));
            cap = L.drawScene(ax, studio); n = numel(ax.Children);
            studio.Coords.commit = 5; L.drawScene(ax, studio);
            tc.verifyEqual(numel(ax.Children), n, 'second draw reuses the scene');
            tc.verifyTrue(contains(cap, '3'));
            tc.verifyEqual(L.timeline(rec), "tick");
        end

        function replayStudioOpensGridRecordingWithGridLayout(tc)
            f = string(tempname) + ".h5"; tc.addTeardown(@() delete(f));
            tc.record(f);
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close()); %#ok<NASGU>
            tc.verifyEqual(s.Layout.name(), "GridWorld");
            s.setCoord("commit", 4);
            caps = OceanTest.OceanViz.ReplayStudioTest.captions(s);
            tc.verifyFalse(any(contains(caps, "could not draw")), strjoin(caps, " | "));
            tc.verifyTrue(any(contains(caps, "stored at commit = 4")), strjoin(caps, " | "));
        end

        function frameSourceHoldsLatestAtOrBeforeTime(tc)
            rec = tc.recordToy(tc, OceanTest.OceanViz.Fixtures.TimedToy());
            src = OceanViz.Playback.FrameSource(rec, "position", 'GapThreshold', 2);
            tc.verifyFalse(src.at(-1).Available, 'before the first sample nothing is shown (no extrapolation)');
            a = src.at(0.55);
            tc.verifyEqual([a.Commit a.Time], [2 0.5], 'holds the sample at 0.5, not the nearer one at 0.6');
            tc.verifyEqual(a.Staleness, 0.05, 'AbsTol', 1e-12);
            tc.verifyEqual(double(a.Values.position), [2 102]);
            b = src.at(2.9);
            tc.verifyEqual(b.Commit, 3);
            tc.verifyTrue(b.Gap, 'staleness 2.3 exceeds the threshold');
            tc.verifyEqual(src.at(99).Commit, 6, 'after the end the last sample is held');
            tc.verifyEqual(src.at(3).Commit, 4, 'exact time hits the sample');
        end

        function frameSourceSeparatesEpisodesAndRejectsTimeReversal(tc)
            rec = tc.recordToy(tc, OceanTest.OceanViz.Fixtures.TimedToy([0 1 2 0 1 2], [1 1 1 2 2 2]));
            src = OceanViz.Playback.FrameSource(rec, "position");
            tc.verifyEqual(src.episodes(), [1 2]);
            tc.verifyEqual(src.at(1.5, 'Episode', 2).Commit, 5);
            tc.verifyEqual(src.at(1.5, 'Episode', 1).Commit, 2);
            bad = tc.recordToy(tc, OceanTest.OceanViz.Fixtures.TimedToy([0 2 1], [1 1 1]));
            tc.verifyError(@() OceanViz.Playback.FrameSource(bad, "position"), 'OceanViz:Playback:timeOrder');
        end

        function cacheIsByteBoundedAndPrefetchIsSuperseded(tc)
            rec = tc.recordToy(tc, OceanTest.OceanViz.Fixtures.TimedToy());
            src = OceanViz.Playback.FrameSource(rec, "position", 'CacheBytes', 40);   % two 16-byte values fit
            g = src.newRequest();
            tc.verifyEqual(src.prefetch(1:6, g), 6);
            tc.verifyLessThanOrEqual(src.CachedBytes, 40);
            g2 = src.newRequest();
            tc.verifyEqual(src.prefetch(1:6, g), 0, 'a superseded prefetch stops before loading');
            tc.verifyEqual(src.prefetch(1:2, g2), 2);
            src.at(0.5); src.at(0.5);
            tc.verifyGreaterThanOrEqual(src.Hits, 1);
        end

        function clockMapsWallToPresentationTime(tc)
            w = containers.Map({'t'}, {0});                                  % a handle, so the wall clock can be advanced
            clk = OceanViz.Playback.PlaybackClock([0 10], 'Wall', @() w('t'), 'Speed', 2);
            tc.verifyEqual(clk.now(), 0);
            clk.play(); w('t') = 1.5; tc.verifyEqual(clk.now(), 3);
            g = clk.Generation; clk.pause(); w('t') = 9; tc.verifyEqual(clk.now(), 3, 'paused: wall time no longer moves it');
            tc.verifyGreaterThan(clk.Generation, g);
            clk.seek(8); clk.play(); w('t') = 10; tc.verifyEqual(clk.now(), 10, 'clamped to the end');
            tc.verifyFalse(clk.Playing, 'reaching the end pauses');
            clk.setSpeed(-1); clk.play(); w('t') = 12; tc.verifyEqual(clk.now(), 8, 'reverse playback');
        end
    end
end
