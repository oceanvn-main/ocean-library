classdef ParityLearningLayout < OceanViz.App.Replay.Layouts.IReplayStudioLayout
    % PARITYLEARNINGLAYOUT  A ParityLearningRecordable recording: how player 2 learns to exploit player 1.
    %   Tile 1: win rates of both players over a sliding window of games up to the current game, with the theory's
    %           benchmarks for a random guesser and for the best response (PG1 contract section 3).
    %   Tile 2: P2's rate of guessing even against the rate at which P1 actually puts even (what it must learn).
    %   Tile 3: what P2 holds after the current game (belief or Q table), drawn by its kind's renderer.
    %   Rates are computed at display time from the stored per-game counts: per TURN for the guess/put rates (a
    %   per-game mean would be biased, PG1 test), per GAME for wins. Window = Window games (default 50).

    properties
        Window (1,1) double = 50
    end

    properties (Access = private)
        Cache_ = []            % struct(Rec, R, Turns, GE, PE, windowed rates) read once per recording
        Plots_ = struct()      % retained graphics per tile: lines, cursor; rebuilt only if the axes was cleared
    end

    methods
        function n = name(~), n = "ParityLearning"; end

        function n = timeline(~, ~), n = "p2Reward"; end

        function tf = appliesTo(~, rec)
            tf = all(ismember(["p2Reward" "turns" "p2GuessEven" "p1PutEven"], rec.names()));
        end

        function tiles = tiles(obj, rec)
            T = @OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile;
            tiles = [T("Who wins (window rates; dashed = theory)", ["p2Reward" "turns"], @obj.drawWins), ...
                     T("What P2 learned: guesses even vs P1 puts even", ["p2GuessEven" "p1PutEven" "turns"], @obj.drawGuess)];
            learned = obj.learnedName(rec);
            if learned ~= "", tiles(end+1) = T("P2's learned values after this game", learned); end
        end

        function lines = notes(obj, rec, coords)
            lines = {'PARITY GUESS -- P2 learns from P1 (PG1 contract)', ...
                'Each turn both put a number (parity) and guess the other''s parity; one right guess wins.', ...
                'P1: puts even 90 %, guesses at random. P2: puts even 70 %, learns its guess.', ...
                'Best response: always guess even  ->  P2 wins 0.474 of games (random guess: 0.333).', ''};
            c = obj.read(rec); g = coords.game; [s, e] = obj.span(c, g);
            lines{end+1} = sprintf('game %g: window of %d games (%g..%g)', g, e - s + 1, s, e);
            lines{end+1} = sprintf('P2 win rate %.3f   P1 win rate %.3f', mean(c.R(s:e) == 1), mean(c.R(s:e) == -1));
            lines{end+1} = sprintf('P2 guesses even %.3f of turns   P1 puts even %.3f of turns', ...
                sum(c.GE(s:e)) / sum(c.Turns(s:e)), sum(c.PE(s:e)) / sum(c.Turns(s:e)));
        end

        function caption = drawWins(obj, ax, studio)
            c = obj.read(studio.Recording); g = studio.Coords.game;
            T = studio.Theme;
            refs = {{obj.theoryWin(0.5), '--', 'P2, random guess'}, {obj.theoryWin(1), ':', 'P2, best response'}};
            obj.curves_("Wins", ax, T, g, {c.W2, c.W1}, {'P2 wins', 'P1 wins'}, 'win rate', refs);
            caption = sprintf('window %d games; stored through game %g', obj.Window, g);
        end

        function caption = drawGuess(obj, ax, studio)
            c = obj.read(studio.Recording); g = studio.Coords.game;
            obj.curves_("Guess", ax, studio.Theme, g, {c.GuessEven, c.PutEven}, {'P2 guesses even', 'P1 puts even'}, ...
                'share of turns', {});
            caption = sprintf('per-turn shares, window %d games', obj.Window);
        end
    end

    methods (Access = private)
        function c = read(obj, rec)
            if ~isempty(obj.Cache_) && obj.Cache_.Rec == rec, c = obj.Cache_; return; end
            [~, R] = rec.series("p2Reward"); [~, T] = rec.series("turns");
            [~, GE] = rec.series("p2GuessEven"); [~, PE] = rec.series("p1PutEven");
            c = struct('Rec', rec, 'R', R, 'Turns', T, 'GE', GE, 'PE', PE);
            c.W2 = obj.windowMean(R == 1); c.W1 = obj.windowMean(R == -1);          % derived once per recording
            c.GuessEven = obj.windowRatio(GE, T); c.PutEven = obj.windowRatio(PE, T);
            obj.Cache_ = c;
        end

        function curves_(obj, key, ax, T, g, ys, names, ylab, refs)
            % Retained plot: build once (axes, legend, reference lines), then only rewrite XData/YData and the cursor.
            n = numel(ys{1}); k = (1:n).'; m = min(max(g, 1), n);
            h = []; if isfield(obj.Plots_, key), h = obj.Plots_.(key); end
            if isempty(h) || ~isvalid(ax) || ~all(isvalid(h.Lines)) || h.Axes ~= ax || ~isequal(ax.Tag, char("parity" + key))
                OceanViz.App.Replay.Renderers.IChannelRenderer.prepare(ax);
                hold(ax, 'on');
                cols = {T.Primary, T.Secondary};
                lines = gobjects(1, numel(ys));
                for i = 1:numel(ys)
                    lines(i) = plot(ax, nan, nan, '-', 'Color', cols{i}, 'LineWidth', 1.6, 'DisplayName', names{i});
                end
                for r = refs
                    yline(ax, r{1}{1}, r{1}{2}, 'Color', T.Text, 'DisplayName', r{1}{3});
                end
                cursor = xline(ax, 1, '-', 'Color', T.Grid, 'HandleVisibility', 'off');
                hold(ax, 'off'); xlim(ax, [1 max(n, 2)]); ylim(ax, [0 1]); grid(ax, 'on');
                xlabel(ax, 'game'); ylabel(ax, ylab);
                OceanViz.App.Replay.Renderers.IChannelRenderer.finish(ax, T, true);
                OceanViz.App.Replay.Renderers.IChannelRenderer.styleLegend(legend(ax, 'Location', 'east'), T);
                ax.Tag = char("parity" + key);                     % marks the axes as ours (prepare() clears it)
                h = struct('Axes', ax, 'Lines', lines, 'Cursor', cursor); obj.Plots_.(key) = h;
            end
            for i = 1:numel(ys)
                h.Lines(i).XData = k(1:m); h.Lines(i).YData = ys{i}(1:m);
            end
            h.Cursor.Value = m;
        end

        function [s, e] = span(obj, c, g)
            e = min(max(g, 1), numel(c.R)); s = max(1, e - obj.Window + 1);
        end

        function m = windowMean(obj, x)
            cs = [0; cumsum(double(x(:)))]; n = numel(x); k = (1:n).'; s = max(1, k - obj.Window + 1);
            m = (cs(k + 1) - cs(s)) ./ (k - s + 1);
        end

        function m = windowRatio(obj, num, den)
            a = [0; cumsum(double(num(:)))]; b = [0; cumsum(double(den(:)))]; n = numel(num); k = (1:n).';
            s = max(1, k - obj.Window + 1); m = (a(k + 1) - a(s)) ./ (b(k + 1) - b(s));
        end

        function n = learnedName(~, rec)
            n = "";
            if isfield(rec.Meta, 'model') && isfield(rec.Meta.model, 'learnedChannel') ...
                    && any(rec.names() == string(rec.Meta.model.learnedChannel))
                n = string(rec.Meta.model.learnedChannel);
            end
        end
    end

    methods (Static)
        function w = theoryWin(g)
            % P2's win probability when it guesses even with probability g and P1 guesses at random (PG1 section 3).
            p = 0.1 + 0.8 * g; w = p / (1 + p);
        end
    end
end
