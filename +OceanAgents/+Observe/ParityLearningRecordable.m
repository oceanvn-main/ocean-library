classdef ParityLearningRecordable < OceanReplay.Record.IRecordable
    % PARITYLEARNINGRECORDABLE  Records how the players of a parity-guess run learn, one point per finished game
    %   (PG1_Parity_Guess_Contract.md). The real JointRunner plays inside the Drive function; the recorder only reads
    %   value records through a ParityGameObserver, so the run is unchanged.
    %
    %   rec = OceanAgents.Observe.ParityLearningRecordable(@drive, env.Factors, 800, ...
    %                                                       'LearnedShape', [3 2], 'LearnedName', "qTable");
    %   OceanReplay.Record.Recorder.run(rec, rec.defaultSpec(), "parity.h5");
    %   function out = drive(observer, flush)
    %       % attach observer to p1 and p2; for each game: runner.run(1); flush(valueP2Holds)
    %
    % flush(learned) records every game the observer finished since the last flush, each with `learned` (what P2
    % holds, copied now; omit when no LearnedName). Playing one game per run(1) gives the same games as run(n).
    % Axis game = 1..Games. Channels per game (raw counts, nothing smoothed):
    %   p2Reward  turns  p2GuessEven (P2's even guesses)  p1PutEven (P1's even puts)  <LearnedName>

    properties (SetAccess = private)
        Drive
        Factors
        Games (1,1) double
        LearnedShape (1,:) double
        LearnedName (1,1) string
        Observer
    end

    properties (Access = private)
        Recorder_ = []
    end

    methods
        function obj = ParityLearningRecordable(drive, factors, games, opts)
            arguments
                drive (1,1) function_handle
                factors (1,1) OceanAgents.Core.ActionFactors
                games (1,1) double {mustBeInteger, mustBePositive}
                opts.LearnedShape (1,:) double = []
                opts.LearnedName (1,1) string = ""
            end
            obj.Drive = drive; obj.Factors = factors; obj.Games = games;
            obj.LearnedShape = opts.LearnedShape; obj.LearnedName = opts.LearnedName;
            obj.Observer = OceanAgents.Observe.ParityGameObserver(factors);
        end

        function axes = axesInfo(obj)
            axes = OceanReplay.Core.Axis("game", 1:obj.Games, 'Symbol', "g", 'Description', "game number (one game = until a deciding turn)");
        end

        function [channels, views] = catalog(obj)
            C = @OceanReplay.Core.ChannelInfo;
            channels = [C("p2Reward", "scalar", 'Axes', "game", 'Symbol', "R_2", 'Description', "player 2's return in the game (+1 win, 0 draw, -1 loss)"), ...
                C("turns", "scalar", 'Axes', "game", 'Description', "turns the game lasted"), ...
                C("p2GuessEven", "scalar", 'Axes', "game", 'Description', "turns in which player 2 guessed even"), ...
                C("p1PutEven", "scalar", 'Axes', "game", 'Description', "turns in which player 1 put an even number")];
            if obj.LearnedName ~= ""
                kind = "vector"; if numel(obj.LearnedShape) == 2 && all(obj.LearnedShape > 1), kind = "matrix"; end
                shape = obj.LearnedShape; if kind == "vector", shape = prod(shape); end
                channels(end+1) = C(obj.LearnedName, kind, 'Axes', "game", 'ValueShape', shape, ...
                    'Description', "what player 2 holds after the game (copied at the game's end)");
            end
            views = OceanReplay.Core.ViewInfo.empty;
        end

        function spec = defaultSpec(obj)
            S = OceanReplay.Core.Sampling;
            names = ["p2Reward" "turns" "p2GuessEven" "p1PutEven"];
            if obj.LearnedName ~= "", names(end+1) = obj.LearnedName; end
            spec = OceanReplay.Core.Spec().addGroup("games", 0, names, struct('game', S.all()));
        end

        function out = run(obj, recorder)
            obj.Recorder_ = recorder;
            cleanup = onCleanup(@() obj.detach_()); %#ok<NASGU>
            out = obj.Drive(obj.Observer, @(varargin) obj.flush_(varargin{:}));
        end

        function d = describe(obj)
            d = struct('model', struct('name', "ParityGuess learning", 'learnedChannel', obj.LearnedName), ...
                'layout', "ParityLearning", ...
                'references', {{"Planning/System/Game Theory/PG1_Parity_Guess_Contract.md"}}, ...
                'notation', struct('game', "one game of up to 200 turns", 'p2Reward', "player 2's return"));
        end
    end

    methods (Access = private)
        function flush_(obj, learned)
            games = obj.Observer.takeGames();
            for k = 1:numel(games)
                g = games(k);
                if isempty(obj.Recorder_) || g.Game > obj.Games, continue; end
                r = g.P2Reward; t = g.Turns; ge = g.P2GuessEven; pe = g.P1PutEven;
                state = struct('p2Reward', @() r, 'turns', @() t, 'p2GuessEven', @() ge, 'p1PutEven', @() pe);
                if obj.LearnedName ~= ""
                    v = reshape(double(learned), obj.LearnedShape);      % copied now, before the next game
                    if numel(obj.LearnedShape) == 2 && ~all(obj.LearnedShape > 1), v = v(:).'; end
                    state.(obj.LearnedName) = @() v;
                end
                obj.Recorder_.point(struct('game', g.Game), state);
            end
        end

        function detach_(obj)
            obj.Recorder_ = [];
        end
    end
end
