classdef ParityGameObserver < OceanAgents.Observe.IPlayerObserver
    % PARITYGAMEOBSERVER  Counts what both parity-guess players do and keeps one row per finished game. Attach the SAME
    %   observer to both players. It holds plain data only (no callback, no player), so the isolation check passes:
    %   whoever drives the run takes the finished games with takeGames().
    %
    %   g = obs.takeGames();   struct array (Game, P2Reward, Turns, P2GuessEven, P1PutEven); the pending list is cleared
    % Reads value records only (decision, feedback); it never touches a player.

    properties (SetAccess = private)
        Factors
        Pending = struct('Game', {}, 'P2Reward', {}, 'Turns', {}, 'P2GuessEven', {}, 'P1PutEven', {})
    end

    properties (Access = private)
        P1_ (1,1) string
        P2_ (1,1) string
        Turns_ (1,1) double = 0
        GuessEven_ (1,1) double = 0
        PutEven_ (1,1) double = 0
        Reward_ (1,1) double = 0
        Episode_ (1,1) double = 0
    end

    methods
        function obj = ParityGameObserver(factors, opts)
            arguments
                factors
                opts.Player1 (1,1) string = "p1"
                opts.Player2 (1,1) string = "p2"
            end
            obj.Factors = factors; obj.P1_ = opts.Player1; obj.P2_ = opts.Player2;
        end

        function onDecision(obj, playerID, ~, decision)
            parts = obj.Factors.parts(decision.Action);        % [put guess]
            if playerID == obj.P1_
                obj.PutEven_ = obj.PutEven_ + (parts(1) == 1);
            elseif playerID == obj.P2_
                obj.Turns_ = obj.Turns_ + 1;
                obj.GuessEven_ = obj.GuessEven_ + (parts(2) == 1);
                obj.Episode_ = decision.EpisodeID;
            end
        end

        function onFeedback(obj, playerID, feedback)
            if playerID ~= obj.P2_, return; end
            obj.Reward_ = obj.Reward_ + feedback.Reward;
            if feedback.ended()
                obj.Pending(end + 1) = struct('Game', obj.Episode_, 'P2Reward', obj.Reward_, 'Turns', obj.Turns_, ...
                    'P2GuessEven', obj.GuessEven_, 'P1PutEven', obj.PutEven_);
                obj.Turns_ = 0; obj.GuessEven_ = 0; obj.PutEven_ = 0; obj.Reward_ = 0;
            end
        end

        function g = takeGames(obj)
            g = obj.Pending;
            obj.Pending = obj.Pending([]);
        end
    end
end
