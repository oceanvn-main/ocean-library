classdef BlackjackEnv < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IEnumerableEnvironment
    % BLACKJACKENV  Blackjack against a fixed dealer policy, infinite deck (Sutton & Barto 2018, Example 5.1).
    %
    % State (observation): [player_sum, dealer_showing, usable_ace]
    %   player_sum 12..21 (below 12 the player always hits, so those hands are never observed), dealer_showing 1..10
    %   (Ace = 1), usable_ace 0/1. A bust is reported as the terminal observation with player_sum > 21.
    % Actions: 0 = stick, 1 = hit (ActionSpace = Discrete(2, 'Start', 0)).
    % Rewards: +1 win, 0 draw, -1 loss/bust, given on the last step. The dealer sticks on 17 or more.
    % Sample-only: the environment does not expose its dynamics (no IModelEnvironment), but its 200 tracked states can be
    % listed (IEnumerableEnvironment); state_to_index returns 0 for a bust (a terminal outside the table).
    %
    % Randomness: after setSeed / reset('Seed') the environment owns an independent stream; until then it draws from
    % MATLAB's global stream, so rng(k) still makes an unseeded run reproducible.

    properties (Constant)
        PLAYER_SUM_MIN = 12    % Minimum tracked player sum
        PLAYER_SUM_MAX = 21    % Maximum valid player sum (bust if > 21)
        DEALER_MIN     = 1     % Min dealer card shown (Ace)
        DEALER_MAX     = 10    % Max dealer card shown (faces count 10)
        N_STATES       = 200   % 10 x 10 x 2
        N_ACTIONS      = 2     % 0 = stick, 1 = hit
    end

    properties (SetAccess = private)
        StateIndexLUT  double   % (21 x 10 x 2) lookup: (player_sum, dealer_card, usable_ace+1) -> index; NaN out of range
        AllStatesCache double   % (200 x 3) pre-built state matrix
        PlayerHand     double   % player card values (1..10)
        DealerHand     double   % dealer card values (1..10)
        EpisodeDone    logical  % true if the episode has ended
    end
    properties (Access = private)
        Stream = []             % empty: use the global stream
    end

    methods
        function obj = BlackjackEnv()
            obj.build_state_lut_();
            obj.EpisodeDone = true;                       % reset() is required before the first step
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Box([4 1 0], [31 10 1]);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(2, 'Start', 0);
        end

        function setSeed(obj, seed)
            obj.Stream = RandStream('mt19937ar', 'Seed', seed);
        end

        function set_state(obj, state)
            % SET_STATE  Force the hands to match [player_sum, dealer_showing, usable_ace].
            ps = state(1);  dc = state(2);  ua = state(3);
            obj.DealerHand = [dc, obj.draw_card_()];      % showing card plus a hidden second card
            if ua == 1
                obj.PlayerHand = [1, ps - 11];
            else
                obj.PlayerHand = [10, ps - 10];
            end
            obj.EpisodeDone = false;
        end

        function [next_state, reward, terminated, truncated, info] = step(obj, action)
            if obj.EpisodeDone
                error('BlackjackEnv:EpisodeEnded', 'Episode is over. Call reset() to start a new episode.');
            end
            if ~obj.ActionSpace.contains(action)
                error('BlackjackEnv:InvalidAction', 'Action must be 0 (stick) or 1 (hit).');
            end
            truncated = false;
            info = struct('action_mask', [false false]);        % every step of this environment ends the episode or is re-masked below
            if action == 1                                % ---- hit
                obj.PlayerHand(end+1) = obj.draw_card_();
                [ps, ua] = obj.hand_sum_(obj.PlayerHand);
                if ps > obj.PLAYER_SUM_MAX                % bust
                    reward = -1;
                    terminated = true;
                    obj.EpisodeDone = true;
                    next_state = [ps, obj.DealerHand(1), double(ua)];
                    return;
                end
                next_state = obj.current_state_();
                reward = 0;
                terminated = false;
                info.action_mask = [true true];
            else                                          % ---- stick: the dealer plays out its fixed policy
                [dv, ~] = obj.hand_sum_(obj.DealerHand);
                while dv < 17
                    obj.DealerHand(end+1) = obj.draw_card_();
                    [dv, ~] = obj.hand_sum_(obj.DealerHand);
                end
                [pv, ~] = obj.hand_sum_(obj.PlayerHand);
                if dv > obj.PLAYER_SUM_MAX || pv > dv
                    reward = 1;
                elseif pv == dv
                    reward = 0;
                else
                    reward = -1;
                end
                terminated = true;
                obj.EpisodeDone = true;
                next_state = obj.current_state_();
            end
        end

        % ---- IEnumerableEnvironment
        function actions = get_valid_actions(~, ~), actions = [0, 1]; end
        function all_states = get_all_states(obj), all_states = obj.AllStatesCache; end
        function idx = state_to_index(obj, state)
            ps = state(1);  dc = state(2);  ua = state(3);
            if ps < obj.PLAYER_SUM_MIN || ps > obj.PLAYER_SUM_MAX || dc < obj.DEALER_MIN || dc > obj.DEALER_MAX
                idx = 0;                                  % bust / out of the tracked range
                return;
            end
            idx = obj.StateIndexLUT(ps, dc, ua + 1);
        end
        function action_idx = action_to_index(~, action), action_idx = action + 1; end
        function action = index_to_action(~, action_idx), action = action_idx - 1; end
        function n = get_num_actions(~), n = 2; end
        function tf = is_terminal(obj, state), tf = state(1) > obj.PLAYER_SUM_MAX; end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.PlayerHand = [obj.draw_card_(), obj.draw_card_()];
            obj.DealerHand = [obj.draw_card_(), obj.draw_card_()];
            [ps, ~] = obj.hand_sum_(obj.PlayerHand);
            while ps < obj.PLAYER_SUM_MIN                 % always correct to hit below the tracked threshold
                obj.PlayerHand(end+1) = obj.draw_card_();
                [ps, ~] = obj.hand_sum_(obj.PlayerHand);
            end
            obj.EpisodeDone = false;
            obs = obj.current_state_();
            info = struct('action_mask', [true true]);
        end
    end

    methods (Access = private)
        function card = draw_card_(obj)
            % one card from an infinite deck: face 1..13 uniform, clamped to 10 (Ace stays 1)
            if isempty(obj.Stream), face = randi(13); else, face = randi(obj.Stream, 13); end
            card = min(face, 10);
        end

        function [total, usable_ace] = hand_sum_(~, hand)
            % best hand total; an ace is "usable" when counting it as 11 keeps the total <= 21
            total = sum(hand);
            usable_ace = false;
            if any(hand == 1) && (total + 10) <= 21
                total = total + 10;
                usable_ace = true;
            end
        end

        function state = current_state_(obj)
            [ps, ua] = obj.hand_sum_(obj.PlayerHand);
            state = [ps, obj.DealerHand(1), double(ua)];
        end

        function build_state_lut_(obj)
            % row order of AllStatesCache: usable_ace outer, dealer_showing middle, player_sum inner
            lut = nan(21, 10, 2);
            all_s = zeros(obj.N_STATES, 3);
            idx = 0;
            for ua = 0:1
                for dc = obj.DEALER_MIN:obj.DEALER_MAX
                    for ps = obj.PLAYER_SUM_MIN:obj.PLAYER_SUM_MAX
                        idx = idx + 1;
                        lut(ps, dc, ua + 1) = idx;
                        all_s(idx, :) = [ps, dc, ua];
                    end
                end
            end
            obj.StateIndexLUT = lut;
            obj.AllStatesCache = all_s;
        end
    end
end
