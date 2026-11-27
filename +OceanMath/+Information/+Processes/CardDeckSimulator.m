classdef CardDeckSimulator < handle
    % CARDDECKSIMULATOR Simulates stateful deck draws sequentially without replacement.
    %
    % This class maintains a card deck with Red and Black cards, shuffling and
    % drawing one card at a time, tracking conditional probabilities prior to each draw.
    %
    % Colors:
    %   1 - Red
    %   2 - Black
    %
    % Properties:
    %   NumRedR        - Total number of Red cards (R)
    %   NumBlackB      - Total number of Black cards (B)
    %   DeckSequence   - 1xN vector of shuffled card colors (1 for Red, 2 for Black)
    %   CurrentIndex   - The current draw index (0 to N)
    %   RemainingRed   - Remaining Red cards (r)
    %   RemainingBlack - Remaining Black cards (b)
    %
    % Examples:
    %   sim = OceanMath.Information.Processes.CardDeckSimulator(26, 26);
    %   [card, pRed, pBlack] = sim.drawNext();
    %
    % See also: OceanMath.Information.Processes.CardDeckBettingOptimizer

    properties
        NumRedR (1,1) double {mustBeNonnegative, mustBeInteger}
        NumBlackB (1,1) double {mustBeNonnegative, mustBeInteger}
        DeckSequence (1,:) double = double.empty(1, 0)
        CurrentIndex (1,1) double {mustBeNonnegative, mustBeInteger} = 0
        RemainingRed (1,1) double {mustBeNonnegative, mustBeInteger} = 0
        RemainingBlack (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    end

    methods
        function obj = CardDeckSimulator(R, B)
            % CARDDECKSIMULATOR Constructor.
            %
            % Initializes a card deck with R red and B black cards, shuffles it,
            % and sets draw pointers. Handles edge cases where R=0 or B=0 safely.
            arguments
                R (1,1) double {mustBeNonnegative, mustBeInteger}
                B (1,1) double {mustBeNonnegative, mustBeInteger}
            end

            obj.NumRedR = R;
            obj.NumBlackB = B;
            obj.DeckSequence = [ones(1, R), 2 * ones(1, B)];
            obj.shuffle();
        end

        function shuffle(obj)
            % SHUFFLE Shuffles the deck sequence randomly and resets the draw state.
            arguments
                obj (1,1) OceanMath.Information.Processes.CardDeckSimulator
            end

            N = obj.NumRedR + obj.NumBlackB;
            if N > 0
                obj.DeckSequence = obj.DeckSequence(randperm(N));
            else
                obj.DeckSequence = double.empty(1, 0);
            end
            obj.CurrentIndex = 0;
            obj.RemainingRed = obj.NumRedR;
            obj.RemainingBlack = obj.NumBlackB;
        end

        function [card, pRed, pBlack] = drawNext(obj)
            % DRAWNEXT Draws the next card from the deck without replacement.
            %
            % Returns:
            %   card   - 1 for Red, 2 for Black
            %   pRed   - Conditional probability of drawing Red *prior* to the draw: r / (r + b)
            %   pBlack - Conditional probability of drawing Black *prior* to the draw: b / (r + b)
            %
            % Decrements the respective remaining card count.
            % Throws an error if no cards remain.
            arguments
                obj (1,1) OceanMath.Information.Processes.CardDeckSimulator
            end

            r = obj.RemainingRed;
            b = obj.RemainingBlack;
            total = r + b;

            if total <= 0
                error('OceanMath:Information:CardDeckSimulator:NoCardsRemaining', ...
                    'Cannot draw next card. The deck has no remaining cards.');
            end

            pRed = r / total;
            pBlack = b / total;

            obj.CurrentIndex = obj.CurrentIndex + 1;
            card = obj.DeckSequence(obj.CurrentIndex);

            if card == 1
                obj.RemainingRed = obj.RemainingRed - 1;
            elseif card == 2
                obj.RemainingBlack = obj.RemainingBlack - 1;
            else
                error('OceanMath:Information:CardDeckSimulator:InvalidCardType', ...
                    'Invalid card color encountered: %g. Must be 1 or 2.', card);
            end
        end

        function seq = getRemainingSequence(obj)
            % GETREMAININGSEQUENCE Returns a vector of the remaining card colors.
            arguments
                obj (1,1) OceanMath.Information.Processes.CardDeckSimulator
            end
            seq = obj.DeckSequence(obj.CurrentIndex + 1 : end);
        end
    end
end
