classdef AddingProblem
    % ADDINGPROBLEM  The classic long-term-dependency benchmark
    %   (Hochreiter & Schmidhuber 1997; Le, Jaitly & Hinton 2015).
    %
    %   Each sequence has T steps and 2 input channels:
    %     channel 1  values v_t ~ U(0, 1)
    %     channel 2  a marker, 1 at exactly two steps -- one in the first half,
    %                one in the second half -- and 0 elsewhere
    %   Target: the SUM of the two marked values, y = v_a + v_b.
    %
    %   The network must remember v_a from early in the sequence across up to
    %   ~T steps of distractors. Reference errors (MSE):
    %     predict the constant 1        E[(v_a + v_b - 1)^2] = 2/12 = 0.1667
    %     "solved" (common threshold)   MSE < 0.01
    %
    %   Output is already in the repo's canonical layout: X (2 x N x T, "CBT"),
    %   y (1 x N) -- no import conversion needed.
    %
    %   [X, y, marks] = OceanData.Generators.Sequences.AddingProblem.generate(64, 100, RandStream('mt19937ar','Seed',1));

    properties (Constant)
        BaselineMSE = 1/6          % variance of v_a + v_b: the "no memory" error
    end

    methods (Static)
        function [X, y, marks] = generate(N, T, stream)
            arguments
                N (1,1) double {mustBePositive, mustBeInteger}
                T (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(T, 2)}
                stream = RandStream.getGlobalStream()
            end
            half = floor(T / 2);
            v = rand(stream, N, T);
            a = randi(stream, half, N, 1);                    % marker in the first half
            b = half + randi(stream, T - half, N, 1);         % marker in the second half
            M = zeros(N, T);
            M(sub2ind([N T], (1:N).', a)) = 1;
            M(sub2ind([N T], (1:N).', b)) = 1;
            X = permute(cat(3, v, M), [3 1 2]);               % (N, T, 2) -> 2 x N x T
            y = (v(sub2ind([N T], (1:N).', a)) + v(sub2ind([N T], (1:N).', b))).';
            marks = [a b];
        end
    end
end
