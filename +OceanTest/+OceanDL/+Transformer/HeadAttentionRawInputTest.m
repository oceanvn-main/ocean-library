classdef HeadAttentionRawInputTest < OceanTest.BaseTestCase
    % HEADATTENTIONRAWINPUTTEST  MultiHeadAttention.headAttention gives the same
    %   attention for raw arrays as for Tensors, and it equals an independent
    %   per-head softmax over KEYS (regression: a bare softmax(S,2) on doubles
    %   resolved to the Deep Learning Toolbox function, which ignores DIM).

    methods (Test)
        function testRawEqualsTensorAndReference(testCase)
            s = RandStream('mt19937ar', 'Seed', 3);
            dk = 4; Tq = 3; Tk = 5; h = 2; N = 2;
            Q = randn(s, dk, Tq, h, N); K = randn(s, dk, Tk, h, N); V = randn(s, dk, Tk, h, N);
            mask = true(N, Tk); mask(2, 4:5) = false;
            [Or, Ar] = OceanDL.Transformer.MultiHeadAttention.headAttention(Q, K, V, 'KeyMask', mask);
            T = @(x) OceanDL.Autograd.Tensor(x);
            [Ot, At] = OceanDL.Transformer.MultiHeadAttention.headAttention(T(Q), T(K), T(V), 'KeyMask', mask);
            raw = @(x) OceanDL.Autograd.Tensor.raw(x);
            testCase.verifyEqual(raw(Ar), raw(At), 'AbsTol', 1e-14);
            testCase.verifyEqual(raw(Or), raw(Ot), 'AbsTol', 1e-14);
            for n = 1:N
                for j = 1:h
                    S = Q(:, :, j, n)' * K(:, :, j, n) / sqrt(dk);
                    S(:, ~mask(n, :)) = -Inf;
                    E = exp(S - max(S, [], 2));
                    Aref = E ./ sum(E, 2);                       % rows (queries) sum to 1 over keys
                    testCase.verifyEqual(raw(Ar(:, :, j, n)), Aref, 'AbsTol', 1e-14);
                    testCase.verifyEqual(raw(Or(:, :, j, n)), V(:, :, j, n) * Aref', 'AbsTol', 1e-13);
                end
            end
        end
    end
end
