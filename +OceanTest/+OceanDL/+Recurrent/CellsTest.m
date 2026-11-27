classdef CellsTest < OceanTest.BaseTestCase
    % CELLSTEST  The five recurrent cells:
    %   - Elman / LSTM / GRU reproduce PyTorch nn.RNN / nn.LSTM / nn.GRU
    %     exactly (external reference fixtures, imported through NpyReader +
    %     DimFormat -- the same path real imported data takes);
    %   - every cell's parameter gradients (BPTT through the autograd
    %     engine) match finite differences;
    %   - Leaky and Echo-state specific properties.

    methods (Test)
        function testElmanMatchesPyTorchRNN(testCase)
            [X, h0] = OceanTest.OceanDL.Recurrent.CellsTest.referenceInputs();
            cell = OceanDL.Recurrent.ElmanCell(4, 6);
            cell.Wx.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('rnn_w_ih'); cell.Wh.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('rnn_w_hh');
            cell.b.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('rnn_b');
            Y = OceanTest.OceanDL.Recurrent.CellsTest.unroll(cell, X, struct('h', h0));
            testCase.verifyEqual(Y, OceanTest.OceanDL.Recurrent.CellsTest.fxSeq('rnn_y'), 'AbsTol', 1e-12);
        end

        function testLstmMatchesPyTorchLSTM(testCase)
            [X, h0, c0] = OceanTest.OceanDL.Recurrent.CellsTest.referenceInputs();
            cell = OceanDL.Recurrent.LSTMCell(4, 6);
            cell.Wx.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('lstm_w_ih'); cell.Wh.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('lstm_w_hh');
            cell.b.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('lstm_b');
            [Y, state] = OceanTest.OceanDL.Recurrent.CellsTest.unroll(cell, X, struct('h', h0, 'c', c0));
            testCase.verifyEqual(Y, OceanTest.OceanDL.Recurrent.CellsTest.fxSeq('lstm_y'), 'AbsTol', 1e-12);
            cn = OceanDL.Layout.DimFormat.toCanonical(OceanTest.OceanDL.Recurrent.CellsTest.fxRaw('lstm_cn'), "BC");
            testCase.verifyEqual(OceanDL.Autograd.Tensor.raw(state.c), cn, 'AbsTol', 1e-12);
        end

        function testGruMatchesPyTorchGRU(testCase)
            [X, h0] = OceanTest.OceanDL.Recurrent.CellsTest.referenceInputs();
            cell = OceanDL.Recurrent.GRUCell(4, 6, 'Variant', "pytorch");
            Whh = OceanTest.OceanDL.Recurrent.CellsTest.fx('gru_w_hh'); bhh = OceanTest.OceanDL.Recurrent.CellsTest.fx('gru_b_hh');
            cell.Wx.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('gru_w_ih'); cell.bx.Data = OceanTest.OceanDL.Recurrent.CellsTest.fx('gru_b_ih');
            cell.Whrz.Data = Whh(1:12, :); cell.bhrz.Data = bhh(1:12);       % weight_hh = [W_hr; W_hz; W_hn]
            cell.Whn.Data = Whh(13:18, :); cell.bhn.Data = bhh(13:18);
            Y = OceanTest.OceanDL.Recurrent.CellsTest.unroll(cell, X, struct('h', h0));
            testCase.verifyEqual(Y, OceanTest.OceanDL.Recurrent.CellsTest.fxSeq('gru_y'), 'AbsTol', 1e-12);
        end

        function testEveryCellBpttMatchesFiniteDifferences(testCase)
            import OceanDL.Recurrent.*
            makers = {@() ElmanCell(3, 4, 'Seed', 1), @() ElmanCell(3, 4, 'Nonlinearity', "relu", 'Seed', 2), ...
                      @() LSTMCell(3, 4, 'Seed', 3), @() GRUCell(3, 4, 'Seed', 4), ...
                      @() GRUCell(3, 4, 'Variant', "original", 'Seed', 5), ...
                      @() LeakyCell(3, 4, 'LearnTimeConstants', true, 'Seed', 6), @() LeakyCell(3, 4, 'Seed', 7)};
            X = randn(RandStream('mt19937ar', 'Seed', 8), 3, 2, 7);
            for k = 1:numel(makers)
                cell = makers{k}();
                [maxErr, nChecked] = OceanTest.OceanDL.Recurrent.CellsTest.paramGradError(cell, X);
                testCase.verifyGreaterThan(nChecked, 0);
                testCase.verifyLessThan(maxErr, 1e-6, sprintf('%s: BPTT vs finite differences', class(cell)));
            end
        end

        function testLeakyWithAlphaOneIsElman(testCase)
            e = OceanDL.Recurrent.ElmanCell(3, 5, 'Seed', 1);
            l = OceanDL.Recurrent.LeakyCell(3, 5, 'Alpha', 1, 'Seed', 2);
            l.Wx.Data = e.Wx.Data; l.Wh.Data = e.Wh.Data; l.b.Data = e.b.Data;
            X = randn(3, 2, 6);
            testCase.verifyEqual(OceanTest.OceanDL.Recurrent.CellsTest.unroll(l, X), OceanTest.OceanDL.Recurrent.CellsTest.unroll(e, X), 'AbsTol', 1e-14);
        end

        function testLeakyTimeScalesAreLogUniform(testCase)
            l = OceanDL.Recurrent.LeakyCell(2, 5, 'AlphaMin', 0.01);
            testCase.verifyEqual(l.alpha(), [0.01; 0.01^0.75; 0.1; 0.01^0.25; 1], 'RelTol', 1e-12);
        end

        function testEchoStateReservoirIsFixedAndScaled(testCase)
            esn = OceanDL.Recurrent.EchoStateCell(3, 50, 'SpectralRadius', 0.8, 'Seed', 1);
            testCase.verifyEmpty(esn.parameters());
            testCase.verifyEqual(OceanDL.Recurrent.Diagnostics.spectralRadius(esn.W), 0.8, 'RelTol', 1e-10);
        end

        function testEchoStatePropertyForgetsInitialState(testCase)
            % Same inputs, two different initial states: with rho(W) < 1 the
            % trajectories converge -- the reservoir "echoes" the input only.
            esn = OceanDL.Recurrent.EchoStateCell(1, 60, 'SpectralRadius', 0.8, 'Seed', 2);
            s = RandStream('mt19937ar', 'Seed', 3);
            x = randn(s, 1, 1, 300);
            a = struct('h', randn(s, 60, 1)); b = struct('h', randn(s, 60, 1));
            d0 = norm(a.h - b.h);
            for t = 1:300
                [~, a] = esn.step(x(:, :, t), a);
                [~, b] = esn.step(x(:, :, t), b);
            end
            testCase.verifyLessThan(norm(a.h - b.h), 1e-6 * d0);
        end

        function testRidgeReadoutRecoversLinearMap(testCase)
            s = RandStream('mt19937ar', 'Seed', 4);
            Hs = randn(s, 10, 500); V = randn(s, 2, 10); c = [0.5; -1];
            [Vh, ch] = OceanDL.Recurrent.EchoStateCell.fitRidgeReadout(Hs, V * Hs + c, 0);
            testCase.verifyEqual(Vh, V, 'AbsTol', 1e-10);
            testCase.verifyEqual(ch, c, 'AbsTol', 1e-10);
        end
    end

    methods (Static)
        function [Y, state] = unroll(cell, X, state)
            if nargin < 3
                state = cell.initialState(size(X, 2), X);
            end
            layer = OceanDL.Recurrent.RecurrentLayer(cell);
            [Yt, state] = layer.forward(X, 'InitialState', state);
            Y = OceanDL.Autograd.Tensor.raw(Yt);
        end

        function [maxErr, n] = paramGradError(cell, X)
            % Autograd BPTT gradient of L = sum(Y .* R) vs central differences
            % on every parameter element (graph recording off for the latter).
            R = randn(RandStream('mt19937ar', 'Seed', 99), cell.HiddenSize, size(X, 2), size(X, 3));
            layer = OceanDL.Recurrent.RecurrentLayer(cell);
            lossOf = @() sum(OceanDL.Autograd.Tensor.raw(layer.forward(X)) .* R, 'all');
            cell.zeroGrad();
            L = sum(layer.forward(X) .* R, 'all');
            L.backward();
            guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            P = cell.parameters();
            maxErr = 0; n = 0; h = 1e-6;
            for k = 1:numel(P)
                p = P{k};
                for i = 1:numel(p.Data)
                    orig = p.Data(i);
                    p.Data(i) = orig + h; Lp = lossOf();
                    p.Data(i) = orig - h; Lm = lossOf();
                    p.Data(i) = orig;
                    maxErr = max(maxErr, abs((Lp - Lm) / (2*h) - p.Grad(i)));
                    n = n + 1;
                end
            end
        end

        function [X, h0, c0] = referenceInputs()
            import OceanDL.Layout.DimFormat
            X = DimFormat.toCanonical(OceanTest.OceanDL.Recurrent.CellsTest.fxRaw('x_TND'), "TBC");        % (T,N,D) -> D x N x T
            h0 = DimFormat.toCanonical(OceanTest.OceanDL.Recurrent.CellsTest.fxRaw('h0'), "BC");           % (N,H)   -> H x N
            c0 = DimFormat.toCanonical(OceanTest.OceanDL.Recurrent.CellsTest.fxRaw('c0'), "BC");
        end

        function A = fxRaw(name)
            A = OceanData.Adapters.NpyReader.read(fullfile(fileparts(mfilename('fullpath')), 'fixtures', [name '.npy']));
        end

        function A = fx(name)
            % Weights / biases: PyTorch (out x in) already matches the repo's W (out x in).
            A = OceanTest.OceanDL.Recurrent.CellsTest.fxRaw(name);
            if isvector(A)
                A = A(:);
            end
        end

        function Y = fxSeq(name)
            Y = OceanDL.Layout.DimFormat.toCanonical(OceanTest.OceanDL.Recurrent.CellsTest.fxRaw(name), "TBC");
        end
    end
end
