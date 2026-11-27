classdef AttentionRollout
    % ATTENTIONROLLOUT  Where does the [CLS] token (the classifier's input)
    %   gather its information from? (Abnar & Zuidema 2020)
    %
    %   One layer's attention A_l (averaged over heads) says how each token
    %   mixes the tokens of the layer below; the residual connection also
    %   passes every token through unchanged, so the effective mixing is
    %     Ahat_l = (A_l + I) / 2   (rows re-normalised to sum to 1).
    %   Chaining layers multiplies these maps:
    %     Rollout = Ahat_L * ... * Ahat_2 * Ahat_1
    %   Row [CLS] of the rollout, over the patch tokens, reshaped to the
    %   patch grid, is a heat map of which image regions reach the decision.
    %
    %   maps = AttentionRollout.compute(vit, X)      % Hp x Wp x N heat maps (rows sum over patches = 1 share)
    %   Needs a ViT with Pooling = "cls".

    methods (Static)
        function [maps, rollout] = compute(vit, X)
            arguments
                vit OceanDL.Vision.ViT
                X
            end
            if vit.Pooling ~= "cls"
                error('OceanDL:Vision:AttentionRollout:NeedsCls', 'Attention rollout reads the [CLS] row; use Pooling = "cls".');
            end
            guard = OceanDL.Autograd.GradMode.disable(); %#ok<NASGU>
            enc = vit.Encoder;
            enc.setStoreAttention(true);
            cleaner = onCleanup(@() enc.setStoreAttention(false));
            [~, info] = vit.forward(X);
            T = size(info.Tokens, 3);
            N = size(info.Tokens, 2);
            rollout = repmat(eye(T), 1, 1, N);                         % T x T x N
            for l = 1:enc.Depth
                A = mean(enc.Layers{l}.Attention.LastAttention, 3);   % Tq x Tk x 1 x N (head average)
                A = reshape(A, T, T, N);
                Ahat = 0.5 * A + 0.5 * eye(T);
                Ahat = Ahat ./ sum(Ahat, 2);
                rollout = pagemtimes(Ahat, rollout);                   % later layers on the left
            end
            cls = reshape(rollout(1, 2:end, :), T - 1, N);             % [CLS] row over patches
            cls = cls ./ sum(cls, 1);
            maps = reshape(cls, info.Grid(1), info.Grid(2), N);        % column-major token order
        end
    end
end
