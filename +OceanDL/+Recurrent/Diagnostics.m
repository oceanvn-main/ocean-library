classdef Diagnostics
    % DIAGNOSTICS  Make the long-term-dependency problem of Goodfellow
    %   Sec. 10.7 visible on a real network.
    %
    %   gradientNormsOverTime(layer)  -> g(t) = mean over the batch of ||dL/dh_t||
    %     For a loss on the LAST step, g(t) measures how much of the training
    %     signal reaches step t. The chain rule gives
    %       dL/dh_t = (dh_{t+1}/dh_t)' ... (dh_T/dh_{T-1})' dL/dh_T,
    %       dh_{k}/dh_{k-1} = diag(1 - h_k.^2) W   (Elman, tanh)
    %     so g(t) shrinks or grows roughly like (gamma * rho(W))^(T-t):
    %     exponential decay = vanishing gradients, growth = exploding.
    %     (Requires RecurrentLayer TrackStates = true, then backward().)
    %
    %   spectralRadius(W) = max |eig(W)| of a recurrent weight matrix.

    methods (Static)
        function g = gradientNormsOverTime(layer)
            arguments
                layer OceanDL.Recurrent.RecurrentLayer
            end
            S = layer.StepStates;
            if isempty(S) || all(cellfun(@isempty, S))
                error('OceanDL:Recurrent:Diagnostics:NoStates', ...
                    'Construct the layer with ''TrackStates'', true and run forward() + backward() first.');
            end
            g = nan(1, numel(S));
            for t = 1:numel(S)
                if ~isempty(S{t}) && ~isempty(S{t}.Grad)
                    g(t) = mean(vecnorm(double(S{t}.Grad), 2, 1));
                end
            end
        end

        function rho = spectralRadius(W)
            W = OceanDL.Autograd.Tensor.raw(W);
            rho = max(abs(eig(double(W))));
        end
    end
end
