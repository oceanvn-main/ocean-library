classdef Notation
    % NOTATION  Textbook notation of a channel as LaTeX (for labels drawn with
    %   Interpreter 'latex'). Uses the channel's SymbolLatex when the recording
    %   carries one (newer recordings); otherwise translates the known ASCII
    %   symbols written by the Ocean recorders (Lai et al. Sec. 2.2; ESL 14.3.6);
    %   anything else is shown as upright text.
    %
    %   s = OceanViz.App.Replay.Notation.latex(info)        % '$\epsilon_\phi(x_i, i)$'

    methods (Static)
        function s = latex(info)
            if isprop(info, 'SymbolLatex') && strlength(info.SymbolLatex) > 0
                s = char(info.SymbolLatex); return;
            end
            s = OceanViz.App.Replay.Notation.ofSymbol(info.Symbol);
        end

        function s = ofSymbol(sym)
            sym = string(sym);
            persistent map
            if isempty(map)
                k = {"x_i", "eps_phi(x_i, i)", "x_phi(x_i, i)", "mu", "x_0", "t_i", "t_{i-1}", ...
                     "alphaBar_i", "alphaBar_{i-1}", "sqrt(1 - alphaBar_i^2)", "sigma^2(i)", "rms(eps_phi)", ...
                     "||x_i - (alphaBar x_phi + sigma eps_phi)||", "occlusion(eps_phi)", ...
                     "m_k", "C(i)", "sum_i ||x_i - m_C(i)||^2", "#changed", "R^2", "f(c; W_epoch)", "W", ...
                     "i", "m", "t", "k", "E"};
                v = {'$x_i$', '$\epsilon_\phi(x_i, i)$', '$x_\phi(x_i, i)$', '$\mu$', '$x_0$', '$t_i$', '$t_{i-1}$', ...
                     '$\bar{\alpha}_i$', '$\bar{\alpha}_{i-1}$', '$\sqrt{1-\bar{\alpha}_i^2}$', '$\sigma^2(i)$', ...
                     '$\mathrm{rms}(\epsilon_\phi)$', '$\|x_i - (\bar{\alpha}_i x_\phi + \sigma_i \epsilon_\phi)\|$', ...
                     '$\mathrm{occlusion}(\epsilon_\phi)$', '$m_k$', '$C(i)$', '$\sum_i \|x_i - m_{C(i)}\|^2$', ...
                     '$\#\,\mathrm{changed}$', '$R^2$', '$f(c;\,W_{\mathrm{epoch}})$', '$W$', ...
                     '$i$', '$m$', '$t$', '$k$', '$E$'};
                map = containers.Map(cellfun(@char, k, 'UniformOutput', false), v);
            end
            if isKey(map, char(sym)), s = map(char(sym)); return; end
            if strlength(sym) == 0, s = ''; return; end
            safe = regexprep(char(sym), '([_#%&{}$\\^~])', '\\$1');      % unknown: upright text
            s = ['$\mathrm{' strrep(safe, ' ', '\ ') '}$'];
        end
    end
end
