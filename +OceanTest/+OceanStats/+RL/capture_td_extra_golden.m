function capture_td_extra_golden()
% CAPTURE_TD_EXTRA_GOLDEN  HISTORICAL RECORD. Freezes V, Q, pi and the full history (including recorded episodes) of every
% td_extra_cases() configuration as produced by the learner BEFORE its decomposition (Phase 2 of the RL architecture refactor).
% Output td_extra_golden.mat is the frozen reference of TDRefactorGoldenTest; it must never be regenerated from the
% refactored code.
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
cases = OceanTest.OceanStats.RL.td_extra_cases();
G = struct('name', {}, 'V', {}, 'Q', {}, 'pi', {}, 'history', {});
for k = 1:size(cases, 1)
    [V, Q, pol, h] = cases{k, 2}().run();
    G(k).name = cases{k, 1};  G(k).V = V;  G(k).Q = Q;  G(k).pi = pol;  G(k).history = h;
end
save(fullfile(here, 'td_extra_golden.mat'), 'G');
fprintf('wrote td_extra_golden.mat (%d cases)\n', numel(G));
end
