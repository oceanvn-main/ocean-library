function k = sampleInverseCdf_(p, u)
% SAMPLEINVERSECDF_  Inverse-CDF sampling shared by every OceanAgents policy: the smallest column k with
% u < cumsum(p)(k), for one uniform draw u in (0, 1). Columns with zero mass are never returned (their cumulative
% value equals the previous column's). If u lies above the total mass because of rounding in cumsum, the last
% column with positive mass is returned.
c = cumsum(p);
k = find(u < c, 1);
if isempty(k), k = find(p > 0, 1, 'last'); end
end
