function v = greedyValue(q, valid)
% GREEDYVALUE  max_a q(a) over the valid actions (logical mask, same size as q).
v = max(q(valid));
end
