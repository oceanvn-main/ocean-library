function v = expectedValue(q, probs, valid)
% EXPECTEDVALUE  sum_a probs(a) * q(a) over the valid actions (probs need not be masked;
% invalid entries are ignored, so an unset q(a) can never poison the sum with 0 * NaN/Inf).
v = sum(probs(valid) .* q(valid));
end
