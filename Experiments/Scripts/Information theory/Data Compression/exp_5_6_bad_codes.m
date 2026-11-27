%% exp_5_6_bad_codes.m
% Problem 5.6 — Bad Codes (Cover & Thomas, Elements of Information Theory)
%
% Question: Which of the following cannot be Huffman codes for ANY
% probability assignment?
%   (a) {0, 10, 11}
%   (b) {00, 01, 10, 110}
%   (c) {01, 10}
%
% A valid Huffman code must satisfy ALL three conditions:
%   1. Prefix-free (no codeword is a prefix of another)
%   2. Complete: Kraft sum = sum(2^(-l_k)) == 1
%   3. Sibling property: deepest codewords come in sibling pairs
%      (same prefix, differ only in last bit)
%
% Usage:
%   >> run('Experiments/Scripts/exp_5_6_bad_codes.m')

clc;
fprintf('==========================================================\n');
fprintf('  Problem 5.6 — Bad Codes\n');
fprintf('==========================================================\n');
fprintf('  Which of these CANNOT be Huffman codes for any p?\n\n');

chk = @OceanMath.Information.Coding.isValidHuffmanCode;

% -------------------------------------------------------------------------
%  (a)  {0, 10, 11}
% -------------------------------------------------------------------------
fprintf('(a)  {0, 10, 11}\n');
chk({'0','10','11'}, 'Verbose', true);

% -------------------------------------------------------------------------
%  (b)  {00, 01, 10, 110}
% -------------------------------------------------------------------------
fprintf('(b)  {00, 01, 10, 110}\n');
chk({'00','01','10','110'}, 'Verbose', true);

% -------------------------------------------------------------------------
%  (c)  {01, 10}
% -------------------------------------------------------------------------
fprintf('(c)  {01, 10}\n');
chk({'01','10'}, 'Verbose', true);

% -------------------------------------------------------------------------
%  Summary table
% -------------------------------------------------------------------------
codes = { {'0','10','11'}, {'00','01','10','110'}, {'01','10'} };
labels = {'(a)', '(b)', '(c)'};

fprintf('==========================================================\n');
fprintf('  Summary\n');
fprintf('==========================================================\n');
fprintf('  %-4s  %-22s  %-12s  %-8s  %-14s  %s\n', ...
    'Set', 'Codewords', 'Kraft sum', 'Complete', 'Sibling Prop', 'Verdict');
fprintf('  %s\n', repmat('-',1,78));

for i = 1:3
    r = chk(codes{i});
    cwStr = strjoin(cellfun(@(c) sprintf('"%s"',c), codes{i}, 'UniformOutput',false), ' ');
    verdict = ternary_(r.isValid, 'CAN be', 'CANNOT be');
    fprintf('  %-4s  %-22s  %-12.4f  %-8s  %-14s  %s\n', ...
        labels{i}, cwStr, r.kraftSum, ...
        yn_(r.isComplete), yn_(r.hasSiblingProp), verdict);
end
fprintf('  %s\n\n', repmat('-',1,78));

function v = ternary_(c, a, b)
    if c, v = a; else, v = b; end
end
function s = yn_(tf)
    if tf, s = 'Yes'; else, s = 'No'; end
end
