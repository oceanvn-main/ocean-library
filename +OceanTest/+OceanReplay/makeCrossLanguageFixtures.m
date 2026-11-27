function makeCrossLanguageFixtures(folder)
% MAKECROSSLANGUAGEFIXTURES  Write the MATLAB side of the MATLAB <-> Python
%   parity fixtures (Master plan Sec. 5.7). The Python tests
%   (Python/ocean_replay/tests/test_cross_language.py) read these files.
%
%   OceanTest.OceanReplay.makeCrossLanguageFixtures()          % default: .../fixtures
%
%   Writes
%     toy_matlab.h5                 ToyRecordable, fullSpec, float64 (exact comparison)
%     toy_matlab_expected.json      every stored value: coords + shape + column-major data
%     toy_matlab_extract.h5         OceanReplay.IO.extract(toy_matlab.h5, extract_spec.json)
%     extract_spec.json             the extract spec (Python extracts with the same file)
%     sampling_cases.json           rules + axis values + the masks MATLAB selects

if nargin < 1, folder = fullfile(fileparts(mfilename('fullpath')), 'fixtures'); end
S = OceanReplay.Core.Sampling;

% --- toy recording (float64 so values compare exactly) ---
spec = OceanTest.OceanReplay.ToyRecordable.fullSpec(); spec.Precision = "float64";
toyFile = fullfile(folder, 'toy_matlab.h5');
OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), spec, toyFile, 'Overwrite', true, ...
    'RunId', "toy-matlab", 'Title', "cross-language fixture (MATLAB)");
r = OceanReplay.Core.Recording(toyFile);
expected = struct();
for n = r.names()
    info = r.info(n); s = r.stored(n);
    pts = cell(1, s.P);
    for p = 1:s.P
        c = struct();
        for a = 1:numel(info.Axes), c.(info.Axes(a)) = s.Values(p, a); end
        v = r.get(n, c, "exact");
        if info.Kind == "tokens"
            pts{p} = struct('coords', c, 'tokens', {cellstr(v)});
        else
            pts{p} = struct('coords', c, 'shape', info.ValueShape, 'data', double(v(:)).');
        end
    end
    expected.(n) = pts;
end
writeText(fullfile(folder, 'toy_matlab_expected.json'), jsonencode(struct('channels', expected, ...
    'axes', struct('t', r.axis("t").Values, 'k', r.axis("k").Values))));

% --- extract (same spec file for both languages) ---
ex = OceanReplay.Core.Spec() ...
    .addGroup("small", 1, ["img" "energy"], struct('t', S.union(S.list([4 6]), S.every(8)))) ...
    .addGroup("tok", 1, "tok", struct('k', S.list(2))) ...
    .addView("noisy", "", struct(), struct('t', S.list(2)));
ex.toFile(fullfile(folder, 'extract_spec.json'));
OceanReplay.IO.extract(toyFile, fullfile(folder, 'extract_spec.json'), fullfile(folder, 'toy_matlab_extract.h5'), ...
    'Overwrite', true);

% --- sampling parity cases ---
axes = {1000:-1:0, 0:30, [10 5 2 1 0.5 0.2 0.1], 10:-2:0};
rules = {S.every(20), S.every(5, 1, 21), S.everyIndex(3), S.window(51, 1), S.list([52 50]), ...
         S.union(S.every(250), S.window(51, 1)), S.window(0.3, 0.2), S.union(S.every(4, 0, 8), S.list(6)), S.all()};
cases = {};
for i = 1:numel(rules)
    for j = 1:numel(axes)
        cases{end+1} = struct('rule', rules{i}.toStruct(), 'axis', axes{j}, 'mask', rules{i}.select(axes{j})); %#ok<AGROW>
    end
end
writeText(fullfile(folder, 'sampling_cases.json'), jsonencode(cases));
fprintf('Cross-language fixtures written to %s\n', folder);
end

function writeText(file, txt)
fid = fopen(file, 'w', 'n', 'UTF-8'); c = onCleanup(@() fclose(fid));
fwrite(fid, unicode2native(char(txt), 'UTF-8'));
end
