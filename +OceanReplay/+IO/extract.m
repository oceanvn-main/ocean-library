function report = extract(archive, spec, outFile, opts)
% EXTRACT  Copy a subset of a recording into a smaller file of the SAME format
%   (Format spec Sec. 5.2). Nothing is recomputed: channels and views are
%   copied at the points the extract spec selects. The MATLAB twin of
%   Python's `python -m ocean_replay.extract`.
%
%   report = OceanReplay.IO.extract(archive, spec, outFile)
%     archive  full recording (.h5)
%     spec     OceanReplay.Core.Spec or a JSON file path; groups name channels +
%              "at" rules; views name recorded views (method ignored)
%     report   struct: Copied (name -> points), NotStored (requested "list"
%              values that the archive does not have, as "name@axis=value")
%
%   Example:
%     S = OceanReplay.Core.Sampling;
%     spec = OceanReplay.Core.Spec().addGroup("small", 1, ["x" "eps"], ...
%            struct('step', S.union(S.every(100), S.list([50 51 52]))));
%     OceanReplay.IO.extract("archive.h5", spec, "extract.h5");

arguments
    archive (1,1) string {mustBeFile}
    spec
    outFile (1,1) string
    opts.Overwrite (1,1) logical = false
end
if ~isa(spec, 'OceanReplay.Core.Spec'), spec = OceanReplay.Core.Spec.fromFile(string(spec)); end

src = OceanReplay.Core.Recording(archive);
for j = 1:numel(spec.Views), spec.Views(j).Method = spec.Views(j).Name; end   % extract refers to recorded views by name
spec.validate(src.Channels, src.Views, src.Axes);          % names must exist in the archive

meta = src.Meta;
runId = ""; if isfield(meta, 'runId'), runId = string(meta.runId); end
meta.extractedFrom = struct('archive', archive, 'archiveRunId', runId, 'extractSpec', spec.toStruct());

prec = "float32"; if isfield(meta, 'spec') && isfield(meta.spec, 'precision'), prec = string(meta.spec.precision); end
w = OceanReplay.IO.Hdf5Writer(outFile, 'Overwrite', opts.Overwrite, 'Precision', prec);
w.begin(meta, src.Axes);                                    % full axes: indices stay valid

report = struct('Copied', struct(), 'NotStored', strings(0, 1));
for n = spec.channelNames()
    [report, ~] = copyOne(src, w, n, "channels", @(v, i) spec.wants(n, v, i), report);
    report.NotStored = [report.NotStored; notStored(src, spec, n)];
end
for j = 1:numel(spec.Views)
    n = spec.Views(j).Name;
    report = copyOne(src, w, n, "views", @(v, i) spec.viewWants(j, v, i), report);
end
w.close();
end

function [report, k] = copyOne(src, w, name, where, wantsFcn, report)
info = src.info(name);
w.define(info, where);
s = src.stored(name);
k = 0;
for p = 1:s.P
    v = struct(); ix = struct();
    for a = 1:numel(info.Axes)
        v.(info.Axes(a)) = s.Values(p, a); ix.(info.Axes(a)) = s.Index(p, a);
    end
    if ~wantsFcn(v, ix), continue; end
    value = src.Reader.readPoint(pathOf(src, name) + "/data", info.ValueShape, p);
    w.append(name, value, s.Index(p, :));
    k = k + 1;
end
report.Copied.(matlab.lang.makeValidName(name)) = k;
end

function p = pathOf(src, name)
if src.isView(name), p = "/views/" + name; else, p = "/channels/" + name; end
end

function bad = notStored(src, spec, name)
% Requested "list" values of `name` (per axis) that no stored point has.
bad = strings(0, 1);
info = src.info(name); s = src.stored(name);
for g = spec.Groups
    if ~ismember(name, g.Channels), continue; end
    f = string(fieldnames(g.At)).';
    for a = f
        k = find(info.Axes == a, 1);
        if isempty(k), continue; end
        for v = g.At.(a).missing(s.Values(:, k).')
            bad(end+1, 1) = sprintf('%s@%s=%g', name, a, v); %#ok<AGROW>
        end
    end
end
end
