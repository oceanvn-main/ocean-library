%% generateSignatures.m — Auto-generate functionSignatures.json
%
%  USAGE:
%    OceanTools.generateSignatures          % writes resources/functionSignatures.json
%    OceanTools.generateSignatures('dry')   % preview only, no file written
%    OceanTools.generateSignatures('write', 'Packages', {'+OceanReplay'}, 'OutFile', tmp)
%                                           % only these package folders, into another file
%                                           % (to MERGE new entries without regenerating all)
%
%  HOW IT WORKS:
%    1. Scans +OceanData, +OceanStats, +OceanPipeline, +OceanViz for .m files
%    2. For each class file: reads classdef + method arguments blocks
%    3. Extracts (paramName, kind, type, purpose) from each arguments section
%    4. Merges auto-generated entries with manually-curated overrides from
%       resources/signatures_overrides.json (if present)
%    5. Writes one valid functionSignatures.json
%
%  LIMITATIONS (by design — keep the script simple and fast):
%    - Only parses the constructor and public methods with arguments blocks
%    - Type is inferred from argument annotations (e.g. "double", "string")
%    - mustBe validators are parsed for choices={} hints
%    - Free-form comments after "%" become the "purpose" field

function generateSignatures(mode, opts)
arguments
    mode (1,1) string {mustBeMember(mode,{'write','dry'})} = "write"
    opts.Packages cell = {}          % package folders relative to the project root ({} = all below)
    opts.OutFile (1,1) string = ""   % "" = resources/functionSignatures.json
end

ROOT     = fileparts(mfilename('fullpath'));          % .../+OceanTools/
PROJROOT = fileparts(ROOT);                          % .../Matlab code practice/
RES_DIR  = fullfile(PROJROOT, 'resources');
OUT_FILE = fullfile(RES_DIR, 'functionSignatures.json');
OVERRIDE = fullfile(RES_DIR, 'signatures_overrides.json');
PACKAGES = {'+OceanData', '+OceanStats', '+OceanPipeline', '+OceanViz', '+OceanMath', '+OceanNLP', '+OceanCS', '+OceanEngine', '+OceanAlgo'};
if ~isempty(opts.Packages), PACKAGES = opts.Packages; end
if opts.OutFile ~= "", OUT_FILE = char(opts.OutFile); end

fprintf('\n[generateSignatures] Scanning packages...\n');

entries = struct();  % the master signatures dictionary

%% 1. Scan all .m files in each package tree
for p = 1:numel(PACKAGES)
    pkgDir = fullfile(PROJROOT, PACKAGES{p});
    if ~isfolder(pkgDir), continue; end

    mfiles = rdir_(pkgDir, '*.m');
    for f = 1:numel(mfiles)
        fpath = mfiles{f};
        fqn   = path2fqn_(PROJROOT, fpath);         % e.g. OceanData.DataArch.FeatureStore
        if isempty(fqn), continue; end

        new = parseClassFile_(fpath, fqn);
        fnames = fieldnames(new);
        for i = 1:numel(fnames)
            key = fnames{i};
            entries.(key) = new.(key);
        end
    end
end

fprintf('[generateSignatures] Found %d signature entries.\n', numel(fieldnames(entries)));

%% 2. Merge with manual overrides (optional file; not for a package subset)
if isfile(OVERRIDE) && isempty(opts.Packages)
    raw = fileread(OVERRIDE);
    ov  = jsondecode(raw);
    ovf = fieldnames(ov);
    for i = 1:numel(ovf)
        % jsondecode turns '.' into '_' in field names.
        % We convert '_' to '__' to match our safeName_ format so they overwrite properly!
        safeKey = strrep(ovf{i}, '_', '__');
        entries.(safeKey) = ov.(ovf{i});
    end
    fprintf('[generateSignatures] Merged %d override entries.\n', numel(ovf));
end

%% 3. Build final JSON struct
out = struct();
out.x_schemaVersion = '2.0.0';
out.x_generated     = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm'));
out.x_generator     = 'resources/generateSignatures.m';

% Merge entries into output (x_ prefix avoids clash with schema fields)
fnames = fieldnames(entries);
for i = 1:numel(fnames)
    entry = entries.(fnames{i});
    % MATLAB jsondecode converts length-1 JSON arrays of objects into scalar structs.
    % jsonencode then converts scalar structs back into JSON objects instead of arrays!
    % We must forcefully wrap any struct 'inputs' into a cell array so it encodes as [ {..} ]
    if isfield(entry, 'inputs') && isstruct(entry.inputs)
        cellInputs = cell(1, numel(entry.inputs));
        for j = 1:numel(entry.inputs)
            cellInputs{j} = entry.inputs(j);
        end
        entry.inputs = cellInputs;
    end
    if isfield(entry, 'outputs') && isstruct(entry.outputs)          % same single-element-array trap for 'outputs'
        cellOutputs = cell(1, numel(entry.outputs));
        for j = 1:numel(entry.outputs)
            cellOutputs{j} = entry.outputs(j);
        end
        entry.outputs = cellOutputs;
    end
    out.(fnames{i}) = entry;
end

%% 4. Write or preview
json = jsonencode(out, 'PrettyPrint', true);
% jsonencode uses underscores internally — restore dots in keys
json = fixKeys_(json);

if mode == "dry"
    fprintf('\n--- DRY RUN (first 200 chars) ---\n%s...\n', json(1:min(200,end)));
    fprintf('[generateSignatures] DRY RUN — no file written.\n\n');
else
    fid = fopen(OUT_FILE, 'w', 'n', 'UTF-8');
    fprintf(fid, '%s\n', json);
    fclose(fid);
    fprintf('[generateSignatures] Written: %s\n\n', OUT_FILE);
end
end

%% ====================================================================== %%
function sigs = parseClassFile_(fpath, fqn)
% Parse a single .m class file and return a struct of signature entries.
sigs = struct();

lines = readlines_(fpath);
if isempty(lines), return; end

% Quick check: is this a classdef file?
cdLine = find(~cellfun(@isempty, regexp(lines, '^\s*classdef\s', 'match')), 1);
if isempty(cdLine), return; end

% Parse description from first comment after classdef
desc = extractDesc_(lines, cdLine);

% Constructor (same name as class basename)
baseName = fqn2base_(fqn);
ctorSig  = parseMethod_(lines, baseName, fqn, desc);
if ~isempty(ctorSig)
    sigs.(safeName_(fqn)) = ctorSig;
end

% Other PUBLIC methods: skip methods declared in a methods block with Access = private / protected, and helpers named with the
% trailing-underscore convention (MATLAB cannot resolve those from outside the class, so listing them is an error)
methStarts = find(~cellfun(@isempty, regexp(lines, '^\s*function\s', 'match')));
blockLines = find(~cellfun(@isempty, regexp(lines, '^\s*(methods|properties|events|enumeration)\b', 'match')));
for i = 1:numel(methStarts)
    methName = extractMethodName_(lines{methStarts(i)});
    if isempty(methName) || strcmp(methName, baseName), continue; end
    if endsWith(methName, '_'), continue; end
    blk = blockLines(find(blockLines < methStarts(i), 1, 'last'));
    if ~isempty(blk) && ~isempty(regexp(lines{blk}, 'Access\s*=\s*(private|protected)', 'once')), continue; end
    skip = {'delete','disp','display','saveobj','loadobj'};
    if any(strcmp(methName, skip)), continue; end

    sig = parseMethod_(lines, methName, fqn, '');
    if ~isempty(sig)
        key = safeName_(sprintf('%s.%s', fqn, methName));
        sigs.(key) = sig;
    end
end
end

%% ====================================================================== %%
function sig = parseMethod_(lines, methodName, fqn, desc)
% Find function declaration + arguments block for methodName.
sig = [];

% Find function line
pat   = sprintf('function.+%s\\s*\\(', regexptranslate('escape', methodName));
fLine = find(~cellfun(@isempty, regexp(lines, pat)), 1);
if isempty(fLine), return; end

% Find arguments block after function line
argStart = [];
for i = fLine+1:min(fLine+30, numel(lines))
    if ~isempty(regexp(lines{i}, '^\s*arguments', 'once'))
        argStart = i; break;
    end
    if ~isempty(regexp(lines{i}, '^\s*(if|for|while|switch|try)', 'once'))
        break;  % gave up: no arguments block
    end
end

inputs = {};
if ~isempty(argStart)
    for i = argStart+1:numel(lines)
        ln = strtrim(lines{i});
        if strcmp(ln, 'end'), break; end
        if isempty(ln) || startsWith(ln, '%'), continue; end

        inp = parseSingleArg_(ln);
        if ~isempty(inp)
            inputs{end+1} = inp; %#ok<AGROW>
        end
    end
end

% Build entry
if isempty(desc)
    desc = sprintf('%s.%s', fqn, methodName);
end
sig.description = desc;
if ~isempty(inputs)
    sig.inputs = inputs;
end
end

%% ====================================================================== %%
function inp = parseSingleArg_(ln)
% Parse one line of an arguments block into an input descriptor.
% Handles:
%   obj                                          → skip (self)
%   ~                                            → skip
%   name type                                    → required
%   opts.name type = default                     → namevalue
%   name (1,1) type {mustBe...} = default        → ordered (optional positional)
inp = [];

% Skip self / ignore (only an argument that IS 'obj' or '~'; a name such as 'objective' is a real argument)
first = regexp(strtrim(ln), '^(~|\w+)', 'match', 'once');
if strcmp(first, 'obj') || strcmp(first, '~')
    return;
end

% Extract trailing comment as purpose
purpose = '';
cmtIdx  = strfind(ln, '%');
if ~isempty(cmtIdx)
    purpose = strtrim(ln(cmtIdx(1)+1:end));
    ln      = strtrim(ln(1:cmtIdx(1)-1));
end

% Namevalue: <structName>.<name>  (opts.X, options.X, params.X, ...): a positional argument name never contains a dot
nvTok = regexp(ln, '^\w+\.(\w+)', 'tokens', 'once');
if ~isempty(nvTok)
    name   = nvTok{1};
    kind   = 'namevalue';
else
    toks   = regexp(ln, '^(\w+)', 'tokens', 'once');
    if isempty(toks), return; end
    name   = toks{1};
    % Has a default => optional positional
    if contains(ln, '=')
        kind = 'ordered';                 % MATLAB's current name for an optional positional argument ('optional' is deprecated)
    else
        kind = 'required';
    end
end

% Extract type annotation
typStr = extractType_(ln);

inp.name = name;
inp.kind = kind;
if ~isempty(typStr), inp.type    = {typStr}; end       % MATLAB's schema wants an array of type strings
if ~isempty(purpose), inp.purpose = purpose; end
end

%% ====================================================================== %%
function t = extractType_(ln)
% Try to pull a type keyword from the argument annotation line.
t = '';

% mustBeMember choices: {'a','b'}, ["a" "b"] or ["a", "b"]
ch = regexp(ln, 'mustBeMember\s*\(\w+[\.\w]*\s*,\s*[\{\[]([^}\]]+)[\}\]]', 'tokens', 'once');
if ~isempty(ch)
    items = regexp(ch{1}, '["'']([^"'']+)["'']', 'tokens');
    if ~isempty(items)
        t = ['choices={' strjoin(cellfun(@(c) ['''' c{1} ''''], items, 'UniformOutput', false), ',') '}'];
        return;
    end
end

% what remains must be the declared class/type: drop validator functions {..} and the default value (= ...), which may
% themselves contain class names or type words
ln = regexprep(ln, '\{[^}]*\}', '');
ln = regexprep(ln, '=.*$', '');

% Known MATLAB types
types = {'double','single','int8','int16','int32','int64', ...
         'uint8','uint16','uint32','uint64', ...
         'logical','char','string','cell','struct', ...
         'numeric','integer','float'};
for i = 1:numel(types)
    if ~isempty(regexp(ln, sprintf('\\<%s\\>', types{i}), 'once'))
        t = types{i}; return;      % (\< \> are MATLAB's word boundaries; \b would be a backspace)
    end
end

% Class reference (like OceanData.DataArch.FeatureStore)
cls = regexp(ln, '([A-Z]\w+(?:\.\w+)+)', 'tokens', 'once');
if ~isempty(cls)
    t = cls{1};
end
end

%% ====================================================================== %%
function desc = extractDesc_(lines, cdLine)
% Try to pull the first comment block after classdef.
desc = '';
for i = cdLine+1:min(cdLine+5, numel(lines))
    ln = strtrim(lines{i});
    if startsWith(ln, '%')
        desc = strtrim(ln(2:end));
        return;
    end
end
end

%% ====================================================================== %%
function name = extractMethodName_(ln)
% Extract the method name from a 'function ...' declaration line.
name = '';
tok  = regexp(strtrim(ln), 'function\s+(?:\[?[\w,\s~]+\]?\s*=\s*)?(\w+)\s*\(', ...
              'tokens', 'once');
if ~isempty(tok)
    n = tok{1};
    if ~strcmp(n, 'obj'), name = n; end
end
end

%% ====================================================================== %%
function fqn = path2fqn_(projRoot, fpath)
% Convert a file path to a fully-qualified class name.
% e.g. .../+OceanData/+DataArch/FeatureStore.m → OceanData.DataArch.FeatureStore
fqn = '';
rel = strrep(fpath, projRoot, '');
rel = strrep(rel, '\', '/');
% Remove leading /
if startsWith(rel, '/'), rel = rel(2:end); end
parts = strsplit(rel, '/');
out   = {};
for i = 1:numel(parts)
    p = parts{i};
    if startsWith(p, '+')
        out{end+1} = p(2:end); %#ok<AGROW>
    elseif endsWith(p, '.m')
        out{end+1} = p(1:end-2); %#ok<AGROW>
    end
end
if numel(out) >= 1
    fqn = strjoin(out, '.');
end
end

%% ====================================================================== %%
function b = fqn2base_(fqn)
parts = strsplit(fqn, '.');
b = parts{end};
end

%% ====================================================================== %%
function s = safeName_(str)
% jsonencode/jsondecode replaces dots with underscores in field names.
% We store them with underscores internally and fix them in JSON post-processing.
s = strrep(str, '.', '__');
end

%% ====================================================================== %%
function json = fixKeys_(json)
% Reverse the __ → DOT substitution that safeName_ applied.
% This operates on the raw JSON text so key names use dots as expected by MATLAB IDE.
json = regexprep(json, '"(\w+)__(\w+)', '"$1.$2');
% Handle multi-segment names (OceanData__DataArch__FeatureStore etc.)
while contains(json, '__')
    json = strrep(json, '__', '.');
end
end

%% ====================================================================== %%
function lines = readlines_(fpath)
fid   = fopen(fpath, 'r', 'n', 'UTF-8');
lines = {};
if fid == -1, return; end
tline = fgetl(fid);
while ischar(tline)
    lines{end+1} = tline; %#ok<AGROW>
    tline = fgetl(fid);
end
fclose(fid);
end

%% ====================================================================== %%
function files = rdir_(root, pattern)
% Recursive directory listing for files matching pattern.
files = {};
d = dir(fullfile(root, pattern));
for i = 1:numel(d)
    files{end+1} = fullfile(d(i).folder, d(i).name); %#ok<AGROW>
end
subs = dir(root);
for i = 1:numel(subs)
    if subs(i).isdir && startsWith(subs(i).name, '+')
        sub = rdir_(fullfile(root, subs(i).name), pattern);
        files = [files, sub]; %#ok<AGROW>
    end
end
end

%% ====================================================================== %%
function r = startsWith(s, p)
r = numel(s) >= numel(p) && strcmp(s(1:numel(p)), p);
end

function r = endsWith(s, p)
r = numel(s) >= numel(p) && strcmp(s(end-numel(p)+1:end), p);
end
