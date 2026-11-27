function m = fsddTestManifest()
% FSDDTESTMANIFEST  Shared FSDD manifest for the Data tests (built once per MATLAB session).
%   SpeechManifest.fromFSDD reads audioinfo for all 2000 recordings (~20 s), so the
%   result is cached in a persistent variable. Callers must not mutate it (the
%   manifest API returns new objects for subset/split operations).
persistent cached
if isempty(cached) || ~isvalid(cached)
    cached = OceanNLP.Speech.Data.SpeechManifest.fromFSDD();
end
m = cached;
end
