classdef Checkpoint
    % CHECKPOINT  Resumable experiments: a finished unit of work is saved
    %   once and reloaded on the next run, so a crash or power cut loses only
    %   the unit in progress.
    %
    %   out = OceanDL.Training.Checkpoint.run(dir, key, @() work())
    %     - if <dir>/<key>.mat exists: load and return its 'out' (no work)
    %     - else: out = work(), saved to <key>.mat.tmp then renamed (atomic --
    %       a half-written file is never mistaken for a finished one)
    %   Put everything that determines the result (steps, sizes, seeds) in the
    %   key; delete the folder to recompute.

    methods (Static)
        function out = run(dir, key, fn)
            if ~isfolder(dir), mkdir(dir); end
            file = fullfile(dir, [matlab.lang.makeValidName(char(key)) '.mat']);
            if isfile(file)
                s = load(file, 'out'); out = s.out;
                fprintf('  [checkpoint] reusing %s\n', key);
                return;
            end
            out = fn();
            tmp = [file '.tmp'];
            save(tmp, 'out', '-v7');
            movefile(tmp, file, 'f');
        end
    end
end
