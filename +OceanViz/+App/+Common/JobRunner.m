classdef JobRunner
    % JOBRUNNER  Run a long computation in small chunks so the app stays
    %   responsive: a progress dialog with a Cancel button, updated between
    %   chunks (e.g. a 1000-step DDPM reverse pass in chunks of 20 steps).
    %
    %   ok = JobRunner.run(fig, 'Denoising...', @() stepFn())
    %     stepFn() -> [done, fraction, message]   one chunk of work
    %   ok = false if the user cancelled. Without a visible figure (tests,
    %   scripts) the chunks simply run to completion.

    methods (Static)
        function ok = run(fig, titleText, stepFn)
            dlg = [];
            if ~isempty(fig) && isvalid(fig) && strcmp(fig.Visible, 'on')
                try
                    dlg = uiprogressdlg(fig, 'Title', titleText, 'Cancelable', 'on', 'Value', 0);
                catch
                    dlg = [];
                end
            end
            cleanup = onCleanup(@() OceanViz.App.Common.JobRunner.closeDialog(dlg));
            ok = true;
            done = false;
            while ~done
                [done, fraction, msg] = stepFn();
                if ~isempty(dlg)
                    dlg.Value = min(1, max(0, fraction));
                    dlg.Message = char(msg);
                    drawnow limitrate;
                    if dlg.CancelRequested
                        ok = false;
                        return;
                    end
                end
            end
        end
    end

    methods (Static, Access = private)
        function closeDialog(dlg)
            if ~isempty(dlg) && isvalid(dlg), close(dlg); end
        end
    end
end
