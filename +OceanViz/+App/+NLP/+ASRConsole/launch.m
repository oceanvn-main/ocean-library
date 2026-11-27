function app = launch(opts)
% LAUNCH  Start the ASR Console (A10): talk to the registry models through the local service.
%
%   addpath('C:\Ocean Library\Matlab code practice');
%   app = OceanViz.App.NLP.ASRConsole.launch();
%   app = OceanViz.App.NLP.ASRConsole.launch('ServiceUrl', "http://127.0.0.1:8009", 'Visible', false)
%
%   First start the service in the ASR repository (Production SI/ASR):
%       uv run asr serve          (this PC)   or   just up a10   (Docker)
%   app is the Views.ConsoleView; app.ViewModel the ConsoleViewModel; app.close() ends it.
%   'Client' / 'RecorderFactory' / 'Player' replace the service, microphone and speaker (tests).
arguments
    opts.Visible (1,1) logical = true
    opts.ServiceUrl (1,1) string = "http://127.0.0.1:8009"
    opts.Connect (1,1) logical = true
    opts.Client = []
    opts.RecorderFactory = []
    opts.Player = []
    opts.SaveFolder (1,1) string = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel.defaultSaveFolder()
end
client = opts.Client;
if isempty(client)
    client = OceanViz.App.NLP.ASRConsole.Models.RecognitionClient('BaseUrl', opts.ServiceUrl);
end
vm = OceanViz.App.NLP.ASRConsole.ViewModels.ConsoleViewModel('Client', client, ...
    'RecorderFactory', opts.RecorderFactory, 'Player', opts.Player, 'SaveFolder', opts.SaveFolder);
app = OceanViz.App.NLP.ASRConsole.Views.ConsoleView(vm, 'Visible', opts.Visible);
if opts.Connect
    % Draw the window BEFORE connecting: a slow or missing service then shows in the status bar instead of as
    % "no window" (finding 2026-10-20: the model list waited on a stopped registry and nothing appeared).
    drawnow;
    vm.connect();     % failures land in the status bar and the Log tab, not in an error
end
end
