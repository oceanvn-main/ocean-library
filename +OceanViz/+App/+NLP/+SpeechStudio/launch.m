function app = launch(opts)
% LAUNCH  Start the Speech Pipeline Studio (model -> view-model -> view).
%
%   addpath('C:\Ocean Library\Matlab code practice');
%   app = OceanViz.App.NLP.SpeechStudio.launch();
%   app = OceanViz.App.NLP.SpeechStudio.launch('Visible', false, 'Recorder', rec, 'Theme', "Dark")
%
%   app is the Views.StudioView; app.ViewModel and app.ViewModel.Session give the
%   ViewModel and Model. app.close() deletes the window and all listeners / timers.
%   'Recorder': an IAudioRecorder (default MicrophoneRecorder; tests inject samples with
%   Models.InjectedRecorder). 'Root': repository root (default: derived from this file).
arguments
    opts.Visible (1,1) logical = true
    opts.Theme (1,1) string = "Dark"
    opts.Recorder = []
    opts.Root (1,1) string = OceanViz.App.NLP.SpeechStudio.Models.StudioSession.repoRoot()
    opts.Manifest = []
end
session = OceanViz.App.NLP.SpeechStudio.Models.StudioSession('Root', opts.Root, 'Manifest', opts.Manifest);
vm = OceanViz.App.NLP.SpeechStudio.ViewModels.StudioViewModel(session, 'Recorder', opts.Recorder);
app = OceanViz.App.NLP.SpeechStudio.Views.StudioView(vm, 'Visible', opts.Visible, 'Theme', opts.Theme);
dlg = [];
if opts.Visible
    dlg = uiprogressdlg(app.Figure, 'Title', 'Speech Pipeline Studio', 'Indeterminate', 'on', ...
        'Message', 'Discovering checkpoints (ModelCatalog), building the FSDD manifest, loading the model...');
end
try
    vm.initialize();
catch err
    if ~isempty(dlg), close(dlg); end
    uialert(app.Figure, err.message, 'Speech Pipeline Studio');
    return;
end
if ~isempty(dlg), close(dlg); end
end
