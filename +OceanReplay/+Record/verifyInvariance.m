function [tf, report] = verifyInvariance(factory, spec, opts)
% VERIFYINVARIANCE  Recording must not change the computation: run a domain
%   once without recording and once with it (same global rng seed) and
%   compare the outputs exactly (isequaln).
%
%   [tf, report] = OceanReplay.Record.verifyInvariance(factory, spec)
%     factory  @() new IRecordable in its initial state (a fresh one per run)
%     spec     OceanReplay.Core.Spec used for the recorded run
%     report   struct: Without, With (the two outputs), File (the recording)
%
%   Example:
%     [ok, r] = OceanReplay.Record.verifyInvariance(@() MyDomain(), spec);
%     assert(ok, 'recording changed the result');

arguments
    factory (1,1) function_handle
    spec (1,1) OceanReplay.Core.Spec
    opts.Seed (1,1) double = 0
    opts.File (1,1) string = string(tempname) + ".h5"
    opts.ViewSeed (1,1) double = 1007
end
saved = rng;
restore = onCleanup(@() rng(saved));

rng(opts.Seed, 'twister');
without = factory().run(OceanReplay.Record.NullRecorder());

rng(opts.Seed, 'twister');
with = OceanReplay.Record.Recorder.run(factory(), spec, opts.File, 'Overwrite', true, 'ViewSeed', opts.ViewSeed);

tf = isequaln(without, with);
report = struct('Without', without, 'With', with, 'File', opts.File);
end
