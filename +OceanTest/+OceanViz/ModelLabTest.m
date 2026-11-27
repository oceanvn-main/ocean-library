classdef ModelLabTest < matlab.unittest.TestCase
    % MODELLABTEST  Headless (hidden uifigure) tests of the Model Lab:
    %   the tabs follow the model's capabilities, and every panel's main
    %   action runs and produces what its equations say.

    methods (Test)
        function testIdealDDPMTabsAndPanels(tc)
            lab = OceanViz.App.ModelLab.launch("Ideal DDPM - shapes 16x16 (no training)", 'Visible', false);
            tc.addTeardown(@() lab.close());
            tc.verifyEqual(lab.tabTitles(), ["Forward process", "Reverse process", "Generate", "Training step"]);

            fwd = lab.panel("Forward process");
            fwd.frame(1000);                                          % ends at (almost) pure noise
            step = lab.panel("Training step");
            loss = step.update();
            tc.verifyGreaterThanOrEqual(loss, 0);
            tc.verifyLessThan(loss, 1, 'the ideal denoiser must beat eps_phi = 0 (loss ~ 1)');

            gen = lab.panel("Generate");
            imgs = gen.generate();
            tc.verifySize(imgs, [16 16 1 16]);
            tc.verifyGreaterThanOrEqual(min(imgs(:)), 0); tc.verifyLessThanOrEqual(max(imgs(:)), 1);
            seed = gen.SeedField.Value;
            gen.generate();
            tc.verifyEqual(gen.SeedField.Value, seed + 1, 'each click must use a new seed');

            rev = lab.panel("Reverse process");
            rev.run();
            tc.verifyNotEmpty(rev.Process.Frames);
            tc.verifyEqual(rev.Process.numSteps(), 1000, 'default sampler = DDPM over all L steps');
            rev.showFrame(numel(rev.Process.Frames));
        end

        function testSaveRecordingEqualsTheRun(tc)
            % "Save recording..." records exactly the run "Run reverse process" shows
            % (same sampler, seed, round-trip start and clipping), and Replay Studio opens it.
            lab = OceanViz.App.ModelLab.launch("Ideal DDPM - shapes 16x16 (no training)", 'Visible', false);
            c = onCleanup(@() delete(lab));
            rev = lab.panel("Reverse process");
            rev.SamplerDropdown.Value = 'DDIM deterministic, 50 steps (Sec. 9.2)';
            rev.StartDropdown.Value = 'Current image noised to t0';
            rev.T0Field.Value = 0.6; rev.SeedField.Value = 4;
            rev.run();
            f = fullfile(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder, "rev.h5");
            rev.saveRecording(f, 'Open', false);
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            r = OceanReplay.Core.Recording(f);
            m = rev.Ctx.Model;
            tc.verifyEqual(double(r.get("result")), double(reshape(rev.Process.X, m.ImageSize)), 'AbsTol', 1e-6);
            steps = r.axis("step").Values;
            tc.verifyNumElements(steps, rev.Process.numSteps());
            for k = [1 numel(rev.Process.Frames)]
                fr = rev.Process.Frames(k);
                tc.verifyEqual(double(r.get("x", struct('step', steps(fr.Step)), "exact")), ...
                    double(reshape(fr.X, m.ImageSize)), 'AbsTol', 1e-6);
            end
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c2 = onCleanup(@() s.close());
            tc.verifyEqual(s.Layout.name(), "Diffusion");
        end

        function testImageChangeReachesPanels(tc)
            lab = OceanViz.App.ModelLab.launch("Ideal DDPM - shapes 16x16 (no training)", 'Visible', false);
            tc.addTeardown(@() lab.close());
            step = lab.panel("Training step");
            before = step.update();
            lab.Context.useDatasetImage(7);
            tc.verifyNotEqual(step.update(), before);
        end

        function testCnnPanelsAllRun(tc)
            lab = OceanViz.App.ModelLab.launch("Dummy CNN (random weights)", 'Visible', false);
            tc.addTeardown(@() lab.close());
            titles = lab.tabTitles();
            tc.verifyNumElements(titles, 8);
            tc.verifyFalse(any(titles == "Generate"), 'a classifier must not get generative tabs');
            for t = titles
                p = lab.panel(t);
                p.run();
                tc.verifyNotEmpty(p.Ax.Children, sprintf('%s drew nothing', t));
            end
        end

        function testLegacyStudioLaunchesModelLab(tc)
            root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, '+OceanViz', '+App', 'Computer Vision')));
            s = CNNInterpretabilityStudio('Visible', false);
            tc.addTeardown(@() s.Lab.close());
            tc.verifyNumElements(s.Lab.tabTitles(), 8);
        end

        function testVaeCardRoundTrip(tc)
            % A small untrained HVAE saved as a ModelCard opens with the VAE tabs.
            recipe = OceanDL.Training.ModelCard.recipe("OceanDL.Generative.HierarchicalVAE", {}, ...
                {'Widths', [8 8 8], 'LatentsPerLevel', [0 1 1], 'LatentChannels', 2, 'Likelihood', "bernoulli", ...
                 'Groups', 4, 'Seed', 3});
            net = OceanDL.Training.ModelCard.build(recipe);
            file = string(tempname) + ".mat";
            tc.addTeardown(@() delete(file));
            OceanDL.Training.ModelCard.save(file, net, recipe, 'Meta', ...
                struct('Dataset', "usps", 'ImageSize', [16 16 1], 'ClassNames', string(0:9)));
            lab = OceanViz.App.ModelLab.launch(file, 'Visible', false);
            tc.addTeardown(@() lab.close());
            tc.verifyEqual(lab.tabTitles(), ["Generate", "VAE explorer", "Inside the VAE"]);
            tc.verifyTrue(lab.Context.Model.has("hierarchical"));
            ex = lab.panel("VAE explorer");
            a = ex.traverse(0); b = ex.traverse(3);
            tc.verifySize(a, [16 16]);
            tc.verifyGreaterThan(max(abs(a(:) - b(:))), 0, 'shifting a latent must change the decoded image');
            tc.verifySize(lab.panel("Generate").generate(), [16 16 1 16]);
        end

        function testInsideVaePanelRunsEveryView(tc)
            % Every view of every section computes (no "Could not compute") on a small HVAE.
            recipe = OceanDL.Training.ModelCard.recipe("OceanDL.Generative.HierarchicalVAE", {}, ...
                {'Widths', [8 8 8], 'LatentsPerLevel', [0 1 1], 'LatentChannels', 2, 'Likelihood', "bernoulli", ...
                 'Groups', 4, 'Seed', 4});
            net = OceanDL.Training.ModelCard.build(recipe);
            file = string(tempname) + ".mat";
            tc.addTeardown(@() delete(file));
            OceanDL.Training.ModelCard.save(file, net, recipe, 'Meta', ...
                struct('Dataset', "usps", 'ImageSize', [16 16 1], 'ClassNames', string(0:9), 'Binarized', true));
            lab = OceanViz.App.ModelLab.launch(file, 'Visible', false);
            tc.addTeardown(@() lab.close());
            p = lab.panel("Inside the VAE");
            tc.assertNotEmpty(p, 'the tab appears for traceable models');
            for section = ["encoder", "latent", "decoder"]
                p.select(section);
                views = string(p.ViewDropdown.Items);
                for v = views
                    p.select(section, [], v, 1);
                    R = p.show();
                    tc.verifyFalse(startsWith(string(R.Text{1}), "Could not compute"), section + " / " + v + ": " + string(R.Text{1}));
                    tc.verifyTrue(all(R.Image(:) >= 0 & R.Image(:) <= 1), section + " / " + v);
                end
            end
        end
    end
end
