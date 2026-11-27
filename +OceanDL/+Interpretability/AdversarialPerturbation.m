classdef AdversarialPerturbation
    % ADVERSARIALPERTURBATION  "Which tiny input direction most changes the
    %   prediction?" FGSM / iterative FGSM (Goodfellow, Shlens & Szegedy,
    %   2014; Kurakin et al., 2016). A robustness/interpretability
    %   diagnostic, framed the same as the other 7 methods -- it answers
    %   "how fragile is this prediction to a bounded input perturbation",
    %   not an attack against a deployed system.
    %
    %   Uses cross-entropy-on-probabilities gradient w.r.t. the input,
    %   through CNNEngine.gradient() -- a direct Tape.gradient() query,
    %   same mechanism as SaliencyMap.
    %
    %   USAGE:
    %     r = AdversarialPerturbation.fgsm(engine, X, trueClass, 0.02);
    %     r2 = AdversarialPerturbation.iterativeFGSM(engine, X, trueClass, 0.05, 'NumSteps', 10);

    methods (Static)
        function result = fgsm(engine, X, trueClass, epsilon, opts)
            arguments
                engine
                X (:,:,:,:) double
                trueClass (1,1) double {mustBePositive, mustBeInteger}
                epsilon (1,1) double {mustBeNonnegative}
                opts.OutputLayer (1,:) char = ''
                opts.ValueRange (1,2) double = [0 1]
            end
            import OceanDL.Interpretability.AdversarialPerturbation
            [dX, Y] = AdversarialPerturbation.ceGradient(engine, X, trueClass, opts.OutputLayer);

            Xadv = X + epsilon * sign(dX);
            Xadv = min(max(Xadv, opts.ValueRange(1)), opts.ValueRange(2));
            Yadv = engine.predict(Xadv);

            result = struct('AdversarialImage', Xadv, 'Perturbation', Xadv - X, ...
                'OriginalScore', Y(trueClass, :), 'AdversarialScore', Yadv(trueClass, :), 'Gradient', dX);
        end

        function result = iterativeFGSM(engine, X, trueClass, epsilon, opts)
            arguments
                engine
                X (:,:,:,:) double
                trueClass (1,1) double {mustBePositive, mustBeInteger}
                epsilon (1,1) double {mustBeNonnegative}
                opts.Alpha (1,1) double {mustBeNonnegative} = 0
                opts.NumSteps (1,1) double {mustBePositive, mustBeInteger} = 10
                opts.OutputLayer (1,:) char = ''
                opts.ValueRange (1,2) double = [0 1]
            end
            import OceanDL.Interpretability.AdversarialPerturbation
            alpha = opts.Alpha;
            if alpha == 0
                alpha = 2.5 * epsilon / opts.NumSteps;   % Kurakin et al. heuristic
            end

            Xadv = X;
            for t = 1:opts.NumSteps %#ok<FXUP>
                dX = AdversarialPerturbation.ceGradient(engine, Xadv, trueClass, opts.OutputLayer);
                Xadv = Xadv + alpha * sign(dX);
                delta = max(min(Xadv - X, epsilon), -epsilon);   % project to the L_inf epsilon-ball around X
                Xadv = min(max(X + delta, opts.ValueRange(1)), opts.ValueRange(2));
            end

            Yorig = engine.predict(X);
            Yadv = engine.predict(Xadv);
            result = struct('AdversarialImage', Xadv, 'Perturbation', Xadv - X, ...
                'OriginalScore', Yorig(trueClass, :), 'AdversarialScore', Yadv(trueClass, :));
        end
    end

    methods (Static, Access = private)
        function [dX, Y] = ceGradient(engine, X, trueClass, outputLayer)
            % CEGRADIENT  d(CrossEntropy(trueClass, softmax-or-given-output))/d(input).
            %   dCE/dY_c = -1/Y_c for c==trueClass, 0 otherwise -- the
            %   standard cross-entropy-on-probabilities gradient, pushed
            %   back to the input through the Tape.
            if isempty(outputLayer)
                outputLayer = engine.Spec.LayerNames{end};
            end
            Y = engine.forward(X);
            seed = zeros(size(Y));
            seed(trueClass, :) = -1 ./ max(Y(trueClass, :), 1e-12);
            dX = engine.gradient(outputLayer, 'input', seed);
        end
    end
end
