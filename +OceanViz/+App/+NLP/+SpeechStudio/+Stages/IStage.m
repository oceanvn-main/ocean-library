classdef (Abstract) IStage < handle
    % ISTAGE  One Speech Pipeline Studio stage (plug-in; the Model Lab IPanel pattern).
    %
    %   A stage declares the model CAPABILITIES it needs; the stepper enables it only for
    %   models that have them, so new recogniser families reuse the stages unchanged.
    %
    %   Implement:
    %     t = title(obj)            stepper / header title
    %     c = requires(obj)         string array of capabilities (all required; [] = any model)
    %     build(obj, parent)        create the stage layout once inside PARENT (a grid cell)
    %     render(obj, ctx)          draw a completed run (Stages.StageContext)
    %   Optional:
    %     setCursor(obj, t)         move the linked time cursor (seconds; NaN hides it)
    %     t = pointerTime(obj, fig) time under the pointer on one of the stage's time axes
    %     onClick(obj, t)           pointer click at time t on a time axis
    %     tf = needsRecognition(obj) default true; false for stages meaningful without speech
    %   Computation belongs in library classes; stages only select recorded artifacts
    %   and draw them through OceanViz.Core strategies.

    methods (Abstract)
        t = title(obj)
        c = requires(obj)
        build(obj, parent)
        render(obj, ctx)
    end

    methods
        function setCursor(~, ~)
        end

        function t = pointerTime(~, ~)
            t = NaN;
        end

        function onClick(~, ~)
        end

        function tf = needsRecognition(~)
            % NEEDSRECOGNITION  true when the stage shows recogniser internals (it is disabled
            %   when nothing was transcribed, e.g. no speech detected).
            tf = true;
        end

        function tf = supports(obj, capabilities)
            tf = all(arrayfun(@(c) any(capabilities == c), obj.requires()));
        end
    end
end
