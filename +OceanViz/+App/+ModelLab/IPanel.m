classdef (Abstract) IPanel < handle
    % IPANEL  One Model Lab feature tab (plug-in). A panel declares the model
    %   CAPABILITIES it needs; the Shell shows it only for models that have
    %   them, so new model families reuse existing panels without GUI edits.
    %
    %   Implement:
    %     t = title(obj)            tab title
    %     c = requires(obj)         string array of capabilities (all required)
    %     build(obj, parent, ctx)   create controls inside parent (a uitab)
    %   Optional:
    %     r = reference(obj)        book / paper section shown in the tab tooltip
    %     onImageChanged(obj, ctx)  react to a new input image
    %   Computation belongs in library classes (e.g. OceanDL.Generative.Inspect);
    %   panels only gather settings and draw results.

    methods (Abstract)
        t = title(obj)
        c = requires(obj)
        build(obj, parent, ctx)
    end

    methods
        function r = reference(~)
            r = "";
        end

        function onImageChanged(~, ~)
        end

        function tf = supports(obj, model)
            tf = all(arrayfun(@(c) model.has(c), obj.requires()));
        end
    end
end
