classdef ComponentType
    % COMPONENTTYPE  The type of one component: storage class, width (columns per entity), physical dimension, how it
    % moves with time (Kind) and who may see it (Visibility). The guard admits an offered component when its type
    % MATCHES the owner's declared type (Class, Width, Dimension, Kind) - never by comparing names.
    %
    %   D = OceanAgents.World.Guard.Dimension;
    %   t = OceanAgents.World.Guard.ComponentType('Width', 2, 'Dimension', D.L());          % a 2-D position
    %   c = OceanAgents.World.Guard.ComponentType('Kind', "linear", 'Dimension', D.E());     % charge moving with time
    %
    %   Kind  "value"  : stored value, changed only by its owner's writes
    %         "linear" : Base + Rate*(now - T0); moves with the timeline at zero cost; rate dimension = Dimension / T
    %         "clock"  : now - T0 (e.g. age); Width 1, Dimension T
    % Contract: W1_Kernel_Contract.md section 3.2.

    properties (SetAccess = immutable)
        Class string = "double"
        Width double = 1
        Dimension = OceanAgents.World.Guard.Dimension()
        Kind string = "value"
        Visibility string = "public"
    end

    properties (Constant)
        Classes = ["double" "logical" "string"]
        Kinds = ["value" "linear" "clock"]
        Visibilities = ["public" "self" "hidden"]
    end

    methods
        function obj = ComponentType(options)
            arguments
                options.Class = "double"
                options.Width = 1
                options.Dimension = OceanAgents.World.Guard.Dimension()
                options.Kind = "value"
                options.Visibility = "public"
            end
            bad = @(msg) error('OceanAgents:Guard:BadType', 'ComponentType: %s', msg);
            obj.Class = string(options.Class);
            if ~any(obj.Class == obj.Classes), bad('Class must be "double", "logical" or "string".'); end
            w = options.Width;
            if ~(isnumeric(w) && isscalar(w) && w >= 1 && w == fix(w)), bad('Width must be a positive integer.'); end
            obj.Width = double(w);
            if ~isa(options.Dimension, 'OceanAgents.World.Guard.Dimension'), bad('Dimension must be a Guard.Dimension.'); end
            obj.Dimension = options.Dimension;
            obj.Kind = string(options.Kind);
            if ~any(obj.Kind == obj.Kinds), bad('Kind must be "value", "linear" or "clock".'); end
            obj.Visibility = string(options.Visibility);
            if ~any(obj.Visibility == obj.Visibilities), bad('Visibility must be "public", "self" or "hidden".'); end
            if obj.Class ~= "double" && ~obj.Dimension.isDimensionless()
                bad('only "double" components may carry a physical dimension.');
            end
            if obj.Kind ~= "value" && obj.Class ~= "double"
                bad('"linear" and "clock" components must be "double".');
            end
            if obj.Kind == "clock" && (obj.Width ~= 1 || obj.Dimension ~= OceanAgents.World.Guard.Dimension.T())
                bad('a "clock" component has Width 1 and Dimension T.');
            end
        end

        function tf = matches(a, b)
            % MATCHES  Same Class, Width, Dimension and Kind (Visibility is a property of the slot, not compared).
            tf = a.Class == b.Class && a.Width == b.Width && a.Dimension == b.Dimension && a.Kind == b.Kind;
        end

        function why = mismatch(a, b)
            % MISMATCH  First differing attribute ("" if they match) - for refusal details.
            why = "";
            if a.Class ~= b.Class, why = "Class"; elseif a.Width ~= b.Width, why = "Width";
            elseif a.Dimension ~= b.Dimension, why = "Dimension"; elseif a.Kind ~= b.Kind, why = "Kind"; end
        end

        function d = rateDimension(obj)
            % RATEDIMENSION  Dimension of the rate of a "linear" component (Dimension per unit time).
            d = obj.Dimension ./ OceanAgents.World.Guard.Dimension.T();
        end
    end
end
