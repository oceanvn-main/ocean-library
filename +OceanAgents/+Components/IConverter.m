classdef (Abstract) IConverter < OceanAgents.Components.IComponent
    % ICONVERTER  A component that turns one kind of signal into another (design section 3; AB1 section 3.6). The
    % router inserts a converter at join time when it bridges an offered type to what a receiver accepts, choosing by
    % type only. Converters are the only parts that change a signal's content.

    properties (SetAccess = protected)
        InType = OceanAgents.Minds.Signals.SignalType()
        OutType = OceanAgents.Minds.Signals.SignalType()
    end
end
