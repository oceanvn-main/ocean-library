classdef (Abstract) IAccumulator < handle
    % IACCUMULATOR  Abstract interface for online/streaming statistical accumulators.
    %
    %   Unifies incremental sample averaging, weighted importance sampling,
    %   exponential moving averages, and Welford running variance into a shared
    %   infrastructure interface across RL, Signal Processing, and Data Analytics.
    %
    %   CORE METHODS:
    %     update(obj, sample, weight, idx) - Online update with new sample and weight
    %     val = getValue(obj)              - Current mean / value estimate
    %     w   = getWeight(obj)             - Cumulative weight / count
    %     reset(obj)                       - Resets accumulator state
    %     meta= getMetadata(obj)           - Pedagogical reflection metadata
    %
    %   Package: OceanEngine.Accumulators
    %   See also: OceanEngine.Accumulators.WeightedMeanAccumulator
    %             OceanEngine.Accumulators.ExponentialMovingAverage
    %             OceanEngine.Accumulators.WelfordVarianceAccumulator

    methods (Abstract)
        update(obj, sample, weight, idx)
        % UPDATE  Perform online state update with new sample and optional weight/indexing.
        %   sample - Scalar, vector, or array of observed values
        %   weight - (optional) Sample weight / importance sampling ratio (default 1.0)
        %   idx    - (optional) Index/subscript for targeted array element updates

        val = getValue(obj)
        % GETVALUE  Returns current value / mean table estimate.

        w = getWeight(obj)
        % GETWEIGHT  Returns cumulative weight / count table.

        reset(obj)
        % RESET  Resets accumulators back to initial state.

        meta = getMetadata(obj)
        % GETMETADATA  Returns self-describing pedagogical metadata.

        spec = getDiagramSpec(obj)
        % GETDIAGRAMSPEC  Returns Pillar 3 diagram representation spec (Ports & Canvas Symbol).
    end
end
