classdef (Abstract) IImportanceWeighting < handle
    % IIMPORTANCEWEIGHTING Abstract base class for Importance Sampling Weighting Schemes.
    %
    % Encapsulates the mathematical calculation and update mechanics for
    % off-policy importance sampling estimators (Sutton & Barto Chapter 5).
    %
    % Concrete subclasses implement:
    %   - OrdinaryISWeighting                 (S&B Eq 5.5: unweighted empirical mean)
    %   - WeightedISWeighting                 (S&B Eq 5.6: normalized weighted average)
    %   - DiscountingAwareOrdinaryISWeighting (S&B Eq 5.9: flat partial return, unweighted)
    %   - DiscountingAwareWeightedISWeighting (S&B Eq 5.10: flat partial return, normalized)
    %
    % 3-PILLAR COMPONENT CONTRACT:
    %   - Pillar 1 (Symbolic φ): getMetadata() returning LaTeX formulas and symbols dict.
    %   - Pillar 2 (Computational f): update() applying numeric weight update.
    %   - Pillar 3 (Diagram D): getPipelineSpec() defining ports spec.
    %
    % Package: OceanStats.RL.MonteCarlo.Weighting
    % Reference: Sutton & Barto (2018) Sections 5.5, 5.6, 5.8

    methods (Abstract)
        % UPDATE Applies importance-weighted return update to an accumulator table.
        %
        % Inputs:
        %   accum - Target accumulator (VAccumulator or QAccumulator)
        %   G     - Return value (or cell array of partial returns for discounting-aware IS)
        %   W     - Cumulative importance sampling weight (or cell array of truncated weights)
        %   s_idx - State linear index
        %   a_idx - Action linear index (optional, 0 if V-table update)
        update(obj, accum, G, W, s_idx, a_idx)
    end

    methods (Static, Abstract)
        meta = getMetadata()
        spec = getPipelineSpec()
    end
end
