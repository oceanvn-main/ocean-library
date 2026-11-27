classdef (Abstract) IBeliefStore < handle
    % IBELIEFSTORE  A mind's private belief memory and the rule that decides how evidence changes it. The substitution
    % point between a fixed-score mind and an evidence-adaptive one.
    %
    %   rec  = store.belief(subjectID, context)   BeliefRecord for the store's Quantity
    %   disp = store.observe(evidence)            1xK string, one disposition per EvidenceRecord, in order:
    %          "duplicate" | "ineligible" | "report" | "unknown" | "frozen" | "recorded-only" | "applied"
    %          (a seen EvidenceID with different content raises OceanAgents:Beliefs:EvidenceConflict)
    %   v    = store.version()                    +1 per APPLIED update (0 at construction)
    %   n    = store.counts()                     struct of disposition counts
    %   store.setUpdatesEnabled(false)            frozen-belief evaluation (every event is still consumed once)
    %   store.beginEpisode(); store.endEpisode()  persistence policy ('Persist' = "run" | "episode")
    %
    % Beliefs are keyed by SubjectID (never by slot). Every observed EvidenceID is marked seen whatever its disposition,
    % so each event is consumed exactly once. Capacities are explicit: exceeding them raises
    % OceanAgents:Beliefs:CapacityExceeded; nothing is silently evicted.
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md section 3.2.

    methods (Abstract)
        d = describe(obj)                 % struct: Name, UpdateRule, Quantity, Persist, UpdatesEnabled
        rec = belief(obj, subjectID, context)
        disp = observe(obj, evidence)
        v = version(obj)
        n = counts(obj)
        setUpdatesEnabled(obj, tf)
    end

    methods
        function beginEpisode(~), end
        function endEpisode(~), end
        function s = exportState(~), s = struct(); end
        function importState(~, ~), end
    end
end
