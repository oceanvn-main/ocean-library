classdef Factory
    % FACTORY  Convention-based creation of TD components by name; adding one = adding ONE file.
    %   OceanStats.RL.TD.Factory.create("target", "qlearning")        -> +Targets/QLearningTarget
    %   OceanStats.RL.TD.Factory.create("stepsize", "harmonic", 'Power', 0.7)
    %   OceanStats.RL.TD.Factory.create("exploration", "epsilongreedy", 0.1)
    %   OceanStats.RL.TD.Factory.create("trace", "replacing")
    %   OceanStats.RL.TD.Factory.create("backup", "nstep")               -> +Backups/NStepBackup (the update scheme, "Mode")
    %   OceanStats.RL.TD.Factory.validNames("target")
    % Lookup is OceanNLP.Text.StrategyResolver (class name = <Name><Suffix>, case/underscore insensitive,
    % abstract interfaces skipped, interface derivation verified).
    methods (Static)
        function obj = create(kind, name, varargin)
            [pkg, suffix, base] = OceanStats.RL.TD.Factory.kindInfo_(kind);
            ctor = OceanNLP.Text.StrategyResolver.resolve(pkg, suffix, name, base);
            obj = ctor(varargin{:});
        end
        function n = validNames(kind)
            [pkg, suffix] = OceanStats.RL.TD.Factory.kindInfo_(kind);
            n = OceanNLP.Text.StrategyResolver.names(pkg, suffix);
        end
        function k = kinds(), k = ["target" "stepsize" "exploration" "trace" "backup"]; end
    end
    methods (Static, Access = private)
        function [pkg, suffix, base] = kindInfo_(kind)
            switch lower(string(kind))
                case "target",      pkg = "OceanStats.RL.TD.Targets";     suffix = "Target";        base = "OceanStats.RL.TD.Targets.ITarget";
                case "stepsize",    pkg = "OceanStats.RL.TD.StepSize";    suffix = "StepSize";      base = "OceanStats.RL.TD.StepSize.IStepSize";
                case "exploration", pkg = "OceanStats.RL.TD.Exploration"; suffix = "Exploration";   base = "OceanStats.RL.TD.Exploration.IExploration";
                case "trace",       pkg = "OceanStats.RL.TD.Traces";      suffix = "Trace";         base = "OceanStats.RL.TD.Traces.ITrace";
                case "backup",      pkg = "OceanStats.RL.TD.Backups";     suffix = "Backup";        base = "OceanStats.RL.TD.Backups.IBackup";
                otherwise
                    error('OceanStats:RL:TD:Factory:UnknownKind', 'Unknown TD component kind "%s". Valid: %s.', kind, strjoin(OceanStats.RL.TD.Factory.kinds(), ', '));
            end
        end
    end
end
