classdef RecencyBonus < OceanStats.RL.Dyna.Bonuses.IPlanningBonus
    % RECENCYBONUS Dyna-Q+ bonus kappa*sqrt(real steps since last visit).
    properties (SetAccess = immutable)
        Kappa (1,1) double
    end
    properties (Access = private)
        LastRealVisit (:,:) double = zeros(0,0)
    end
    methods
        function obj=RecencyBonus(kappa,validActionMask)
            arguments
                kappa (1,1) double {mustBeFinite,mustBeNonnegative}
                validActionMask (:,:) logical = true(0,0)
            end
            obj.Kappa=kappa; obj.LastRealVisit=zeros(size(validActionMask));
            obj.LastRealVisit(~validActionMask)=NaN;
        end
        function bonus=evaluate(obj,s,a,currentRealStep,lastRealVisit)
            arguments
                obj
                s (1,1) double {mustBeInteger,mustBePositive}
                a (1,1) double {mustBeInteger,mustBePositive}
                currentRealStep (1,1) double {mustBeInteger,mustBeNonnegative}
                lastRealVisit double = []
            end
            if isempty(lastRealVisit)
                obj.validatePair_(s,a); lastRealVisit=obj.LastRealVisit(s,a);
            end
            if ~(isscalar(lastRealVisit)&&isfinite(lastRealVisit)&&lastRealVisit>=0&&lastRealVisit==fix(lastRealVisit))
                error('OceanStats:RL:Dyna:Bonuses:BadLastVisit','Last real visit must be a non-negative integer.');
            end
            if lastRealVisit>currentRealStep
                error('OceanStats:RL:Dyna:Bonuses:FutureVisit','Last real visit cannot exceed current real step.');
            end
            bonus=obj.Kappa*sqrt(currentRealStep-lastRealVisit);
        end
        function onRealVisit(obj,s,a,currentRealStep)
            arguments
                obj
                s (1,1) double {mustBeInteger,mustBePositive}
                a (1,1) double {mustBeInteger,mustBePositive}
                currentRealStep (1,1) double {mustBeInteger,mustBeNonnegative}
            end
            obj.validatePair_(s,a); obj.LastRealVisit(s,a)=currentRealStep;
        end
        function reset(obj)
            valid=~isnan(obj.LastRealVisit); obj.LastRealVisit(valid)=0;
        end
        function snapshot=inspect(obj), snapshot=struct('Kappa',obj.Kappa,'LastRealVisit',obj.LastRealVisit); end
    end
    methods (Access=private)
        function validatePair_(obj,s,a)
            if isempty(obj.LastRealVisit)
                error('OceanStats:RL:Dyna:Bonuses:Unconfigured','Supply LastRealVisit or construct with a valid-action mask.');
            end
            if s>size(obj.LastRealVisit,1)||a>size(obj.LastRealVisit,2)
                error('OceanStats:RL:Dyna:Bonuses:PairOutOfRange','State/action index exceeds bonus dimensions.');
            end
            if isnan(obj.LastRealVisit(s,a))
                error('OceanStats:RL:Dyna:Bonuses:InvalidAction','The state-action pair is invalid.');
            end
        end
    end
end
