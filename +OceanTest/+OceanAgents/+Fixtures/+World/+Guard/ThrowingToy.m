classdef ThrowingToy < OceanTest.OceanAgents.Fixtures.World.Guard.GoodToy
    % THROWINGTOY  apply always errors: the TM-2 predicate throws, which must become a FAILED axiom with the error
    % message as detail, never an escaping exception.
    methods
        function y = apply(~, ~) %#ok<STOUT>
            error('OceanTest:Fixtures:ToyBroken', 'apply is not implemented');
        end
    end
end
