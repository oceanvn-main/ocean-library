classdef PrivateHolder < handle
    % PRIVATEHOLDER  Test fixture: keeps a value in a PRIVATE property, to check that the isolation checker
    % sees through access restrictions (struct(obj)).
    properties (Access = private)
        Secret = []
    end
    methods
        function obj = PrivateHolder(secret)
            if nargin > 0, obj.Secret = secret; end
        end
        function setSecret(obj, secret)
            obj.Secret = secret;
        end
    end
end
