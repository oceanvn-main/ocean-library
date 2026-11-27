classdef Dimension
    % DIMENSION  Physical dimension as integer exponents over (L length, M mass, T time, E energy, C currency, N count).
    % Dimensions are compared and combined by algebra on the exponent vector, never by name - this is how the guard
    % tells a speed (L^1 T^-1) from a fuel amount (E^1) without reading any label.
    %
    %   D = OceanAgents.World.Guard.Dimension;  v = D.L() / D.T();                                % L^1 T^-1
    %   v == OceanAgents.World.Guard.Dimension([1 0 -1 0 0 0])                                   % true
    % Reference: Kennedy, "Dimension types", ESOP 1994. Contract: W1_Kernel_Contract.md section 3.1.

    properties (SetAccess = immutable)
        Exponents (1,6) double = zeros(1, 6)
    end

    properties (Constant)
        Bases = ["L" "M" "T" "E" "C" "N"]
    end

    methods
        function obj = Dimension(exponents)
            if nargin == 0, return; end
            if ~(isnumeric(exponents) && isequal(size(exponents), [1 6]) && all(isfinite(exponents)) ...
                    && all(exponents == fix(exponents)))
                error('OceanAgents:Guard:BadDimension', 'A dimension is a 1x6 vector of integer exponents (L M T E C N).');
            end
            obj.Exponents = double(exponents);
        end

        function c = times(a, b), c = OceanAgents.World.Guard.Dimension(a.Exponents + b.Exponents); end
        function c = mtimes(a, b), c = times(a, b); end
        function c = rdivide(a, b), c = OceanAgents.World.Guard.Dimension(a.Exponents - b.Exponents); end
        function c = mrdivide(a, b), c = rdivide(a, b); end
        function c = power(a, k)
            if ~(isnumeric(k) && isscalar(k) && k == fix(k))
                error('OceanAgents:Guard:BadDimension', 'Dimensions can only be raised to integer powers.');
            end
            c = OceanAgents.World.Guard.Dimension(a.Exponents * k);
        end
        function c = mpower(a, k), c = power(a, k); end
        function tf = eq(a, b), tf = isequal(a.Exponents, b.Exponents); end
        function tf = ne(a, b), tf = ~eq(a, b); end
        function tf = isDimensionless(obj), tf = all(obj.Exponents == 0); end

        function s = char(obj)
            % CHAR  Human-readable form for messages ONLY (never compared by the guard).
            if obj.isDimensionless(), s = '1'; return; end
            parts = strings(1, 0);
            for k = find(obj.Exponents ~= 0)
                parts(end + 1) = sprintf('%s^%d', obj.Bases(k), obj.Exponents(k)); %#ok<AGROW>
            end
            s = char(strjoin(parts, ' '));
        end
        function s = string(obj), s = string(char(obj)); end
    end

    methods (Static)
        function d = none(), d = OceanAgents.World.Guard.Dimension(); end
        % Base dimensions by letter (a static named "length" would shadow MATLAB's length() on Dimension objects)
        function d = L(), d = OceanAgents.World.Guard.Dimension([1 0 0 0 0 0]); end   % length
        function d = M(), d = OceanAgents.World.Guard.Dimension([0 1 0 0 0 0]); end   % mass
        function d = T(), d = OceanAgents.World.Guard.Dimension([0 0 1 0 0 0]); end   % time
        function d = E(), d = OceanAgents.World.Guard.Dimension([0 0 0 1 0 0]); end   % energy
        function d = C(), d = OceanAgents.World.Guard.Dimension([0 0 0 0 1 0]); end   % currency
        function d = N(), d = OceanAgents.World.Guard.Dimension([0 0 0 0 0 1]); end   % count
    end
end
