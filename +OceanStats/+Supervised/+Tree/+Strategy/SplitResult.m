classdef SplitResult
    % SPLITRESULT Represents the outcome of a split decision.
    %
    % This class abstracts the "routing" logic. The Tree does not need to know
    % if the split is "Threshold < 5" or "Category == 'Red'". It just asks
    % "Where does this row go?".
    %
    % Properties:
    %   IsTerminal (logical): True if this node should be a leaf (no split found).
    %   Description (string): Human-readable description of the split.
    %   FeatureIndex (double): Index of feature used (for importance tracking).
    %   Gain (double): Improvement/Gain from this split.
    %
    % Methods:
    %   route(X_row) -> childIndex (1..K)
    %
    % Author: Ocean Library System
    % Date: 2026-02-17
    
    properties
        IsTerminal (1,1) logical = true
        Description (1,1) string = "Terminal"
        FeatureIndex (1,1) double = NaN
        Gain (1,1) double = 0
        
        % Common Properties for implementations:
        Threshold (1,1) double = NaN       % For Binary: X <= Threshold -> 1, Else -> 2
        Categories cell = {}               % For Multiway: List of unique levels
        CategoryMap containers.Map         % For Multiway: Map level -> childIndex
        RouteFn function_handle            % Custom routing: @(x) -> index
        
        % Metadata for pruning/display
        LeftChildName string = "Left"
        RightChildName string = "Right"
        
        % Surrogates: Alternative splits to use if primary key is missing
        Surrogates = [] % Array of SplitResult objects
    end
    
    methods
        function obj = SplitResult(isTerminal)
            if nargin > 0
                obj.IsTerminal = isTerminal;
            end
        end
        
        function childIdx = route(obj, x_row)
            % ROUTE Determine which child node a sample belongs to.
            %
            % Input:
            %   x_row: A single row of data (1 x P)
            % Output:
            %   childIdx: Integer 1..K (or NaN if missing & no surrogate)
            
            if obj.IsTerminal
                childIdx = 0; 
                return;
            end
            
            val = x_row(obj.FeatureIndex);
            
            % Check for Missing Value
            isMissing = false;
            if isnumeric(val) && isnan(val)
                isMissing = true;
            elseif isstring(val) && ismissing(val)
                isMissing = true;
            elseif iscategorical(val) && isundefined(val)
                isMissing = true;
            end
            
            if isMissing
                % Try Surrogates
                if ~isempty(obj.Surrogates)
                    for k = 1:length(obj.Surrogates)
                        surr = obj.Surrogates(k);
                        % Route using surrogate
                        % Ensure surrogate feature is valid before routing
                        
                        sVal = x_row(surr.FeatureIndex);
                        sMissing = (isnumeric(sVal) && isnan(sVal)) || ...
                                   (isstring(sVal) && ismissing(sVal)) || ...
                                   (iscategorical(sVal) && isundefined(sVal));
                                   
                        if ~sMissing
                            try
                                childIdx = surr.route(x_row);
                                if ~isnan(childIdx)
                                    % NOTIFY USER as requested
                                    fprintf('NOTICE: Missing Feature %d. Using Surrogate Feature %d (%s).\n', ...
                                        obj.FeatureIndex, surr.FeatureIndex, surr.Description);
                                    return;
                                end
                            catch
                                % Ignore surrogate errors, try next
                            end
                        end
                    end
                end
                
                % Still missing? Return NaN (Tree handles via Majority/Imputation)
                childIdx = NaN;
                return;
            end
            
            % 1. Standard Binary Split (Fast path)
            if ~isnan(obj.Threshold)
                if val <= obj.Threshold
                    childIdx = 1; % Left
                else
                    childIdx = 2; % Right
                end
                return;
            end
            
            % 2. Categorical / Multi-way Split
            if ~isempty(obj.Categories) && ~isempty(obj.CategoryMap)
                try
                    key = val;
                    if ischar(key), key = string(key); end
                    if obj.CategoryMap.isKey(key)
                        childIdx = obj.CategoryMap(key);
                        return;
                    end
                catch
                    % Fallback to missing logic below
                end
            end
            
            % 3. Custom Function (Generic)
            if ~isempty(obj.RouteFn)
                childIdx = obj.RouteFn(val);
                return;
            end
            
            error('OceanStats:SplitResult:NoRoute', 'No routing logic defined.');
        end
    end
end
