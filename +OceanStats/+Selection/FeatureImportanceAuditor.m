classdef FeatureImportanceAuditor < handle
    % FEATUREIMPORTANCEAUDITOR High-level orchestrator for Feature Selection & Validation.
    %
    % OVERVIEW:
    %   Orchestrates the analysis by combining a Feature Selector (Search)
    %   with one or more Validation Strategies (NestedCV, Bootstrap).
    
    properties
        Selector OceanStats.Base.IFeatureSelector
        Validators cell % Cell array of OceanStats.Base.IValidator
    end
    
    methods
        function obj = FeatureImportanceAuditor(selector, validators)
            % Constructor
            % Inputs:
            %   selector   : IFeatureSelector (Default: BestSubsetSelector)
            %   validators : Cell array of IValidator or single IValidator
            
            if nargin < 1 || isempty(selector)
                obj.Selector = OceanStats.Selection.FeatureSelection.BestSubsetSelector();
            else
                obj.Selector = selector;
            end
            
            if nargin < 2 || isempty(validators)
                % Default to K-Fold (5)
                obj.Validators = {OceanStats.Selection.Validation.NestedCV(OceanStats.Selection.CrossValidation.KFold(5))};
            elseif ~iscell(validators)
                obj.Validators = {validators};
            else
                obj.Validators = validators;
            end
        end
        
        function report = run(obj, X, Y, strategy, featureNames)
            [N, P] = size(X);
            if nargin < 5, featureNames = "Feat" + (1:P); end
            
            fprintf('Auditor: Analyzing Feature Importance (%d features)...\n', P);
            
            % 1. Global Search
            fprintf(' > Global Search...\n');
            global_history = obj.Selector.select(X, Y, strategy); 
            % Note: select usually returns mask, but we need history for plotting.
            % BestSubsetSelector returns (mask, history).
            [~, global_history] = obj.Selector.select(X, Y, strategy);

            % 2. Run Validators
            validation_results = struct();
            
            % Prepare Table
            tbl = table((0:P)', 'VariableNames', {'K'});
            
            % Add Training Error from Global Search
            if ~isempty(global_history)
                 tbl.TrainingError = [global_history.Error]' / N;
                 % Feature Names
                 feats = strings(P+1, 1);
                 for i = 1:(P+1)
                     m = global_history(i);
                     if m.Size == 0, feats(i) = "(Null)"; else, feats(i) = strjoin(featureNames(m.Mask), ', '); end
                 end
                 tbl.Features = feats;
            end
            
            for i = 1:length(obj.Validators)
                v = obj.Validators{i};
                res = v.validate(obj.Selector, strategy, X, Y);
                validation_results.(regexprep(v.Name, '\W', '_')) = res;
                
                % Append to Table if it matches K structure
                if isfield(res, 'MeanError')
                    tbl.CV_Error = res.MeanError;
                    tbl.CV_SE = res.SE;
                elseif isfield(res, 'Error')
                    tbl.Bootstrap_Error = res.Error;
                end
            end
            
            report = struct();
            report.Table = tbl;
            report.GlobalHistory = global_history;
            report.Validation = validation_results;
            report.Data = struct('X', X, 'Y', Y, 'Names', featureNames);
            
            fprintf('Auditor: Complete.\n');
        end
        
        function plot(obj, report, viz)
            viz.createFigure('Audit: Feature Importance');
            ax = gca;
            
            if ismember('CV_Error', report.Table.Properties.VariableNames)
                errorbar(ax, report.Table.K, report.Table.CV_Error, report.Table.CV_SE, '-o', ...
                    'LineWidth', 2, 'DisplayName', 'Nested CV Error');
                hold(ax, 'on');
            end
            
            if ismember('Bootstrap_Error', report.Table.Properties.VariableNames)
                plot(ax, report.Table.K, report.Table.Bootstrap_Error, '-s', ...
                    'LineWidth', 2, 'DisplayName', '.632+ Bootstrap');
                 hold(ax, 'on');
            end
            
            if ismember('TrainingError', report.Table.Properties.VariableNames)
                plot(ax, report.Table.K, report.Table.TrainingError, '--', ...
                    'DisplayName', 'Training Error (Biased)');
                 hold(ax, 'on');
            end
            
            xlabel('Number of Features');
            ylabel('MsE / Prediction Error');
            title('Feature Selection Stability');
            legend(ax);
            viz.styleAxes(ax);
            hold(ax, 'off');
        end
    end
end
