classdef BiasVarianceAuditor < handle
    % BIASVARIANCEAUDITOR High-level orchestrator for Complexity Analysis.
    % (Delegates to ComplexitySweeper)
    
    properties
        Sweeper
    end
    
    methods
        function obj = BiasVarianceAuditor()
            obj.Sweeper = OceanStats.Selection.HyperParameter.ComplexitySweeper();
        end
        
        function report = run(obj, X, Y, strategyFactory, range)
            % RUN Executes the audit.
            fprintf('Auditor: Running Complexity Sweep...\n');
            tbl = obj.Sweeper.sweep(X, Y, strategyFactory, 'Complexity', range);
            
            % Package Report
            report = struct();
            report.Table = tbl;
            report.Best = tbl.Properties.UserData; % Extract struct from UserData
            report.Data = struct('X', X, 'Y', Y);
            report.StrategyFactory = strategyFactory;
            report.Range = range;
            
            fprintf('Auditor: Complete. Best AIC=%d, Best BIC=%d\n', ...
                report.Best.BestAIC, report.Best.BestBIC);
        end
        
        function plot(obj, report, viz)
            % Visualize Metrics
            viz.createFigure('Audit: Bias-Variance Metrics');
            ax = gca;
            plot(ax, report.Table.Complexity, report.Table.AIC, '-o', 'DisplayName', 'AICc');
            hold(ax, 'on');
            plot(ax, report.Table.Complexity, report.Table.BIC, '-s', 'DisplayName', 'BIC');
            legend(ax); viz.styleAxes(ax); hold(ax, 'off');
        end
    end
end
