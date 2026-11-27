classdef ModelFactory
    %MODELFACTORY Static factory for creating ML Strategy objects.
    %   USAGE:
    %       strat = OceanStats.Selection.Strategies.ModelFactory.create('Tree');
    %       strat = OceanStats.Selection.Strategies.ModelFactory.create('Linear');

    methods (Static)
        function strat = create(type)
            arguments
                type (1, 1) string
            end
            switch lower(type)
                case {'linear', 'ols', 'regression', 'linreg'}
                    strat = OceanStats.Selection.Strategies.LinearReg();
                case {'tree', 'cart', 'decisiontree', 'ctree'}
                    strat = OceanStats.Selection.Strategies.DecisionTree();
                case {'rtree', 'regressiontree'}
                    strat = OceanStats.Selection.Strategies.RegressionTree();
                case {'rf', 'randomforest', 'random forest'}
                    strat = OceanStats.Selection.Strategies.RandomForestStrategy();
                case {'et', 'extratrees', 'extra trees', 'extra-trees'}
                    strat = OceanStats.Selection.Strategies.ExtraTreesStrategy();
                case {'knn', 'k-nn', 'knearest'}
                    strat = OceanStats.Selection.Strategies.KNNStrategy();
                % Future expansion placeholders
                % case 'svm', strat = OceanStats.Selection.Strategies.SVM();
                % case 'glm', strat = OceanStats.Selection.Strategies.GLM();
                otherwise
                    error('OceanStats:Factory:UnknownType', ...
                        'Unknown model type: "%s". Supported: Linear, Tree.', type);
            end
        end

    end
end
