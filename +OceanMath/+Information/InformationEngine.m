classdef InformationEngine < handle
    % INFORMATIONENGINE  Orchestrator for the Information Theoretic architecture.
    %
    % This is the top-level class the end-user interacts with. It binds a
    % data context (an IDistribution or IStochasticProcess) with a pluggable
    % mathematical metric (an IMetricStrategy) and executes the calculation
    % via Dependency Injection.
    %
    % The engine never contains if-else logic for which formula to run.
    % It blindly delegates to whichever strategy is currently loaded.
    % This is the Command + Strategy Patterns working together.
    %
    % --- Quick-start (Fluent API) ---
    %
    %   % 1. Shannon entropy of a discrete distribution
    %   dist   = OceanMath.Information.Distributions.DiscreteDistribution.fromData([1 2 2 3 3 3]');
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(dist) ...
    %                .usingStrategy('ShannonEntropy') ...
    %                .run();
    %
    %   % 2. Entropy rate of a Markov chain
    %   P  = [0.7 0.3; 0.4 0.6];
    %   mc = OceanMath.Information.Processes.MarkovChain(P);
    %   result = OceanMath.Information.InformationEngine ...
    %                .withContext(mc) ...
    %                .usingStrategy('MarkovEntropyRate') ...
    %                .run();
    %
    %   % 3. Inject a custom strategy directly
    %   myStrat = MyCustomStrategy();
    %   engine  = OceanMath.Information.InformationEngine(mc, myStrat);
    %   result  = engine.run();
    %
    % --- Constructor ---
    %   engine = InformationEngine()                   - empty, configure later
    %   engine = InformationEngine(context)            - context set, no strategy
    %   engine = InformationEngine(context, strategy)  - fully configured
    %
    % See also: OceanMath.Information.Strategies.IMetricStrategy
    %           OceanMath.Information.Distributions.IDistribution
    %           OceanMath.Information.Processes.IStochasticProcess

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Context_  : the data/process object being analysed
        % Strategy_ : the IMetricStrategy implementation to run
        Context_  = []
        Strategy_ = []
    end

    % ------------------------------------------------------------------ %
    % Fluent builder methods — allow engine.withContext(...).usingStrategy(...).run()
    methods (Static)
        function engine = withContext(context)
            % WITHCONTEXT  Start a fluent chain with a given context.
            %
            % INPUTS:
            %   context - IDistribution or IStochasticProcess
            engine = OceanMath.Information.InformationEngine();
            engine = engine.setContext(context);
        end
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = InformationEngine(context, strategy)
            % INFORMATIONENGINE  Optionally inject context and/or strategy.
            arguments
                context  = []
                strategy = []
            end
            if ~isempty(context),  obj.Context_  = context;  end
            if ~isempty(strategy), obj.Strategy_ = strategy; end
        end

        % ---- Fluent setters (each returns obj for chaining) ---- %

        function obj = setContext(obj, context)
            % SETCONTEXT  Inject the data/process context. Chainable.
            obj.Context_ = context;
        end

        function obj = usingStrategy(obj, strategyOrName, options)
            % USINGSTRATEGY  Set the strategy by name alias or direct object.
            %
            % INPUTS:
            %   strategyOrName - either an IMetricStrategy instance,
            %                    or one of the built-in string aliases:
            %
            %   Alias                  -> Class
            %   "ShannonEntropy"       -> ShannonEntropyStrategy
            %   "MarkovEntropyRate"    -> MarkovEntropyRateStrategy
            %   "KLDivergence"         -> KLDivergenceStrategy (when implemented)
            %
            %   --- Coding metrics (context must be a CodeTable) ---
            %   "ExpectedLength"       -> ExpectedCodeLengthStrategy
            %   "Redundancy"           -> RedundancyStrategy
            %   "CodingEfficiency"     -> CodingEfficiencyStrategy
            %   "KraftSum"             -> KraftInequalityStrategy
            %   "CodingGap"            -> CodingGapStrategy
            %
            % Name-Value Options (forwarded to strategy constructor):
            %   Base (double): log base. Default: 2.
            arguments
                obj
                strategyOrName
                options.Base (1,1) double {mustBePositive} = 2
            end

            if ischar(strategyOrName) || isstring(strategyOrName)
                obj.Strategy_ = OceanMath.Information.InformationEngine.resolveStrategy_( ...
                    string(strategyOrName), options);
            elseif isa(strategyOrName, 'OceanMath.Information.Strategies.IMetricStrategy')
                obj.Strategy_ = strategyOrName;
            else
                error('OceanMath:Information:InformationEngine:BadStrategy', ...
                    'Expected a strategy alias string or IMetricStrategy instance. Got: %s', ...
                    class(strategyOrName));
            end
        end

        % ---- Execution ---- %

        function result = run(obj)
            % RUN  Validate and execute the configured strategy on the context.
            %
            % OUTPUTS:
            %   result - scalar (or struct) returned by the strategy

            obj.assertReady_();
            result = obj.Strategy_.safeCalculate(obj.Context_);
        end

        function results = runAll(obj, varargin)
            % RUNALL  Run multiple strategies on the same context.
            %
            % INPUTS:
            %   varargin - list of strategy aliases or IMetricStrategy objects
            %
            % OUTPUTS:
            %   results - struct where each field name is the strategy name
            %             and value is the calculated result
            %
            % Example:
            %   results = engine.runAll('ShannonEntropy', 'MarkovEntropyRate');
            %   results.ShannonEntropy    % H(X)
            %   results.MarkovEntropyRate % entropy rate
            if isempty(obj.Context_)
                error('OceanMath:Information:InformationEngine:NoContext', ...
                    'No context loaded. Call setContext() first.');
            end

            results = struct();
            for k = 1:numel(varargin)
                s = varargin{k};

                % Resolve alias to strategy object if needed
                if ischar(s) || isstring(s)
                    strat = OceanMath.Information.InformationEngine.resolveStrategy_( ...
                        string(s), struct('Base', 2));
                    fieldName = char(s);
                else
                    strat = s;
                    fieldName = strrep(strat.getName(), ' ', '_');
                end

                results.(fieldName) = strat.safeCalculate(obj.Context_);
            end
        end

        function S = encodeSummary(~, symbols, probs, options)
            % ENCODESUMMARY  One-shot Huffman encode + run all coding metrics.
            %
            % Convenience static-style method: encodes the source with
            % HuffmanCoder and immediately returns all five coding metrics.
            %
            % INPUTS:
            %   symbols - (K x 1) cell of symbol labels
            %   probs   - (1 x K) probabilities summing to 1
            %
            % Name-Value Options:
            %   Base (double): code base. Default: 2.
            %
            % OUTPUTS:
            %   S.table           - CodeTable from HuffmanCoder
            %   S.metrics         - struct: ExpectedLength, Redundancy,
            %                       CodingEfficiency, KraftSum, CodingGap
            arguments
                ~
                symbols (:,1) cell
                probs   (1,:) double {mustBeNonnegative}
                options.Base (1,1) double {mustBePositive, mustBeInteger} = 2
            end
            coder  = OceanMath.Information.Coding.HuffmanCoder('Base', options.Base);
            table  = coder.encode(symbols, probs);
            engine = OceanMath.Information.InformationEngine(table);
            S.table   = table;
            S.metrics = engine.runAll( ...
                'ExpectedLength', 'Redundancy', ...
                'CodingEfficiency', 'KraftSum', 'CodingGap');
        end

        % ---- Display ---- %

        function disp(obj)
            fprintf('InformationEngine\n');
            if isempty(obj.Context_)
                fprintf('  Context  : (none)\n');
            else
                fprintf('  Context  : %s\n', class(obj.Context_));
            end
            if isempty(obj.Strategy_)
                fprintf('  Strategy : (none)\n');
            else
                fprintf('  Strategy : %s\n', obj.Strategy_.getName());
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Static, Access = private)
        function strategy = resolveStrategy_(alias, options)
            % RESOLVESTRATEGY_  Map a string alias to a concrete strategy instance.
            switch lower(alias)
                case {'shannonentropy', 'shannon', 'entropy'}
                    strategy = OceanMath.Information.Strategies.ShannonEntropyStrategy( ...
                        'Base', options.Base);

                case {'markoventropyrate', 'entropyrate', 'markovrate'}
                    strategy = OceanMath.Information.Strategies.MarkovEntropyRateStrategy( ...
                        'Base', options.Base);

                % ---- Coding metrics (CodeTable context) ---- %
                case {'expectedlength', 'expectedcodelength', 'lavg'}
                    strategy = OceanMath.Information.Strategies.ExpectedCodeLengthStrategy();

                case {'redundancy', 'coderedundancy'}
                    strategy = OceanMath.Information.Strategies.RedundancyStrategy();

                case {'codingefficiency', 'efficiency', 'eta'}
                    strategy = OceanMath.Information.Strategies.CodingEfficiencyStrategy();

                case {'kraftsum', 'kraft', 'kraftinequality'}
                    strategy = OceanMath.Information.Strategies.KraftInequalityStrategy();

                case {'codinggap', 'gap'}
                    strategy = OceanMath.Information.Strategies.CodingGapStrategy();

                otherwise
                    error('OceanMath:Information:InformationEngine:UnknownStrategy', ...
                        ['Unknown strategy alias: "%s". ' ...
                         'Known aliases: ShannonEntropy, MarkovEntropyRate, ' ...
                         'ExpectedLength, Redundancy, CodingEfficiency, KraftSum, CodingGap. ' ...
                         'Pass an IMetricStrategy object directly for custom strategies.'], ...
                        alias);
            end
        end
    end

    methods (Access = private)
        function assertReady_(obj)
            if isempty(obj.Context_)
                error('OceanMath:Information:InformationEngine:NoContext', ...
                    'No context loaded. Call setContext() or use withContext().');
            end
            if isempty(obj.Strategy_)
                error('OceanMath:Information:InformationEngine:NoStrategy', ...
                    'No strategy set. Call usingStrategy() with an alias or object.');
            end
        end
    end
end
