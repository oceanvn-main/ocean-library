classdef BootstrapSampler < handle
    % BOOTSTRAPSAMPLER A generic bootstrap engine. 
    % This class provides a unified interface for both Non-parametric (resampling data)
    % and Parametric (resampling from model) bootstrap.
    % 
    % == USAGE ==
    %   sampler = OceanStats.Data.BootstrapSampler(data);
    %   
    %   % 1. Non-parametric (Pairs/Resampling indices)
    %   %    statisticFcn: function handle that takes (dataSubset) -> statistics
    %   stats = sampler.drawNonParametric(@myStatistic, 'Replicates', 1000);
    %   
    %   % 2. Parametric (Resampling from fitted model)
    %   %    fitFcn:      (data) -> params
    %   %    genFcn:      (params, n) -> syntheticData
    %   %    statisticFcn: (data) -> statistics
    %   stats = sampler.drawParametric(@myFit, @myGen, @myStatistic, 'Replicates', 1000);
    %   
    % Author: Ocean Library System
    % Date: 2026-02-10

    properties (SetAccess = private)
        % The input data (Matrix or Table)
        % If Matrix: Rows are observations.
        % If Table: Rows are observations.
        Data
        
        % Sample size (Number of rows)
        N (1, 1) double {mustBeNonnegative, mustBeInteger}
    end

    methods
        function obj = BootstrapSampler(data)
            % BOOTSTRAPSAMPLER Constructor.
            %
            % Inputs:
            %   data - Numeric matrix or Table. Rows are observations.
            arguments
                data {mustBeNonempty}
            end
            obj.Data = data;
            obj.N = size(data, 1);          
        end

        function samples = drawNonParametric(obj, statisticFcn, options)
            % DRAWNONPARAMETRIC Standard Resampling (Pairs Bootstrap).
            % Resamples rows of Data with replacement and applies statisticFcn.
            %
            % Inputs:
            %   statisticFcn - Handle: @(dataSubset) -> [1 x K] results
            %   'Replicates' - Number of bootstrap estimates.
            %
            % Outputs:
            %   samples - (Replicates x K) matrix of bootstrap estimates.

            arguments
                obj
                statisticFcn function_handle
                options.Replicates (1, 1) double {mustBePositive, mustBeInteger} = 1000
                options.OutputFormat string {mustBeMember(options.OutputFormat, ["numeric", "cell"])} = "numeric"
            end

            B = options.Replicates;
            n_obs = obj.N;
            use_cell = strcmp(options.OutputFormat, "cell");

            % --- Dry Run ---
            idx_dry = randsample(n_obs, n_obs, true);
            val_dry = statisticFcn(obj.subset(idx_dry));
            

            
            % Always use cell array for parfor stability
            % (MATLAB parfor handles cell slicing robustly)
            samples_cell = cell(B, 1);

            %Extract data for efficiency parfor broadcasting
            inputData = obj.Data;
            
            % Determine parallel strategy
            % Only use parfor if the job is significant
            if B > 100
                numWorkers = Inf; 
            else
                numWorkers = 0;   
            end

            %Main loop
            parfor (b = 1:B, numWorkers)
                % Resample
                idx = randsample(n_obs, n_obs, true);
                data_b = inputData(idx, :);

                % Compute
                res = statisticFcn(data_b);
                
                % Store in cell
                samples_cell{b} = res;
            end
            
            % Post-process: Convert to request format
            if strcmp(options.OutputFormat, "cell")
                samples = samples_cell;
            else
                % Numeric: flexible conversion
                % If res is [1 x K], cell2mat works perfectly
                try
                    samples = cell2mat(samples_cell);
                    % Ensure shape is [B x K]
                    % cell2mat on { [1xK]; [1xK] } -> [B x K]
                catch ME
                    error("BootstrapSampler:ConversionFailed", ...
                        "Failed to convert bootstrap results to numeric matrix. Use 'OutputFormat', 'cell'.\nDetails: %s", ME.message);
                end
            end
        end  

    function samples = drawParametric(obj, fitFcn, genFcn, statisticFcn, options)
        % DRAWPARAMETRIC Model-Based Bootstrap
        %   1. Fit models to original data: params = fitFcn(Data)
        %   2. Generate B datasets: data* = genFcn(params, N)
        %   3. Computes statistic: stat* = statisticFcn(data*)
        % Inputs:
        %   fitFcn      - Handle: @(data) - > params(struct or vector)
        %   getFcn      - Handle: @(params, n) -> syntheticData
        %   statisticFcn - Handle: @(data) -> [1 x K] results
        %   'Replicates' - Default 1000
        %Ouputs:
        %   samples - (Replicates x K) matrix.
        arguments
            obj
            fitFcn function_handle
            genFcn function_handle
            statisticFcn function_handle
            options.Replicates (1, 1) double {mustBePositive, mustBeInteger} = 1000
        end
        B = options.Replicates;
        n_obs = obj.N;

        %1. Fit Model Once
        params = fitFcn(obj.Data);

        %Dry Run
        data_dry = genFcn(params, n_obs);
        val_dry = statisticFcn(data_dry);
        k = numel(val_dry);
        samples = zeros(B, k);

        % Determine parallel strategy
        if B > 1000
            numWorkers = Inf; 
        else
            numWorkers = 0;   
        end

        %Main loop (parallel)
        parfor (b = 1:B, numWorkers)
            data_b = genFcn(params, n_obs);

            %Compute Statistic
            res = statisticFcn(data_b);
            samples(b, :) = res(:)';
        end

    end


    end

    methods (Access = private)
        function d_sub = subset(obj, indices)
            % SUBSET Helper to index into Data (Matrix or Table)
            if istable(obj.Data)
                d_sub = obj.Data(indices, :);
            else
                d_sub = obj.Data(indices, :);
            end
        end
    end
end
