classdef PolyMARSForwardSelector < OceanStats.Base.IForwardPass
    % POLYMARSFORSARDSELECTOR Forward pass strategy for PolyMARS.
    %
    % Implements the PolyMARS algorithm (Stone et al., 1997) as described in
    % ESL Section 9.4.4. Maximizes Multinomial Log-Likelihood (or any objective
    % defined by the injected Link function) via an inner Score Test search.
    %
    % INJECTABLE STRATEGIES:
    %   Link          (ILinkFunction)          — default: SoftmaxLink
    %   Approximation (IObjectiveApproximation) — default: ScoreTestApproximation
    %
    % INPUTS to build():
    %   X       - (N x P) feature matrix
    %   Y       - (N x K) encoded response matrix (from IDataEncoder)
    %   options - struct with MaxTerms, MaxInteractionDegree, CategoricalFeatures
    
    properties
        % Inverse-link: maps linear scores to probabilities, provides weights.
        Link OceanStats.Base.ILinkFunction = OceanStats.Link.SoftmaxLink()
        
        % Pluggable gain approximation strategy (Score Test by default).
        Approximation OceanStats.Base.IObjectiveApproximation = ...
            OceanMath.Approximation.ScoreTestApproximation()
    end
    
    methods
        function basisList = build(obj, X, Y, options)
            import OceanStats.Supervised.MARS.Basis.*
            [N, P] = size(X);
            K      = size(Y, 2);
            
            %%% MATH: THE NULL MODEL %%%
            % We initialize the design matrix B with just the intercept.
            % B = [1] (an N x 1 matrix of ones).
            basisList = { ConstantBasis() };
            B         = ones(N, 1);
            
            beta  = zeros(1, K);
            
            %%% MATH: CALCULATING THE INITIAL STATE %%%
            % 1. Raw Score (theta): theta = B * beta
            % 2. Probabilities: P = Link(theta) 
            P_mat = obj.Link.apply(B * beta);       % (N x K)
            
            % 3. The Inner Derivative (Score / Error): s = dL/d(theta) = Y - P
            s     = Y - P_mat;                      % (N x K)
            
            % 4. The Fisher Variance: W = P * (1 - P)
            w     = obj.Link.weights(P_mat);        % (N x 1)
            
            % Boot up the caching engine to solve the initial Gradient and Hessian
            wls = OceanMath.LinearAlgebra.IncrementalWLS(B, w, s);
            
            fprintf('PolyMARS Forward Pass: Growing up to %d terms...\n', options.MaxTerms);
            
            while length(basisList) < options.MaxTerms - 1
                bestGain = -inf;
                bestNew1 = []; bestNew2 = [];
                best_h1  = []; best_h2  = [];
                
                %%% MATH: THE PARENT LOOP (INTERACTIONS) %%%
                % MARS builds interactions by multiplying a new split by an existing
                % basis function. If parent = 1, it's a main effect. If parent = X1>5,
                % the new term is an interaction.
                for m = 1:length(basisList)
                    parent = basisList{m};
                    if parent.getDegree() >= options.MaxInteractionDegree; continue; end
                    parent_eval = B(:, m);
                    
                    for j = 1:P
                        x_j = X(:, j);
                        
                        % Categorical Features
                        if isfield(options, 'CategoricalFeatures') && ismember(j, options.CategoricalFeatures)
                            unique_vals = unique(x_j);
                            Kc = length(unique_vals);
                            if Kc > 15 || Kc < 2; continue; end
                            
                            %%% MATH: COMBINATORIAL SPLITS %%%
                            % We must test every way to split Kc categories into 2 groups.
                            % Number of unique splits = 2^(Kc - 1) - 1.
                            for p_idx = 1:(2^(Kc-1) - 1)
                                bin_str = dec2bin(p_idx, Kc);
                                subset1 = []; subset2 = [];
                                for kk = 1:Kc
                                    if bin_str(kk)=='1'; subset1=[subset1,unique_vals(kk)]; %#ok<AGROW>
                                    else;                subset2=[subset2,unique_vals(kk)]; end %#ok<AGROW>
                                end
                                
                                % Create the binary indicator vectors, multiplied by parent
                                h1_val = parent_eval .* double(ismember(x_j, subset1));
                                h2_val = parent_eval .* double(ismember(x_j, subset2));
                                
                                % Statistical Guardrail: Require minimum support
                                if sum(h1_val>0)<5 || sum(h2_val>0)<5; continue; end
                                
                                %%% MATH: THE RAO SCORE TEST %%%
                                % Instead of doing a full matrix inversion for every candidate,
                                % we use the Taylor Expansion shortcut: Gain = (Gradient^2) / Curvature
                                gain = obj.Approximation.computeGain(wls,h1_val,w,s) + ...
                                       obj.Approximation.computeGain(wls,h2_val,w,s);
                                       
                                if gain > bestGain
                                    bestGain=gain; best_h1=h1_val; best_h2=h2_val;
                                    isC = isa(parent,'OceanStats.Supervised.MARS.Basis.ConstantBasis');
                                    b1=CategoricalBasis(j,subset1); b2=CategoricalBasis(j,subset2);
                                    if isC; bestNew1=b1; bestNew2=b2;
                                    else;   bestNew1=ProductBasis(parent,b1); bestNew2=ProductBasis(parent,b2); end
                                end
                            end
                            continue;
                        end
                        
                        % Continuous Features
                        uv = unique(x_j);
                        if length(uv) > 50; t_cands = quantile(uv, linspace(0.05,0.95,50));
                        else;               t_cands = uv; end
                        
                        for i = 1:length(t_cands)
                            t = t_cands(i);
                            
                            %%% MATH: CONTINUOUS SPLINE EQUATIONS (HINGES) %%%
                            % h1 = parent * max(x - t, 0)
                            % h2 = parent * max(t - x, 0)
                            h1_val = parent_eval .* max(x_j - t, 0);
                            h2_val = parent_eval .* max(t - x_j, 0);
                            
                            if sum(h1_val>0)<5 || sum(h2_val>0)<5; continue; end
                            
                            %%% MATH: THE RAO SCORE TEST %%%
                            % Gain = (Gradient^2) / Curvature
                            gain = obj.Approximation.computeGain(wls,h1_val,w,s) + ...
                                   obj.Approximation.computeGain(wls,h2_val,w,s);
                                   
                            if gain > bestGain
                                bestGain=gain; best_h1=h1_val; best_h2=h2_val;
                                hinge1=HingeBasis(j,t,1); hinge2=HingeBasis(j,t,-1);
                                isC=isa(parent,'OceanStats.Supervised.MARS.Basis.ConstantBasis');
                                if isC; bestNew1=hinge1; bestNew2=hinge2;
                                else;   bestNew1=ProductBasis(parent,hinge1); bestNew2=ProductBasis(parent,hinge2); end
                            end
                        end
                    end
                end
                
                if isempty(bestNew1); break; end
                
                %%% MATH: MODEL EXPANSION AND RECALIBRATION %%%
                
                % 1. Add the winning splines to the Design Matrix (B = [B_old, H_new])
                basisList{end+1} = bestNew1; %#ok<AGROW>
                basisList{end+1} = bestNew2; %#ok<AGROW>
                H_new = [best_h1, best_h2];
                B     = [B, H_new];
                
                % 2. Update the cached WLS matrices (Gradient and Hessian) using block algebra
                wls.addColumns(H_new, w, s);
                
                % 3. Run a full Newton-Raphson update step to find the optimal weights:
                % beta = (Hessian)^-1 * (Gradient)
                beta  = wls.solveForBeta();
                
                % 4. Recalculate reality with the new weights
                P_mat = obj.Link.apply(B * beta);
                s     = Y - P_mat;                      % New Error (dL/dtheta)
                w     = obj.Link.weights(P_mat);        % New Variance P(1-P)
                
                % 5. Overwrite the WLS cache with the new, perfectly calibrated matrices
                wls.refresh(B, w, s);
                
                fprintf('  Added terms %d & %d. Score Gain: %.4f\n', ...
                    length(basisList)-1, length(basisList), bestGain);
            end
        end
    end
end