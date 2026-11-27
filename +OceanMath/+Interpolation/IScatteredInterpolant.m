classdef (Abstract) IScatteredInterpolant < handle
    % ISCATTEREDINTERPOLANT  Abstract strategy interface for N-D scattered
    %   data interpolation.
    %
    %   All concrete interpolants must implement:
    %     fitCore_(obj, X, y)      — store model from training data
    %     y_hat = predictCore_(obj, Xq)  — evaluate at query points
    %
    %   The public fit() / predict() methods are provided by this base class
    %   and handle dimension validation + state guarding.
    %
    %   Contract:
    %     X   : (N x D) double  — training site coordinates
    %     y   : (N x 1) double  — observed values at X
    %     Xq  : (M x D) double  — query site coordinates
    %     y_hat : (M x 1) double — interpolated values at Xq
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties (Access = protected)
        X_train_   % (N x D) stored training sites
        y_train_   % (N x 1) stored training values
        D_         % Dimensionality of input space
        IsFitted_  (1,1) logical = false
    end

    % ── Public API ────────────────────────────────────────────────────────
    methods

        function fit(obj, X, y)
            % FIT  Fit the interpolant to scattered training data.
            %
            %   obj.fit(X, y)
            %
            %   Inputs:
            %     X  - (N x D) training coordinates
            %     y  - (N x 1) observed scalar values

            arguments
                obj
                X (:,:) double
                y (:,1) double
            end

            N = size(X, 1);
            if size(y, 1) ~= N
                error('IScatteredInterpolant:DimMismatch', ...
                    'X has %d rows but y has %d rows.', N, size(y,1));
            end
            if N < 2
                error('IScatteredInterpolant:TooFewPoints', ...
                    'At least 2 training points are required.');
            end

            obj.X_train_ = X;
            obj.y_train_ = y;
            obj.D_       = size(X, 2);
            obj.IsFitted_ = false;   % Reset before fit attempt

            obj.fitCore_(X, y);
            obj.IsFitted_ = true;
        end

        function y_hat = predict(obj, Xq)
            % PREDICT  Evaluate the interpolant at query locations.
            %
            %   y_hat = obj.predict(Xq)
            %
            %   Inputs:
            %     Xq  - (M x D) query coordinates
            %
            %   Outputs:
            %     y_hat - (M x 1) interpolated values

            arguments
                obj
                Xq (:,:) double
            end

            if ~obj.IsFitted_
                error('IScatteredInterpolant:NotFitted', ...
                    'Call fit() before predict().');
            end
            if size(Xq, 2) ~= obj.D_
                error('IScatteredInterpolant:DimMismatch', ...
                    'Query dimensionality %d does not match training dimensionality %d.', ...
                    size(Xq,2), obj.D_);
            end

            y_hat = obj.predictCore_(Xq);
        end

        function tf = isFitted(obj)
            % ISFITTED  Returns true if fit() has been successfully called.
            tf = obj.IsFitted_;
        end

        function D = getDim(obj)
            % GETDIM  Returns the input space dimensionality.
            D = obj.D_;
        end

    end

    % ── Abstract Hooks (subclasses implement these) ───────────────────────
    methods (Abstract, Access = protected)

        % FITCORE_  Perform the actual model fitting.
        %   Subclass receives validated (X, y) and stores any model state.
        fitCore_(obj, X, y)

        % PREDICTCORE_  Evaluate model at validated query points Xq.
        %   Returns (M x 1) predictions.
        y_hat = predictCore_(obj, Xq)

    end

    methods (Abstract)
        % GETNAME  Returns a short human-readable identifier string.
        name = getName(obj)
    end

end
