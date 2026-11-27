classdef IDataEncoder < handle
    % IDATAENCODER Interface for encoding and decoding the response variable Y.
    %
    % Separates the responsibility of Y preparation (classification detection,
    % indicator matrix creation, label storage) from MARS.fit() logic.
    %
    % After encode() is called, the encoder holds any metadata needed to
    % reverse the transformation in decode().
    %
    % Implementations:
    %   IndicatorEncoder — converts class labels to N×K indicator matrix
    %   RawEncoder       — pass-through for regression
    
    methods (Abstract)
        % ENCODE Transform raw Y into the format the forward pass expects.
        %
        % INPUTS:
        %   Y - (N x 1) raw response (labels, numeric, categorical, etc.)
        %
        % OUTPUTS:
        %   Y_enc - (N x K) encoded response matrix
        Y_enc = encode(obj, Y)
        
        % DECODE Transform linear model scores back to the original output space.
        %
        % INPUTS:
        %   scores - (N x K) raw linear scores (B * Coefficients)
        %
        % OUTPUTS:
        %   y_pred - (N x 1) predictions in original space (labels or numeric)
        y_pred = decode(obj, scores)
        
        % ISCLASSIFICATION Returns true if this encoder represents a classifier.
        tf = isClassification(obj)
    end
end
