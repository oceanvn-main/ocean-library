classdef Flatten < handle
    % FLATTEN  Reshapes a spatial (H x W x C x N) tensor to dense (D x N),
    %   D = H*W*C -- the bridge between the Conv package's spatial
    %   (H,W,C,N) convention and the rest of +OceanDL's dense
    %   (d x N) column-major convention (matches VectorizedEngine/
    %   BackpropEngine exactly, so a Dense head composes naturally after it).
    %
    %   Buffer-on-self forward/backward pairing, same contract as Conv2D/
    %   MaxPool2D: forward() caches the input size; backward() reshapes back.
    %
    %   USAGE:
    %     flat = OceanDL.Conv.Flatten();
    %     A    = flat.forward(X);     % X: (H x W x C x N) -> A: (D x N)
    %     dX   = flat.backward(dA);   % dA: (D x N) -> dX: (H x W x C x N)

    properties (Access = private)
        InputSize_ = []   % [H W C N] of the most recent forward()
    end

    methods
        function A = forward(obj, X)
            arguments
                obj
                X (:,:,:,:) double
            end
            [H, W, C, N] = size(X);
            obj.InputSize_ = [H, W, C, N];
            A = reshape(X, H*W*C, N);
        end

        function dX = backward(obj, dA)
            arguments
                obj
                dA (:,:) double
            end
            if isempty(obj.InputSize_)
                error('OceanDL:Conv:Flatten:NoForwardCache', ...
                    'backward() called before forward().');
            end
            dX = reshape(dA, obj.InputSize_);
        end
    end
end
