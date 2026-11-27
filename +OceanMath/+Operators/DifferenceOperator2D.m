classdef DifferenceOperator2D < OceanMath.Operators.IDifferenceOperator
    % DIFFERENCEOPERATOR2D  2D spatial difference operator for images and 2D grids.
    %
    %   Builds sparse D matrix (K x N) stacking horizontal diffs above vertical diffs:
    %     D * f(:) = [diff(f,1,2)(:); diff(f,1,1)(:)]
    %
    %   Numbering & Matrix Values:
    %     - Every row k of D contains EXACTLY TWO non-zero numbers: -1 and +1.
    %     - Row k computes: (-1)*f(u) + (+1)*f(v) = f(v) - f(u) for neighbor pair (u,v).
    %
    %   Visual 2x2 Image Example:
    %     Given 2x2 image F = [f11 f12; f21 f22], flattened f = [f11; f21; f12; f22]:
    %
    %        Col 1(f11)  Col 2(f21)  Col 3(f12)  Col 4(f22)
    %       ┌                                               ┐  ┌     ┐    ┌             ┐
    %  Row 1│   -1          0          +1          0        │  │ f11 │    │  f12 - f11  │  (H-1)
    %  Row 2│    0         -1           0         +1        │  │ f21 │    │  f22 - f21  │  (H-2)
    %  Row 3│   -1         +1           0          0        │  │ f12 │ =  │  f21 - f11  │  (V-1)
    %  Row 4│    0          0          -1         +1        │  │ f22 │    │  f22 - f12  │  (V-2)
    %       └                                               ┘  └     ┘    └             ┘
    %
    %   Sparsity & Efficiency:
    %     For a 512x512 image (N = 262,144 pixels), D has 523,264 rows and 262,144 cols.
    %     99.999% of entries are zero! Stored in MATLAB as a sparse matrix.
    %
    %   Package: OceanMath.Operators
    %   Author: Ocean Library  |  2026-08-05

    methods
        function D = buildMatrix(~, shape)
            % BUILDMATRIX Build sparse 2D spatial difference operator D.
            %   shape — [rows, cols]
            rows = shape(1);
            cols = shape(2);
            N = rows * cols;

            % ── Horizontal differences: f(r,c+1) - f(r,c) ──────────────────
            Nh = rows * (cols - 1);
            [r_src, c_src] = ndgrid(1:rows, 1:cols-1);
            src_h = r_src(:) + (c_src(:) - 1) * rows;
            tgt_h = r_src(:) +  c_src(:)      * rows;

            edge_h = (1:Nh)';
            Dh = sparse([edge_h; edge_h], [src_h; tgt_h], [-ones(Nh,1); ones(Nh,1)], Nh, N);

            % ── Vertical differences: f(r+1,c) - f(r,c) ────────────────────
            Nv = (rows - 1) * cols;
            [r_src2, c_src2] = ndgrid(1:rows-1, 1:cols);
            src_v = r_src2(:) + (c_src2(:) - 1) * rows;
            tgt_v = r_src2(:) + (c_src2(:) - 1) * rows + 1;

            edge_v = (1:Nv)';
            Dv = sparse([edge_v; edge_v], [src_v; tgt_v], [-ones(Nv,1); ones(Nv,1)], Nv, N);

            % ── Stack horizontal above vertical ──────────────────────────────
            D = [Dh; Dv];
        end

        function d = apply(obj, f, shape)
            if nargin < 3
                shape = size(f);
            end
            D = obj.buildMatrix(shape);
            d = D * f(:);
        end

        function str = getName(~)
            str = "2D Spatial Difference";
        end
    end
end
