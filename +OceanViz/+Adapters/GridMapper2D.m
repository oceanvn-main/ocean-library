classdef GridMapper2D < OceanViz.Adapters.ISpatialMapper
    % GRIDMAPPER2D  Maps linear state indices to (col, row) on a 2D grid.
    %
    % This is the simplest and most commonly useful spatial adapter. It
    % treats the abstract 1D state space of a Markov chain as a flat NxM grid:
    %   State k  →  (col, row)  in physical 2D space
    %
    % --- WHY THIS MAPPING MAKES SENSE (THEORY & PHYSICS) ---
    % 1. Preserving Local Adjacency: 
    %    By flattening the grid using row-major ordering (left-to-right, then down),
    %    physical movements map perfectly to simple algebraic integer offsets in 
    %    the transition matrix P. For any state 's':
    %      - Move Right : s -> s + 1
    %      - Move Left  : s -> s - 1
    %      - Move Down  : s -> s + numCols
    %      - Move Up    : s -> s - numCols
    %    This makes generating massive environment matrices programmable via 
    %    simple loops rather than complex spatial geometry.
    %
    % 2. O(1) Time Complexity:
    %    Instead of calculating modulo and floor divisions for millions of steps 
    %    during a simulation, this class pre-computes the entire geometry once.
    %    Mapping a trajectory becomes a pure memory lookup, preventing the CPU 
    %    from bottlenecking the AI renderer.
    %
    % Optionally, you can provide a physical cell size in meters so the
    % coordinates reflect real-world units rather than grid indices.
    %
    % Usage:
    %   % 4x4 grid (16 states), each cell is 1 metre x 1 metre
    %   mapper = OceanViz.Adapters.GridMapper2D(4, 4);
    %   coords = mapper.mapState(6);    % [2, 1] (col 2, row 1)
    %
    %   % Batch translate a trajectory from SimulationHistory
    %   seq    = hist.getStateSequence();
    %   coords = mapper.mapSequence(seq);  % (N x 2) matrix

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        NumCols_   (1,1) double {mustBePositive, mustBeInteger} = 1
        NumRows_   (1,1) double {mustBePositive, mustBeInteger} = 1
        CellSize_  (1,1) double {mustBePositive} = 1.0   % metres per cell
    end

    % Pre-computed lookup table: row K maps to [col, row] coordinates.
    % Initialized once at construction — O(1) lookup per state thereafter.
    properties (Access = private)
        LookupTable_ (:,2) double
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = GridMapper2D(numCols, numRows, options)
            % GRIDMAPPER2D  Construct a 2D grid mapper.
            %
            % INPUTS:
            %   numCols - integer, number of columns in the grid
            %   numRows - integer, number of rows in the grid
            arguments
                numCols  (1,1) double {mustBePositive, mustBeInteger}
                numRows  (1,1) double {mustBePositive, mustBeInteger}
                options.CellSize (1,1) double {mustBePositive} = 1.0
            end

            obj.NumCols_  = numCols;
            obj.NumRows_  = numRows;
            obj.CellSize_ = options.CellSize;

            % --- PRE-COMPUTE THE LOOKUP TABLE ---
            % We calculate the (x,y) coordinates for the entire universe exactly once.
            N = numCols * numRows;
            
            % THE ALGEBRAIC TRICK (0-Based Indexing):
            % MATLAB uses 1-based indexing, which breaks standard modulo wrapping 
            % (e.g., 4 mod 4 = 0, but there is no Column 0). 
            % We temporarily shift the state indices to start at 0 (0:N-1), apply 
            % the pure math, and then shift the resulting coordinates back to 1.
            
            % The Modulo acts as a "typewriter carriage return". It counts across 
            % the columns and wraps back to the left edge when it hits numCols.
            col = mod((0:N-1)', numCols) + 1;        % 1-indexed column
            
            % The Floor division acts as the row counter. It strictly counts how 
            % many entire rows have been fully saturated so far.
            row = floor((0:N-1)' / numCols) + 1;      % 1-indexed row
            
            % Scale the abstract grid indices into physical real-world metrics.
            obj.LookupTable_ = [col, row] * options.CellSize;
        end

        % ---- ISpatialMapper Implementation ---- %

        function coords = mapState(obj, stateIndex)
            % MAPSTATE  Return [x, y] physical coordinates for a state.
            if ~obj.isValidState(stateIndex)
                error('OceanViz:Adapters:GridMapper2D:InvalidState', ...
                    'State %d is out of range [1, %d].', ...
                    stateIndex, obj.getNumStates());
            end
            coords = obj.LookupTable_(stateIndex, :);
        end

        function coordSeq = mapSequence(obj, stateSeq)
            % MAPSEQUENCE  Batch translate a state trajectory.
            % Implemented as a highly optimized vectorized lookup. Zero for-loops.
            idx      = double(stateSeq(:));
            coordSeq = obj.LookupTable_(idx, :);
        end

        function D = getDimensions(~)
            D = 2;
        end

        function N = getNumStates(obj)
            N = obj.NumCols_ * obj.NumRows_;
        end

        function valid = isValidState(obj, s)
            valid = (s >= 1) && (s <= obj.getNumStates());
        end

        function disp(obj)
            fprintf('GridMapper2D  [%d cols x %d rows = %d states]  CellSize: %.2f m\n', ...
                obj.NumCols_, obj.NumRows_, obj.getNumStates(), obj.CellSize_);
        end
    end
end