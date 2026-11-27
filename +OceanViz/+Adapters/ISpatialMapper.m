classdef (Abstract) ISpatialMapper < handle
    % ISPATIALMAPPER  Abstract adapter between state indices and physical space.
    %
    % This is the Adapter Pattern layer that separates abstract math from
    % geometry. The Markov chain only ever knows integers (State 1, 2, 3...).
    % The GUI only ever knows physical coordinates (x, y, z).
    % This adapter is the translator between them.
    %
    % Design Principle:
    %   By inserting ISpatialMapper between SimulationHistory and the GUI
    %   renderer, you can swap the geometry without touching the math engine.
    %   The same MarkovChain can run in a 2D grid, a 3D terrain, or a hex map
    %   by simply switching the mapper.
    %
    %   Math Engine ──► SimulationHistory ──► ISpatialMapper ──► GUI Renderer
    %                   (state indices)       (coordinates)       (pixels)
    %
    % Contract:
    %   coords = mapState(stateIndex)
    %     - stateIndex : integer, the abstract state from the process
    %     - coords     : numeric vector, physical position in space
    %                    [x, y] for 2D, [x, y, z] for 3D, [x, y, z, roll, pitch, yaw] for kinematic
    %
    %   numDimensions = getDimensions()
    %     - Returns the spatial dimensionality (2, 3, or 6).
    %
    %   numStates = getNumStates()
    %     - Returns how many abstract states this mapper can translate.
    %
    % See also: OceanViz.Adapters.GridMapper2D
    %           OceanViz.Adapters.KinematicMapper3D

    methods (Abstract)
        % MAPSTATE  Translate a state index to physical coordinates.
        %
        % INPUTS:
        %   stateIndex - positive integer, 1-indexed state from the process
        %
        % OUTPUTS:
        %   coords - (1 x D) numeric vector of physical coordinates
        coords = mapState(obj, stateIndex)

        % MAPSEQUENCE  Batch-translate an entire state trajectory.
        %
        % INPUTS:
        %   stateSeq - (N x 1) integer vector (e.g., from SimulationHistory)
        %
        % OUTPUTS:
        %   coordSeq - (N x D) numeric matrix of coordinates, one per row
        coordSeq = mapSequence(obj, stateSeq)

        % GETDIMENSIONS  Return D, the number of spatial dimensions.
        %
        % OUTPUTS:
        %   D - integer (2 for 2D, 3 for 3D, 6 for full kinematic pose)
        D = getDimensions(obj)

        % GETNUMSTATES  Return the number of mappable abstract states.
        %
        % OUTPUTS:
        %   N - integer, total states this mapper was configured for
        N = getNumStates(obj)
    end

    methods
        % ISVALIDSTATE  Check if a state index is within the mapper's range.
        %
        % INPUTS:
        %   stateIndex - integer to validate
        %
        % OUTPUTS:
        %   tf - logical scalar
        function tf = isValidState(obj, stateIndex)
            tf = isscalar(stateIndex) && ...
                 mod(stateIndex, 1) == 0 && ...
                 stateIndex >= 1 && ...
                 stateIndex <= obj.getNumStates();
        end
    end
end
