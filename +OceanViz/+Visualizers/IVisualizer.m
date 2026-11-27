classdef (Abstract) IVisualizer < handle
    % IVISUALIZER  Abstract rendering contract for simulation visualizers.
    %
    % Extends the existing OceanViz.Interfaces.IVizStrategy pattern for
    % simulation-aware, time-driven rendering. Concrete visualizers listen
    % to an ISimulator event stream and update a MATLAB axes handle
    % to reflect the current state of the simulation.
    %
    % Design:
    %   IVisualizer sits at the end of the full data pipeline:
    %
    %   MarkovChain → MarkovSimulator → SimulationHistory
    %                       |
    %                  StateChanged event
    %                       |
    %                  IVisualizer.onUpdate_()  →  render on UIAxes
    %
    %   The visualizer knows:
    %     - An ISpatialMapper (where each state lives in space)
    %     - A SimulationHistory (the full recorded trajectory)
    %     - A target UIAxes handle (where to paint)
    %
    %   The visualizer does NOT know:
    %     - Anything about the math engine (MarkovChain, entropy, etc.)
    %     - Anything about the GUI layout or App Designer structure
    %
    % Contract:
    %   initialize(axesHandle)  - First-time render, draw background geometry
    %   updateToStep(t)         - Re-render for time step t
    %   clear()                 - Wipe the axes
    %
    % See also: OceanViz.Visualizers.ParticleVisualizer
    %           OceanViz.Adapters.ISpatialMapper
    %           OceanMath.Simulation.SimulationHistory

    % ------------------------------------------------------------------ %
    properties (SetAccess = protected)
        AxesHandle_    % matlab.graphics.axis.Axes — target rendering surface
        Mapper_        % OceanViz.Adapters.ISpatialMapper
        History_       % OceanMath.Simulation.SimulationHistory
        IsInitialized_ (1,1) logical = false
        Listener_      % event listener handle
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = IVisualizer(mapper, history)
            % IVISUALIZER  Inject the mapper and history dependencies.
            arguments
                mapper  OceanViz.Adapters.ISpatialMapper
                history OceanMath.Simulation.SimulationHistory
            end
            obj.Mapper_  = mapper;
            obj.History_ = history;
        end

        function attachToAxes(obj, axesHandle)
            % ATTACHTOAXES  Bind this visualizer to a MATLAB axes handle.
            %
            % Accepts both Axes (figure) and UIAxes (uifigure/App Designer).
            obj.AxesHandle_    = axesHandle;
            obj.IsInitialized_ = false;
            obj.initialize(axesHandle);
            obj.IsInitialized_ = true;
        end

        function attachToSimulator(obj, simulator)
            % ATTACHTOSIMULATOR  Subscribe to the simulator's StateChanged event.
            obj.Listener_ = addlistener(simulator, 'StateChanged', ...
                @(src, evt) obj.onSimulatorStep_(evt));
        end

        function detach(obj)
            % DETACH  Remove the simulator listener safely.
            if ~isempty(obj.Listener_)
                delete(obj.Listener_);
                obj.Listener_ = [];
            end
        end

        function delete(obj)
            obj.detach();
        end
    end

    % ------------------------------------------------------------------ %
    methods (Abstract)
        % INITIALIZE  Draw the static background geometry (grid, graph edges).
        % Called once when the axes handle is first attached.
        initialize(obj, axesHandle)

        % UPDATETOSTEP  Re-render for a specific time step t.
        %
        % INPUTS:
        %   t - integer >= 1, the step index into SimulationHistory
        updateToStep(obj, t)

        % CLEAR  Wipe all dynamic content from the axes.
        clear(obj)
    end

    % ------------------------------------------------------------------ %
    methods (Access = protected)
        function onSimulatorStep_(obj, eventData)
            % ONSIMULATORSTEP_  Listener callback — called on every sim.step().
            % Updates the render to match the current time step.
            if obj.IsInitialized_ && isvalid(obj.AxesHandle_)
                t = eventData.Time + 1;   % History is 1-indexed (t=0 → row 1)
                obj.updateToStep(t);
            end
        end
    end
end
