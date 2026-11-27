classdef NNStudio < handle
    % NNSTUDIO  Interactive neural network visualization studio (pure OOP uifigure).
    %
    %   A self-contained MATLAB GUI that lets you:
    %     - Build a feedforward network with sliders (layers, width) and dropdowns
    %       (activation function, weight initialization)
    %     - Render it live via NNLayeredLayout + NetworkGraphStrategy (dark theme)
    %     - Click any node to see its info in the right panel
    %     - Enable/Disable nodes with a button click (undoable)
    %     - Highlight a node and Swap it with another (undoable)
    %     - Undo / Redo any structural change from the toolbar
    %     - Run a forward pass and watch the activation heatmap update live
    %
    %   All mutations are channeled through NNGraph's Command system —
    %   enabling full undo/redo history. The canvas auto-redraws whenever
    %   NNGraph fires its TopologyChanged or ActivationsUpdated events.
    %
    %   Usage (interactive MATLAB desktop only):
    %     studio = OceanViz.App.NNStudio();
    %     studio = OceanViz.App.NNStudio('Spec', NNTopologySpec.make([4,6,2],...));
    %
    %   See also: OceanDL.Graph.NNGraph
    %             OceanViz.Strategies.NetworkGraphStrategy
    %             OceanDL.Viz.NNLayeredLayout

    % ------------------------------------------------------------------ %
    properties (Access = private)
        % --- Core model
        Graph_      % OceanDL.Graph.NNGraph
        Layout_     % OceanDL.Viz.NNLayeredLayout
        Strategy_   % OceanViz.Strategies.NetworkGraphStrategy
        Listeners_  % cell of event listener handles

        % --- UI handles
        Fig_
        CanvasAx_       % main network graph (UIAxes)
        FlowAx_         % activation flow bar chart (UIAxes)

        % -- Toolbar buttons
        BtnUndo_
        BtnRedo_
        BtnFwdPass_
        BtnRebuild_

        % -- Hyperparameter panel (left sidebar)
        SldrLayers_
        SldrWidth_
        DdActFn_
        DdWtInit_
        LblLayersVal_
        LblWidthVal_

        % -- Node info panel (right sidebar)
        LblNodeID_
        LblNodeAct_
        LblNodePreAct_  % Phase 5: pre-activation value
        LblNodeVal_
        LblNodeState_
        BtnToggle_      % Enable / Disable
        BtnHighlight_   % Highlight / Deselect
        BtnSwap_        % Swap with highlighted

        % -- Input panel (bottom of left sidebar)
        InputField_     % uieditfield for X vector

        % -- Status bar
        StatusBar_

        % --- Phase 5: Forward pass engine + flow strategy
        FwdPassEngine_  % OceanDL.Simulation.ForwardPassEngine
        FlowStrategy_   % OceanViz.Strategies.ActivationFlowStrategy
        LastResult_     % most recent ForwardPassResult struct

        % --- Phase 6: Snapshot + weight histogram
        HistAx_         % UIAxes for weight distribution chart (right panel)
        BtnSnapshot_    % capture snapshot button
        BtnCompare_     % compare with snapshot button
        BtnPrune_       % prune small weights button
        HistStrategy_   % OceanViz.Strategies.WeightHistogramStrategy
        Snapshot_       % NNGraphMemento (or empty)
        ColorbarAx_     % dedicated thin axes for edge-weight colorbar

        % --- Interaction state
        SelectedNodeID_  (1,:) char = ''   % ID of clicked node
        HighlightedID_   (1,:) char = ''   % ID of node pending swap

        % --- Phase 7: Exploration
        LayoutDropDown_   % uidropdown
        LayoutEngines_    % containers.Map
        NodeDisplayDrop_  % uidropdown — Heatmap/Numbers/Both

        % --- Input/Output size controls
        InputSizeField_   % uieditfield (numeric)
        OutputClassField_ % uieditfield (numeric)

        % --- Themes
        ThemeCore_        % OceanViz.Core
        Colors_           % struct mapped from ThemeCore_.Colors
        ThemeDropdown_    % uidropdown
    end

    properties (Access = private, Constant)
        % ---- UI Sizes ------------------------------------------------ %

        FIG_W    = 1280
        FIG_H    = 720       % taller to fit flow panel
        PANEL_W  = 220       % left + right sidebar width each
        TLBR_H   = 52        % toolbar height
        SBAR_H   = 24        % status bar height
        FLOW_H   = 140       % activation flow panel height

        DEFAULT_LAYERS  = 3
        DEFAULT_WIDTH   = 4
        DEFAULT_ACT     = 'relu'
        DEFAULT_WTINIT  = 'He Normal'
    end

    % ================================================================== %
    methods
        function obj = NNStudio(options)
            % NNSTUDIO  Construct and open the studio window.
            %
            % Name-Value:
            %   Spec - NNTopologySpec to pre-load (optional).
            if nargin < 1, options = struct(); end

            obj.Layout_        = OceanDL.Viz.NNLayeredLayout();
            obj.Strategy_      = OceanViz.Strategies.NetworkGraphStrategy();
            obj.FwdPassEngine_ = OceanDL.Simulation.ForwardPassEngine();
            obj.FlowStrategy_  = OceanViz.Strategies.ActivationFlowStrategy();
            obj.HistStrategy_  = OceanViz.Strategies.WeightHistogramStrategy();
            obj.LastResult_    = [];
            obj.Snapshot_      = [];
            obj.Graph_         = [];   % empty until user clicks Rebuild

            % Build Theme
            themeName = 'Dark';
            if isfield(options, 'Theme')
                themeName = options.Theme;
            end
            obj.ThemeCore_ = OceanViz.Core('Theme', themeName);
            obj.updateColors_();

            % Build layout engine lookup
            obj.LayoutEngines_ = containers.Map( ...
                {'Layered', 'Circular'}, ...
                {OceanDL.Viz.NNLayeredLayout(), ...
                 OceanDL.Viz.NNCircularLayout()});

            % If a Spec was passed in, pre-load it
            if isfield(options, 'Spec')
                obj.buildGraph_(options.Spec);
            end

            % Build the GUI
            obj.buildUI_();

            % Attach event listeners only if a graph was pre-loaded
            if ~isempty(obj.Graph_)
                obj.attachListeners_();
                obj.redrawCanvas_();
                obj.setStatus_('NNStudio ready. Click a node to inspect it.');
            else
                % No graph yet — show placeholder, disable action buttons
                obj.setActionButtonsEnabled_(false);
                obj.drawCanvasPlaceholder_();
                obj.setStatus_('Configure hyperparameters above and click  Rebuild Network  to begin.');
            end
        end

        function show(obj)
            % SHOW  Bring the studio window to front.
            figure(obj.Fig_);
        end

        function delete(obj)
            % DELETE  Clean teardown — detach listeners, close window.
            obj.detachListeners_();
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                delete(obj.Fig_);
            end
        end
    end

    % ================================================================== %
    methods (Access = private)

        % ---- Graph Lifecycle ----------------------------------------- %

        function spec = buildDefaultSpec_(obj) %#ok<MANU>
            spec = OceanDL.Graph.NNTopologySpec.make( ...
                [4, obj.DEFAULT_WIDTH, obj.DEFAULT_WIDTH, 2], ...
                {'linear', obj.DEFAULT_ACT, obj.DEFAULT_ACT, 'softmax'});
        end

        function buildGraph_(obj, spec)
            obj.Graph_ = OceanDL.Graph.NNGraph.fromSpec(spec);
        end

        function rebuildGraph_(obj)
            % Read current slider / dropdown values and rebuild the graph.
            nLayers = round(obj.SldrLayers_.Value);
            width   = round(obj.SldrWidth_.Value);
            actStr  = obj.DdActFn_.Value;
            initStr = obj.DdWtInit_.Value;

            % Read user-defined input and output dimensions
            dIn  = max(1, round(obj.InputSizeField_.Value));
            dOut = max(2, round(obj.OutputClassField_.Value));

            sizes    = [dIn, width * ones(1, max(1, nLayers - 2)), dOut];
            acts     = [{'linear'}, repmat({actStr}, 1, max(1, nLayers - 2)), {'softmax'}];

            spec = OceanDL.Graph.NNTopologySpec.make(sizes, acts);
            % Note: weight initialization is driven automatically by NNGraph.fromSpec()
            % based on each layer's activation function (He for relu, Glorot otherwise).
            % The initStr dropdown is retained in the UI for future use.

            obj.detachListeners_();
            obj.buildGraph_(spec);
            obj.attachListeners_();

            obj.SelectedNodeID_  = '';
            obj.HighlightedID_   = '';
            obj.resetNodeInfoPanel_();

            % Reset input field to zeros matching new d_in
            obj.InputField_.Value = num2str(zeros(1, dIn), '%.1f ');

            % First Rebuild: enable action buttons that were locked on startup
            obj.setActionButtonsEnabled_(true);

            obj.redrawCanvas_();
            obj.setStatus_(sprintf('Network rebuilt: %d layers, width %d, %s.', ...
                nLayers, width, actStr));
        end

        % ---- Event Listeners ----------------------------------------- %

        function attachListeners_(obj)
            obj.Listeners_ = { ...
                addlistener(obj.Graph_, 'TopologyChanged',    @(~,~) obj.onTopologyChanged_()), ...
                addlistener(obj.Graph_, 'WeightsChanged',     @(~,~) obj.onTopologyChanged_()), ...
                addlistener(obj.Graph_, 'ActivationsUpdated', @(~,~) obj.onActivationsUpdated_()) ...
            };
        end

        function detachListeners_(obj)
            if ~isempty(obj.Listeners_)
                cellfun(@delete, obj.Listeners_);
                obj.Listeners_ = {};
            end
        end

        function onTopologyChanged_(obj)
            obj.redrawCanvas_();
            obj.updateUndoRedoButtons_();
            obj.refreshNodeInfoPanel_();
            % Refresh weight histogram with current weights
            obj.onWeightsChanged_();
        end

        function onWeightsChanged_(obj)
            obj.redrawCanvas_();
            obj.updateUndoRedoButtons_();
            obj.refreshNodeInfoPanel_();
            try
                data = obj.Graph_.getWeightHistData();
                obj.redrawHistogram_(data);
            catch
            end
        end

        function onActivationsUpdated_(obj)
            obj.redrawCanvas_();
            obj.redrawFlow_();
        end

        % ---- Canvas Rendering ---------------------------------------- %

        function setActionButtonsEnabled_(obj, tf)
            % Enable or disable action buttons that require a built network.
            % BtnRebuild_ is intentionally excluded — it's the trigger to BUILD.
            state = 'off';
            if tf, state = 'on'; end
            btns = {obj.BtnFwdPass_, obj.BtnSnapshot_, obj.BtnCompare_, ...
                    obj.BtnPrune_,   obj.BtnUndo_,     obj.BtnRedo_};
            for bi = 1:numel(btns)
                if ~isempty(btns{bi}) && isvalid(btns{bi})
                    btns{bi}.Enable = state;
                end
            end
        end

        function drawCanvasPlaceholder_(obj)
            % Draw a styled 'no network' prompt on the canvas.
            if isempty(obj.CanvasAx_) || ~isvalid(obj.CanvasAx_), return; end
            cla(obj.CanvasAx_, 'reset');
            ax = obj.CanvasAx_;
            ax.Color  = [0.08 0.08 0.10];
            ax.XColor = 'none';  ax.YColor = 'none';
            ax.XLim   = [0 1];   ax.YLim   = [0 1];
            % Primary instruction
            text(ax, 0.5, 0.56, 'No network built yet', ...
                'Color',               [0.55 0.58 0.65], ...
                'FontSize',            18, ...
                'FontWeight',          'bold', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment',   'middle', ...
                'Interpreter',         'none');
            % Sub-instruction
            text(ax, 0.5, 0.44, 'Set hyperparameters on the left, then click  Rebuild Network', ...
                'Color',               [0.38 0.42 0.50], ...
                'FontSize',            11, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment',   'middle', ...
                'Interpreter',         'none');
        end

        function redrawCanvas_(obj)
            if isempty(obj.Graph_) || ~isvalid(obj.Graph_), return; end
            try
                cla(obj.CanvasAx_, 'reset');
                data = obj.Layout_.toData(obj.Graph_, struct('Colors', obj.Colors_));
                % Read Node Display mode (default 'Heatmap' before UI is built)
                if ~isempty(obj.NodeDisplayDrop_) && isvalid(obj.NodeDisplayDrop_)
                    dispMode = obj.NodeDisplayDrop_.Value;
                else
                    dispMode = 'Both';
                end
                obj.Strategy_.plot(obj.CanvasAx_, data, struct( ...
                    'ShowWeights', true, ...
                    'ShowLabels',  true, ...
                    'NodeDisplay', dispMode, ...
                    'Colors',      obj.Colors_, ...
                    'ColorbarAxes', obj.ColorbarAx_));
                obj.attachNodeClickHandlers_(data);
            catch ex
                obj.setStatus_(sprintf('Render error: %s', ex.message));
            end
        end

        function attachNodeClickHandlers_(obj, data)
            % Attach ButtonDownFcn to each scatter object (the nodes).
            %
            % LARGE-NETWORK GUARD: if data.IsInteractive is false, the
            % layout compressed some layers. In that case node clicks are
            % meaningless (a visual node may represent many real nodes), so
            % we skip handler attachment and inform the user via status bar.
            if isfield(data, 'IsInteractive') && ~data.IsInteractive
                obj.setStatus_( ...
                    'Node interaction disabled: network too large. ' + ...
                    'Use Input≤16, Output≤16, Nodes/layer≤16 to enable clicking.');
                return;
            end

            % data.NodeRealIDs gives the correct 'L#_N#' ID for each
            % visual node in draw order — avoids the old flatIndexToID_ bug.
            if ~isfield(data, 'NodeRealIDs')
                return;   % layout did not provide ID map — skip safely
            end

            ch = obj.CanvasAx_.Children;
            scatObjs = ch(arrayfun( ...
                @(c) isa(c, 'matlab.graphics.chart.primitive.Scatter'), ch));

            nN = min(numel(scatObjs), data.NumNodes);
            for k = 1:nN
                % scatObjs are in reverse order (last drawn = first in Children)
                nodeIdx = nN - k + 1;
                if nodeIdx < 1 || nodeIdx > numel(data.NodeRealIDs), continue; end
                nodeID = data.NodeRealIDs{nodeIdx};
                scatObjs(k).ButtonDownFcn = @(~,~) obj.onNodeClick_(nodeID);
                scatObjs(k).PickableParts = 'all';
                scatObjs(k).HitTest       = 'on';
            end
        end

        function id = flatIndexToID_(obj, data, k) %#ok<INUSL>
            % Reconstruct node ID from flat index using the graph spec.
            spec  = obj.Graph_.toSpec();
            sizes = spec.LayerSizes;
            total = 0;
            for L = 1:numel(sizes)
                if k <= total + sizes(L)
                    ni = k - total;
                    id = sprintf('L%d_N%d', L, ni);
                    return;
                end
                total = total + sizes(L);
            end
            id = '';
        end

        function L = parseLayer_(~, nodeID)
            % PARSELAYER_  Extract layer index from a node ID string.
            %   'L3_N7' → 3
            tok = regexp(nodeID, '^L(\d+)_N\d+$', 'tokens');
            if isempty(tok)
                L = -1;
            else
                L = str2double(tok{1}{1});
            end
        end

        % ---- Node Click Interaction ----------------------------------- %

        function onNodeClick_(obj, nodeID)
            obj.SelectedNodeID_ = nodeID;
            obj.refreshNodeInfoPanel_();

            % Guide the user through the Swap flow via the status bar.
            if isempty(obj.HighlightedID_)
                % No node pinned yet — show Step 1 hint
                obj.setStatus_(sprintf( ...
                    'Selected: %s.  Click [Highlight] to pin it as Swap target.', nodeID));
                obj.BtnSwap_.Enable = 'off';
            else
                % A node is already pinned — check layer compatibility
                hLayer = obj.parseLayer_(obj.HighlightedID_);
                sLayer = obj.parseLayer_(nodeID);
                if strcmp(nodeID, obj.HighlightedID_)
                    obj.setStatus_(sprintf('Selected: %s (same as pinned). Pick a different node.', nodeID));
                    obj.BtnSwap_.Enable = 'off';
                elseif hLayer ~= sLayer
                    obj.setStatus_(sprintf( ...
                        'Layer mismatch: %s (L%d) vs %s (L%d). Swap requires same layer.', ...
                        obj.HighlightedID_, hLayer, nodeID, sLayer));
                    obj.BtnSwap_.Enable = 'off';
                else
                    obj.setStatus_(sprintf( ...
                        'Ready to swap: %s ↔ %s  (both in Layer %d). Click [? Swap].', ...
                        obj.HighlightedID_, nodeID, sLayer));
                    obj.BtnSwap_.Enable = 'on';
                    obj.BtnSwap_.BackgroundColor = obj.Colors_.Primary;
                end
            end
        end

        function refreshNodeInfoPanel_(obj)
            if isempty(obj.SelectedNodeID_), return; end
            try
                nd = obj.Graph_.getNodeByID(obj.SelectedNodeID_);
                obj.LblNodeID_.Text      = obj.SelectedNodeID_;
                obj.LblNodeAct_.Text     = sprintf('Act:     %s', nd.Activation);
                obj.LblNodePreAct_.Text  = sprintf('PreAct:  %.4f', nd.PreActivation_);
                obj.LblNodeVal_.Text     = sprintf('Value:   %.4f', nd.Value_);
                obj.LblNodeState_.Text   = sprintf('State:   %s', nd.getState());

                if strcmp(nd.getState(), 'Disabled')
                    obj.BtnToggle_.Text            = 'Enable';
                    obj.BtnToggle_.BackgroundColor = obj.Colors_.Primary;
                else
                    obj.BtnToggle_.Text            = 'Disable';
                    obj.BtnToggle_.BackgroundColor = obj.Colors_.Secondary;
                end
            catch
                obj.resetNodeInfoPanel_();
            end
        end

        function resetNodeInfoPanel_(obj)
            obj.LblNodeID_.Text      = '(none selected)';
            obj.LblNodeAct_.Text     = 'Act:     —';
            obj.LblNodePreAct_.Text  = 'PreAct:  —';
            obj.LblNodeVal_.Text     = 'Value:   —';
            obj.LblNodeState_.Text   = 'State:   —';
            obj.BtnToggle_.Text    = 'Disable';
            obj.BtnToggle_.BackgroundColor = obj.Colors_.Button;
        end

        % ---- Button Callbacks ---------------------------------------- %

        function onToggleNode_(obj)
            if isempty(obj.SelectedNodeID_), return; end
            try
                nd = obj.Graph_.getNodeByID(obj.SelectedNodeID_);
                if strcmp(nd.getState(), 'Disabled')
                    obj.Graph_.enableNode(obj.SelectedNodeID_);
                    obj.setStatus_(sprintf('%s enabled.', obj.SelectedNodeID_));
                else
                    obj.Graph_.disableNode(obj.SelectedNodeID_);
                    obj.setStatus_(sprintf('%s disabled.', obj.SelectedNodeID_));
                end
            catch ex
                obj.setStatus_(sprintf('Error: %s', ex.message));
            end
        end

        function onHighlight_(obj)
            if isempty(obj.SelectedNodeID_), return; end
            if strcmp(obj.HighlightedID_, obj.SelectedNodeID_)
                % Toggle off: deselect
                obj.Graph_.deselectNode(obj.SelectedNodeID_);
                obj.HighlightedID_ = '';
                obj.BtnHighlight_.Text = 'Highlight';
                obj.BtnSwap_.Enable   = 'off';
                obj.BtnSwap_.BackgroundColor = obj.Colors_.Button;
                obj.setStatus_('Pin cleared.');
            else
                if ~isempty(obj.HighlightedID_)
                    try; obj.Graph_.deselectNode(obj.HighlightedID_); catch; end
                end
                obj.Graph_.highlightNode(obj.SelectedNodeID_);
                obj.HighlightedID_     = obj.SelectedNodeID_;
                obj.BtnHighlight_.Text = 'Deselect';
                obj.BtnSwap_.Enable    = 'off';   % need a second node first
                obj.BtnSwap_.BackgroundColor = obj.Colors_.Button;
                obj.setStatus_(sprintf( ...
                    'Pinned: %s.  Now click a second node IN THE SAME LAYER, then [? Swap].', ...
                    obj.SelectedNodeID_));
            end
        end

        function onSwap_(obj)
            if isempty(obj.SelectedNodeID_) || isempty(obj.HighlightedID_)
                obj.setStatus_('Pin a node first ([Highlight]), then select a second and [? Swap].');
                return;
            end
            if strcmp(obj.SelectedNodeID_, obj.HighlightedID_)
                obj.setStatus_('Cannot swap a node with itself.');
                return;
            end

            % Explicit cross-layer guard — give a clear message
            hLayer = obj.parseLayer_(obj.HighlightedID_);
            sLayer = obj.parseLayer_(obj.SelectedNodeID_);
            if hLayer ~= sLayer
                obj.setStatus_(sprintf( ...
                    'Cross-layer swap not supported: %s (L%d) ↔ %s (L%d). Pick nodes in the same layer.', ...
                    obj.HighlightedID_, hLayer, obj.SelectedNodeID_, sLayer));
                return;
            end

            try
                obj.Graph_.swapNodes(obj.HighlightedID_, obj.SelectedNodeID_);
                obj.setStatus_(sprintf('Swapped %s ↔ %s. (Undoable)', ...
                    obj.HighlightedID_, obj.SelectedNodeID_));
            catch ex
                obj.setStatus_(sprintf('Swap error: %s', ex.message));
            end

            % Reset swap state regardless of outcome
            obj.HighlightedID_            = '';
            obj.BtnHighlight_.Text        = 'Highlight';
            obj.BtnSwap_.Enable           = 'off';
            obj.BtnSwap_.BackgroundColor  = obj.Colors_.Button;
        end

        function onUndo_(obj)
            if obj.Graph_.canUndo()
                obj.Graph_.undo();
                obj.setStatus_('Undo.');
            else
                obj.setStatus_('Nothing to undo.');
            end
        end

        function onRedo_(obj)
            if obj.Graph_.canRedo()
                obj.Graph_.redo();
                obj.setStatus_('Redo.');
            else
                obj.setStatus_('Nothing to redo.');
            end
        end

        function onFwdPass_(obj)
            try
                spec = obj.Graph_.toSpec();
                d_in = spec.LayerSizes(1);

                % Parse X from InputField_, fall back to randn
                X = obj.parseInputField_(d_in);

                % Store back formatted values for user reference
                obj.InputField_.Value = num2str(X(1,:), '%.3f ');

                % Run engine (multi-sample diagnostics)
                result = obj.FwdPassEngine_.run(obj.Graph_, X);
                obj.LastResult_ = result;

                % Run single-sample pass on row 1 so node values match the
                % output shown in the status bar (avoids last-sample mismatch).
                obj.Graph_.forwardPass(X(1,:));

                obj.setStatus_(sprintf( ...
                    'Forward pass done. Output: [%s]. Layers: %d.', ...
                    num2str(result.Output(1,:), '%.3f '), numel(spec.LayerSizes)));

                % ---- Console report ------------------------------------- %
                fprintf('\n=== NNStudio Forward Pass Report ===\n');
                fprintf('  Time  : %s\n', datestr(now, 'HH:MM:SS'));
                fprintf('  Input : [%s]  (1 sample of %d shown)\n', ...
                    num2str(X(1,:), '%.4f '), size(X,1));
                fprintf('\n  %-10s %-10s %-8s %-8s %-8s %s\n', ...
                    'Layer', 'Act.Fn', 'Mean|a|', 'Std', 'Sat%', 'Dead neurons');
                fprintf('  %s\n', repmat('-', 1, 58));
                for li = 1:result.NumLayers
                    deadCount = 0;
                    if ~isempty(result.DeadNeuronMask{li})
                        deadCount = sum(result.DeadNeuronMask{li});
                    end
                    fprintf('  %-10s %-10s %-8.4f %-8.4f %-8.1f %d\n', ...
                        result.LayerNames{li}, ...
                        result.Activations{li}, ...
                        result.LayerMeans(li), ...
                        result.LayerStd(li), ...
                        result.SaturationRate(li) * 100, ...
                        deadCount);
                end
                fprintf('\n  Output (sample 1): [%s]\n', ...
                    num2str(result.Output(1,:), '%.4f '));
                [~, predClass] = max(result.Output(1,:));
                fprintf('  Predicted class  : %d  (0-indexed: %d)\n', ...
                    predClass, predClass - 1);
                fprintf('=====================================\n\n');
            catch ex
                obj.setStatus_(sprintf('Forward pass error: %s', ex.message));
            end
        end

        function X = parseInputField_(obj, d_in)
            % Parse the input field text into an (N x d_in) matrix.
            % Falls back to randn(5, d_in) if empty or invalid.
            try
                raw = strtrim(obj.InputField_.Value);
                if isempty(raw)
                    X = randn(5, d_in);
                else
                    vals = str2num(raw); %#ok<ST2NM>
                    if isempty(vals) || numel(vals) ~= d_in
                        X = randn(5, d_in);
                        obj.setStatus_(sprintf('Input parse failed. Using random (%d features).', d_in));
                    else
                        X = repmat(vals(:)', 5, 1);  % replicate single row 5x
                    end
                end
            catch
                X = randn(5, d_in);
            end
        end


        function onPrune_(obj)
            res = uiconfirm(obj.Fig_, 'Zero-out weights with |w| < 0.05?', 'Prune Weights', ...
                'Options', {'Prune', 'Cancel'}, 'DefaultOption', 1);
            if strcmp(res, 'Prune')
                obj.Graph_.prune(0.05);
                obj.setStatus_('Applied pruning (Threshold=0.05). Check Weight Histogram.');
            end
        end

        function onLayoutChanged_(obj)
            val = obj.LayoutDropDown_.Value;
            obj.Layout_ = obj.LayoutEngines_(val);
            obj.redrawCanvas_();
            obj.setStatus_(sprintf('Switched to %s layout.', val));
        end

        function redrawFlow_(obj)
            % Render activation flow bar chart in FlowAx_.
            if isempty(obj.LastResult_) || isempty(obj.FlowAx_) || ~isvalid(obj.FlowAx_)
                return;
            end
            try
                obj.FlowStrategy_.plot(obj.FlowAx_, obj.LastResult_, struct('Colors', obj.Colors_));
            catch
            end
        end

        function onSnapshot_(obj)
            % Capture current graph parameters as a snapshot.
            try
                obj.Snapshot_ = obj.Graph_.saveSnapshot();
                obj.BtnCompare_.FontColor = obj.Colors_.Text;
                obj.setStatus_(sprintf('Snapshot saved at %s. Modify weights, then Compare.', ...
                    obj.Snapshot_.Timestamp));
            catch ex
                obj.setStatus_(sprintf('Snapshot error: %s', ex.message));
            end
        end

        function onCompare_(obj)
            % Show weight histogram with snapshot overlay.
            if isempty(obj.Snapshot_)
                obj.setStatus_('No snapshot yet. Click [Snapshot] first.');
                return;
            end
            try
                opts.Snapshot = obj.Snapshot_;
                data = obj.Graph_.getWeightHistData(opts);
                obj.redrawHistogram_(data);
                obj.setStatus_(sprintf('Comparing current weights vs snapshot from %s.', ...
                    obj.Snapshot_.Timestamp));
            catch ex
                obj.setStatus_(sprintf('Compare error: %s', ex.message));
            end
        end

        function redrawHistogram_(obj, data)
            % Render weight histogram in HistAx_.
            if isempty(obj.HistAx_) || ~isvalid(obj.HistAx_)
                return;
            end
            try
                obj.HistStrategy_.plot(obj.HistAx_, data, struct('Colors', obj.Colors_));
            catch
            end
        end

        function updateUndoRedoButtons_(obj)
            obj.BtnUndo_.Enable = 'on';
            obj.BtnRedo_.Enable = 'on';
            if ~obj.Graph_.canUndo(), obj.BtnUndo_.Enable = 'off'; end
            if ~obj.Graph_.canRedo(), obj.BtnRedo_.Enable = 'off'; end
        end

        % ---- Slider Callbacks ---------------------------------------- %

        function onLayersSlider_(obj)
            v = round(obj.SldrLayers_.Value);
            obj.LblLayersVal_.Text = num2str(v);
        end

        function onWidthSlider_(obj)
            v = round(obj.SldrWidth_.Value);
            obj.LblWidthVal_.Text = num2str(v);
        end

        % ---- Status Bar --------------------------------------------- %

        function setStatus_(obj, msg)
            if ~isempty(obj.StatusBar_) && isvalid(obj.StatusBar_)
                obj.StatusBar_.Text = msg;
            end
        end

        % ---- Close --------------------------------------------------- %

        function onClose_(obj)
            obj.delete();
        end

        % ================================================================ %
        % ---- UI Construction ------------------------------------------- %
        % ================================================================ %

        function buildUI_(obj)
            scr = get(0, 'ScreenSize');
            x0  = max(10, (scr(3) - obj.FIG_W) / 2);
            y0  = max(10, (scr(4) - obj.FIG_H) / 2);

            obj.Fig_ = uifigure( ...
                'Name',            'NNStudio — Ocean Library', ...
                'Position',        [x0, y0, obj.FIG_W, obj.FIG_H], ...
                'Color',           obj.Colors_.Background, ...
                'Resize',          'on', ...
                'CloseRequestFcn', @(~,~) obj.onClose_());

            obj.buildToolbar_();
            obj.buildCanvas_();
            obj.buildHyperparamPanel_();
            obj.buildNodeInfoPanel_();
            obj.buildStatusBar_();
        end

        % ---- Toolbar ------------------------------------------------- %

        function buildToolbar_(obj)
            tlH = obj.TLBR_H;
            W   = obj.FIG_W;

            % Background strip
            uilabel(obj.Fig_, ...
                'Position',  [0, obj.FIG_H - tlH, W, tlH], ...
                'Text',      '', ...
                'BackgroundColor', obj.Colors_.Panel);

            bw = 100; bh = 34; by = obj.FIG_H - tlH + 9;
            bx = 12;

            obj.BtnUndo_ = uibutton(obj.Fig_, ...
                'Position',          [bx, by, bw, bh], ...
                'Text',              '⟵ Undo', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          13, ...
                'ButtonPushedFcn',   @(~,~) obj.onUndo_());
            bx = bx + bw + 8;

            obj.BtnRedo_ = uibutton(obj.Fig_, ...
                'Position',          [bx, by, bw, bh], ...
                'Text',              'Redo ⟶', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          13, ...
                'ButtonPushedFcn',   @(~,~) obj.onRedo_());
            bx = bx + bw + 24;

            obj.BtnFwdPass_ = uibutton(obj.Fig_, ...
                'Position',          [bx, by, 140, bh], ...
                'Text',              '▶ Run Forward Pass', ...
                'BackgroundColor',   obj.Colors_.Primary, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          13, ...
                'FontWeight',        'bold', ...
                'ButtonPushedFcn',   @(~,~) obj.onFwdPass_());
            bx = bx + 140 + 8;

            obj.BtnRebuild_ = uibutton(obj.Fig_, ...
                'Position',          [bx, by, 130, bh], ...
                'Text',              'Rebuild Network', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          13, ...
                'ButtonPushedFcn',   @(~,~) obj.rebuildGraph_());
            bx = bx + 130 + 16;

            obj.BtnSnapshot_ = uibutton(obj.Fig_, ...
                'Position',          [bx, by, 110, bh], ...
                'Text',              'Snapshot', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          13, ...
                'ButtonPushedFcn',   @(~,~) obj.onSnapshot_());
            bx = bx + 110 + 6;

            obj.BtnCompare_ = uibutton(obj.Fig_, ...
                'Position',          [bx, by, 110, bh], ...
                'Text',              'Compare', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.DimText, ...
                'FontSize',          13, ...
                'ButtonPushedFcn',   @(~,~) obj.onCompare_());
            bx = bx + 110 + 6;

            obj.BtnPrune_ = uibutton(obj.Fig_, 'push', ...
                'Position',          [bx, by, 80, bh], ...
                'Text',              '✂ Prune', ...
                'BackgroundColor',   obj.Colors_.Secondary, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          13, ...
                'ButtonPushedFcn',   @(~,~) obj.onPrune_());
            bx = bx + 80 + 16;

            obj.ThemeDropdown_ = uidropdown(obj.Fig_, ...
                'Position',          [bx, by+4, 100, 26], ...
                'Items',             {'Dark', 'Scientific', 'Blueprint', 'Solarized', 'Midnight', 'Cyberpunk', 'Nature', 'GrayScale'}, ...
                'Value',             obj.ThemeCore_.ThemeName, ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11, ...
                'ValueChangedFcn',   @(~,~) obj.onThemeChanged_());
            bx = bx + 100 + 16;

            % Title label (right side of toolbar)
            uilabel(obj.Fig_, ...
                'Position',          [obj.FIG_W - 260, by - 2, 250, bh + 4], ...
                'Text',              'NNStudio  ·  Ocean Library', ...
                'FontColor',         obj.Colors_.Accent, ...
                'FontSize',          14, ...
                'FontWeight',        'bold', ...
                'HorizontalAlignment','right');
        end

        % ---- Canvas -------------------------------------------------- %

        function buildCanvas_(obj)
            % Canvas: full height minus toolbar + status + flow panels.
            % A dedicated 50px ColorbarAx_ strip sits immediately to the
            % right of the canvas — keeps the main axes toolbar unobstructed.
            CB_W      = 55;   % colorbar strip width (px)
            canvasX   = obj.PANEL_W;
            canvasY   = obj.SBAR_H + obj.FLOW_H;
            canvasW   = obj.FIG_W - 2 * obj.PANEL_W - CB_W - 4;
            
            % Add 30px padding at top so the native axes toolbar handles
            % (which float slightly above the axes) aren't clipped by the app toolbar.
            TOP_PAD   = 30;
            canvasH   = obj.FIG_H - obj.TLBR_H - obj.SBAR_H - obj.FLOW_H - TOP_PAD;

            obj.CanvasAx_ = uiaxes(obj.Fig_, ...
                'Position',   [canvasX, canvasY, canvasW, canvasH], ...
                'Color',      [0.08 0.08 0.10], ...
                'XColor',     'none', ...
                'YColor',     'none', ...
                'Box',        'off', ...
                'XLimMode',   'auto', ...
                'YLimMode',   'auto');
            obj.CanvasAx_.Toolbar.Visible = 'on';  % zoom / pan / restore

            % Dedicated colorbar strip — separate axes, no canvas squeezing
            cbX = canvasX + canvasW + 4;
            obj.ColorbarAx_ = uiaxes(obj.Fig_, ...
                'Position',  [cbX, canvasY, CB_W, canvasH], ...
                'Color',     [0.08 0.08 0.10], ...
                'XColor',    'none', ...
                'YColor',    'none', ...
                'Box',       'off');
            obj.ColorbarAx_.Toolbar.Visible = 'off';

            % Flow panel (below canvas)
            obj.FlowAx_ = uiaxes(obj.Fig_, ...
                'Position',   [canvasX, obj.SBAR_H, canvasW, obj.FLOW_H], ...
                'Color',      [0.08 0.08 0.10], ...
                'XColor',     'none', ...
                'YColor',     'none', ...
                'Box',        'off');
            obj.FlowAx_.Toolbar.Visible = 'off';

            % Placeholder label in flow panel
            uilabel(obj.Fig_, ...
                'Position',  [canvasX + 10, obj.SBAR_H + 4, canvasW - 20, 20], ...
                'Text',      'Run a Forward Pass to see the Activation Flow chart.', ...
                'FontColor', [0.40 0.42 0.45], ...
                'FontSize',  10, ...
                'HorizontalAlignment', 'center');
        end

        % ---- Left Sidebar — Hyperparameters -------------------------- %

        function buildHyperparamPanel_(obj)
            x  = 0;
            y  = obj.SBAR_H;
            w  = obj.PANEL_W;
            h  = obj.FIG_H - obj.TLBR_H - obj.SBAR_H;

            % Panel background
            uilabel(obj.Fig_, 'Position', [x, y, w, h], 'Text', '', ...
                'BackgroundColor', obj.Colors_.Panel);

            % Header
            uilabel(obj.Fig_, ...
                'Position',   [x + 8, y + h - 36, w - 16, 28], ...
                'Text',       'Hyperparameters', ...
                'FontColor',  obj.Colors_.Accent, ...
                'FontSize',   13, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            cy = y + h - 68;
            pad = 8; fw = w - 2*pad;

            % -- Layers slider
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 20], ...
                'Text', 'Layers', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            cy = cy - 22;
            obj.SldrLayers_ = uislider(obj.Fig_, ...
                'Position',     [x+pad, cy, fw, 3], ...
                'Limits',       [2, 8], 'Value', obj.DEFAULT_LAYERS, ...
                'MajorTicks',   2:8, 'MinorTicks', [], ...
                'FontColor',    obj.Colors_.Text, ...
                'ValueChangedFcn', @(~,~) obj.onLayersSlider_());
            cy = cy - 28;
            obj.LblLayersVal_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], ...
                'Text',      num2str(obj.DEFAULT_LAYERS), ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 11, ...
                'HorizontalAlignment', 'center');
            cy = cy - 36;

            % -- Width slider
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 20], ...
                'Text', 'Nodes per layer', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            cy = cy - 22;
            obj.SldrWidth_ = uislider(obj.Fig_, ...
                'Position',     [x+pad, cy, fw, 3], ...
                'Limits',       [2, 16], 'Value', obj.DEFAULT_WIDTH, ...
                'MajorTicks',   [2, 4, 8, 12, 16], 'MinorTicks', [], ...
                'FontColor',    obj.Colors_.Text, ...
                'ValueChangedFcn', @(~,~) obj.onWidthSlider_());
            cy = cy - 28;
            obj.LblWidthVal_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], ...
                'Text',      num2str(obj.DEFAULT_WIDTH), ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 11, ...
                'HorizontalAlignment', 'center');
            cy = cy - 36;

            % -- Activation function
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 20], ...
                'Text', 'Hidden Activation', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            cy = cy - 30;
            obj.DdActFn_ = uidropdown(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, 26], ...
                'Items',             {'relu', 'sigmoid', 'tanh', 'linear'}, ...
                'Value',             obj.DEFAULT_ACT, ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11);
            cy = cy - 40;

            % -- Weight init
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 20], ...
                'Text', 'Weight Init', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            cy = cy - 30;
            obj.DdWtInit_ = uidropdown(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, 26], ...
                'Items',             {'He Normal', 'Glorot Uniform', 'Random'}, ...
                'Value',             obj.DEFAULT_WTINIT, ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11);
            cy = cy - 40;

            % -- Input Size
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw*0.55, 20], ...
                'Text', 'Input Size', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            uilabel(obj.Fig_, 'Position', [x+pad+fw*0.55, cy, fw*0.45, 20], ...
                'Text', 'Output Classes', 'FontColor', obj.Colors_.Text, 'FontSize', 10);
            cy = cy - 30;
            obj.InputSizeField_ = uieditfield(obj.Fig_, 'numeric', ...
                'Position',          [x+pad, cy, fw*0.45, 26], ...
                'Value',             2, ...
                'Limits',            [1 1024], ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11);
            obj.OutputClassField_ = uieditfield(obj.Fig_, 'numeric', ...
                'Position',          [x+pad+fw*0.55, cy, fw*0.45, 26], ...
                'Value',             2, ...
                'Limits',            [2 1000], ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11);
            cy = cy - 40;

            % -- Layout Style
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 20], ...
                'Text', 'Layout Style', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            cy = cy - 30;
            obj.LayoutDropDown_ = uidropdown(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, 26], ...
                'Items',             {'Layered', 'Circular'}, ...
                'Value',             'Layered', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11, ...
                'ValueChangedFcn',   @(~,~) obj.onLayoutChanged_());
            cy = cy - 46;

            % -- Node Display mode
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 20], ...
                'Text', 'Node Display', 'FontColor', obj.Colors_.Text, 'FontSize', 11);
            cy = cy - 30;
            obj.NodeDisplayDrop_ = uidropdown(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, 26], ...
                'Items',             {'Heatmap', 'Numbers', 'Both'}, ...
                'Value',             'Heatmap', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          11, ...
                'ValueChangedFcn',   @(~,~) obj.redrawCanvas_());
        end

        % ---- Right Sidebar — Node Info ------------------------------- %

        function buildNodeInfoPanel_(obj)
            x = obj.FIG_W - obj.PANEL_W;
            y = obj.SBAR_H;
            w = obj.PANEL_W;
            h = obj.FIG_H - obj.TLBR_H - obj.SBAR_H;

            % Panel background
            uilabel(obj.Fig_, 'Position', [x, y, w, h], 'Text', '', ...
                'BackgroundColor', obj.Colors_.Panel);

            % Header
            uilabel(obj.Fig_, ...
                'Position',  [x+8, y+h-36, w-16, 28], ...
                'Text',      'Node Inspector', ...
                'FontColor', obj.Colors_.Accent, ...
                'FontSize',  13, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            pad = 10; fw = w - 2*pad;
            cy  = y + h - 72;

            % Info labels
            obj.LblNodeID_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 22], 'Text', '(none selected)', ...
                'FontColor', obj.Colors_.Text, 'FontSize', 12, 'FontWeight', 'bold');
            cy = cy - 26;

            obj.LblNodeAct_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], 'Text', 'Act:     —', ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 11);
            cy = cy - 22;

            obj.LblNodePreAct_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], 'Text', 'PreAct:  —', ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 11);
            cy = cy - 22;

            obj.LblNodeVal_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], 'Text', 'Value:   —', ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 11);
            cy = cy - 22;

            obj.LblNodeState_ = uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], 'Text', 'State:   —', ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 11);
            cy = cy - 38;

            % Action buttons
            bh = 32;
            obj.BtnToggle_ = uibutton(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, bh], ...
                'Text',              'Disable', ...
                'BackgroundColor',   obj.Colors_.Secondary, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          12, ...
                'ButtonPushedFcn',   @(~,~) obj.onToggleNode_());
            cy = cy - bh - 8;

            obj.BtnHighlight_ = uibutton(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, bh], ...
                'Text',              'Highlight', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          12, ...
                'ButtonPushedFcn',   @(~,~) obj.onHighlight_());
            cy = cy - bh - 8;

            obj.BtnSwap_ = uibutton(obj.Fig_, ...
                'Position',          [x+pad, cy, fw, bh], ...
                'Text',              '? Swap', ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          12, ...
                'ButtonPushedFcn',   @(~,~) obj.onSwap_());

            % -- Weight Histogram panel (above input section)
            histH = 140;
            histY = y + 135;  % sits above the input field section
            uilabel(obj.Fig_, ...
                'Position',  [x+pad, histY + histH + 4, fw, 16], ...
                'Text',      'Weights  (click Compare)', ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 9, ...
                'HorizontalAlignment', 'center');
            obj.HistAx_ = uiaxes(obj.Fig_, ...
                'Position', [x+pad, histY, fw, histH], ...
                'Color',    [0.08 0.08 0.10], ...
                'XColor',   [0.45 0.48 0.52], ...
                'YColor',   'none', 'Box', 'off');
            obj.HistAx_.Toolbar.Visible = 'off';

            % -- Input section (bottom of right panel)
            cy = y + 100;
            uilabel(obj.Fig_, 'Position', [x+pad, cy, fw, 1], 'Text', '', ...
                'BackgroundColor', obj.Colors_.Button);
            cy = cy - 18;
            uilabel(obj.Fig_, ...
                'Position',  [x+pad, cy, fw, 20], ...
                'Text',      'Input X (space-separated)', ...
                'FontColor', obj.Colors_.DimText, 'FontSize', 10, ...
                'HorizontalAlignment', 'center');
            cy = cy - 28;

            % d_in is unknown before first Rebuild — default to 2 as placeholder
            if ~isempty(obj.Graph_) && isvalid(obj.Graph_)
                spec = obj.Graph_.toSpec();
                d_in = spec.LayerSizes(1);
            else
                d_in = 2;
            end
            defaultX = num2str(zeros(1, d_in), '%.1f ');
            obj.InputField_ = uieditfield(obj.Fig_, 'text', ...
                'Position',          [x+pad, cy, fw, 28], ...
                'Value',             defaultX, ...
                'BackgroundColor',   obj.Colors_.Button, ...
                'FontColor',         obj.Colors_.Text, ...
                'FontSize',          10);
        end

        % ---- Status Bar --------------------------------------------- %

        function buildStatusBar_(obj)
            obj.StatusBar_ = uilabel(obj.Fig_, ...
                'Position',           [8, 0, obj.FIG_W - 8, obj.SBAR_H], ...
                'Text',               'Initializing...', ...
                'BackgroundColor',    obj.Colors_.Panel, ...
                'FontColor',          obj.Colors_.DimText, ...
                'FontSize',           11, ...
                'HorizontalAlignment','left', ...
                'VerticalAlignment',  'center');
        end

        % ---- Theme Engine ------------------------------------------- %

        function updateColors_(obj)
            % Derive UI application colors from the active OceanViz Theme
            c = obj.ThemeCore_.Colors;
            name = string(obj.ThemeCore_.ThemeName);

            obj.Colors_.Background = c.Background;
            obj.Colors_.Primary    = c.Primary;
            obj.Colors_.Secondary  = c.Secondary;
            obj.Colors_.Text       = c.Text;
            obj.Colors_.Accent     = c.Primary;

            % UI Panels need subtle contrast from the main background based on light/dark modes
            isLight = mean(c.Background) > 0.5;
            if isLight
                % Darken slightly for panels and buttons
                obj.Colors_.Panel   = max(c.Background - 0.05, 0);
                obj.Colors_.Button  = max(c.Background - 0.12, 0);
                obj.Colors_.DimText = c.Text * 0.6 + c.Background * 0.4;
            else
                % Lighten slightly for panels and buttons
                obj.Colors_.Panel   = min(c.Background + 0.05, 1.0);
                obj.Colors_.Button  = min(c.Background + 0.12, 1.0);
                obj.Colors_.DimText = c.Text * 0.6 + c.Background * 0.4;
            end
        end

        function onThemeChanged_(obj)
            % Called when the user selects a new Theme from the dropdown
            themeName = obj.ThemeDropdown_.Value;
            oldInputStr = obj.InputField_.Value;
            
            obj.ThemeCore_ = OceanViz.Core('Theme', themeName);
            obj.updateColors_();

            % Rebuild UI to apply new colors
            if ~isempty(obj.Fig_) && isvalid(obj.Fig_)
                pos = obj.Fig_.Position;
                delete(obj.Fig_);    % Tear down old window
                obj.buildUI_();      % Build new styled window
                obj.Fig_.Position = pos; % Restore position
                obj.InputField_.Value = oldInputStr;

                % Restore graph state & listeners
                if ~isempty(obj.Graph_)
                    obj.attachListeners_();
                    obj.setActionButtonsEnabled_(true);
                    
                    obj.redrawCanvas_();
                    obj.setStatus_(sprintf('Theme changed to %s.', themeName));
                else
                    obj.setActionButtonsEnabled_(false);
                    obj.drawCanvasPlaceholder_();
                end
            end
        end
    end
end
