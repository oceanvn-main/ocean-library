classdef NNNode < handle
    % NNNODE  Leaf node — a single neuron in the NNGraph.
    %
    %   OWNS:
    %     - One scalar bias parameter (Bias_)
    %     - Its current activation value (read after forward pass)
    %     - Its own StateEnum (Active / Disabled / Highlighted)
    %
    %   STATE MACHINE:
    %
    %     ┌──────────────────────────────────────────────────┐
    %     │   [Active] ──disable()──▶ [Disabled]             │
    %     │      ▲                        │                  │
    %     │      └────────enable()────────┘                  │
    %     │      │                                           │
    %     │   highlight()                                    │
    %     │      ▼                                           │
    %     │   [Highlighted] ──deselect()──▶ [Active]         │
    %     └──────────────────────────────────────────────────┘
    %
    %   STATISTICAL CONCERN:
    %     - GradMagnitude_: magnitude of last gradient through this node.
    %       Used by isDead() to detect dead ReLU neurons.
    %     - PreActivation_: z = W*a_prev + b (pre-activation, for diagnostics).
    %
    %   NAMING CONVENTION:
    %     ID = "L{layerIndex}_N{nodeIndex}", e.g. "L2_N3"
    %
    %   See also: OceanDL.Graph.NNLayer
    %             OceanDL.Graph.NNEdge

    % ------------------------------------------------------------------ %
    properties (Access = public)
        ID          (1,1) string   % "L2_N3" — globally unique within a graph
        LayerIndex  (1,1) int32    % 1-indexed layer position
        NodeIndex   (1,1) int32    % 1-indexed position within its layer
        Activation  (1,1) string   % 'relu'|'sigmoid'|'tanh'|'softmax'|'linear'
        Bias_       (1,1) double = 0   % learnable bias parameter
    end

    properties (Access = public)
        % Transient runtime diagnostics — populated by NNGraph.forwardPass()
        Value_          (1,1) double = 0  % mean post-activation a = f(z) over batch
        PreActivation_  (1,1) double = 0  % mean pre-activation z = W*a_prev + b
        GradMagnitude_  (1,1) double = 0  % |dL/da| (set by training layer; 0 initially)
    end

    properties (Access = private)
        State_  (1,1) string = "Active"   % "Active" | "Disabled" | "Highlighted"
    end

    % ================================================================== %
    methods
        function obj = NNNode(layerIdx, nodeIdx, activation, bias)
            % NNNODE  Construct a neuron.
            %
            %   node = NNNode(layerIdx, nodeIdx, activation)
            %   node = NNNode(layerIdx, nodeIdx, activation, bias)
            arguments
                layerIdx   (1,1) int32
                nodeIdx    (1,1) int32
                activation (1,1) string
                bias       (1,1) double = 0
            end
            obj.LayerIndex = layerIdx;
            obj.NodeIndex  = nodeIdx;
            obj.Activation = lower(activation);
            obj.Bias_      = bias;
            obj.ID         = sprintf("L%d_N%d", layerIdx, nodeIdx);
        end

        % ---- State transitions --------------------------------------- %

        function disable(obj)
            % DISABLE  Deactivate this node (edges become inactive).
            if ~strcmp(obj.State_, 'Disabled')
                obj.State_ = 'Disabled';
            end
        end

        function enable(obj)
            % ENABLE  Re-activate a previously disabled node.
            if strcmp(obj.State_, 'Disabled')
                obj.State_ = 'Active';
            end
        end

        function highlight(obj)
            % HIGHLIGHT  Mark for permutation / swap selection.
            if strcmp(obj.State_, 'Active')
                obj.State_ = 'Highlighted';
            end
        end

        function deselect(obj)
            % DESELECT  Clear highlight, return to Active.
            if strcmp(obj.State_, 'Highlighted')
                obj.State_ = 'Active';
            end
        end

        % ---- State queries ------------------------------------------ %

        function tf = isEnabled(obj)
            % ISENABLED  True if Active or Highlighted (contributes to fwd pass).
            tf = ~strcmp(obj.State_, 'Disabled');
        end

        function tf = isHighlighted(obj)
            tf = strcmp(obj.State_, 'Highlighted');
        end

        function tf = isDisabled(obj)
            tf = strcmp(obj.State_, 'Disabled');
        end

        function tf = isDead(obj)
            % ISDEAD  True if this is a ReLU neuron that never fires.
            %   Detected by GradMagnitude_ < threshold.
            %   Used by NNStudio for visual dead-neuron warning.
            tf = strcmp(obj.Activation, 'relu') && obj.GradMagnitude_ < 1e-8;
        end

        function s = getState(obj)
            s = obj.State_;
        end

        % ---- Activation (scalar / vector) --------------------------- %

        function a = applyActivation(obj, z)
            % APPLYACTIVATION  Apply this node's activation to input z.
            %
            %   z can be scalar or a 1×N vector (one value per batch sample).
            %   NOTE: softmax is applied at layer level in NNGraph.forwardPass()
            %         because it needs the full layer's z. Per-node call here
            %         returns exp(z) (unnormalized) as a safe fallback.
            switch obj.Activation
                case 'relu',    a = max(0, z);
                case 'sigmoid', a = 1 ./ (1 + exp(-z));
                case 'tanh',    a = tanh(z);
                case 'linear',  a = z;
                case 'softmax'
                    % Caller should use layer-level softmax. Fallback:
                    expz = exp(z - max(z));
                    a    = expz / sum(expz);
                otherwise
                    error('OceanDL:Graph:NNNode:UnknownActivation', ...
                        'Unknown activation: "%s"', obj.Activation);
            end
        end

        % ---- Serialization ------------------------------------------ %

        function s = toStruct(obj)
            % TOSTRUCT  Export node state to a plain struct (for save/load).
            s.id           = obj.ID;
            s.layerIndex   = double(obj.LayerIndex);
            s.nodeIndex    = double(obj.NodeIndex);
            s.activation   = obj.Activation;
            s.bias         = obj.Bias_;
            s.state        = obj.State_;
        end

        % ---- Display ------------------------------------------------- %

        function disp(obj)
            fprintf('NNNode  %s  [Layer %d, Node %d]  act=%s  bias=%.4f  state=%s  value=%.4f\n', ...
                obj.ID, obj.LayerIndex, obj.NodeIndex, obj.Activation, ...
                obj.Bias_, obj.State_, obj.Value_);
        end
    end
end
