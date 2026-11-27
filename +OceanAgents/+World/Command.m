classdef Command
    % COMMAND  A structural change to the world - spawn, remove, attach components, detach components - applied only at
    % a synchronisation point, after validation and (for spawn/attach) admission by the guard. Commands never change a
    % component's VALUE; values change only through the owning door's writes.
    %
    %   offer = struct('Position', struct('Value', [2 3], 'Type', posType), ...
    %                  'Charge',   struct('Value', struct('Value', 10, 'Rate', -0.5), 'Type', chargeType));
    %   c = OceanAgents.World.Command.spawn(offer);                 % applied at this phase's synchronisation point
    %   c = OceanAgents.World.Command.remove(id, 'At', 370);        % scheduled on the Calendar
    %   c = OceanAgents.World.Command.remove(id, 'Guard', @(v) v.TargetAlive);
    % Offer values: value kind -> 1 x Width row; linear -> struct(Value, Rate); clock -> T0.
    % The World stamps Issuer, IssuerEntity and Seq. Contract: W1_Kernel_Contract.md section 4.4.

    properties (SetAccess = immutable)
        Kind string = ""
        Time double = []
        Phase string = ""
        Target double = 0
        Offer struct = struct()
        Detach string = strings(1, 0)
        Guard = []
        Issuer string = ""
        IssuerEntity double = 0
        Seq double = 0
    end

    properties (Constant)
        Kinds = ["spawn" "remove" "attach" "detach"]
    end

    methods
        function obj = Command(options)
            arguments
                options.Kind
                options.Time = []
                options.Phase = ""
                options.Target = 0
                options.Offer = struct()
                options.Detach = strings(1, 0)
                options.Guard = []
                options.Issuer = ""
                options.IssuerEntity = 0
                options.Seq = 0
            end
            bad = @(msg) error('OceanAgents:World:BadValue', 'Command: %s', msg);
            if ~isfield(options, 'Kind'), bad('Kind is required.'); end
            obj.Kind = string(options.Kind);
            if ~isscalar(obj.Kind) || ~any(obj.Kind == obj.Kinds), bad('Kind must be spawn, remove, attach or detach.'); end
            t = options.Time;
            if ~isempty(t) && ~(isnumeric(t) && isscalar(t) && t >= 0 && t == fix(t) && isfinite(t))
                bad('Time must be empty or a non-negative integer tick.');
            end
            obj.Time = double(t);
            obj.Phase = string(options.Phase);
            tg = options.Target;
            if ~(isnumeric(tg) && isscalar(tg) && tg >= 0 && tg == fix(tg)), bad('Target must be an entity ID or 0.'); end
            obj.Target = double(tg);
            if obj.Kind == "spawn" && obj.Target ~= 0, bad('spawn has no Target.'); end
            if obj.Kind ~= "spawn" && obj.Target == 0, bad(sprintf('%s needs a Target entity ID.', obj.Kind)); end
            of = options.Offer;
            if ~isstruct(of) || ~isscalar(of), bad('Offer must be a scalar struct.'); end
            if any(obj.Kind == ["spawn" "attach"])
                f = fieldnames(of);
                if isempty(f), bad(sprintf('%s needs a non-empty Offer.', obj.Kind)); end
                for k = 1:numel(f)
                    o = of.(f{k});
                    if ~(isstruct(o) && isscalar(o) && all(isfield(o, {'Value', 'Type'})) && ...
                            isa(o.Type, 'OceanAgents.World.Guard.ComponentType'))
                        bad(sprintf('Offer.%s must be struct(Value, Type) with a Guard.ComponentType.', f{k}));
                    end
                    OceanAgents.Core.assertValueData(struct('v', {o.Value}), "Offer." + f{k});
                end
            elseif ~isempty(fieldnames(of))
                bad(sprintf('%s takes no Offer.', obj.Kind));
            end
            obj.Offer = of;
            obj.Detach = reshape(string(options.Detach), 1, []);
            if isempty(obj.Detach), obj.Detach = strings(1, 0); end
            if obj.Kind == "detach" && isempty(obj.Detach), bad('detach needs component names.'); end
            if ~isempty(options.Guard) && ~isa(options.Guard, 'function_handle'), bad('Guard must be a function handle.'); end
            obj.Guard = options.Guard;
            obj.Issuer = string(options.Issuer);
            ie = options.IssuerEntity;
            if ~(isnumeric(ie) && isscalar(ie) && ie >= 0 && ie == fix(ie)), bad('IssuerEntity must be an entity ID or 0.'); end
            obj.IssuerEntity = double(ie);
            sq = options.Seq;
            if ~(isnumeric(sq) && isscalar(sq) && sq >= 0 && sq == fix(sq)), bad('Seq must be a non-negative integer.'); end
            obj.Seq = double(sq);
        end

        function c = stamp(obj, issuer, issuerEntity, seq, time, phase)
            % STAMP  Copy with World-assigned issuer, sequence and (resolved) time/phase (called by the World only).
            c = OceanAgents.World.Command('Kind', obj.Kind, 'Time', time, 'Phase', phase, 'Target', obj.Target, ...
                'Offer', obj.Offer, 'Detach', obj.Detach, 'Guard', obj.Guard, 'Issuer', issuer, ...
                'IssuerEntity', issuerEntity, 'Seq', seq);
        end
    end

    methods (Static)
        function c = spawn(offer, varargin), c = OceanAgents.World.Command.make_("spawn", 0, offer, strings(1, 0), varargin{:}); end
        function c = remove(id, varargin), c = OceanAgents.World.Command.make_("remove", id, struct(), strings(1, 0), varargin{:}); end
        function c = attach(id, offer, varargin), c = OceanAgents.World.Command.make_("attach", id, offer, strings(1, 0), varargin{:}); end
        function c = detach(id, names, varargin), c = OceanAgents.World.Command.make_("detach", id, struct(), names, varargin{:}); end
    end

    methods (Static, Access = private)
        function c = make_(kind, target, offer, detach, options)
            arguments
                kind
                target
                offer
                detach
                options.At = []
                options.Phase = ""
                options.Guard = []
            end
            c = OceanAgents.World.Command('Kind', kind, 'Target', target, 'Offer', offer, 'Detach', detach, ...
                'Time', options.At, 'Phase', options.Phase, 'Guard', options.Guard);
        end
    end
end
