classdef Certificate
    % CERTIFICATE  Record of one module admission: which axioms were checked, how, with which probe seeds, and the
    % verdict. It never over-claims: an assumed axiom is reported as "assumed", never as verified; a certificate is
    % valid only for the exact module and door versions it was issued for.
    %
    %   Axioms: struct array  ID, Mode, Status ("verified-static"|"probed"|"partial"|"assumed"|"failed"), Probes, Seeds, Detail,
    %           Coverage (condition -> number of probes that reached it). "partial" = every probe passed but some
    %           condition was never reached, so the axiom is not reported as probed.
    % Contract: W1_Kernel_Contract.md section 3.4.

    properties (SetAccess = immutable)
        Subject string = ""
        ModuleVersion string = ""
        Door string = ""
        DoorVersion string = ""
        Axioms struct = struct('ID', {}, 'Mode', {}, 'Status', {}, 'Probes', {}, 'Seeds', {}, 'Detail', {}, 'Coverage', {})
        Admitted logical = false
        IssuedAt string = ""
    end

    properties (Constant)
        Statuses = ["verified-static" "probed" "partial" "assumed" "failed"]
    end

    methods
        function obj = Certificate(options)
            arguments
                options.Subject
                options.ModuleVersion
                options.Door
                options.DoorVersion
                options.Axioms
                options.IssuedAt = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'))
            end
            bad = @(msg) error('OceanAgents:Guard:BadCertificate', 'Certificate: %s', msg);
            need = {'Subject', 'ModuleVersion', 'Door', 'DoorVersion', 'Axioms'};
            if ~all(isfield(options, need)), bad('Subject, ModuleVersion, Door, DoorVersion and Axioms are required.'); end
            obj.Subject = string(options.Subject);
            obj.ModuleVersion = string(options.ModuleVersion);
            obj.Door = string(options.Door);
            obj.DoorVersion = string(options.DoorVersion);
            ax = options.Axioms;
            fields = {'ID', 'Mode', 'Status', 'Probes', 'Seeds', 'Detail'};
            if ~isstruct(ax) || (~isempty(ax) && ~all(isfield(ax, fields)))
                bad('Axioms must be a struct array with fields ID, Mode, Status, Probes, Seeds, Detail.');
            end
            for k = 1:numel(ax)
                if ~any(string(ax(k).Status) == obj.Statuses), bad(sprintf('unknown status for %s.', ax(k).ID)); end
                if string(ax(k).Mode) == "assumed" && string(ax(k).Status) ~= "assumed"
                    bad(sprintf('%s is assumed and cannot be reported as %s.', ax(k).ID, ax(k).Status));
                end
            end
            if ~isfield(ax, 'Coverage'), [ax.Coverage] = deal(struct()); end
            obj.Axioms = reshape(ax, 1, []);
            obj.Admitted = isempty(ax) || ~any(string({ax.Status}) == "failed");
            obj.IssuedAt = string(options.IssuedAt);
        end

        function tf = validFor(obj, moduleVersion, doorVersion, options)
            % VALIDFOR  True only for the exact module and door versions this certificate was issued for (K18), and -
            % when given - the same subject (module class) and door (critic W1 F11: versions alone are not identity).
            %   cert.validFor("1", "2", 'Subject', "ToyGood", 'Door', "modules")
            arguments
                obj
                moduleVersion
                doorVersion
                options.Subject = []
                options.Door = []
            end
            tf = obj.Admitted && obj.ModuleVersion == string(moduleVersion) && obj.DoorVersion == string(doorVersion);
            if ~isempty(options.Subject), tf = tf && obj.Subject == string(options.Subject); end
            if ~isempty(options.Door), tf = tf && obj.Door == string(options.Door); end
        end

        function ids = unverified(obj)
            % UNVERIFIED  IDs of the axioms that did not fail but were not fully checked ("assumed" or "partial").
            if isempty(obj.Axioms), ids = strings(1, 0); return; end
            st = string({obj.Axioms.Status});
            ids = string({obj.Axioms(st == "assumed" | st == "partial").ID});
        end

        function ids = failed(obj)
            % FAILED  IDs of the axioms that failed.
            if isempty(obj.Axioms), ids = strings(1, 0); return; end
            ids = string({obj.Axioms(string({obj.Axioms.Status}) == "failed").ID});
        end
    end
end
