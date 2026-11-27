classdef ModuleAdmissionTest < OceanTest.BaseTestCase
    % MODULEADMISSIONTEST  GuardEngine.admitModule on the toy module interface (W1_Kernel_Contract.md section 9.3;
    % K16 probe mechanics in W1 scope, K18 certificates). Each deliberately broken implementation is refused with
    % exactly its axiom's ID; a module is admitted by passing the axioms, not by its name.

    properties (Constant)
        G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors
        Pkg = 'OceanTest.OceanAgents.Fixtures.World.Guard.'
    end

    methods
        function m = toy(tc, name)
            m = feval([tc.Pkg name]);
        end

        function cert = admit(~, module, door, varargin)
            cert = OceanAgents.World.Guard.GuardEngine.admitModule(module, door, varargin{:});
        end
    end

    methods (Test)
        function goodModuleCertifiedWithTrueStatuses(tc)
            cert = tc.admit(tc.toy('GoodToy'), tc.G.modules(), 'ModuleVersion', "1.0", 'Seed', 3);
            tc.verifyTrue(cert.Admitted);
            tc.verifyEqual(cert.Subject, string([tc.Pkg 'GoodToy']));
            tc.verifyEqual([cert.Door cert.DoorVersion cert.ModuleVersion], ["modules" "1" "1.0"]);
            tc.verifyEqual([cert.Axioms.ID], ["TM-1" "TM-2" "TM-3" "TM-4" "TM-5"]);
            tc.verifyEqual([cert.Axioms.Status], ["verified-static" "probed" "probed" "probed" "assumed"]);
            tc.verifyEqual([cert.Axioms.Probes], [1 16 8 4 0]);
            tc.verifyEqual(cert.Axioms(2).Seeds, 1:16);                      % stream indices actually run
            tc.verifyEmpty(cert.Axioms(5).Seeds);                            % assumed: never run
            tc.verifyTrue(cert.validFor("1.0", "1"));
        end

        function eachBrokenModuleRefusedWithExactlyItsAxiom(tc)
            cases = {'BrokenCapacityToy', "TM-1"; 'BrokenOutputToy', "TM-2"; 'BrokenSeedToy', "TM-3"; ...
                'BrokenGlobalToy', "TM-4"; 'ThrowingToy', "TM-2"};
            g = RandStream.getGlobalStream();
            saved = g.State;
            restore = onCleanup(@() set(g, 'State', saved));                % BrokenGlobalToy draws globally on purpose
            for k = 1:size(cases, 1)
                cert = tc.admit(tc.toy(cases{k, 1}), tc.G.modules());
                tc.verifyFalse(cert.Admitted, cases{k, 1});
                tc.verifyEqual(cert.failed(), cases{k, 2}, cases{k, 1});
                tc.verifyFalse(cert.validFor("1", "1"));
                tc.verifyEqual(cert.Axioms(5).Status, "assumed");            % never over-claimed, even on failure
            end
            clear restore
        end

        function failureDetailsAndEarlyStop(tc)
            cert = tc.admit(tc.toy('ThrowingToy'), tc.G.modules());
            tm2 = cert.Axioms(2);
            tc.verifyEqual(tm2.Status, "failed");
            tc.verifySubstring(char(tm2.Detail), 'apply is not implemented');     % the error became a failed axiom
            tc.verifySubstring(char(tm2.Detail), 'probe 1');
            tc.verifyEqual(tm2.Probes, 1);                                   % stops at the first failing probe
            cert = tc.admit(tc.toy('BrokenOutputToy'), tc.G.modules());
            tm2 = cert.Axioms(2);
            tc.verifyEqual(tm2.Seeds, 1:tm2.Probes);
            tc.verifySubstring(char(tm2.Detail), sprintf('probe %d', tm2.Probes));
            tc.verifySubstring(char(tm2.Detail), 'NaN');
        end

        function probeStreamsAreThePerProbeSubstreams(tc)
            spy = OceanAgents.World.Guard.AxiomSpec('ID', "TM-9", 'Scope', "module", 'Mode', "probe", 'Probes', 5, ...
                'Predicate', @OceanTest.OceanAgents.Fixtures.World.Guard.ToyAxioms.logFirstDraw);
            door = tc.G.modules('Extra', spy);
            m = tc.toy('GoodToy');
            g = RandStream.getGlobalStream();
            before = g.State;
            tc.admit(m, door, 'Seed', 7);
            tc.verifyEqual(g.State, before);                                 % the guard never uses the global stream
            expected = arrayfun(@(k) rand(RandStream.create('mrg32k3a', 'Seed', 7, 'NumStreams', 5, 'StreamIndices', k)), 1:5);
            tc.verifyEqual(m.DrawLog, expected);
            % the same seed reproduces the certificate; another seed gives other probe streams
            c1 = tc.admit(tc.toy('BrokenOutputToy'), tc.G.modules(), 'Seed', 11);
            c2 = tc.admit(tc.toy('BrokenOutputToy'), tc.G.modules(), 'Seed', 11);
            tc.verifyEqual(c2.Axioms, c1.Axioms);
        end

        function versionsInvalidateCertificatesK18(tc)
            cert = tc.admit(tc.toy('GoodToy'), tc.G.modules('Version', "2"), 'ModuleVersion', "0.3");
            tc.verifyTrue(cert.validFor("0.3", "2"));
            tc.verifyFalse(cert.validFor("0.4", "2"));                       % changed module version
            tc.verifyFalse(cert.validFor("0.3", "3"));                       % changed door version
        end

        function notAdmissibleByInterface(tc)
            tc.verifyError(@() tc.admit(tc.toy('NotAToy'), tc.G.modules()), 'OceanAgents:Guard:NotAdmissible');
            tc.verifyError(@() tc.admit(tc.toy('GoodToy'), tc.G.road()), 'OceanAgents:Guard:NotAdmissible');   % admits nothing
        end

        function predicateFormsAndEntityAxiomsIgnored(tc)
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            extra = [A('ID', "TM-6", 'Scope', "module", 'Mode', "static", 'Predicate', @(m) m.capacity() > 2), ...  % 1 output
                     A('ID', "TM-7", 'Scope', "module", 'Mode', "static", 'Predicate', @(m) "yes"), ...            % not logical
                     A('ID', "TM-8", 'Scope', "module", 'Mode', "probe", 'Probes', 3, ...
                       'Predicate', @(m, s) rand(s) < 2), ...                                                        % 1 output, probe
                     A('ID', "EN-1", 'Component', "Nothing", 'Range', [0 1])];                                      % entity scope
            cert = tc.admit(tc.toy('GoodToy'), tc.G.modules('Extra', extra));
            ids = [cert.Axioms.ID];
            tc.verifyFalse(any(ids == "EN-1"));
            st = @(id) cert.Axioms(ids == id).Status;
            tc.verifyEqual(st("TM-6"), "verified-static");
            tc.verifyEqual(st("TM-7"), "failed");
            tc.verifyEqual(st("TM-8"), "probed");
            tc.verifyEqual(cert.failed(), "TM-7");
        end
    end
end
