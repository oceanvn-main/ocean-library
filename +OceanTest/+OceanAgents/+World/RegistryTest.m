classdef RegistryTest < OceanTest.BaseTestCase
    % REGISTRYTEST  OceanAgents.World.Registry (W1_Kernel_Contract.md section 5.2): IDs increase and are never reused
    % (not even across episodes), id <-> row mapping, UnknownEntity for unknown/unbound/removed IDs, atomic bind.

    methods (Test)
        function allocateBindLookup(tc)
            r = OceanAgents.World.Registry();
            tc.verifyEqual(r.allocate(0), zeros(0, 1));
            ids = r.allocate(3);
            tc.verifyEqual(ids, [1; 2; 3]);
            tc.verifyEqual(r.isAlive(ids), false(3, 1), 'allocated but unbound IDs are not alive');
            r.bind(ids, [5; 1; 2]);
            tc.verifyEqual(r.rowsOf([3 1]), [2; 5]);
            tc.verifyEqual(r.idsOf([1; 2; 5; 3; 99]), [2; 3; 1; 0; 0]);
            tc.verifyEqual(r.isAlive([1 2 3 4 0 -1 1.5]), [true; true; true; false; false; false; false]);
            tc.verifyEqual(r.numAlive(), 3);
            tc.verifyEqual(r.allocate(2), [4; 5]);
            tc.verifyEqual(r.LastID, 5);
        end

        function unknownEntities(tc)
            r = OceanAgents.World.Registry();
            ids = r.allocate(3);
            r.bind(ids(1:2), [1; 2]);
            id = 'OceanAgents:World:UnknownEntity';
            tc.verifyError(@() r.rowsOf(7), id);            % never allocated
            tc.verifyError(@() r.rowsOf(3), id);            % allocated, never bound
            r.retire(1);
            tc.verifyError(@() r.rowsOf(1), id);            % removed
            tc.verifyError(@() r.rowsOf([2 1]), id);        % one bad ID fails the whole lookup
            tc.verifyError(@() r.retire(1), id);            % retired twice
            tc.verifyError(@() r.retire(9), id);
            tc.verifyEqual(r.rowsOf(2), 2);
            tc.verifyEqual(r.idsOf(1), 0, 'a removed entity''s row maps to no ID');
            tc.verifyError(@() r.rowsOf(0), 'OceanAgents:World:BadValue');
            tc.verifyError(@() r.rowsOf("1"), 'OceanAgents:World:BadValue');
        end

        function idsAreNeverReused(tc)
            r = OceanAgents.World.Registry();
            a = r.allocate(4);
            r.bind(a, (1:4)');
            r.retire(a([2 4]));
            b = r.allocate(2);
            tc.verifyEqual(b, [5; 6]);
            r.clearEpisode();
            tc.verifyEqual(r.isAlive((1:6)'), false(6, 1));
            tc.verifyError(@() r.rowsOf(1), 'OceanAgents:World:UnknownEntity');
            c = r.allocate(3);
            tc.verifyEqual(c, [7; 8; 9], 'the counter survives clearEpisode');
            r.bind(c, (1:3)');                                   % rows are free again after the reset
            tc.verifyEqual(r.idsOf((1:4)'), [7; 8; 9; 0]);
            tc.verifyError(@() r.bind(5, 4), 'OceanAgents:World:BadValue');   % released IDs cannot come back
            tc.verifyEqual(numel(unique([a; b; c])), 9);
        end

        function bindIsValidatedAndAtomic(tc)
            r = OceanAgents.World.Registry();
            ids = r.allocate(4);
            r.bind(ids(1), 1);
            before = r.snapshot();
            id = 'OceanAgents:World:BadValue';
            tc.verifyError(@() r.bind(ids(2:3), 2), id);                  % count mismatch
            tc.verifyError(@() r.bind(ids(2:3), [1; 3]), id);             % row 1 is taken
            tc.verifyError(@() r.bind(ids([2 2]), [2; 3]), id);           % duplicate ID
            tc.verifyError(@() r.bind(ids(2:3), [3; 3]), id);             % duplicate row
            tc.verifyError(@() r.bind(ids(1), 5), id);                    % already bound
            tc.verifyError(@() r.bind(ids(2), 0), id);                    % bad row
            tc.verifyError(@() r.bind(10, 5), 'OceanAgents:World:UnknownEntity');
            tc.verifyEqual(r.snapshot(), before);
            r.bind(ids(2:4), [4; 3; 2]);
            tc.verifyEqual(r.rowsOf(ids), [1; 4; 3; 2]);
            r.retire(ids(1));
            r.bind(r.allocate(1), 1);                                     % a released row can hold a new entity
            tc.verifyEqual(r.idsOf(1), 5);
        end

        function snapshotRestore(tc)
            r = OceanAgents.World.Registry();
            r.bind(r.allocate(3), [1; 2; 3]);
            snap = r.snapshot();
            OceanAgents.Core.assertValueData(snap, "snap");
            r.retire(2);
            r.bind(r.allocate(2), [7; 8]);
            r.restore(snap);
            tc.verifyEqual(r.snapshot(), snap);
            tc.verifyEqual(r.allocate(1), 4);
            tc.verifyError(@() r.restore(struct('LastID', 1)), 'OceanAgents:World:BadValue');
        end

        function vectorisedAtScale(tc)
            n = 1e5;
            r = OceanAgents.World.Registry();
            t = tic;
            ids = r.allocate(n);
            r.bind(ids, flipud((1:n)'));
            rows = r.rowsOf(ids);
            back = r.idsOf(rows);
            elapsed = toc(t);
            tc.verifyEqual(back, ids);
            tc.verifyEqual(rows(1), n);
            tc.log(matlab.unittest.Verbosity.Terse, sprintf('Registry allocate+bind+rowsOf+idsOf on 1e5 IDs: %.3f ms', 1e3 * elapsed));
            tc.verifyLessThan(elapsed, 0.5);
        end

        function pillars(tc)
            r = OceanAgents.World.Registry();
            m = r.getMetadata();
            tc.verifyFalse(m.IsSymbolic);
            tc.verifyEqual(r.getPipelineSpec().ComponentName, "Registry");
        end
    end
end
