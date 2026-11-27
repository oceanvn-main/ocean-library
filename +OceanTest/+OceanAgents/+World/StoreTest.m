classdef StoreTest < OceanTest.BaseTestCase
    % STORETEST  OceanAgents.World.Store (W1_Kernel_Contract.md section 5.1): component kinds with exact hand values,
    % linear rebasing, validation (OceanAgents:World:BadValue / WriteConflict) with no partial changes, dirty tracking,
    % row stability, snapshot/restore bit-for-bit, and a timing report on 1e5 rows.

    methods (Static, Access = private)
        function s = store()
            s = OceanAgents.World.Store(OceanTest.OceanAgents.Fixtures.World.Data.storeTypes());
        end
        function o = full(pos, charge, rate, born)
            % A complete offer in the Command.Offer format (Type fields omitted: the Store ignores them).
            o = struct('Position', struct('Value', pos), 'Charge', struct('Value', struct('Value', charge, 'Rate', rate)), ...
                'Born', struct('Value', born));
        end
        function w = write(name, rows, values, varargin)
            w = struct('Component', name, 'Rows', rows, 'Values', values, 'Rates', []);
            if ~isempty(varargin), w.Rates = varargin{1}; end
        end
    end

    methods (Test)
        function constructionAndEmptyStore(tc)
            s = tc.store();
            tc.verifyEqual(s.Names, ["Position" "Velocity" "Charge" "Level" "Born" "Name" "Flags"]);
            tc.verifyEqual(s.NumRows, 0);
            tc.verifyEqual(s.addRows(0), zeros(0, 1));
            tc.verifyEqual(s.rowsWith("Position"), zeros(0, 1));
            tc.verifyError(@() OceanAgents.World.Store(struct('X', 3)), 'OceanAgents:World:BadValue');
            e = OceanAgents.World.Store();
            tc.verifyEqual(e.addRows(2), [1; 2]);
            tc.verifyEqual(e.rowsWith(), [1; 2]);
        end

        function offerAndHandValues(tc)
            % Charge offered 10 with rate -1 at now 5; Born T0 = 3.
            s = tc.store();
            rows = s.addRows(2);
            tc.verifyEqual(rows, [1; 2]);
            offer = tc.full([2 3], 10, -1, 3);
            offer.Name = struct('Value', "bob");
            offer.Flags = struct('Value', [true false true]);
            s.setOffer(1, offer, 5);
            tc.verifyEqual(s.valueAt("Charge", 1, 8), 7);              % 10 + (-1)(8 - 5)
            tc.verifyEqual(s.valueAt("Charge", 1, 5), 10);
            tc.verifyEqual(s.valueAt("Born", 1, 8), 5);                % 8 - 3
            tc.verifyEqual(s.valueAt("Position", 1, 8), [2 3]);
            tc.verifyEqual(s.valueAt("Position", 1), [2 3]);           % value kind needs no now
            tc.verifyEqual(s.valueAt("Name", 1), "bob");
            tc.verifyEqual(s.valueAt("Flags", 1), [true false true]);
            r = s.raw("Charge");
            tc.verifyEqual([r.Base(1) r.Rate(1) r.T0(1)], [10 -1 5]);
            tc.verifyEqual(r.Has, [true; false]);
            tc.verifyClass(s.raw("Name").Data, 'string');
            tc.verifyClass(s.raw("Flags").Data, 'logical');
            tc.verifyEqual(s.has("Velocity", [1 2]), [false; false]);
            tc.verifyEqual(s.has("Position", [1 2]), [true; false]);
            tc.verifyEqual(s.rowsWith(["Position" "Charge"]), 1);
            tc.verifyEqual(s.rowsWith(strings(1, 0)), [1; 2]);
            % a one-row offer broadcasts to several rows
            more = s.addRows(3);
            s.setOffer(more, struct('Level', struct('Value', struct('Value', [1 2], 'Rate', [0.5 -1]))), 4);
            tc.verifyEqual(s.valueAt("Level", more, 6), repmat([2 0], 3, 1));
            tc.verifyEqual(s.rowsWith("Level"), [3; 4; 5]);
        end

        function linearWritesRebase(tc)
            s = tc.store();
            s.addRows(1);
            s.setOffer(1, tc.full([0 0], 10, -1, 0), 5);
            % at 8 the value is 7; write 7 with a new rate -2 -> Base 7, T0 8, Rate -2
            s.applyWrites(tc.write("Charge", 1, 7, -2), 8);
            r = s.raw("Charge");
            tc.verifyEqual([r.Base r.Rate r.T0], [7 -2 8]);
            tc.verifyEqual(s.valueAt("Charge", 1, 10), 3);             % 7 - 2 * (10 - 8)
            % write without Rates keeps the old rate: Base 20, T0 10, Rate -2
            s.applyWrites(tc.write("Charge", 1, 20), 10);
            r = s.raw("Charge");
            tc.verifyEqual([r.Base r.Rate r.T0], [20 -2 10]);
            tc.verifyEqual(s.valueAt("Charge", 1, 12), 16);
            % two-column linear
            s.setOffer(1, struct('Level', struct('Value', struct('Value', [1 1], 'Rate', [1 2]))), 12);
            s.applyWrites(tc.write("Level", 1, [5 6], [0 -1]), 14);
            tc.verifyEqual(s.valueAt("Level", 1, 17), [5 3]);
        end

        function clockWritesSetT0(tc)
            s = tc.store();
            s.addRows(2);
            s.setOffer([1; 2], struct('Born', struct('Value', [0; 4])));
            tc.verifyEqual(s.valueAt("Born", [1; 2], 10), [10; 6]);
            s.applyWrites(tc.write("Born", 2, 9), 10);
            tc.verifyEqual(s.valueAt("Born", [2; 1], 12), [3; 12]);
            tc.verifyError(@() s.valueAt("Born", 1), 'OceanAgents:World:BadValue');   % a clock needs now
        end

        function valueWritesAndClasses(tc)
            s = tc.store();
            s.addRows(3);
            s.setOffer((1:3)', struct('Position', struct('Value', [1 1; 2 2; 3 3]), 'Name', struct('Value', ["a"; "b"; "c"]), ...
                'Flags', struct('Value', false(3, 3))));
            s.applyWrites([tc.write("Position", [3; 1], [30 31; 10 11]), tc.write("Name", 2, "B"), ...
                tc.write("Flags", [1; 2], logical([1 0 0; 0 1 0]))], 0);
            tc.verifyEqual(s.valueAt("Position", (1:3)'), [10 11; 2 2; 30 31]);
            tc.verifyEqual(s.valueAt("Name", (1:3)'), ["a"; "B"; "c"]);
            tc.verifyEqual(s.valueAt("Flags", (1:3)'), logical([1 0 0; 0 1 0; 0 0 0]));
        end

        function badOffersChangeNothing(tc)
            s = tc.store();
            s.addRows(3);
            s.setOffer(1, tc.full([1 1], 5, 0, 0), 0);
            s.killRows(3);
            before = s.snapshot();
            id = 'OceanAgents:World:BadValue';
            bads = { ...
                {2, struct('Unknown', struct('Value', 1))}, ...
                {2, struct('Position', struct('Value', single([1 2])))}, ...        % class
                {2, struct('Position', struct('Value', [1 2 3]))}, ...              % width
                {2, struct('Position', struct('Value', [1 NaN]))}, ...              % finiteness
                {2, struct('Position', struct('Value', [1 2; 3 4; 5 6]))}, ...      % rows (2 or 1 expected? 1 row)
                {2, struct('Flags', struct('Value', [1 0 1]))}, ...                 % logical expected
                {2, struct('Name', struct('Value', 'bob'))}, ...                    % string expected, char given
                {2, struct('Name', struct('Value', string(missing)))}, ...
                {2, struct('Charge', struct('Value', 3))}, ...                      % linear needs struct(Value, Rate)
                {2, struct('Charge', struct('Value', struct('Value', 3, 'Rate', Inf)))}, ...
                {2, struct('Position', [1 2])}, ...                                 % not the Command.Offer format
                {3, struct('Position', struct('Value', [1 2]))}, ...                % dead row
                {4, struct('Position', struct('Value', [1 2]))}, ...                % no such row
                {[2 2.5], struct('Position', struct('Value', [1 2]))}, ...
                {2, struct('Position', struct('Value', [1 2]), 'Born', struct('Value', Inf))}};  % second field bad
            for k = 1:numel(bads)
                tc.verifyError(@() s.setOffer(bads{k}{:}, 0), id, sprintf('bad offer %d', k));
                tc.verifyEqual(s.snapshot(), before, sprintf('bad offer %d changed the store', k));
            end
            % linear offer without a valid now
            tc.verifyError(@() s.setOffer(2, struct('Charge', struct('Value', struct('Value', 1, 'Rate', 0)))), id);
            tc.verifyEqual(s.snapshot(), before);
        end

        function badWritesChangeNothing(tc)
            s = tc.store();
            s.addRows(3);
            s.setOffer([1; 2], tc.full([1 1], 5, 0, 0), 0);
            s.setOffer(1, struct('Name', struct('Value', "n1")));
            s.killRows(2);
            before = s.snapshot();
            id = 'OceanAgents:World:BadValue';
            bads = { ...
                tc.write("Nope", 1, 1), ...
                tc.write("Position", 2, [1 1]), ...                 % dead row
                tc.write("Position", 3, [1 1]), ...                 % row without the component
                tc.write("Position", 1, [1 1 1]), ...               % width
                tc.write("Position", 1, int32([1 1])), ...          % class
                tc.write("Position", 1, [NaN 1]), ...
                tc.write("Position", [1; 1], [1 1]), ...            % handled below as WriteConflict
                tc.write("Position", 1, [1 1], [1 1]), ...          % Rates on a value component
                tc.write("Charge", 1, 3, [1 2]), ...                % Rates width
                tc.write("Charge", 1, 3, Inf), ...
                tc.write("Born", 1, [1 2])};                        % clock takes n x 1
            for k = 1:numel(bads)
                exp = id;
                if k == 7, exp = 'OceanAgents:World:WriteConflict'; end
                tc.verifyError(@() s.checkWrites(bads{k}, 3), exp, sprintf('bad write %d', k));
                tc.verifyError(@() s.applyWrites([tc.write("Name", 1, "valid"), bads{k}], 3), exp, ...
                    sprintf('bad write %d (in a list)', k));
                tc.verifyEqual(s.snapshot(), before, sprintf('bad write %d changed the store', k));
            end
            % the same (component, row) written by two entries -> WriteConflict
            tc.verifyError(@() s.applyWrites([tc.write("Position", 1, [1 1]), tc.write("Position", 1, [2 2])], 3), ...
                'OceanAgents:World:WriteConflict');
            tc.verifyError(@() s.applyWrites(tc.write("Charge", 1, 3), NaN), id);     % linear rebase needs now
            tc.verifyError(@() s.applyWrites(struct('Rows', 1, 'Values', 1), 3), id);
            tc.verifyEqual(s.snapshot(), before);
            s.checkWrites(tc.write("Position", 1, [4 4]), 3);                         % valid: no change either
            tc.verifyEqual(s.snapshot(), before);
            s.applyWrites([], 3);
            tc.verifyEqual(s.snapshot(), before);
        end

        function dirtyTracking(tc)
            s = tc.store();
            s.addRows(3);
            s.setOffer((1:3)', tc.full([0 0], 1, 1, 0), 0);
            tc.verifyEqual(s.takeDirty("Charge"), [1; 2; 3]);            % newly offered rows
            tc.verifyEqual(s.takeDirty("Born"), [1; 2; 3]);
            tc.verifyEqual(s.takeDirty("Charge"), zeros(0, 1));           % cleared by the take
            s.applyWrites(tc.write("Position", 2, [5 5]), 1);
            tc.verifyEqual(s.takeDirty("Charge"), zeros(0, 1));           % value writes do not count
            s.applyWrites([tc.write("Charge", 3, 2, 0), tc.write("Born", 1, 1)], 2);
            tc.verifyEqual(s.takeDirty("Charge"), 3);
            tc.verifyEqual(s.takeDirty("Born"), 1);
            s.applyWrites(tc.write("Charge", [1; 2], [1; 1]), 3);
            s.killRows(1);
            tc.verifyEqual(s.takeDirty("Charge"), 2, 'dead rows are never returned');
            s.clearComponent(2, "Charge");
            tc.verifyEqual(s.takeDirty("Charge"), 2, 'a detach changes the parameters');
            tc.verifyEqual(s.takeDirty("Position"), zeros(0, 1));
        end

        function killAndDetach(tc)
            s = tc.store();
            s.addRows(3);
            s.setOffer((1:3)', tc.full([7 7], 4, 2, 1), 0);
            s.killRows(2);
            tc.verifyEqual(s.isAlive((1:3)'), [true; false; true]);
            tc.verifyEqual(s.has("Position", (1:3)'), [true; false; true]);
            tc.verifyEqual(s.rowsWith("Position"), [1; 3]);
            tc.verifyEqual(s.addRows(1), 4, 'rows are appended, never reused or reordered');
            tc.verifyEqual(s.valueAt("Position", 3), [7 7]);
            s.clearComponent([1; 3], "Charge");
            tc.verifyEqual(s.has("Charge", [1; 3]), [false; false]);
            r = s.raw("Charge");
            tc.verifyEqual([r.Base(1) r.Rate(1) r.T0(1)], [0 0 0]);
            tc.verifyEqual(s.rowsWith(["Position" "Charge"]), zeros(0, 1));
            tc.verifyEqual(s.rowsWith("Born"), [1; 3]);
            tc.verifyError(@() s.clearComponent(1, "Nope"), 'OceanAgents:World:BadValue');
            tc.verifyError(@() s.has("Position", 9), 'OceanAgents:World:BadValue');
        end

        function rowsStableUnderGrowth(tc)
            % 300 single-row spawns (storage grows by doubling several times); every row keeps its data.
            s = tc.store();
            for i = 1:300
                r = s.addRows(1);
                tc.assertEqual(r, i);
                s.setOffer(r, struct('Position', struct('Value', [i -i]), 'Name', struct('Value', "e" + i)));
                if mod(i, 7) == 0, s.killRows(r); end
            end
            raw = s.raw("Position");
            alive = mod((1:300)', 7) ~= 0;
            expected = [(1:300)' -(1:300)'];
            tc.verifyEqual(raw.Data(alive, :), expected(alive, :));
            tc.verifyEqual(raw.Data(~alive, :), zeros(nnz(~alive), 2), 'killed rows are reset to defaults');
            tc.verifyEqual(s.rowsWith("Position"), find(alive));
            tc.verifyEqual(s.valueAt("Name", 299), "e299");
        end

        function snapshotRestoreIsExact(tc)
            s = tc.store();
            s.addRows(4);
            s.setOffer((1:4)', tc.full([1 2], 3, -0.25, 1), 2);
            s.takeDirty("Charge");
            s.applyWrites(tc.write("Charge", 2, 1.5, 0.125), 4);          % leaves row 2 dirty
            snap = s.snapshot();
            OceanAgents.Core.assertValueData(snap, "snap");
            s.addRows(40);
            s.setOffer(10, tc.full([9 9], 1, 1, 1), 9);
            s.applyWrites(tc.write("Position", 1, [100 100]), 9);
            s.killRows(3);
            s.clearComponent(4, "Born");
            s.restore(snap);
            tc.verifyEqual(s.snapshot(), snap);                           % bit-for-bit
            tc.verifyEqual(s.NumRows, 4);
            tc.verifyEqual(s.takeDirty("Charge"), 2);                     % dirty marks are state too
            tc.verifyEqual(s.addRows(1), 5, 'the store keeps working after restore');
            s.setOffer(5, tc.full([5 5], 5, 5, 5), 5);
            tc.verifyEqual(s.valueAt("Charge", [2; 5], 6), [1.5 + 0.125 * 2; 10]);
            other = OceanAgents.World.Store(struct('X', OceanAgents.World.Guard.ComponentType()));
            tc.verifyError(@() other.restore(snap), 'OceanAgents:World:BadValue');
            s.clearAll();
            tc.verifyEqual(s.NumRows, 0);
            tc.verifyEqual(s.addRows(1), 1);
        end

        function performanceOn1e5Rows(tc)
            % Reported, with only a loose sanity bound (the contract asks for vectorised code, not a benchmark).
            n = 1e5;
            s = tc.store();
            rows = s.addRows(n);
            rs = RandStream('mrg32k3a', 'Seed', 1);
            s.setOffer(rows, struct('Position', struct('Value', rand(rs, n, 2)), ...
                'Charge', struct('Value', struct('Value', rand(rs, n, 1), 'Rate', -rand(rs, n, 1)))), 0);
            vals = rand(rs, n, 2);
            w = [tc.write("Position", rows, vals), tc.write("Charge", rows, rand(rs, n, 1))];
            reps = 7;
            tv = zeros(1, reps);  tw = zeros(1, reps);
            for k = 1:reps
                t = tic;  v = s.valueAt("Charge", rows, 10 + k);  tv(k) = toc(t);
                t = tic;  s.applyWrites(w, 10 + k);                tw(k) = toc(t);
            end
            tc.verifyEqual(size(v), [n 1]);
            tc.verifyEqual(s.valueAt("Position", rows), vals);
            tc.log(matlab.unittest.Verbosity.Terse, sprintf( ...
                'Store on 1e5 rows (median of %d): valueAt(linear) %.3f ms, applyWrites(value 1e5x2 + linear 1e5) %.3f ms', ...
                reps, 1e3 * median(tv), 1e3 * median(tw)));
            tc.verifyLessThan(median(tv), 0.05);
            tc.verifyLessThan(median(tw), 0.05);
        end

        function pillars(tc)
            s = tc.store();
            m = s.getMetadata();
            tc.verifyFalse(m.IsSymbolic);
            tc.verifyTrue(all(isfield(m, {'SymbolicLatex', 'Symbols', 'Assumptions', 'Reference'})));
            p = s.getPipelineSpec();
            tc.verifyEqual(p.ComponentName, "Store");
        end
    end
end
