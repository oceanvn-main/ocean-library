classdef TestDa2 < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            % --- DWSchema tests -------------------------------------------------- %
            RESULT_FILE = 'c:\Ocean Library\Matlab code practice\da2_result.txt';
            warehouseDir = tempname();
            mdmDir       = tempname();
            % Build a reusable schema
            schema = OceanData.DataArch.DWSchema( ...
            {'age', 'income', 'label', 'gender'}, ...
            {'double', 'double', 'logical', 'category'}, ...
            'Nullable',    [false, false, false, true], ...
            'MinValues',   [0, 0, NaN, NaN], ...
            'MaxValues',   [120, 1e7, NaN, NaN], ...
            'Categories',  struct('gender', {{'M','F','Other'}}));
            % T1: Good row passes validation
            row = struct('age', 35, 'income', 55000, 'label', true, 'gender', 'M');
            schema.validate(row);
            testCase.verifyTrue(true, "T1_good_row_passes");
            % T2: Wrong type (string for numeric)
            thrown = false;
            schema.validate(struct('age','old','income',1000,'label',true,'gender','M'));
            testCase.verifyTrue(thrown, "T2_wrong_type_rejected");
            % T3: Null in non-nullable column
            thrown = false;
            schema.validate(struct('age',NaN,'income',1000,'label',true,'gender','M'));
            testCase.verifyTrue(thrown, "T3_null_rejected");
            % T4: Out of range (age > 120)
            thrown = false;
            schema.validate(struct('age',200,'income',1000,'label',true,'gender','M'));
            testCase.verifyTrue(thrown, "T4_range_rejected");
            % T5: Invalid category
            thrown = false;
            schema.validate(struct('age',30,'income',1000,'label',true,'gender','X'));
            testCase.verifyTrue(thrown, "T5_category_rejected");
            % --- SchemaRegistry tests -------------------------------------------- %
            % T6: Register / get / has / list
            reg = OceanData.DataArch.SchemaRegistry();
            reg.register('users', schema);
            s   = reg.get('users');
            keys = reg.list();
            results.T6_registry_works = reg.has('users') && ...
            ~reg.has('unknown') && ...
            numel(keys) == 1 && ...
            s.NumColumns == 4;
            % --- DataWarehouse tests --------------------------------------------- %
            dw = OceanData.DataArch.DataWarehouse('rootDir', string(warehouseDir));
            dw.createTable('users', schema);
            row1 = struct('age',25,'income',40000,'label',true, 'gender','M');
            row2 = struct('age',40,'income',80000,'label',false,'gender','F');
            row3 = struct('age',60,'income',30000,'label',true, 'gender','Other');
            % T7: Insert valid rows and query all back
            dw.insert('users', row1);
            dw.insert('users', row2);
            dw.insert('users', row3);
            rows = dw.query('users');
            testCase.verifyTrue(numel(rows) == 3, "T7_insert_query");
            % T8: Invalid row is rejected
            thrown = false;
            dw.insert('users', struct('age', 999, 'income', 0, 'label', true, 'gender', 'M'));
            testCase.verifyTrue(thrown, "T8_invalid_rejected");
            % T9: Query with filter
            filtered_label = dw.query('users', 'label == 1');
            filtered_age   = dw.query('users', 'age > 30');
            filtered_neq   = dw.query('users', 'income != 40000');
            results.T9_filter_works = numel(filtered_label) == 2 && ...
            numel(filtered_age)   == 2 && ...
            numel(filtered_neq)   == 2;
            % T10: rowCount increments
            testCase.verifyTrue(dw.rowCount('users') == 3, "T10_rowcount");
            % T11: delete clears rows (table still exists)
            dw.delete('users');
            results.T11_delete_clears = dw.exists('users') && ...
            numel(dw.query('users')) == 0;
            % --- MDMStore tests -------------------------------------------------- %
            mdm = OceanData.DataArch.MDMStore('rootDir', string(mdmDir));
            % T12: merge 'latest' overwrites
            mdm.merge('e001', struct('name','Alice','age',30), 'latest');
            mdm.merge('e001', struct('name','Alice','age',32), 'latest');
            rec = mdm.resolve('e001');
            testCase.verifyTrue(rec.age == 32, "T12_merge_latest");
            % T13: merge 'union' preserves existing, adds new fields
            mdm.merge('e002', struct('name','Bob','age',25), 'latest');
            mdm.merge('e002', struct('name','Bob','age',30,'email','b@x.com'), 'union');
            rec = mdm.resolve('e002');
            % 'union': existing age=25 preserved, email added
            testCase.verifyTrue(rec.age == 25 && isfield(rec,'email'), "T13_merge_union");
            % T14: merge 'max' keeps maximum numeric values
            mdm.merge('e003', struct('score',80,'level',3), 'latest');
            mdm.merge('e003', struct('score',90,'level',2), 'max');
            rec = mdm.resolve('e003');
            testCase.verifyTrue(rec.score == 90 && rec.level == 3, "T14_merge_max");
            % T15: resolve unknown entity throws EntityNotFound
            thrown = false;
            mdm.resolve('UNKNOWN_XYZ');
            testCase.verifyTrue(thrown, "T15_resolve_unknown_throws");

        end
    end
end
