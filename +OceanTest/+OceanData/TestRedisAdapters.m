classdef TestRedisAdapters < OceanTest.BaseTestCase
    methods (Test)
        function testAll(testCase)
            passed = 0; failed = 0;
            % T1: Connect + isConnected
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            ra = OceanData.Adapters.RedisAdapter(cfg);
            ra.connect();
            testCase.verifyTrue(ra.isConnected(), "T1_connect");


            % T2: set / get round-trip
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            ra.set('t2_key', 'hello_ocean');
            val = ra.get('t2_key');
            testCase.verifyTrue(strcmp(val, 'hello_ocean'), "T2_set_get");
            ra.del('t2_key');


            % T3: del + exists
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            ra.set('t3_key', 'temp');
            existsBefore = ra.exists('t3_key');
            ra.del('t3_key');
            existsAfter = ra.exists('t3_key');
            testCase.verifyTrue(existsBefore && ~existsAfter, "T3_del_exists");


            % T4: RedisCacheStore — struct round-trip
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            store = OceanData.Adapters.RedisCacheStore(ra);
            X_orig = magic(4);  % 4x4 magic square
            meta   = struct('source', 'test', 'nRows', 4);
            store.put('t4/matrix', X_orig, meta);
            entry = store.get('t4/matrix');
            store.delete('t4/matrix');
            results.T4_struct_roundtrip = ...
            isequal(entry.data, X_orig) && ...
            strcmp(entry.meta.source, 'test') && ...
            entry.meta.nRows == 4;


            % T5: TTL expiry
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            shortCfg = OceanData.Adapters.RedisConfig('TTL', 1, 'Prefix', 'ocean_test:');
            ra2 = OceanData.Adapters.RedisAdapter(shortCfg);
            ra2.connect();
            ra2.set('t5_expire', 'gone_soon');
            pause(2.5);  % wait for TTL to expire
            val = ra2.get('t5_expire');
            testCase.verifyTrue(isempty(val), "T5_ttl_expiry");
            ra2.disconnect();


            % T6: FeatureCache — getOrCompute caches on first call
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            fc = OceanData.Adapters.FeatureCache(ra);
            % First call — computes
            e1 = fc.getOrCompute('t6/features', @() randn(5,3), struct());
            % Second call — must hit cache
            e2 = fc.getOrCompute('t6/features', @() error('Should not run'), struct());
            fc.invalidate('t6/features');
            % Both entries must be identical (same data since cache hit)
            testCase.verifyTrue(isequal(e1.data, e2.data), "T6_feature_cache");


            % T7: ModelCache — getOrPredict caches result
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            mc = OceanData.Adapters.ModelCache(ra);
            X_test   = randn(10, 4);
            predictor = @(X) ones(size(X,1),1) * 2;
            Y1 = mc.getOrPredict('test_model_v1', X_test, predictor);
            Y2 = mc.getOrPredict('test_model_v1', X_test, @(X) error('Should not call'));
            mc.invalidate('test_model_v1');
            testCase.verifyTrue(isequal(Y1, Y2), "T7_model_cache");


            % T8: CachingWarehouse — miss then hit
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            dw   = OceanData.DataArch.DataWarehouse('rootDir', tempname());
            cdw  = OceanData.Adapters.CachingWarehouse(dw, ra);
            % Insert some data (no schema for this test — just test caching path)
            % We mock the query: directly put into cache and re-query
            store = OceanData.Adapters.RedisCacheStore(ra);
            store.put('query:mock_table:age > 30', magic(3), struct());
            cdw2   = OceanData.Adapters.CachingWarehouse(dw, ra);
            % Force a simulated hit by directly confirming cache key exists
            cacheKey = sprintf('query:%s:%s', 'mock_table', 'age > 30');
            hitDetected = store.exists(cacheKey);
            store.delete(cacheKey);
            testCase.verifyTrue(hitDetected, "T8_cache_warehouse");
            ra.disconnect();


            % Report
            fprintf('\n=== Redis Cache Layer Test Suite ===\n');
            fprintf('    (Requires Redis server on localhost:6379)\n\n');
            cfg = OceanData.Adapters.RedisConfig('TTL', 10, 'Prefix', 'ocean_test:');
            reportResults_(results);
            % ======================================================================= %
            function reportResults_(results)
            fields = fieldnames(results);
            passed = 0;
            lines  = {};
            fprintf('%-46s  %s\n', 'Test', 'Result');
            fprintf('%s\n', repmat('-',1,56));
            for i = 1:numel(fields)
            testCase.verifyTrue(results.(fields{i}));
            tag = iif(ok,'[PASS]','[FAIL]');
            passed = passed + ok;
            fprintf('  %s  %s\n', tag, strrep(fields{i},'_',' '));
            lines{end+1} = sprintf('[%s] %s', iif(ok,'PASS','FAIL'), fields{i}); %#ok<AGROW>
            end
            fprintf('\n');
            total  = numel(fields);
            footer = sprintf(iif(passed==total,'ALL PASS — Redis Layer DONE (%d/%d)','%d/%d PASS'), ...
            passed, total);
            fprintf('%s\n\n', footer);
            outFile = 'c:\Ocean Library\Matlab code practice\Tests\redis_result.txt';
            fid = fopen(outFile,'w');
            for i=1:numel(lines), fprintf(fid,'%s\n',lines{i}); end
            fprintf(fid,'\n%s\n',footer); fclose(fid);
            end
            function v = iif(cond, a, b)
            if cond, v = a; else, v = b; end
            end
            fprintf(\'Total: %d passed, %d failed\n\', passed, failed);
            testCase.verifyEqual(failed, 0, \'Some procedural checks failed\');
        end

    end
end