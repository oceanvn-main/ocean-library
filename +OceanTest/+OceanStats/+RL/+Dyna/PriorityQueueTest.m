classdef PriorityQueueTest < OceanTest.BaseTestCase
    methods (Test)
        function testMaxOrderAndDeterministicTie(testCase)
            q=OceanStats.RL.Dyna.Planning.MaxPriorityQueue();
            q.push(4,2); q.push(2,2); q.push(3,3);
            [id1,p1]=q.pop(); [id2,p2]=q.pop(); [id3,p3]=q.pop();
            testCase.verifyEqual([id1 id2 id3],[3 2 4]);
            testCase.verifyEqual([p1 p2 p3],[3 2 2],'AbsTol',0);
        end
        function testReplacementUsesLatestPriority(testCase)
            q=OceanStats.RL.Dyna.Planning.MaxPriorityQueue();
            q.push(2,5); q.push(2,1); q.push(3,2);
            [id1,p1]=q.pop(); [id2,p2]=q.pop(); stats=q.inspect();
            testCase.verifyEqual([id1 id2],[3 2]);
            testCase.verifyEqual([p1 p2],[2 1],'AbsTol',0);
            testCase.verifyGreaterThanOrEqual(stats.StaleDiscardCount,1);
        end
        function testEmptyPopHasStableID(testCase)
            q=OceanStats.RL.Dyna.Planning.MaxPriorityQueue();
            testCase.verifyError(@()q.pop(),'OceanStats:RL:Dyna:Planning:EmptyQueue');
        end
        function testRepeatedReplacementCompactsStaleEntries(testCase)
            q=OceanStats.RL.Dyna.Planning.MaxPriorityQueue();
            OceanTest.OceanStats.RL.Dyna.PriorityQueueTest.replacePair_(q,300);
            before=q.inspect(); [id,priority]=q.pop(); after=q.inspect();
            testCase.verifyLessThanOrEqual(before.HeapEntryCount,64);
            testCase.verifyEqual(before.ActiveCount,1);
            testCase.verifyEqual([id priority],[1 300],'AbsTol',0);
            testCase.verifyEqual(after.ActiveCount,0);
        end
    end
    methods (Static, Access=private)
        function replacePair_(queue,n)
            for k=1:n, queue.push(1,k); end
        end
    end
end
