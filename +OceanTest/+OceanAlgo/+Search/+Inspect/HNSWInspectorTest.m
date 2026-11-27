classdef HNSWInspectorTest < OceanTest.BaseTestCase
    methods (Test)
        function testObserverPreservesResultsAndReportsDecisions(testCase)
            X=[0 0; 1 0; 0 1; 2 1; -1 1; 1 2];
            plain=OceanAlgo.Search.HNSWIndex('MaxNeighbors',2,'EfConstruction',4,'RandomSeed',7);
            traced=OceanAlgo.Search.HNSWIndex('MaxNeighbors',2,'EfConstruction',4,'RandomSeed',7);
            inspector=OceanAlgo.Search.Inspect.HNSWInspector(); inspector.attach(traced);
            plain.fit(X); traced.fit(X);
            [a,sa]=plain.search(X(1:2,:),3); [b,sb]=traced.search(X(1:2,:),3);
            testCase.verifyEqual(a,b); testCase.verifyEqual(sa,sb);
            testCase.verifyEqual(plain.Layers,traced.Layers);
            names=cellfun(@(e)char(e.Name),inspector.Events,'UniformOutput',false);
            testCase.verifyTrue(any(strcmp(names,'neighborsSelected')));
            testCase.verifyTrue(any(strcmp(names,'candidateExpanded')));
            testCase.verifyTrue(any(strcmp(names,'queryResult')));
            event=inspector.Events{find(strcmp(names,'neighborsSelected'),1)};
            testCase.verifyEqual(numel(event.Data.Candidates),numel(event.Data.Distances));
        end
        function testGraphNeighborhoodUsesActualLinks(testCase)
            X=[0 0; 1 0; 0 1; 2 1; -1 1];
            index=OceanAlgo.Search.HNSWIndex('MaxNeighbors',2,'RandomSeed',5); index.fit(X);
            inspector=OceanAlgo.Search.Inspect.HNSWInspector();
            [nodes,edges,G]=inspector.graphNeighborhood(index,1,0,1,3);
            testCase.verifyLessThanOrEqual(numel(nodes),3);
            testCase.verifyEqual(numnodes(G),numel(nodes));
            testCase.verifyEqual(G.Nodes.NodeId,nodes(:));
            for i=1:size(edges,1)
                testCase.verifyTrue(ismember(edges(i,2),index.Layers{1}{edges(i,1)}) || ...
                    ismember(edges(i,1),index.Layers{1}{edges(i,2)}));
            end
        end
        function testBoundedCaptureAndReaderIntegration(testCase)
            X=[0 0; 1 0; 0 1; 2 1; -1 1];
            index=OceanAlgo.Search.HNSWIndex('MaxNeighbors',2,'RandomSeed',5);
            inspector=OceanAlgo.Search.Inspect.HNSWInspector('MaxEvents',3);
            inspector.attach(index); index.fit(X); index.search(X(1,:),2);
            testCase.verifyEqual(numel(inspector.Events),3);
            testCase.verifyGreaterThan(inspector.DroppedCount,0);
            reader=OceanAlgo.Search.Inspect.HNSWReaderAdapter.fromInspector( ...
                inspector,index,'CenterNode',1,'Layer',0,'Depth',1,'NodeLimit',3);
            steps=reader.listSteps();
            testCase.verifyEqual(numel(steps),4);
            for i=1:3
                artifact=reader.getArtifact(steps(i).OutputIds);
                testCase.verifyEqual(artifact.Value,inspector.Events{i}.Data);
                testCase.verifyEqual(artifact.Status,"recorded");
                testCase.verifyGreaterThan(artifact.ByteCount,0);
                testCase.verifyEmpty(steps(i).InputIds);
            end
            graphArtifact=reader.getArtifact(steps(end).OutputIds);
            testCase.verifyEqual(graphArtifact.Kind,"graph");
            testCase.verifyEqual(graphArtifact.Status,"recorded");
            testCase.verifyGreaterThan(graphArtifact.ByteCount,0);
            testCase.verifyEmpty(steps(end).InputIds);
            testCase.verifyEqual(numnodes(graphArtifact.Value.Graph), ...
                numel(graphArtifact.Value.Nodes));
        end
        function testByteBudgetDropsWithoutExceedingLimit(testCase)
            inspector=OceanAlgo.Search.Inspect.HNSWInspector( ...
                'MaxEvents',100,'MaxBytes',500);
            event=struct('Name',"candidateExpanded", ...
                'Data',struct('Candidates',1:1000));
            inspector.record(event);
            testCase.verifyEmpty(inspector.Events);
            testCase.verifyEqual(inspector.DroppedCount,1);
            testCase.verifyLessThanOrEqual(inspector.CapturedBytes,inspector.MaxBytes);
            inspector.clear();
            testCase.verifyEqual(inspector.DroppedCount,0);
            testCase.verifyEqual(inspector.CapturedBytes,0);
        end
        function testFailedRefitClearsFittedState(testCase)
            X=[0 0; 1 0; 0 1];
            index=OceanAlgo.Search.HNSWIndex(); index.fit(X);
            index.setObserver(@(~)error('OceanTest:ObserverFailure','Observer failed'));
            testCase.verifyError(@()index.fit(X),'OceanTest:ObserverFailure');
            testCase.verifyFalse(index.IsFitted);
            testCase.verifyError(@()index.search(X(1,:),1),'OceanAlgo:HNSWIndex:NotFitted');
        end
    end
end
