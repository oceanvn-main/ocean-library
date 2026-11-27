classdef HNSWReaderAdapter
    %HNSWREADERADAPTER Map observed HNSW events into the NLP explorer reader.
    methods (Static)
        function reader=fromInspector(inspector,index,options)
            arguments
                inspector (1,1) OceanAlgo.Search.Inspect.HNSWInspector
                index (1,1) OceanAlgo.Search.HNSWIndex
                options.CenterNode (1,1) double {mustBeInteger,mustBePositive}=1
                options.Layer (1,1) double {mustBeInteger,mustBeNonnegative}=0
                options.Depth (1,1) double {mustBeInteger,mustBeNonnegative}=1
                options.NodeLimit (1,1) double {mustBeInteger,mustBePositive}=100
            end
            count=numel(inspector.Events);
            steps=repmat(struct('Id',"",'Name',"",'Stage',"", ...
                'InputIds',strings(1,0),'OutputIds',strings(1,0), ...
                'Summary',"",'Status',"",'Sequence',0),1,count+1);
            artifacts=repmat(struct('Id',"",'Kind',"",'Name',"", ...
                'Value',[],'Summary',"",'Status',"recorded",'ByteCount',0),1,count+1);
            for i=1:count
                event=inspector.Events{i};
                artifactId="hnsw-event-"+i;
                name=string(event.Name);
                summary=OceanAlgo.Search.Inspect.HNSWReaderAdapter.summary_(event);
                eventData=event.Data; eventInfo=whos('eventData');
                artifacts(i)=struct('Id',artifactId,'Kind',"structure", ...
                    'Name',name+" data",'Value',eventData,'Summary',summary, ...
                    'Status',"recorded",'ByteCount',eventInfo.bytes);
                steps(i)=struct('Id',"hnsw-step-"+i,'Name',name, ...
                    'Stage',OceanAlgo.Search.Inspect.HNSWReaderAdapter.stage_(event), ...
                    'InputIds',strings(1,0),'OutputIds',artifactId, ...
                    'Summary',summary,'Status',"completed",'Sequence',i);
            end
            if index.IsFitted
                [nodes,edges,G]=inspector.graphNeighborhood(index,options.CenterNode, ...
                    options.Layer,options.Depth,options.NodeLimit);
                graphValue=struct('Graph',G,'Nodes',nodes,'Edges',edges, ...
                    'Layer',options.Layer,'CenterNode',options.CenterNode, ...
                    'SelectedNodes',options.CenterNode,'VisitedNodes',nodes, ...
                    'CandidateNodes',zeros(1,0));
                artifactId="hnsw-graph";
                summary="Layer "+options.Layer+" neighborhood: "+numel(nodes)+ ...
                    " nodes, "+size(edges,1)+" edges";
                graphInfo=whos('graphValue');
                artifacts(count+1)=struct('Id',artifactId,'Kind',"graph", ...
                    'Name',"Index neighborhood",'Value',graphValue,'Summary',summary, ...
                    'Status',"recorded",'ByteCount',graphInfo.bytes);
                steps(count+1)=struct('Id',"hnsw-step-"+(count+1), ...
                    'Name',"Graph neighborhood",'Stage',"Index structure", ...
                    'InputIds',strings(1,0),'OutputIds',artifactId, ...
                    'Summary',summary,'Status',"completed",'Sequence',count+1);
            else
                steps=steps(1:count); artifacts=artifacts(1:count);
            end
            reader=OceanNLP.Inspect.MemoryReader(steps,artifacts);
        end
    end
    methods (Static, Access=private)
        function stage=stage_(event)
            if startsWith(string(event.Name),"query")
                stage="HNSW query";
            elseif isfield(event.Data,'Phase') && strcmp(event.Data.Phase,'query')
                stage="HNSW query";
            else
                stage="HNSW build";
            end
        end
        function summary=summary_(event)
            data=event.Data; name=string(event.Name);
            if isfield(data,'Node') && isfield(data,'Layer')
                summary=name+" · node "+data.Node+" · layer "+data.Layer;
            elseif isfield(data,'Node')
                summary=name+" · node "+data.Node;
            elseif isfield(data,'Query')
                summary=name+" · query "+data.Query;
            else
                summary=name;
            end
        end
    end
end
