classdef HNSWInspector < handle
    %HNSWINSPECTOR Collect real HNSW decisions and expose bounded graph data.
    properties (SetAccess=private)
        Events cell = cell(1,0)
        MaxEvents (1,1) double = 10000
        MaxBytes (1,1) double = 64 * 1024 * 1024
        CapturedBytes (1,1) double = 0
        DroppedCount (1,1) double = 0
    end
    methods
        function obj=HNSWInspector(options)
            arguments
                options.MaxEvents (1,1) double {mustBeInteger,mustBeNonnegative}=10000
                options.MaxBytes (1,1) double {mustBeInteger,mustBeNonnegative}=64 * 1024 * 1024
            end
            obj.MaxEvents=options.MaxEvents;
            obj.MaxBytes=options.MaxBytes;
        end
        function attach(obj,index)
            arguments
                obj
                index (1,1) OceanAlgo.Search.HNSWIndex
            end
            index.setObserver(@(event)obj.record(event));
        end
        function record(obj,event)
            arguments
                obj
                event (1,1) struct
            end
            if numel(obj.Events)>=obj.MaxEvents
                obj.DroppedCount=obj.DroppedCount+1;
                return
            end
            info=whos('event');
            eventBytes=info.bytes;
            if eventBytes<=obj.MaxBytes-obj.CapturedBytes
                obj.Events{end+1}=event;
                obj.CapturedBytes=obj.CapturedBytes+eventBytes;
            else
                obj.DroppedCount=obj.DroppedCount+1;
            end
        end
        function clear(obj)
            arguments
                obj
            end
            obj.Events=cell(1,0);
            obj.CapturedBytes=0;
            obj.DroppedCount=0;
        end
        function [nodes,edges,G]=graphNeighborhood(obj,index,node,layer,depth,limit)
            arguments
                obj
                index (1,1) OceanAlgo.Search.HNSWIndex
                node (1,1) double {mustBeInteger,mustBePositive}
                layer (1,1) double {mustBeInteger,mustBeNonnegative}=0
                depth (1,1) double {mustBeInteger,mustBeNonnegative}=1
                limit (1,1) double {mustBeInteger,mustBePositive}=100
            end
            if layer+1>numel(index.Layers) || node>size(index.Points,1)
                error('OceanAlgo:HNSWInspector:Range','Node or layer is outside the index.');
            end
            seen=false(1,size(index.Points,1)); seen(node)=true;
            frontier=node; nodes=node;
            for step=1:depth
                next=zeros(1,0);
                for v=frontier
                    adjacent=index.Layers{layer+1}{v};
                    for u=adjacent
                        if ~seen(u) && numel(nodes)<limit
                            seen(u)=true; nodes(end+1)=u; next(end+1)=u; %#ok<AGROW>
                        end
                    end
                end
                frontier=next;
                if isempty(frontier) || numel(nodes)>=limit, break, end
            end
            edges=zeros(0,2);
            for v=nodes
                adjacent=index.Layers{layer+1}{v};
                adjacent=adjacent(seen(adjacent));
                for u=adjacent
                    pair=sort([v,u]);
                    if ~ismember(pair,edges,'rows'), edges(end+1,:)=pair; end %#ok<AGROW>
                end
            end
            if isempty(edges)
                G=graph([],[],[],numel(nodes));
            else
                [~,s]=ismember(edges(:,1),nodes); [~,t]=ismember(edges(:,2),nodes);
                G=graph(s,t,[],numel(nodes));
            end
            G.Nodes.NodeId=nodes(:);
        end
    end
end
