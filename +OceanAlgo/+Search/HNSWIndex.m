classdef HNSWIndex < OceanAlgo.Search.IVectorIndex
    %HNSWINDEX Hierarchical Navigable Small World ANN index.
    properties (SetAccess=private)
        Points double=zeros(0,0)
        Layers cell=cell(1,0)
        Levels double=zeros(1,0)
        EntryPoint (1,1) double=0
        MaxLevel (1,1) double=-1
        Metric string="euclidean"
        MaxNeighbors (1,1) double=16
        EfConstruction (1,1) double=64
        RandomSeed (1,1) double=17
        IsFitted (1,1) logical=false
        Observer = []
    end
    methods
        function obj=HNSWIndex(options)
            arguments
                options.Metric string {mustBeMember(options.Metric,["euclidean","inner-product"])}="euclidean"
                options.MaxNeighbors (1,1) double {mustBeInteger,mustBePositive}=16
                options.EfConstruction (1,1) double {mustBeInteger,mustBePositive}=64
                options.RandomSeed (1,1) double {mustBeInteger}=17
                options.Observer = []
            end
            obj.Metric=options.Metric; obj.MaxNeighbors=options.MaxNeighbors;
            obj.EfConstruction=max(options.EfConstruction,options.MaxNeighbors); obj.RandomSeed=options.RandomSeed;
            obj.setObserver(options.Observer);
        end
        function setObserver(obj,observer)
            arguments
                obj
                observer
            end
            if ~isempty(observer) && ~isa(observer,'function_handle')
                error('OceanAlgo:HNSWIndex:Observer','Observer must be a function handle.');
            end
            obj.Observer=observer;
        end
        function fit(obj,X)
            arguments
                obj
                X (:,:) double
            end
            n=size(X,1); if n==0, error('OceanAlgo:HNSWIndex:EmptyInput','X must not be empty.'); end
            obj.IsFitted=false;
            obj.Points=X; obj.Levels=zeros(1,n); obj.Layers={}; obj.EntryPoint=1; obj.MaxLevel=0;
            stream=RandStream('mt19937ar','Seed',obj.RandomSeed);
            % Sample all levels and allocate the layer storage once. Mature
            % implementations (hnswlib/FAISS) use contiguous per-node link
            % storage; this two-pass layout removes dynamic cell growth from
            % the hot insertion loop while retaining inspectable MATLAB data.
            for i=1:n, obj.Levels(i)=obj.randomLevel_(stream); end
            obj.MaxLevel=max(obj.Levels); obj.Layers=cell(1,obj.MaxLevel+1);
            for layer=1:numel(obj.Layers), obj.Layers{layer}=cell(1,n); end
            for i=1:n
                level=obj.Levels(i);
                obj.emit_('sampleLevel',struct('Node',i,'Level',level));
                if i==1, obj.MaxLevel=level; obj.EntryPoint=1; continue, end
                ep=obj.EntryPoint;
                obj.emit_('insertStart',struct('Node',i,'Level',level,'EntryPoint',ep,'TopLayer',obj.MaxLevel));
                % Descend the existing hierarchy before connecting the new node.
                % This is the critical upper-layer HNSW construction step:
                % a node that lands below the current top must traverse every
                % existing layer from MaxLevel down to its own level + 1.
                for layer=obj.MaxLevel:-1:level+1, ep=obj.greedy_(X(i,:),ep,layer,i,'build'); end

                %Connects new point into the graph
                for layer=min(level,obj.MaxLevel):-1:0
                    [cand,cd]=obj.searchLayer_(X(i,:),ep,obj.EfConstruction,layer,i,'build');
                    chosen=obj.selectNeighbors_(X(i,:),cand,cd,obj.maxNeighbors_(layer)); obj.Layers{layer+1}{i}=chosen;
                    obj.emit_('neighborsSelected',struct('Node',i,'Layer',layer,'Candidates',cand,'Distances',cd,'Selected',chosen));
                    %Reciprocal links
                    for v=chosen
                        before=obj.Layers{layer+1}{v};
                        obj.Layers{layer+1}{v}=obj.prune_(before,i,layer,v);
                        obj.emit_('reciprocalLink',struct('Node',i,'Neighbor',v,'Layer',layer,'Before',before,'After',obj.Layers{layer+1}{v}));
                    end
                    if ~isempty(cand), ep=cand(1); end
                end
                if level>obj.MaxLevel
                    old=obj.EntryPoint; obj.MaxLevel=level; obj.EntryPoint=i;
                    obj.emit_('entryPointChanged',struct('Previous',old,'Current',i,'MaxLevel',level));
                end
            end
            obj.IsFitted=true;
        end
        function [ids,scores]=search(obj,Q,k,options)
            arguments
                obj
                Q (:,:) double
                k (1,1) double {mustBeInteger,mustBePositive}
                options.EfSearch (1,1) double {mustBeInteger,mustBePositive}=64
            end
            if ~obj.IsFitted, error('OceanAlgo:HNSWIndex:NotFitted','Call fit first.'); end
            if size(Q,2)~=size(obj.Points,2), error('OceanAlgo:HNSWIndex:DimMismatch','Query dimension mismatch.'); end
            k=min(k,size(obj.Points,1)); ids=zeros(size(Q,1),k); scores=-inf(size(Q,1),k);
            for r=1:size(Q,1)
                %Start highest level, greedy descent
                ep=obj.EntryPoint;
                obj.emit_('queryStart',struct('Query',r,'EntryPoint',ep,'TopLayer',obj.MaxLevel,'K',k,'EfSearch',max(options.EfSearch,k)));
                for layer=obj.MaxLevel:-1:1, ep=obj.greedy_(Q(r,:),ep,layer,r,'query'); end
                [cand,dist]=obj.searchLayer_(Q(r,:),ep,max(options.EfSearch,k),0,r,'query');
                take=min(k,numel(cand)); ids(r,1:take)=cand(1:take); scores(r,1:take)=-dist(1:take);
                obj.emit_('queryResult',struct('Query',r,'Ids',ids(r,:),'Scores',scores(r,:)));
            end
        end
        function meta=getMetadata(obj)
            meta=struct('Name','HNSWIndex','IsApproximate',true,'IsHierarchical',true,'Metric',obj.Metric,'M',obj.MaxNeighbors,'M0',2*obj.MaxNeighbors,'MaxNeighbors',obj.MaxNeighbors,'EfConstruction',obj.EfConstruction,'MaxLevel',obj.MaxLevel,'EntryPoint',obj.EntryPoint);
        end
    end
    methods (Access=private)
        function level=randomLevel_(obj,stream)
            % Canonical HNSW level law: P(L >= l) ~= M^(-l), where M is
            % the upper-layer degree parameter. Inverse-CDF sampling gives
            % L=floor(-log(U)/log(M)) for U~Uniform(0,1).
            base=max(obj.MaxNeighbors,2); u=max(rand(stream),realmin);
            level=min(floor(-log(u)/log(base)),32);
        end
        function v=greedy_(obj,q,v,layer,subject,phase)
            changed=true; while changed, changed=false; dv=obj.distance_(q,obj.Points(v,:));
                for u=obj.Layers{layer+1}{v}
                    du=obj.distance_(q,obj.Points(u,:));
                    obj.emit_('greedyCompare',struct('Phase',phase,'Subject',subject,'Layer',layer,'From',v,'Candidate',u,'CurrentDistance',dv,'CandidateDistance',du,'Accepted',du<dv));
                    if du<dv, v=u; dv=du; changed=true; end
                end
            end
        end

        function [ids,dist]=searchLayer_(obj,q,entry,ef,layer,subject,phase)
            % Canonical HNSW searchLayer: a min-distance candidate queue and
            % a max-distance result queue. A candidate is expanded while it
            % can still improve the current worst result.
            n=size(obj.Points,1); visited=false(1,n); visited(entry)=true;
            cand=entry; candDist=obj.distance_(q,obj.Points(entry,:));
            result=entry; resultDist=candDist;
            while ~isempty(cand)
                [d,p]=min(candDist); v=cand(p); cand(p)=[]; candDist(p)=[];
                [worst,~]=max(resultDist);
                obj.emit_('candidateExpanded',struct('Phase',phase,'Subject',subject,'Layer',layer,'Node',v,'Distance',d,'WorstDistance',worst,'ResultCount',numel(result),'Ef',ef));
                %If the best candidate still worst than the wosrt in the list, early stop.
                if numel(result)>=ef && d>worst
                    obj.emit_('searchStopped',struct('Phase',phase,'Subject',subject,'Layer',layer,'Node',v,'Distance',d,'WorstDistance',worst));
                    break
                end
                for u=obj.Layers{layer+1}{v}
                    if visited(u), continue, end
                    visited(u)=true; du=obj.distance_(q,obj.Points(u,:));
                    [worst,wp]=max(resultDist);
                    if numel(result)<ef || du<worst
                        obj.emit_('candidateAccepted',struct('Phase',phase,'Subject',subject,'Layer',layer,'From',v,'Node',u,'Distance',du,'WorstDistance',worst));
                        cand(end+1)=u; candDist(end+1)=du;
                        result(end+1)=u; resultDist(end+1)=du;
                        if numel(result)>ef, result(wp)=[]; resultDist(wp)=[]; end
                    end
                end
            end
            [dist,o]=sort(resultDist,'ascend'); ids=result(o);
        end

        function chosen=selectNeighbors_(obj,q,candidates,candidateDist,M)
            % HNSW heuristic: retain a candidate only when it is not already
            % covered by a closer selected neighbor. This preserves long,
            % geometrically diverse edges instead of taking plain k-NN.
            [candidateDist,o]=sort(candidateDist,'ascend'); candidates=candidates(o);
            chosen=zeros(1,0);
            for c=candidates
                if numel(chosen)>=M, break, end
                dqc=obj.distance_(q,obj.Points(c,:)); accept=true;
                if ~isempty(chosen)
                    dcs=obj.distance_(obj.Points(c,:),obj.Points(chosen,:));
                    accept=all(dcs>dqc);
                end
                if accept, chosen(end+1)=c; end
            end
            if numel(chosen)<M
                for c=candidates
                    if ~ismember(c,chosen), chosen(end+1)=c; end
                    if numel(chosen)>=M, break, end
                end
            end
        end
        function nb=prune_(obj,nb,newPoint,layer,owner)
            nb=unique([nb,newPoint],'stable'); M=obj.maxNeighbors_(layer);
            if numel(nb)<=M, return, end
            q=obj.Points(owner,:); ids=nb; d=obj.distance_(q,obj.Points(ids,:));
            nb=obj.selectNeighbors_(q,ids,d,M);
        end
        function M=maxNeighbors_(obj,layer)
            if layer==0, M=2*obj.MaxNeighbors; else, M=obj.MaxNeighbors; end
        end
        function d=distance_(obj,q,X)
            if obj.Metric=="euclidean", d=sum((X-q).^2,2); else, d=-(X*q'); end
        end
        function emit_(obj,name,data)
            if isempty(obj.Observer), return, end
            obj.Observer(struct('Name',string(name),'Data',data));
        end
    end
end
