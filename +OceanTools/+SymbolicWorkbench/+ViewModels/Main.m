classdef Main < handle
    %MAIN Presentation state for the native scalar symbolic workbench.
    properties (SetObservable)
        Expression string = ""
        Operation string = "inspect"
        Variable string = "x"
        Substitutions string = ""
        Status string = "Declare symbols or import a file to begin. Source files are never executed."
        ReportText string = ""
        AIDraftText string = "No AI draft requested. Symbolic calculations are local."
        CompactLatex string = ""
        ExpandedLatex string = ""
        ResultLatex string = ""
        SymbolRows cell = cell(0,2)
        BlockRows cell = cell(0,4)
        Notices string = ""
        SourceText string = ""
        SourcePath string = ""
        Candidates string = strings(0,1)
        SourceSummary string = "No MATLAB or Python source scanned."
        FunctionRows cell = cell(0,6)
        FormulaRows cell = cell(0,5)
        SelectedFunctionDetail string = "Select a function to inspect its symbolic interface and evidence."
        AcceptedRows cell = cell(0,5)
    end
    properties (SetAccess=private)
        Workspace
        Registry
        CandidateRecords struct = struct('expression',{},'provenance',{})
        SourceFunctions struct = struct([])
        SelectedFunctionIndex double = 0
        RejectedFunctionIds string = strings(0,1)
        History cell = {}
        StagedExpression string = ""
        StagedProvenance struct = struct()
    end
    methods
        function obj = Main(workspace,registry)
            arguments
                workspace (1,1) OceanMath.Symbolic.Workspace
                registry (1,1) OceanMath.Symbolic.Registry = OceanMath.Symbolic.Registry()
            end
            obj.Workspace = workspace;
            obj.Registry = registry;
            obj.refresh();
        end

        function selectSourceFunction(obj,index)
            arguments
                obj
                index (1,1) double {mustBeInteger,mustBePositive}
            end
            if index>numel(obj.SourceFunctions), return; end
            obj.SelectedFunctionIndex=index;
            record=obj.SourceFunctions(index);
            obj.FormulaRows=obj.formulaRows(record);
            obj.SelectedFunctionDetail=obj.describeRecord(record);
            obj.Status="Selected "+string(record.signature)+". Review the abstract interface separately from formula candidates.";
        end

        function selectFormula(obj,index)
            arguments
                obj
                index (1,1) double {mustBeInteger,mustBePositive}
            end
            if obj.SelectedFunctionIndex<1, return; end
            record=obj.SourceFunctions(obj.SelectedFunctionIndex);
            formulas=obj.allFormulas(record);
            if index>numel(formulas), return; end
            formula=formulas(index);
            obj.Expression=string(formula.expression);
            obj.StagedExpression=obj.Expression;
            obj.StagedProvenance=struct('kind',char(formula.origin),'sourcePath',char(record.file), ...
                'sourceLines',formula.sourceLines,'functionId',char(record.id), ...
                'computationalSupport',char(formula.computationalSupport));
            obj.ReportText="Formula staged for review."+newline+string(jsonencode(obj.StagedProvenance,'PrettyPrint',true));
            obj.Status="Staged a "+string(formula.origin)+" formula. It is not accepted and its contract is not yet proven.";
        end

        function acceptSelectedAbstract(obj,requestedAlias)
            arguments
                obj
                requestedAlias (1,1) string = ""
            end
            if obj.SelectedFunctionIndex<1
                obj.Status="Select a function first."; return;
            end
            record=obj.SourceFunctions(obj.SelectedFunctionIndex);
            if any(obj.RejectedFunctionIds==string(record.id))
                obj.Status="This function was rejected for the current review session."; return;
            end
            if record.classification=="not-extractable"
                obj.Status="This function has no usable symbolic output contract and cannot be accepted."; return;
            end
            record.classification="opaque";
            record.expression="";
            record.computationalSupport="abstract-only";
            try
                alias=obj.Registry.accept(record,requestedAlias);
                obj.Status="Accepted "+alias+" as an opaque operator. It can be composed and contract-checked, but not expanded or computed through.";
            catch err
                obj.Status="Cannot accept abstract operator: "+string(err.message);
            end
            obj.refresh();
        end

        function acceptSelectedFormula(obj,formulaIndex,requestedAlias,variableMapping)
            arguments
                obj
                formulaIndex (1,1) double {mustBeInteger,mustBePositive}
                requestedAlias (1,1) string = ""
                variableMapping (1,1) string = ""
            end
            if obj.SelectedFunctionIndex<1
                obj.Status="Select a function first."; return;
            end
            record=obj.SourceFunctions(obj.SelectedFunctionIndex);
            if any(obj.RejectedFunctionIds==string(record.id))
                obj.Status="This function was rejected for the current review session."; return;
            end
            formulas=obj.allFormulas(record);
            if formulaIndex>numel(formulas)
                obj.Status="Select a formula candidate first."; return;
            end
            formula=formulas(formulaIndex);
            if formula.origin=="documentation-derived"
                [valid,mapped,detail]=obj.validateDocumentationMapping(string(formula.expression),record,variableMapping);
                if ~valid
                    obj.Status="Documentation formula remains display-only: "+detail;
                    return;
                end
                formula.expression=mapped;
                formula.computationalSupport="display-only";
            end
            record.id=record.id+"#formula"+formulaIndex;
            record.classification="transparent";
            record.expression=string(formula.expression);
            record.computationalSupport=string(formula.computationalSupport);
            record.expressionSourceLine=formula.sourceLines(1);
            try
                alias=obj.Registry.accept(record,requestedAlias);
                obj.Expression=string(formula.expression);
                obj.StagedExpression=obj.Expression;
                obj.StagedProvenance=struct('kind',char(formula.origin),'sourcePath',char(record.file), ...
                    'sourceLines',formula.sourceLines,'functionId',char(record.id), ...
                    'computationalSupport',char(formula.computationalSupport));
                if any(string(formula.computationalSupport)==["conditional-symbolic","matlab-symbolic"])
                    obj.Status="Accepted formula as "+alias+". Confirm scalar input/output contracts before using the MATLAB computation adapter.";
                else
                    obj.Status="Accepted formula as "+alias+" for composition/display. Confirm its contracts; this formula is not supported by the scalar MATLAB computation adapter.";
                end
            catch err
                obj.Status="Cannot accept formula: "+string(err.message);
            end
            obj.refresh();
        end

        function rejectSelected(obj)
            arguments
                obj
            end
            if obj.SelectedFunctionIndex<1
                obj.Status="Select a function first."; return;
            end
            record=obj.SourceFunctions(obj.SelectedFunctionIndex);
            obj.RejectedFunctionIds(end+1,1)=string(record.id);
            obj.FunctionRows{obj.SelectedFunctionIndex,3}='review-rejected';
            obj.Status="Rejected "+string(record.signature)+" for this review session. The source file was not changed.";
            obj.SelectedFunctionDetail=obj.describeRecord(record)+newline+newline+"REVIEW DECISION: rejected";
        end

        function composeOperator(obj,alias,argumentsCSV,domainsCSV,shapesCSV)
            arguments
                obj
                alias (1,1) string
                argumentsCSV (1,1) string
                domainsCSV (1,1) string = ""
                shapesCSV (1,1) string = ""
            end
            expressions=obj.csv(argumentsCSV);
            domains=obj.csv(domainsCSV);
            if isempty(domains), domains=repmat("unknown",size(expressions)); end
            shapes=obj.csv(shapesCSV);
            if isempty(shapes), shapes=repmat("unknown",size(expressions)); end
            if numel(domains)~=numel(expressions)
                obj.Status="Provide one domain per argument, or leave domains empty."; return;
            end
            if numel(shapes)~=numel(expressions)
                obj.Status="Provide one shape per argument, or leave shapes empty."; return;
            end
            argumentsList=struct('expression',{},'domain',{},'shape',{});
            for k=1:numel(expressions)
                argumentsList(k)=struct('expression',expressions(k),'domain',domains(k),'shape',shapes(k)); %#ok<AGROW>
            end
            try
                composition=obj.Registry.compose(alias,argumentsList);
                obj.Expression=string(composition.expression);
                lines=["Formal composition: "+obj.Expression; "Output domain: "+string(composition.domain); ...
                    "Output shape: "+string(composition.shape); ""; "Contract checks (unknown is not a pass):"];
                for k=1:numel(composition.checks)
                    c=composition.checks(k);
                    lines(end+1)=string(c.label)+" — "+string(c.status)+": "+string(c.detail); %#ok<AGROW>
                end
                obj.ReportText=join(lines,newline);
                entry=obj.Registry.Records(find(string({obj.Registry.Records.alias})==alias,1));
                contracts=[entry.record.contracts.inputs entry.record.contracts.outputs];
                shapes=string({contracts.shape});
                scalarReady=~isempty(shapes) && all(shapes=="scalar");
                if entry.record.classification=="transparent" && ...
                        entry.record.computationalSupport=="conditional-symbolic" && scalarReady
                    expanded=obj.Registry.expand(alias,expressions);
                    obj.StagedExpression=expanded;
                    obj.StagedProvenance=struct('kind','registry-expansion','operatorId',char(entry.record.id), ...
                        'sourcePath',char(entry.record.file),'sourceLines',entry.record.sourceLines);
                    obj.Status="Composition contracts checked. The supported transparent expansion is staged for MATLAB symbolic work.";
                else
                    obj.Status="Composition contracts checked. This operator remains formal; MATLAB cannot compute through it.";
                end
            catch err
                obj.Status="Cannot compose: "+string(err.message);
            end
        end

        function updateOperatorContracts(obj,alias,inputDomainsCSV,inputShapesCSV,outputDomainsCSV,outputShapesCSV)
            arguments
                obj
                alias (1,1) string
                inputDomainsCSV (1,1) string
                inputShapesCSV (1,1) string
                outputDomainsCSV (1,1) string
                outputShapesCSV (1,1) string
            end
            try
                index=find(string({obj.Registry.Records.alias})==alias,1);
                if isempty(index), error('OceanTools:SymbolicWorkbench:Alias','Unknown accepted alias.'); end
                record=obj.Registry.Records(index).record;
                inputDomains=obj.csv(inputDomainsCSV); inputShapes=obj.csv(inputShapesCSV);
                outputDomains=obj.csv(outputDomainsCSV); outputShapes=obj.csv(outputShapesCSV);
                if numel(inputDomains)~=numel(record.inputs) || numel(inputShapes)~=numel(record.inputs) || ...
                        numel(outputDomains)~=numel(record.outputs) || numel(outputShapes)~=numel(record.outputs)
                    error('OceanTools:SymbolicWorkbench:Contracts','Contract lists must match the source input/output counts.');
                end
                allowedDomains=["unknown","complex","real","positive","nonnegative","integer"];
                allowedShapes=["unknown","scalar","vector","matrix","function"];
                if any(~ismember([inputDomains outputDomains],allowedDomains))
                    error('OceanTools:SymbolicWorkbench:Contracts','Use a controlled domain: unknown, complex, real, positive, nonnegative, or integer.');
                end
                if any(~ismember([inputShapes outputShapes],allowedShapes))
                    error('OceanTools:SymbolicWorkbench:Contracts','Use a controlled shape: unknown, scalar, vector, matrix, or function.');
                end
                inputs=struct('name',{},'kind',{},'domain',{},'shape',{});
                outputs=struct('name',{},'kind',{},'domain',{},'shape',{});
                for k=1:numel(record.inputs)
                    inputs(k)=struct('name',record.inputs(k),'kind',"value",'domain',inputDomains(k),'shape',inputShapes(k)); %#ok<AGROW>
                end
                for k=1:numel(record.outputs)
                    outputs(k)=struct('name',record.outputs(k),'kind',"value",'domain',outputDomains(k),'shape',outputShapes(k)); %#ok<AGROW>
                end
                obj.Registry.updateContracts(alias,inputs,outputs);
                obj.Status="Updated contracts for "+alias+". Re-compose the operator to validate arguments.";
            catch err
                obj.Status="Cannot update contracts: "+string(err.message);
            end
            obj.refresh();
        end

        function useStagedExpansion(obj)
            arguments
                obj
            end
            if strlength(obj.StagedExpression)==0
                obj.Status="No supported transparent expansion is staged."; return;
            end
            obj.Expression=obj.StagedExpression;
            obj.Status="Expanded expression moved to the MATLAB symbolic editor. Confirm symbol domains before calculation.";
        end
        function addSymbol(obj,name,domain)
            arguments
                obj
                name (1,1) string
                domain (1,1) string
            end
            obj.guard(@() obj.Workspace.addSymbol(name,domain),"Symbol declared.");
        end
        function defineBlock(obj,name,paramsCSV,domainsCSV,expression,outputDomain)
            arguments
                obj
                name (1,1) string
                paramsCSV (1,1) string
                domainsCSV (1,1) string
                expression (1,1) string
                outputDomain (1,1) string
            end
            params = obj.csv(paramsCSV); domains = obj.csv(domainsCSV);
            provenance = struct('kind','user-defined','sourcePath',char(obj.SourcePath));
            if expression==obj.StagedExpression && ~isempty(fieldnames(obj.StagedProvenance))
                provenance=obj.StagedProvenance;
                provenance.reviewStatus='user-promoted; equivalence to source not proven';
            end
            obj.guard(@() obj.Workspace.define(name,params,expression,domains,outputDomain,provenance),"Block defined; its full expression remains available.");
        end
        function renameBlock(obj,oldName,newName)
            arguments
                obj
                oldName (1,1) string
                newName (1,1) string
            end
            try
                actual = obj.Workspace.rename(oldName,newName);
                obj.Expression = string(regexprep(char(obj.Expression),['\<' regexptranslate('escape',char(oldName)) '\>'],char(actual)));
                obj.Status = "Renamed " + oldName + " to " + actual + ". Dependent definitions updated.";
            catch err
                obj.Status = "Cannot rename: " + string(err.message);
            end
            obj.refresh();
        end
        function definition=getBlock(obj,index)
            arguments
                obj
                index (1,1) double {mustBeInteger,mustBePositive}
            end
            definition=obj.Workspace.Definitions(index);
            obj.ReportText=string(jsonencode(definition,'PrettyPrint',true));
            obj.StagedExpression=string(definition.expression);
            obj.StagedProvenance=definition.provenance;
        end
        function calculate(obj)
            arguments
                obj
            end
            obj.ReportText=""; obj.CompactLatex=""; obj.ExpandedLatex=""; obj.ResultLatex="";
            try
                replacements = struct();
                parts = obj.csv(obj.Substitutions);
                for k=1:numel(parts)
                    pair = split(parts(k),"=");
                    if numel(pair)~=2 || ~isvarname(strtrim(pair(1)))
                        error('OceanTools:SymbolicWorkbench:Substitution','Use substitutions such as x=2, sigma=1.');
                    end
                    if isfield(replacements,strtrim(pair(1)))
                        error('OceanTools:SymbolicWorkbench:Substitution','A substitution symbol may only appear once.');
                    end
                    replacements.(strtrim(pair(1))) = strtrim(pair(2));
                end
                if obj.Operation=="inspect"
                    report=obj.Workspace.inspect(obj.Expression);
                else
                    report=obj.Workspace.calculate(obj.Expression,obj.Operation,obj.Variable,replacements);
                end
                obj.CompactLatex=string(report.compactLatex);
                obj.ExpandedLatex=string(report.expandedLatex);
                obj.ResultLatex=string(report.resultLatex);
                lines=["Compact: " + string(report.compact); "Expanded: " + string(report.expanded); "Result: " + string(report.result); ""; "Contract checks (unknown is not a pass):"];
                for k=1:numel(report.checks)
                    c=report.checks(k);
                    lines(end+1)=string(c.label)+" — "+string(c.status)+": "+string(c.detail); %#ok<AGROW>
                end
                obj.ReportText=join(lines,newline);
                obj.History{end+1}=struct('expression',obj.Expression,'operation',obj.Operation,'report',report,'bundle',obj.Workspace.exportBundle());
                if numel(obj.History)>100, obj.History=obj.History(end-99:end); end
                obj.Status="Calculated with MATLAB Symbolic Math Toolbox. History entries: "+numel(obj.History);
            catch err
                obj.Status="Cannot calculate: "+string(err.message);
                obj.ReportText="No result was produced. Correct the input or its assumptions and try again.";
            end
            obj.refresh();
        end
        function importFile(obj,path)
            arguments
                obj
                path (1,1) string
            end
            try
                info=dir(path);
                if isempty(info)||info.bytes>2000000
                    error('OceanTools:SymbolicWorkbench:File','Choose a readable text file no larger than 2 MB.');
                end
                content=string(fileread(path));
                [~,~,ext]=fileparts(path);
                if strcmpi(ext,'.json')
                    bundle=jsondecode(content);
                    kind=""; if isfield(bundle,'kind'), kind=string(bundle.kind); end
                    if kind=="ocean-symbolic-extraction"
                        if ~isfield(bundle,'source') || ~isstruct(bundle.source) || ...
                                ~all(isfield(bundle.source,{'filename','text','language'}))
                            error('OceanTools:SymbolicWorkbench:Packet','Extraction packet requires source filename, text, and language.');
                        end
                        language=string(bundle.source.language);
                        if ~any(language==["matlab","python"])
                            error('OceanTools:SymbolicWorkbench:Packet','Only MATLAB and Python extraction packets are supported.');
                        end
                        packetText=string(bundle.source.text);
                        packetFile=string(bundle.source.filename);
                        scan=OceanMath.Symbolic.SourceExtractor.scanText(packetText,language,packetFile);
                        obj.loadScan(scan,packetText,path);
                        obj.Status="Imported browser extraction packet and safely rescanned its embedded source without executing it.";
                    else
                        obj.Workspace.importBundle(bundle);
                        obj.SourcePath=""; obj.SourceText="";
                        obj.Candidates=strings(0,1); obj.CandidateRecords=struct('expression',{},'provenance',{});
                        obj.SourceFunctions=struct([]); obj.FunctionRows=cell(0,6); obj.FormulaRows=cell(0,5);
                        obj.SourceSummary="Imported a symbolic bundle; no source scan is active.";
                        obj.Status="Symbolic bundle imported. See notation notices for any renamed identifiers.";
                    end
                else
                    if ~any(strcmpi(ext,{'.m','.py'}))
                        error('OceanTools:SymbolicWorkbench:File','Source-symbol extraction supports MATLAB .m and Python .py files.');
                    end
                    scan=OceanMath.Symbolic.SourceExtractor.scanFile(path);
                    obj.loadScan(scan,content,path);
                    obj.SelectedFunctionDetail="Select a function. Every extractable function has an abstract signature; formulas remain separate evidence.";
                    obj.Status="Scanned source text without executing it. Select a function and review its abstract interface or formula evidence; this is not a complete symbolic translation.";
                end
                obj.StagedExpression=""; obj.StagedProvenance=struct();
                obj.AIDraftText="No AI draft requested for this import.";
                obj.ReportText=""; obj.CompactLatex=""; obj.ExpandedLatex=""; obj.ResultLatex="";
            catch err
                obj.Status="Cannot import: "+string(err.message);
            end
            obj.refresh();
        end
        function saveBundle(obj,path)
            arguments
                obj
                path (1,1) string
            end
            try
                text=jsonencode(obj.Workspace.exportBundle(),'PrettyPrint',true);
                fid=fopen(path,'w','n','UTF-8');
                if fid<0, error('OceanTools:SymbolicWorkbench:Write','Cannot open destination.'); end
                cleanup=onCleanup(@() fclose(fid));
                fwrite(fid,unicode2native(text,'UTF-8'),'uint8');
                obj.Status="Symbolic bundle saved: "+path;
            catch err
                obj.Status="Cannot save: "+string(err.message);
            end
        end
        function useCandidate(obj,index)
            arguments
                obj
                index (1,1) double {mustBeInteger,mustBePositive}
            end
            if index>numel(obj.CandidateRecords), return; end
            obj.Expression=obj.CandidateRecords(index).expression;
            obj.StagedExpression=obj.Expression;
            obj.StagedProvenance=obj.CandidateRecords(index).provenance;
            obj.Status="Candidate staged, not accepted as equivalent to the file. Review notation, declare symbols and check domains before calculating.";
            obj.ReportText=string(jsonencode(obj.CandidateRecords(index).provenance,'PrettyPrint',true));
        end
        function gaussianExample(obj)
            arguments
                obj
            end
            try
                x=obj.Workspace.addSymbol("x","real");
                mu=obj.Workspace.addSymbol("mu","real");
                sigma=obj.Workspace.addSymbol("sigma","positive");
                g=obj.Workspace.define("G",["t","m","s"],"exp(-((t-m)/s)^2/2)/(s*sqrt(2*pi))",["real","real","positive"],"positive",struct('kind','user-selected-mathematical-example','description','Gaussian PDF; scalar standard deviation must be positive.'));
                obj.Expression=g+"("+x+","+mu+","+sigma+")"; obj.Variable=x;
                obj.Operation="inspect"; obj.calculate();
            catch err
                obj.Status="Cannot create example: "+string(err.message);
            end
            obj.refresh();
        end
        function requestAI(obj,question)
            arguments
                obj
                question (1,1) string
            end
            obj.AIDraftText="No new AI draft received.";
            try
                if strlength(obj.SourceText)==0, error('OceanTools:SymbolicWorkbench:Source','Import source text first.'); end
                obj.Status="Sending source text to the configured AI service for a draft…"; drawnow;
                draft=OceanTools.SymbolicWorkbench.aiDraft(obj.SourceText,obj.SourcePath,question);
                obj.Candidates=strings(0,1); obj.CandidateRecords=struct('expression',{},'provenance',{});
                for k=1:numel(draft.candidates)
                    c=draft.candidates(k);
                    obj.Candidates(end+1,1)="AI draft · "+string(c.name)+" · "+string(c.expression);
                    obj.CandidateRecords(end+1)=struct('expression',string(c.expression),'provenance',struct('kind','ai-draft','sourcePath',char(obj.SourcePath),'candidate',c,'assumptions',{draft.assumptions},'limitations',{draft.limitations},'model',draft.model));
                end
                obj.AIDraftText=string(jsonencode(draft,'PrettyPrint',true));
                obj.Status="AI draft received, not verified source truth. Review assumptions and contracts before defining a block.";
            catch err
                obj.Status="AI unavailable: "+string(err.message);
            end
        end
    end
    methods (Access=private)
        function loadScan(obj,scan,content,sourcePath)
            obj.SourcePath=sourcePath; obj.SourceText=content;
            obj.SourceFunctions=scan.functions;
            obj.FunctionRows=obj.functionRows(scan.functions);
            obj.FormulaRows=cell(0,5); obj.SelectedFunctionIndex=0;
            obj.Candidates=strings(0,1); obj.CandidateRecords=struct('expression',{},'provenance',{});
            obj.SourceSummary=string(scan.language)+" · "+numel(scan.functions)+" functions · "+ ...
                scan.counts.transparent+" transparent · "+scan.counts.opaque+" opaque · "+ ...
                scan.counts.notExtractable+" not extractable";
            lines=splitlines(content);
            for k=1:numel(lines)
                line=strtrim(lines(k));
                tok=regexp(char(line),'^(?:[%#]\s*)?([A-Za-z]\w*)\s*=\s*(.+?);?$','tokens','once');
                if isempty(tok), continue; end
                rhs=string(regexprep(tok{2},';$',''));
                if ~isempty(regexp(char(rhs),'^[A-Za-z0-9_\s+*/^().,\-]+$','once'))
                    obj.Candidates(end+1,1)="Line "+k+" · "+line; %#ok<AGROW>
                    obj.CandidateRecords(end+1)=struct('expression',rhs,'provenance',struct( ...
                        'kind','source-candidate','sourcePath',char(sourcePath),'sourceLines',k,'literal',char(line))); %#ok<AGROW>
                end
            end
        end

        function [valid,mapped,detail]=validateDocumentationMapping(obj,expression,record,mappingText)
            equalsIndex=strfind(char(expression),'=');
            if isempty(equalsIndex)
                valid=false; mapped=expression; detail="The documented formula has no equation boundary."; return;
            end
            rhs=strtrim(extractAfter(expression,equalsIndex(1)));
            pairs=obj.csv(mappingText); keys=strings(1,0); values=strings(1,0);
            for k=1:numel(pairs)
                pair=strtrim(split(pairs(k),"="));
                if numel(pair)~=2 || ~isvarname(pair(1)) || ~isvarname(pair(2))
                    valid=false; mapped=expression; detail="Use mappings such as X=p, theta=theta0."; return;
                end
                if any(keys==pair(1)) || ~any(string(record.inputs)==pair(2))
                    valid=false; mapped=expression; detail="Each formula variable must map once to a declared function input."; return;
                end
                keys(end+1)=pair(1); values(end+1)=pair(2); %#ok<AGROW>
            end
            identifiers=unique(string(regexp(char(rhs),'[A-Za-z][A-Za-z0-9_]*','match')));
            ignored=["sum","prod","log","exp","sin","cos","sqrt","abs","pi","e", ...
                string(record.displayName),string(record.outputs)];
            required=setdiff(identifiers,ignored);
            if isempty(keys) || any(~ismember(required,keys))
                missing=setdiff(required,keys);
                valid=false; mapped=expression;
                detail="Provide an explicit mapping for every formula variable";
                if ~isempty(missing), detail=detail+": "+join(missing,", "); end
                return;
            end
            mapped=rhs;
            for k=1:numel(keys)
                mapped=string(regexprep(char(mapped),['(?<![A-Za-z0-9_])' regexptranslate('escape',char(keys(k))) '(?![A-Za-z0-9_])'],char(values(k))));
            end
            valid=true; detail="Variables mapped to source inputs.";
        end

        function guard(obj,action,message)
            try
                action(); obj.Status=message;
            catch err
                obj.Status="Cannot apply: "+string(err.message);
            end
            obj.refresh();
        end
        function refresh(obj)
            symbols=obj.Workspace.Symbols; blocks=obj.Workspace.Definitions;
            rows=cell(numel(symbols),2);
            for k=1:numel(symbols), rows(k,:)={char(symbols(k).name),char(symbols(k).domain)}; end
            obj.SymbolRows=rows; rows=cell(numel(blocks),4);
            for k=1:numel(blocks)
                rows(k,:)={char(blocks(k).name),char(join(string(blocks(k).parameters),", ")),char(blocks(k).expression),char(blocks(k).outputDomain)};
            end
            obj.BlockRows=rows;
            entries=obj.Registry.Records;
            accepted=cell(numel(entries),5);
            for k=1:numel(entries)
                r=entries(k).record;
                accepted(k,:)={char(entries(k).alias),char(r.signature),char(r.classification), ...
                    char(r.computationalSupport),char(string(r.file)+":"+string(r.sourceLines(1)))};
            end
            obj.AcceptedRows=accepted;
            obj.Notices=join([obj.Workspace.Notices; obj.Registry.Notices],newline);
        end
    end
    methods (Static,Access=private)
        function values=csv(text)
            values=strtrim(split(text,","))'; values(values=="")=[];
        end

        function rows=functionRows(records)
            rows=cell(numel(records),6);
            for k=1:numel(records)
                r=records(k);
                rows(k,:)={char(r.displayName),char(r.signature),char(r.classification), ...
                    char(r.computationalSupport),sprintf('%.0f%%',100*r.confidence), ...
                    sprintf('%d-%d',r.sourceLines(1),r.sourceLines(end))};
            end
        end

        function formulas=allFormulas(record)
            formulas=struct('label',{},'expression',{},'origin',{},'sourceLines',{}, ...
                'computationalSupport',{},'freeVariables',{},'validated',{});
            if strlength(string(record.expression))>0
                formulas(end+1)=struct('label',"Normalized traced output",'expression',string(record.expression), ...
                    'origin',"source-derived-normalized",'sourceLines',[record.expressionSourceLine record.expressionSourceLine], ...
                    'computationalSupport',string(record.computationalSupport), ...
                    'freeVariables',string(record.inputs),'validated',true);
            end
            if ~isempty(record.formulaCandidates)
                formulas=[formulas record.formulaCandidates];
            end
        end

        function rows=formulaRows(record)
            formulas=OceanTools.SymbolicWorkbench.ViewModels.Main.allFormulas(record);
            rows=cell(numel(formulas),5);
            for k=1:numel(formulas)
                f=formulas(k);
                rows(k,:)={char(f.label),char(f.expression),char(f.origin), ...
                    char(f.computationalSupport),sprintf('%d-%d',f.sourceLines(1),f.sourceLines(end))};
            end
        end

        function text=describeRecord(record)
            inputText="none"; outputText="none";
            if ~isempty(record.inputs), inputText=join(string(record.inputs),", "); end
            if ~isempty(record.outputs), outputText=join(string(record.outputs),", "); end
            text=join(["ABSTRACT SYMBOLIC INTERFACE"; string(record.signature)+" -> "+outputText; ...
                "Qualified identity: "+string(record.qualifiedName); ...
                "Classification: "+string(record.classification); ...
                "Inputs: "+inputText; "Outputs: "+outputText; ...
                "Contracts: domain and shape are unknown until declared or inferred."; ...
                "Computation support: "+string(record.computationalSupport); ...
                "Confidence: "+compose("%.0f%%",100*record.confidence); ...
                "Source: "+string(record.file)+" lines "+record.sourceLines(1)+"-"+record.sourceLines(end); ...
                "Rationale: "+string(record.rationale); ""; ...
                "Opaque acceptance preserves only the interface. Formula acceptance preserves separate source evidence."],newline);
        end
    end
end
