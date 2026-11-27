classdef Registry < handle
    %REGISTRY Accepted source-linked transparent and opaque operators.

    properties (SetAccess=private)
        Records struct = struct('record',{},'alias',{})
        Notices string = strings(0,1)
    end

    methods
        function alias = accept(obj, record, requestedAlias)
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
                record (1,1) struct
                requestedAlias (1,1) string = ""
            end
            required = {'id','displayName','inputs','outputs','classification','contracts'};
            if ~all(isfield(record,required)) || record.classification == "not-extractable"
                error('OceanMath:Symbolic:RegistryRecord', 'Only extractable source records with contracts can be accepted.');
            end
            for k=1:numel(record.contracts.inputs)
                OceanMath.Symbolic.Registry.validateDomain(string(record.contracts.inputs(k).domain));
                OceanMath.Symbolic.Registry.validateShape(string(record.contracts.inputs(k).shape));
            end
            for k=1:numel(record.contracts.outputs)
                OceanMath.Symbolic.Registry.validateDomain(string(record.contracts.outputs(k).domain));
                OceanMath.Symbolic.Registry.validateShape(string(record.contracts.outputs(k).shape));
            end
            existingIds = strings(1,numel(obj.Records));
            for k=1:numel(obj.Records), existingIds(k)=string(obj.Records(k).record.id); end
            if any(existingIds == string(record.id))
                error('OceanMath:Symbolic:DuplicateRecord', 'This source representation is already registered.');
            end
            if requestedAlias == "", requestedAlias = string(record.displayName); end
            if isempty(regexp(char(requestedAlias),'^[A-Za-z][A-Za-z0-9_]{0,39}$','once'))
                error('OceanMath:Symbolic:Identifier', 'Aliases must begin with a letter and contain only letters, digits, or underscores.');
            end
            alias = obj.availableAlias(requestedAlias);
            obj.Records(end+1) = struct('record',record,'alias',alias);
            if alias ~= requestedAlias
                obj.Notices(end+1,1) = "Alias " + requestedAlias + " was already used; registered this operator as " + alias + ".";
            end
        end

        function composition = compose(obj, alias, argumentsList)
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
                alias (1,1) string
                argumentsList struct = struct('expression',{},'domain',{},'shape',{})
            end
            item = obj.lookup(alias); record = item.record;
            if numel(argumentsList) ~= numel(record.inputs)
                error('OceanMath:Symbolic:Arity', '%s requires %d inputs.', alias, numel(record.inputs));
            end
            checks = struct('label',{},'status',{},'detail',{});
            for k=1:numel(argumentsList)
                expected = string(record.contracts.inputs(k).domain);
                actual = string(argumentsList(k).domain);
                expectedShape=string(record.contracts.inputs(k).shape);
                actualShape=string(argumentsList(k).shape);
                OceanMath.Symbolic.Registry.validateDomain(expected);
                OceanMath.Symbolic.Registry.validateShape(expectedShape);
                OceanMath.Symbolic.Registry.validateDomain(actual);
                OceanMath.Symbolic.Registry.validateShape(actualShape);
                status = "unknown";
                if expected ~= "unknown" && actual ~= "unknown"
                    if OceanMath.Symbolic.Registry.domainAccepts(expected,actual), status="passed"; else, status="invalid"; end
                end
                shapeStatus="unknown";
                if expectedShape~="unknown" && actualShape~="unknown"
                    if OceanMath.Symbolic.Registry.shapeAccepts(expectedShape,actualShape), shapeStatus="passed"; else, shapeStatus="invalid"; end
                end
                if shapeStatus=="invalid", status="invalid"; end
                if status=="passed" && shapeStatus=="unknown", status="unknown"; end
                checks(end+1)=struct('label',"Input "+record.inputs(k),'status',status, ...
                    'detail',"Expected domain "+expected+" and shape "+expectedShape+ ...
                    "; received "+actual+" and "+actualShape+"."); %#ok<AGROW>
            end
            if any(string({checks.status})=="invalid")
                error('OceanMath:Symbolic:ContractMismatch', 'The composition violates an input domain contract.');
            end
            args = string({argumentsList.expression});
            outputDomain = "unknown"; outputShape = "unknown";
            if ~isempty(record.contracts.outputs)
                outputDomain=string(record.contracts.outputs(1).domain);
                outputShape=string(record.contracts.outputs(1).shape);
            end
            ast=struct('kind',"call",'operator',alias,'operatorId',string(record.id), ...
                'arguments',{argumentsList},'transparent',record.classification=="transparent");
            composition = struct('expression',alias+"("+join(args,", ")+")", ...
                'domain',outputDomain,'shape',outputShape,'checks',checks,'operatorId',string(record.id),'ast',ast);
        end

        function updateContracts(obj, alias, inputContracts, outputContracts)
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
                alias (1,1) string
                inputContracts struct
                outputContracts struct
            end
            index=find(string({obj.Records.alias})==alias,1);
            if isempty(index), error('OceanMath:Symbolic:UnknownOperator','Unknown registered operator %s.',alias); end
            record=obj.Records(index).record;
            if numel(inputContracts)~=numel(record.inputs) || numel(outputContracts)~=numel(record.outputs)
                error('OceanMath:Symbolic:ContractArity','Contract counts must match the source signature.');
            end
            required={'name','kind','domain','shape'};
            if ~all(isfield(inputContracts,required)) || ~all(isfield(outputContracts,required))
                error('OceanMath:Symbolic:ContractFields','Contracts require name, kind, domain, and shape.');
            end
            for k=1:numel(inputContracts)
                OceanMath.Symbolic.Registry.validateDomain(string(inputContracts(k).domain));
                OceanMath.Symbolic.Registry.validateShape(string(inputContracts(k).shape));
            end
            for k=1:numel(outputContracts)
                OceanMath.Symbolic.Registry.validateDomain(string(outputContracts(k).domain));
                OceanMath.Symbolic.Registry.validateShape(string(outputContracts(k).shape));
            end
            record.contracts=struct('inputs',inputContracts,'outputs',outputContracts);
            obj.Records(index).record=record;
            obj.Notices(end+1,1)="Contracts for "+alias+" were updated by the user.";
        end

        function result = combine(obj, left, operator, right) %#ok<INUSD>
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
                left (1,1) struct
                operator (1,1) string {mustBeMember(operator,["+","-","*","/"])}
                right (1,1) struct
            end
            if ~all(isfield(left,{'expression','domain','shape','ast'})) || ~all(isfield(right,{'expression','domain','shape','ast'}))
                error('OceanMath:Symbolic:Composition','Operands must be typed Registry compositions.');
            end
            status="unknown"; detail="Compatibility cannot be proved from the available contracts.";
            outputShape="unknown";
            if operator=="+" || operator=="-"
                if string(left.shape)~="unknown" && string(right.shape)~="unknown"
                    if string(left.shape)~=string(right.shape)
                        error('OceanMath:Symbolic:ShapeMismatch','Addition and subtraction require matching shapes.');
                    end
                    status="passed"; detail="Operand shapes match."; outputShape=string(left.shape);
                end
            elseif operator=="*"
                [compatible,outputShape,known]=OceanMath.Symbolic.Registry.multiplyShapes(string(left.shape),string(right.shape));
                if known && ~compatible, error('OceanMath:Symbolic:ShapeMismatch','Multiplication dimensions are incompatible.'); end
                if known, status="passed"; detail="Multiplication dimensions are compatible."; end
            elseif operator=="/"
                if string(right.shape)=="scalar"
                    status="passed"; detail="Division by a scalar preserves the numerator shape."; outputShape=string(left.shape);
                elseif string(right.shape)~="unknown"
                    error('OceanMath:Symbolic:ShapeMismatch','Division currently requires a scalar denominator.');
                end
            end
            ast=struct('kind',"binary",'operator',operator,'left',left.ast,'right',right.ast);
            check=struct('label',"Algebraic composition",'status',status,'detail',detail);
            result=struct('expression',"("+string(left.expression)+") "+operator+" ("+string(right.expression)+")", ...
                'domain',"unknown",'shape',outputShape,'checks',check,'operatorId',"",'ast',ast);
        end

        function report = manipulate(obj, composition, operation, variable) %#ok<INUSD>
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
                composition (1,1) struct
                operation (1,1) string {mustBeMember(operation,["simplify","differentiate","inspect"])}
                variable (1,1) string = "x"
            end
            if ~isfield(composition,'ast')
                error('OceanMath:Symbolic:Composition','A typed Registry composition is required.');
            end
            hasOpaque=OceanMath.Symbolic.Registry.containsOpaque(composition.ast);
            if operation=="differentiate" && hasOpaque
                status="unsupported";
                result="D["+string(composition.expression)+", "+variable+"]";
                detail="The derivative remains formal because an opaque operator has no registered derivative rule.";
            elseif operation=="simplify" && hasOpaque
                status="conditional"; result=string(composition.expression);
                detail="Opaque calls were preserved as atomic mathematical nodes; only their contracts are known.";
            else
                status="passed"; result=string(composition.expression);
                detail="The typed expression is valid under the reported input checks.";
            end
            report=struct('expression',string(composition.expression),'result',result,'status',status, ...
                'detail',detail,'ast',composition.ast,'checks',composition.checks);
        end

        function text = expand(obj, alias, argumentExpressions)
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
                alias (1,1) string
                argumentExpressions string
            end
            item=obj.lookup(alias); record=item.record;
            if record.classification ~= "transparent" || strlength(string(record.expression))==0
                error('OceanMath:Symbolic:OpaqueExpansion', 'Opaque operators cannot be expanded.');
            end
            support=string(record.computationalSupport);
            if ~any(support==["conditional-symbolic","matlab-symbolic"])
                error('OceanMath:Symbolic:UnsupportedExpansion', ...
                    'This representation is retained for composition or display and cannot be expanded computationally.');
            end
            if numel(argumentExpressions)~=numel(record.inputs)
                error('OceanMath:Symbolic:Arity', '%s requires %d inputs.',alias,numel(record.inputs));
            end
            text=string(record.expression);
            for k=1:numel(record.inputs)
                text=string(regexprep(char(text),['(?<![A-Za-z0-9_])' regexptranslate('escape',char(record.inputs(k))) '(?![A-Za-z0-9_])'],['(' char(argumentExpressions(k)) ')']));
            end
        end

        function bundle = exportBundle(obj)
            arguments
                obj (1,1) OceanMath.Symbolic.Registry
            end
            bundle=struct('schemaVersion',"1.0",'kind',"ocean-symbolic-registry",'entries',obj.Records);
        end
    end

    methods (Access=private)
        function item=lookup(obj,alias)
            index=find(string({obj.Records.alias})==alias,1);
            if isempty(index), error('OceanMath:Symbolic:UnknownOperator','Unknown registered operator %s.',alias); end
            item=obj.Records(index);
        end
        function alias=availableAlias(obj,requested)
            used=string({obj.Records.alias}); alias=requested; suffix=2;
            while any(used==alias), alias=requested+"_"+string(suffix); suffix=suffix+1; end
        end
    end

    methods (Static, Access=private)
        function validateDomain(domain)
            if ~any(domain==["unknown","complex","real","positive","nonnegative","integer"])
                error('OceanMath:Symbolic:ContractDomain','Unsupported contract domain %s.',domain);
            end
        end
        function validateShape(shape)
            valid=any(shape==["unknown","scalar","vector","matrix","function"]); 
            valid=valid || ~isempty(regexp(char(shape),'^vector\[[A-Za-z0-9_]+\]$','once')) ...
                || ~isempty(regexp(char(shape),'^matrix\[[A-Za-z0-9_]+,[A-Za-z0-9_]+\]$','once'));
            if ~valid, error('OceanMath:Symbolic:ContractShape','Unsupported contract shape %s.',shape); end
        end
        function tf=shapeAccepts(expected,actual)
            tf=expected==actual || (expected=="vector" && startsWith(actual,"vector[")) ...
                || (expected=="matrix" && startsWith(actual,"matrix["));
        end
        function [compatible,output,known]=multiplyShapes(left,right)
            compatible=true; output="unknown"; known=true;
            if left=="unknown" || right=="unknown", known=false; return; end
            if left=="scalar", output=right; return; end
            if right=="scalar", output=left; return; end
            lm=regexp(char(left),'^matrix\[([^,]+),([^\]]+)\]$','tokens','once');
            rm=regexp(char(right),'^matrix\[([^,]+),([^\]]+)\]$','tokens','once');
            rv=regexp(char(right),'^vector\[([^\]]+)\]$','tokens','once');
            if ~isempty(lm) && ~isempty(rm)
                compatible=string(lm{2})==string(rm{1}); output="matrix["+string(lm{1})+","+string(rm{2})+"]";
            elseif ~isempty(lm) && ~isempty(rv)
                compatible=string(lm{2})==string(rv{1}); output="vector["+string(lm{1})+"]";
            else
                known=false;
            end
        end
        function tf=domainAccepts(expected,actual)
            if expected=="complex"
                tf=any(actual==["complex","real","positive","nonnegative","integer"]);
            elseif expected=="real"
                tf=any(actual==["real","positive","nonnegative","integer"]);
            elseif expected=="nonnegative"
                tf=any(actual==["nonnegative","positive"]);
            else
                tf=expected==actual;
            end
        end
        function tf=containsOpaque(ast)
            if ast.kind=="call"
                tf=~ast.transparent;
            elseif ast.kind=="binary"
                tf=OceanMath.Symbolic.Registry.containsOpaque(ast.left) || OceanMath.Symbolic.Registry.containsOpaque(ast.right);
            else
                tf=false;
            end
        end
    end
end
