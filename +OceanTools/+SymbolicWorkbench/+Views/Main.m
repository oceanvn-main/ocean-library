classdef Main < handle
    %MAIN Native view for a scalar symbolic expression workspace.
    %   The view forwards commands to its ViewModel. Imported files are
    %   displayed as source text; this class never executes their contents.

    properties (SetAccess = private)
        Figure
    end

    properties (Access = private)
        ViewModel_
        Listeners_ cell = {}
        Expression_
        Operation_
        Variable_
        Substitutions_
        Status_
        Notices_
        Compact_
        Expanded_
        Result_
        Report_
        SymbolTable_
        SymbolName_
        SymbolDomain_
        BlockTable_
        BlockName_
        BlockParameters_
        BlockDomains_
        BlockExpression_
        BlockOutputDomain_
        RenameFrom_
        RenameTo_
        SourcePath_
        SourceText_
        CandidateList_
        UseCandidate_
        SourceSummary_
        FunctionTable_
        FormulaTable_
        FunctionDetail_
        AcceptedTable_
        Alias_
        FormulaMapping_
        AcceptAbstract_
        AcceptFormula_
        RejectFunction_
        SelectedFormulaIndex_ double = 0
        ComposeAlias_
        ComposeArguments_
        ComposeDomains_
        ComposeShapes_
        ContractInputDomains_
        ContractInputShapes_
        ContractOutputDomains_
        ContractOutputShapes_
        AIQuestion_
        AIDraft_
        AIButton_
        Tabs_
        ExpressionTab_
        SourceTab_
    end

    methods
        function obj = Main(viewModel)
            arguments
                viewModel (1,1) OceanTools.SymbolicWorkbench.ViewModels.Main
            end
            obj.ViewModel_ = viewModel;
            try
                obj.createUI();
                obj.bindViewModel();
                obj.syncAll();
            catch err
                delete(obj);
                rethrow(err);
            end
        end

        function show(obj)
            arguments
                obj
            end
            obj.Figure.Visible = 'on';
        end

        function delete(obj)
            arguments
                obj
            end
            for k = 1:numel(obj.Listeners_)
                delete(obj.Listeners_{k});
            end
            obj.Listeners_ = {};
            if ~isempty(obj.Figure) && isgraphics(obj.Figure)
                obj.Figure.CloseRequestFcn = [];
                delete(obj.Figure);
            end
        end
    end

    methods (Access = private)
        function createUI(obj)
            obj.Figure = uifigure('Name', 'Ocean Library | Symbolic Workbench', ...
                'Visible', 'off', 'Position', [60 60 1280 820], ...
                'CloseRequestFcn', @(~,~) delete(obj));
            root = uigridlayout(obj.Figure, [3 2]);
            root.ColumnWidth = {280, '1x'};
            root.RowHeight = {58, '1x', 48};
            root.Padding = [18 14 18 14];
            root.RowSpacing = 12;
            root.ColumnSpacing = 18;

            header = uigridlayout(root, [2 3]);
            obj.place(header, 1, [1 2]);
            header.ColumnWidth = {'1x', 140, 140};
            header.RowHeight = {28, 22};
            header.Padding = [0 0 0 0];
            title = uilabel(header, 'Text', 'Symbolic Workbench', ...
                'FontSize', 23, 'FontWeight', 'bold');
            obj.place(title, 1, 1);
            scope = uilabel(header, 'Text', ...
                'Scalar expressions and reusable named blocks | MATLAB Symbolic Math Toolbox', ...
                'FontSize', 12);
            obj.place(scope, 2, [1 3]);
            open = uibutton(header, 'Text', 'Import file...', ...
                'ButtonPushedFcn', @(~,~) obj.importSource());
            obj.place(open, 1, 2);
            save = uibutton(header, 'Text', 'Save bundle...', ...
                'ButtonPushedFcn', @(~,~) obj.saveBundle());
            obj.place(save, 1, 3);

            side = uigridlayout(root, [14 1]);
            obj.place(side, 2, 1);
            side.RowHeight = {24, 125, 22, 28, 22, 28, 22, 55, 34, 32, 50, 22, 110, '1x'};
            side.ColumnWidth = {'1x'};
            side.Padding = [0 0 0 0];
            side.Scrollable = 'on';
            obj.label(side, 'Expression', 1);
            obj.Expression_ = uitextarea(side, 'FontName', 'Consolas', ...
                'ValueChangedFcn', @(source,~) obj.updateText('Expression', source.Value));
            obj.place(obj.Expression_, 2, 1);
            obj.label(side, 'Operation', 3);
            obj.Operation_ = uidropdown(side, ...
                'Items', {'inspect','simplify','expand','factor','differentiate','integrate','solve','substitute','evaluate'}, ...
                'ValueChangedFcn', @(source,~) obj.updateText('Operation', source.Value));
            obj.place(obj.Operation_, 4, 1);
            obj.label(side, 'Variable', 5);
            obj.Variable_ = uieditfield(side, 'text', ...
                'ValueChangedFcn', @(source,~) obj.updateText('Variable', source.Value));
            obj.place(obj.Variable_, 6, 1);
            obj.label(side, 'Substitutions (x=2, sigma=1)', 7);
            obj.Substitutions_ = uitextarea(side, 'FontName', 'Consolas', ...
                'ValueChangedFcn', @(source,~) obj.updateText('Substitutions', source.Value));
            obj.place(obj.Substitutions_, 8, 1);
            calculate = uibutton(side, 'Text', 'Calculate / inspect', ...
                'FontWeight', 'bold', 'ButtonPushedFcn', @(~,~) obj.calculate());
            obj.place(calculate, 9, 1);
            example = uibutton(side, 'Text', 'Load Gaussian PDF example', ...
                'ButtonPushedFcn', @(~,~) obj.runExample());
            obj.place(example, 10, 1);
            hint = uilabel(side, 'Text', ...
                'Declare symbols and domains in Symbols & blocks. Imported candidates need review.', ...
                'WordWrap', 'on', 'FontSize', 11);
            obj.place(hint, 11, 1);
            obj.label(side, 'Notation notices', 12);
            obj.Notices_ = uitextarea(side, 'Editable', 'off', 'FontSize', 11);
            obj.place(obj.Notices_, 13, 1);

            obj.Tabs_ = uitabgroup(root);
            obj.place(obj.Tabs_, 2, 2);
            obj.createExpressionTab();
            obj.createDefinitionsTab();
            obj.createSourceTab();
            obj.createAITab();
            obj.Status_ = uitextarea(root, 'Editable', 'off', 'FontSize', 12);
            obj.place(obj.Status_, 3, [1 2]);
        end

        function createExpressionTab(obj)
            obj.ExpressionTab_ = uitab(obj.Tabs_, 'Title', 'Expressions & checks');
            grid = uigridlayout(obj.ExpressionTab_, [5 1]);
            grid.RowHeight = {115, 135, 135, 22, '1x'};
            grid.ColumnWidth = {'1x'};
            grid.Scrollable = 'on';
            obj.Compact_ = obj.formulaPanel(grid, 'Compact notation', 1);
            obj.Expanded_ = obj.formulaPanel(grid, 'Expanded expression', 2);
            obj.Result_ = obj.formulaPanel(grid, 'Result', 3);
            obj.label(grid, 'Copyable report and contract checks', 4);
            obj.Report_ = uitextarea(grid, 'Editable', 'off', 'FontName', 'Consolas');
            obj.place(obj.Report_, 5, 1);
        end

        function createDefinitionsTab(obj)
            tab = uitab(obj.Tabs_, 'Title', 'Symbols & blocks');
            grid = uigridlayout(tab, [1 2]);
            grid.ColumnWidth = {235, '1x'};
            grid.RowHeight = {'1x'};
            grid.Scrollable = 'on';

            symbolPanel = uipanel(grid, 'Title', 'Declared symbols');
            obj.place(symbolPanel, 1, 1);
            symbols = uigridlayout(symbolPanel, [7 1]);
            symbols.RowHeight = {180, 22, 28, 22, 28, 32, '1x'};
            symbols.ColumnWidth = {'1x'};
            symbols.Scrollable = 'on';
            obj.SymbolTable_ = uitable(symbols, 'Data', cell(0,2), ...
                'ColumnName', {'Symbol','Domain'}, 'ColumnEditable', [false false], ...
                'RowName', {}, 'ColumnWidth', {'auto','auto'});
            obj.place(obj.SymbolTable_, 1, 1);
            obj.label(symbols, 'New symbol name', 2);
            obj.SymbolName_ = uieditfield(symbols, 'text');
            obj.place(obj.SymbolName_, 3, 1);
            obj.label(symbols, 'Domain', 4);
            obj.SymbolDomain_ = uidropdown(symbols, 'Items', obj.domains());
            obj.place(obj.SymbolDomain_, 5, 1);
            add = uibutton(symbols, 'Text', 'Declare symbol', ...
                'ButtonPushedFcn', @(~,~) obj.command(@() obj.ViewModel_.addSymbol( ...
                string(obj.SymbolName_.Value), string(obj.SymbolDomain_.Value))));
            obj.place(add, 6, 1);

            blockPanel = uipanel(grid, 'Title', 'Reusable scalar blocks');
            obj.place(blockPanel, 1, 2);
            blocks = uigridlayout(blockPanel, [11 2]);
            blocks.ColumnWidth = {155, '1x'};
            blocks.RowHeight = {160, 42, 28, 28, 28, 28, 90, 32, 22, 28, 32};
            blocks.Scrollable = 'on';
            obj.BlockTable_ = uitable(blocks, 'Tag', 'SymbolicBlockTable', 'Data', cell(0,4), ...
                'ColumnName', {'Name','Parameters','Expression','Output domain'}, ...
                'ColumnEditable', false(1,4), 'RowName', {}, ...
                'ColumnWidth', {'auto','auto',250,'auto'}, ...
                'CellSelectionCallback', @(~,event) obj.selectBlock(event));
            obj.place(obj.BlockTable_, 1, [1 2]);
            hint = uilabel(blocks, 'Text', ...
                'Select a row to copy its definition and domains. Define creates a new block; name collisions receive visible suffixes. Full provenance appears in Expressions & checks.', ...
                'WordWrap', 'on', 'FontSize', 11);
            obj.place(hint, 2, [1 2]);
            obj.label(blocks, 'Block name', 3);
            obj.BlockName_ = uieditfield(blocks, 'text');
            obj.place(obj.BlockName_, 3, 2);
            obj.label(blocks, 'Parameters (comma-separated)', 4);
            obj.BlockParameters_ = uieditfield(blocks, 'text');
            obj.place(obj.BlockParameters_, 4, 2);
            obj.label(blocks, 'Parameter domains (in order)', 5);
            obj.BlockDomains_ = uieditfield(blocks, 'text', 'Tag', 'SymbolicBlockDomains', ...
                'Placeholder', 'real, real, positive');
            obj.place(obj.BlockDomains_, 5, 2);
            obj.label(blocks, 'Output domain', 6);
            obj.BlockOutputDomain_ = uidropdown(blocks, 'Items', obj.domains());
            obj.place(obj.BlockOutputDomain_, 6, 2);
            obj.label(blocks, 'Scalar expression', 7);
            obj.BlockExpression_ = uitextarea(blocks, 'FontName', 'Consolas');
            obj.place(obj.BlockExpression_, 7, 2);
            define = uibutton(blocks, 'Text', 'Define new block', ...
                'ButtonPushedFcn', @(~,~) obj.defineBlock());
            obj.place(define, 8, 2);
            renameHint = uilabel(blocks, 'Text', 'Rename an existing block and its dependent calls', ...
                'FontWeight', 'bold', 'FontSize', 11);
            obj.place(renameHint, 9, [1 2]);
            obj.RenameFrom_ = uieditfield(blocks, 'text', 'Placeholder', 'Current block name');
            obj.place(obj.RenameFrom_, 10, 1);
            obj.RenameTo_ = uieditfield(blocks, 'text', 'Placeholder', 'New block name');
            obj.place(obj.RenameTo_, 10, 2);
            rename = uibutton(blocks, 'Text', 'Rename block', ...
                'ButtonPushedFcn', @(~,~) obj.command(@() obj.ViewModel_.renameBlock( ...
                string(obj.RenameFrom_.Value), string(obj.RenameTo_.Value))));
            obj.place(rename, 11, 2);
        end

        function createSourceTab(obj)
            obj.SourceTab_ = uitab(obj.Tabs_, 'Title', 'Source symbolic interfaces');
            grid = uigridlayout(obj.SourceTab_, [5 2]);
            grid.ColumnWidth = {'1x','1x'};
            grid.RowHeight = {46, 240, 270, 132, 220};
            grid.Scrollable = 'on';
            summaryGrid=uigridlayout(grid,[2 1]);
            obj.place(summaryGrid,1,[1 2]);
            summaryGrid.RowHeight={22,22}; summaryGrid.Padding=[0 0 0 0];
            obj.SourcePath_ = uilabel(summaryGrid, 'Text', 'No source imported.', 'WordWrap', 'on');
            obj.place(obj.SourcePath_, 1, 1);
            obj.SourceSummary_=uilabel(summaryGrid,'Text','No MATLAB or Python source scanned.','FontWeight','bold');
            obj.place(obj.SourceSummary_,2,1);

            inventory=uipanel(grid,'Title','Function inventory — select one to review');
            obj.place(inventory,2,1);
            inventoryGrid=uigridlayout(inventory,[1 1]);
            obj.FunctionTable_=uitable(inventoryGrid,'Tag','SymbolicSourceFunctionTable', ...
                'Data',cell(0,6),'ColumnName',{'Function','Abstract signature','Kind','Compute','Confidence','Lines'}, ...
                'ColumnEditable',false(1,6),'RowName',{},'CellSelectionCallback',@(~,event)obj.selectFunction(event));

            detail=uipanel(grid,'Title','Abstract interface & contract evidence');
            obj.place(detail,2,2);
            detailGrid=uigridlayout(detail,[1 1]);
            obj.FunctionDetail_=uitextarea(detailGrid,'Tag','SymbolicFunctionDetail','Editable','off','FontName','Consolas');

            formulas=uipanel(grid,'Title','Formula representations — separate from the abstract operator');
            obj.place(formulas,3,1);
            formulaGrid=uigridlayout(formulas,[1 1]);
            obj.FormulaTable_=uitable(formulaGrid,'Tag','SymbolicFormulaTable','Data',cell(0,5), ...
                'ColumnName',{'Candidate','Expression','Origin','Compute','Lines'}, ...
                'ColumnEditable',false(1,5),'RowName',{},'CellSelectionCallback',@(~,event)obj.selectFormula(event));

            accepted=uipanel(grid,'Title','Accepted symbolic library');
            obj.place(accepted,3,2);
            acceptedGrid=uigridlayout(accepted,[7 4]);
            acceptedGrid.RowHeight={'1x',24,24,28,24,28,30};
            obj.AcceptedTable_=uitable(acceptedGrid,'Tag','AcceptedSymbolicLibrary','Data',cell(0,5), ...
                'ColumnName',{'Alias','Signature','Kind','Compute','Source'},'ColumnEditable',false(1,5),'RowName',{});
            acceptedGrid.ColumnWidth={120,'1x','1x','1x'};
            obj.place(obj.AcceptedTable_,1,[1 4]);
            obj.ComposeAlias_=uieditfield(acceptedGrid,'text','Placeholder','Alias'); obj.place(obj.ComposeAlias_,2,1);
            obj.ComposeArguments_=uieditfield(acceptedGrid,'text','Placeholder','Arguments: loss, theta0'); obj.place(obj.ComposeArguments_,2,[2 3]);
            obj.ComposeDomains_=uieditfield(acceptedGrid,'text','Placeholder','Argument domains, optional'); obj.place(obj.ComposeDomains_,3,[1 2]);
            obj.ComposeShapes_=uieditfield(acceptedGrid,'text','Placeholder','Argument shapes, optional'); obj.place(obj.ComposeShapes_,3,[3 4]);
            composeButton=uibutton(acceptedGrid,'Text','Compose and check contracts', ...
                'ButtonPushedFcn',@(~,~)obj.composeOperator()); obj.place(composeButton,4,[1 3]);
            expandButton=uibutton(acceptedGrid,'Text','Use staged expansion', ...
                'ButtonPushedFcn',@(~,~)obj.command(@()obj.ViewModel_.useStagedExpansion())); obj.place(expandButton,4,4);
            obj.ContractInputDomains_=uieditfield(acceptedGrid,'text','Placeholder','Input domains'); obj.place(obj.ContractInputDomains_,5,1);
            obj.ContractInputShapes_=uieditfield(acceptedGrid,'text','Placeholder','Input shapes'); obj.place(obj.ContractInputShapes_,5,2);
            obj.ContractOutputDomains_=uieditfield(acceptedGrid,'text','Placeholder','Output domains'); obj.place(obj.ContractOutputDomains_,5,3);
            obj.ContractOutputShapes_=uieditfield(acceptedGrid,'text','Placeholder','Output shapes'); obj.place(obj.ContractOutputShapes_,5,4);
            contractButton=uibutton(acceptedGrid,'Text','Update alias contracts', ...
                'ButtonPushedFcn',@(~,~)obj.updateContracts()); obj.place(contractButton,6,[1 4]);
            composeHint=uilabel(acceptedGrid,'Text', ...
                'Opaque calls remain formal. A transparent scalar formula can be expanded into the calculation editor.', ...
                'WordWrap','on','FontSize',10); obj.place(composeHint,7,[1 4]);

            actions=uipanel(grid,'Title','Review decision');
            obj.place(actions,4,[1 2]);
            actionGrid=uigridlayout(actions,[3 4]);
            actionGrid.RowHeight={26,30,32}; actionGrid.ColumnWidth={180,'1x','1x','1x'};
            obj.Alias_=uieditfield(actionGrid,'text','Placeholder','Optional alias, e.g. H');
            obj.place(obj.Alias_,1,1);
            obj.FormulaMapping_=uieditfield(actionGrid,'text', ...
                'Placeholder','Documentation variable mapping, e.g. X=p, theta=theta0');
            obj.place(obj.FormulaMapping_,1,[2 4]);
            note=uilabel(actionGrid,'Text', ...
                'Opaque: composable contract only. Formula: expand only when supported. Unknown contracts are never treated as passed.', ...
                'WordWrap','on','FontSize',11);
            obj.place(note,2,[1 4]);
            obj.AcceptAbstract_=uibutton(actionGrid,'Text','Accept abstract operator', ...
                'ButtonPushedFcn',@(~,~)obj.acceptAbstract()); obj.place(obj.AcceptAbstract_,3,2);
            obj.AcceptFormula_=uibutton(actionGrid,'Text','Accept selected formula', ...
                'ButtonPushedFcn',@(~,~)obj.acceptFormula()); obj.place(obj.AcceptFormula_,3,3);
            obj.RejectFunction_=uibutton(actionGrid,'Text','Reject for this session', ...
                'ButtonPushedFcn',@(~,~)obj.command(@()obj.ViewModel_.rejectSelected())); obj.place(obj.RejectFunction_,3,4);

            sourcePanel=uipanel(grid,'Title','Literal source (read-only, never executed)');
            obj.place(sourcePanel,5,[1 2]);
            sourceGrid=uigridlayout(sourcePanel,[1 1]);
            obj.SourceText_ = uitextarea(sourceGrid, 'Editable', 'off', 'FontName', 'Consolas');

            % Kept hidden for backward-compatible programmatic staging of
            % simple top-level assignments; function review uses the tables.
            obj.CandidateList_ = uilistbox(sourceGrid,'Visible','off','Items',{'No candidates available'},'ItemsData',0,'Value',0);
            obj.UseCandidate_ = uibutton(sourceGrid,'Visible','off','Text','Stage assignment','ButtonPushedFcn',@(~,~)obj.useCandidate());
        end

        function createAITab(obj)
            tab = uitab(obj.Tabs_, 'Title', 'Optional AI draft');
            grid = uigridlayout(tab, [6 1]);
            grid.ColumnWidth = {'1x'};
            grid.RowHeight = {66, 24, 90, 34, 42, '1x'};
            disclosure = uilabel(grid, 'Text', ...
                'Clicking Request AI draft sends the imported source text, its path, and your question to OpenAI. Drafts require review. Symbolic algebra works locally without AI.', ...
                'WordWrap', 'on');
            obj.place(disclosure, 1, 1);
            obj.label(grid, 'Question about the imported source', 2);
            obj.AIQuestion_ = uitextarea(grid, 'Placeholder', ...
                'Describe the scalar mathematics you want to extract or explain.');
            obj.place(obj.AIQuestion_, 3, 1);
            obj.AIButton_ = uibutton(grid, 'Text', 'Request AI draft', ...
                'ButtonPushedFcn', @(~,~) obj.requestAI());
            obj.place(obj.AIButton_, 4, 1);
            note = uilabel(grid, 'Text', ...
                'Requires OPENAI_API_KEY in the MATLAB environment. Returned expressions appear under Source & candidates; assumptions and limitations appear below.', ...
                'WordWrap', 'on', 'FontSize', 11);
            obj.place(note, 5, 1);
            obj.AIDraft_ = uitextarea(grid, 'Editable', 'off', 'FontName', 'Consolas');
            obj.place(obj.AIDraft_, 6, 1);
        end

        function bindViewModel(obj)
            properties = {'Expression','Operation','Variable','Substitutions','Status', ...
                'Notices','CompactLatex','ExpandedLatex','ResultLatex','ReportText','AIDraftText', ...
                'SymbolRows','BlockRows','SourceText','SourcePath','Candidates', ...
                'SourceSummary','FunctionRows','FormulaRows','SelectedFunctionDetail','AcceptedRows'};
            obj.Listeners_ = cell(size(properties));
            for k = 1:numel(properties)
                name = properties{k};
                obj.Listeners_{k} = addlistener(obj.ViewModel_, name, 'PostSet', ...
                    @(~,~) obj.syncProperty(name));
            end
        end

        function syncAll(obj)
            names = {'Expression','Operation','Variable','Substitutions','Status', ...
                'Notices','CompactLatex','ExpandedLatex','ResultLatex','ReportText','AIDraftText', ...
                'SymbolRows','BlockRows','SourceText','SourcePath','Candidates', ...
                'SourceSummary','FunctionRows','FormulaRows','SelectedFunctionDetail','AcceptedRows'};
            for k = 1:numel(names)
                obj.syncProperty(names{k});
            end
        end

        function syncProperty(obj, name)
            if ~isgraphics(obj.Figure), return; end
            vm = obj.ViewModel_;
            switch name
                case 'Expression', obj.Expression_.Value = obj.lines(vm.Expression);
                case 'Operation', obj.Operation_.Value = char(vm.Operation);
                case 'Variable', obj.Variable_.Value = char(vm.Variable);
                case 'Substitutions', obj.Substitutions_.Value = obj.lines(vm.Substitutions);
                case 'Status', obj.Status_.Value = obj.lines(vm.Status);
                case 'Notices', obj.Notices_.Value = obj.lines(vm.Notices);
                case 'CompactLatex', obj.setFormula(obj.Compact_, vm.CompactLatex);
                case 'ExpandedLatex', obj.setFormula(obj.Expanded_, vm.ExpandedLatex);
                case 'ResultLatex', obj.setFormula(obj.Result_, vm.ResultLatex);
                case 'ReportText'
                    obj.Report_.Value = obj.lines(vm.ReportText);
                case 'AIDraftText'
                    obj.AIDraft_.Value = obj.lines(vm.AIDraftText);
                case 'SymbolRows', obj.SymbolTable_.Data = vm.SymbolRows;
                case 'BlockRows', obj.BlockTable_.Data = vm.BlockRows;
                case 'SourceText'
                    obj.SourceText_.Value = obj.lines(vm.SourceText);
                    if strlength(vm.SourceText) > 0
                        obj.AIButton_.Enable = 'on';
                    else
                        obj.AIButton_.Enable = 'off';
                    end
                case 'SourcePath'
                    if strlength(vm.SourcePath) == 0
                        obj.SourcePath_.Text = 'No source imported.';
                    else
                        obj.SourcePath_.Text = char(vm.SourcePath);
                    end
                case 'SourceSummary', obj.SourceSummary_.Text=char(vm.SourceSummary);
                case 'FunctionRows', obj.FunctionTable_.Data=vm.FunctionRows;
                case 'FormulaRows', obj.FormulaTable_.Data=vm.FormulaRows;
                case 'SelectedFunctionDetail', obj.FunctionDetail_.Value=obj.lines(vm.SelectedFunctionDetail);
                case 'AcceptedRows', obj.AcceptedTable_.Data=vm.AcceptedRows;
                case 'Candidates'
                    obj.CandidateList_.ItemsData = [];
                    if isempty(vm.Candidates)
                        obj.CandidateList_.Items = {'No candidates available'};
                        obj.CandidateList_.ItemsData = 0;
                        obj.CandidateList_.Value = 0;
                        obj.CandidateList_.Enable = 'off';
                        obj.UseCandidate_.Enable = 'off';
                    else
                        obj.CandidateList_.Items = cellstr(vm.Candidates);
                        obj.CandidateList_.ItemsData = 1:numel(vm.Candidates);
                        obj.CandidateList_.Value = 1;
                        obj.CandidateList_.Enable = 'on';
                        obj.UseCandidate_.Enable = 'on';
                    end
            end
        end

        function updateText(obj, property, value)
            obj.ViewModel_.(property) = join(string(value), newline);
        end

        function calculate(obj)
            obj.updateText('Expression', obj.Expression_.Value);
            obj.updateText('Operation', obj.Operation_.Value);
            obj.updateText('Variable', obj.Variable_.Value);
            obj.updateText('Substitutions', obj.Substitutions_.Value);
            obj.command(@() obj.ViewModel_.calculate());
            obj.Tabs_.SelectedTab = obj.ExpressionTab_;
        end

        function runExample(obj)
            obj.command(@() obj.ViewModel_.gaussianExample());
            obj.Tabs_.SelectedTab = obj.ExpressionTab_;
        end

        function defineBlock(obj)
            obj.command(@() obj.ViewModel_.defineBlock( ...
                string(obj.BlockName_.Value), string(obj.BlockParameters_.Value), ...
                string(obj.BlockDomains_.Value), join(string(obj.BlockExpression_.Value), newline), ...
                string(obj.BlockOutputDomain_.Value)));
        end

        function selectBlock(obj, event)
            if isempty(event.Indices), return; end
            row = event.Indices(1,1);
            rows = obj.ViewModel_.BlockRows;
            if row > size(rows,1), return; end
            definition = obj.ViewModel_.getBlock(row);
            obj.BlockName_.Value = char(string(definition.name));
            obj.BlockParameters_.Value = char(join(string(definition.parameters), ', '));
            obj.BlockExpression_.Value = obj.lines(string(definition.expression));
            obj.BlockOutputDomain_.Value = char(string(definition.outputDomain));
            obj.BlockDomains_.Value = char(join(string(definition.parameterDomains), ', '));
            obj.RenameFrom_.Value = char(string(definition.name));
            obj.ViewModel_.Status = "Definition and its parameter domains copied for review. Full provenance is in Expressions & checks.";
        end

        function importSource(obj)
            filters = {'*.json;*.m;*.py', ...
                'Symbolic bundles, MATLAB, and Python'; '*.*', 'All files'};
            [file, folder] = uigetfile(filters, 'Import a symbolic bundle or source text');
            if isequal(file, 0), return; end
            obj.command(@() obj.ViewModel_.importFile(string(fullfile(folder,file))));
            obj.Tabs_.SelectedTab = obj.SourceTab_;
        end

        function saveBundle(obj)
            [file, folder] = uiputfile('*.json', 'Save symbolic bundle');
            if isequal(file, 0), return; end
            obj.command(@() obj.ViewModel_.saveBundle(string(fullfile(folder,file))));
        end

        function useCandidate(obj)
            index = obj.CandidateList_.Value;
            if isempty(index) || index < 1, return; end
            obj.command(@() obj.ViewModel_.useCandidate(index));
            obj.Tabs_.SelectedTab = obj.ExpressionTab_;
        end

        function requestAI(obj)
            obj.command(@() obj.ViewModel_.requestAI(join(string(obj.AIQuestion_.Value), newline)));
        end

        function selectFunction(obj,event)
            if isempty(event.Indices), return; end
            obj.command(@()obj.ViewModel_.selectSourceFunction(event.Indices(1,1)));
        end

        function selectFormula(obj,event)
            if isempty(event.Indices), return; end
            obj.SelectedFormulaIndex_=event.Indices(1,1);
            obj.command(@()obj.ViewModel_.selectFormula(obj.SelectedFormulaIndex_));
        end

        function acceptAbstract(obj)
            obj.command(@()obj.ViewModel_.acceptSelectedAbstract(string(obj.Alias_.Value)));
        end

        function acceptFormula(obj)
            index=obj.SelectedFormulaIndex_;
            if index<1
                obj.ViewModel_.Status="Select a formula row first."; return;
            end
            obj.command(@()obj.ViewModel_.acceptSelectedFormula(index,string(obj.Alias_.Value), ...
                string(obj.FormulaMapping_.Value)));
        end

        function composeOperator(obj)
            obj.command(@()obj.ViewModel_.composeOperator(string(obj.ComposeAlias_.Value), ...
                string(obj.ComposeArguments_.Value),string(obj.ComposeDomains_.Value), ...
                string(obj.ComposeShapes_.Value)));
        end

        function updateContracts(obj)
            obj.command(@()obj.ViewModel_.updateOperatorContracts(string(obj.ComposeAlias_.Value), ...
                string(obj.ContractInputDomains_.Value),string(obj.ContractInputShapes_.Value), ...
                string(obj.ContractOutputDomains_.Value),string(obj.ContractOutputShapes_.Value)));
        end

        function command(obj, action)
            try
                action();
            catch err
                obj.ViewModel_.Status = "Cannot complete action: " + string(err.message);
            end
        end
    end

    methods (Static, Access = private)
        function place(component, row, column)
            component.Layout.Row = row;
            component.Layout.Column = column;
        end

        function label(parent, text, row)
            component = uilabel(parent, 'Text', text, 'FontWeight', 'bold', ...
                'FontSize', 11, 'WordWrap', 'on');
            component.Layout.Row = row;
            component.Layout.Column = 1;
        end

        function label = formulaPanel(parent, title, row)
            panel = uipanel(parent, 'Title', title);
            panel.Layout.Row = row;
            panel.Layout.Column = 1;
            grid = uigridlayout(panel, [1 1]);
            grid.RowHeight = {'fit'};
            grid.ColumnWidth = {'fit'};
            grid.Scrollable = 'on';
            label = uilabel(grid, 'Text', '', 'FontSize', 18, 'Interpreter', 'none');
            label.Layout.Row = 1;
            label.Layout.Column = 1;
        end

        function setFormula(label, value)
            if strlength(value) == 0
                label.Interpreter = 'none';
                label.Text = 'No expression calculated.';
                label.FontSize = 12;
            else
                label.FontSize = 18;
                label.Interpreter = 'latex';
                label.Text = char("$$" + value + "$$");
            end
        end

        function values = domains()
            values = {'real','positive','nonnegative','integer','complex'};
        end

        function value = lines(text)
            value = cellstr(splitlines(string(text)));
            if isempty(value), value = {''}; end
        end
    end
end
