classdef TransactionalReader < OceanPipeline.Ingestion.IDataReader
    % TRANSACTIONALREADER  Generic reader for transactional / basket data.
    %
    %   Reads data where each row is a (transaction, item) pair, groups by
    %   transaction ID to form baskets, and optionally enriches item IDs
    %   via a lookup join. Works with ANY dataset — not tied to a specific
    %   schema.
    %
    %   Supports two ingestion modes:
    %     'relational' — Reads a primary CSV of (transaction, item) pairs
    %     'flat'       — One transaction per line (e.g., Groceries dataset)
    %
    %   OUTPUT CONTRACT:
    %     result.baskets     — (N x 1) cell array, each cell = (1 x k_i) uint32 item IDs
    %     result.vocabulary  — containers.Map(uint32 → string)
    %     result.profile     — struct: nTransactions, nUniqueItems, avgBasketSize,
    %                          sparsityRatio, memoryEstimate, sparseRecommended
    %     result.data        — [] (transactional data is not dense)
    %     result.meta        — struct: source, timestamp, format
    %
    %   USAGE (Instacart — Aisle level):
    %     reader = OceanPipeline.Ingestion.TransactionalReader( ...
    %         'Data/InstaCart', ...
    %         'Mode', 'relational', ...
    %         'PrimaryFile', 'order_products__prior.csv', ...
    %         'EnrichFile', 'products.csv', ...
    %         'EnrichFromCol', 1, ...       % product_id
    %         'EnrichToCol', 3, ...         % aisle_id
    %         'LookupFile', 'aisles.csv');
    %     result = reader.read();
    %
    %   USAGE (MovieLens — same reader, different dataset):
    %     reader = OceanPipeline.Ingestion.TransactionalReader( ...
    %         'Data/MovieLens', ...
    %         'Mode', 'relational', ...
    %         'PrimaryFile', 'ratings.csv', ...
    %         'LookupFile', 'movies.csv', ...
    %         'LookupLabelCol', 2);
    %     result = reader.read();
    %
    %   USAGE (Flat — Groceries):
    %     reader = OceanPipeline.Ingestion.TransactionalReader( ...
    %         'groceries.csv', 'Mode', 'flat', 'Delimiter', ',');
    %     result = reader.read();
    %
    %   See also: OceanPipeline.Ingestion.IDataReader
    
    properties
        SourcePath         (1,1) string
        Mode               (1,1) string  = "relational"
        
        % ----- Relational mode: Primary file (transaction-item pairs) -----
        PrimaryFile        (1,1) string  = ""        % CSV with (transID, itemID) rows
        TransactionCol     (1,1) double  = 1         % Column index for transaction ID
        ItemCol            (1,1) double  = 2         % Column index for item ID
        
        % ----- Enrichment (optional): remap item IDs via a join -----
        EnrichFile         (1,1) string  = ""        % Lookup table for remapping
        EnrichFromCol      (1,1) double  = 1         % Column in EnrichFile to match on
        EnrichToCol        (1,1) double  = 2         % Column in EnrichFile to remap to
        
        % ----- Vocabulary (optional): human-readable item names -----
        LookupFile         (1,1) string  = ""        % File with ID → name mapping
        LookupIdCol        (1,1) double  = 1         % Column in LookupFile with item IDs
        LookupLabelCol     (1,1) double  = 2         % Column in LookupFile with names
        
        % ----- Flat mode -----
        Delimiter          (1,1) string  = ","
        
        % ----- Streaming config -----
        ChunkSize          (1,1) double  = 500000
        StreamThresholdMB  (1,1) double  = 250
    end
    
    methods
        function obj = TransactionalReader(sourcePath, options)
            arguments
                sourcePath                      (1,1) string
                options.Mode                    (1,1) string {mustBeMember(options.Mode, ["relational","flat"])} = "relational"
                options.PrimaryFile             (1,1) string  = ""
                options.TransactionCol          (1,1) double  = 1
                options.ItemCol                 (1,1) double  = 2
                options.EnrichFile              (1,1) string  = ""
                options.EnrichFromCol           (1,1) double  = 1
                options.EnrichToCol             (1,1) double  = 2
                options.LookupFile              (1,1) string  = ""
                options.LookupIdCol             (1,1) double  = 1
                options.LookupLabelCol          (1,1) double  = 2
                options.Delimiter               (1,1) string  = ","
                options.ChunkSize               (1,1) double  = 500000
                options.StreamThresholdMB       (1,1) double  = 250
            end
            
            obj.SourcePath        = sourcePath;
            obj.Mode              = options.Mode;
            obj.PrimaryFile       = options.PrimaryFile;
            obj.TransactionCol    = options.TransactionCol;
            obj.ItemCol           = options.ItemCol;
            obj.EnrichFile        = options.EnrichFile;
            obj.EnrichFromCol     = options.EnrichFromCol;
            obj.EnrichToCol       = options.EnrichToCol;
            obj.LookupFile        = options.LookupFile;
            obj.LookupIdCol       = options.LookupIdCol;
            obj.LookupLabelCol    = options.LookupLabelCol;
            obj.Delimiter         = options.Delimiter;
            obj.ChunkSize         = options.ChunkSize;
            obj.StreamThresholdMB = options.StreamThresholdMB;
        end
        
        function result = read(obj)
            % READ  Load transactional data and compute dimensional profile.
            
            fprintf('[TransactionalReader] Mode: %s\n', obj.Mode);
            tStart = tic;
            
            switch obj.Mode
                case "relational"
                    [baskets, vocabulary] = obj.readRelational();
                case "flat"
                    [baskets, vocabulary] = obj.readFlat();
            end
            
            elapsedSec = toc(tStart);
            fprintf('[TransactionalReader] Ingestion complete in %.1f seconds\n', elapsedSec);
            
            % Compute profile
            prof = obj.computeProfile(baskets, vocabulary.Count);
            
            % Assemble output
            result.baskets    = baskets;
            result.vocabulary = vocabulary;
            result.profile    = prof;
            result.data       = [];
            result.meta       = obj.buildTransactionalMeta(prof);
        end
        
        function src = getSource(obj)
            src = obj.SourcePath;
        end
        
        function tf = isReady(obj)
            if obj.Mode == "relational"
                tf = isfolder(obj.SourcePath);
            else
                tf = isfile(obj.SourcePath);
            end
        end
    end
    
    %% ---- RELATIONAL MODE ----
    methods (Access = private)
        function [baskets, vocabulary] = readRelational(obj)
            % READRELATIONAL  Read primary file, optionally enrich, group by transaction.
            
            % Validate: PrimaryFile must be specified
            assert(obj.PrimaryFile ~= "", ...
                'OceanPipeline:TransactionalReader:NoPrimaryFile', ...
                'PrimaryFile must be specified for relational mode.');
            
            % Step 1: Read the primary (transaction, item) pairs
            primaryPath = fullfile(obj.SourcePath, obj.PrimaryFile);
            obj.assertFileExists(primaryPath);
            
            fprintf('[TransactionalReader] Reading primary file: %s\n', obj.PrimaryFile);
            fprintf('[TransactionalReader]   TransactionCol=%d, ItemCol=%d\n', ...
                obj.TransactionCol, obj.ItemCol);
            [transIds, itemIds] = obj.readPrimary(primaryPath);
            fprintf('[TransactionalReader]   -> %d rows loaded\n', numel(transIds));
            
            % Step 2: Enrich item IDs if an enrichment file is specified
            %Mapping many to one: Such as many products can belongs to one aisle
            if obj.EnrichFile ~= ""
                itemIds = obj.enrichItemIds(itemIds);
            end
            
            % Step 3: Group by transaction ID -> baskets
            baskets = obj.groupByTransaction(transIds, itemIds);
            
            % Step 4: Build vocabulary
            vocabulary = obj.buildVocabulary(itemIds);
        end
        
        function [transIds, itemIds] = readPrimary(obj, filepath)
            % READPRIMARY  Read (transactionID, itemID) columns from CSV.
            %   Uses chunked streaming if file exceeds threshold.
            
            fileSizeMB = dir(filepath).bytes / 1e6;
            useStreaming = fileSizeMB > obj.StreamThresholdMB;
            
            % Count columns from header
            fid = fopen(filepath, 'r');
            header = fgetl(fid); %Get first line only
            fclose(fid); %Close immediately
            nCols = numel(strsplit(header, ',')); %Count commas + 1
            
            % Build textscan format: read only TransactionCol and ItemCol
            %This use format flag as flag to skip the unwanted columns, still read all, but don't store temporary
            [fmtStr, transOutIdx, itemOutIdx] = ...
                OceanPipeline.Ingestion.TransactionalReader.buildTextscanFormat(nCols, obj.TransactionCol, obj.ItemCol);
            
            if useStreaming
                fprintf('[TransactionalReader]   File size %.0f MB > %d MB -> STREAMING\n', ...
                    fileSizeMB, obj.StreamThresholdMB); %Streaming read in 500k-row chunks
                [transIds, itemIds] = obj.readPrimaryChunked(filepath, fmtStr, transOutIdx, itemOutIdx);
            else
                fprintf('[TransactionalReader]   File size %.0f MB -> direct read\n', fileSizeMB);
                [transIds, itemIds] = obj.readPrimaryDirect(filepath, fmtStr, transOutIdx, itemOutIdx); 
            end
        end
        
        function [transIds, itemIds] = readPrimaryDirect(~, filepath, fmtStr, transOutIdx, itemOutIdx)
            % Direct read — small enough to load at once
            fid = fopen(filepath, 'r');
            fgetl(fid); % Read line 1 and throw it away
            data = textscan(fid, fmtStr, 'Delimiter', ',');
            fclose(fid);
            transIds = data{transOutIdx};
            itemIds  = data{itemOutIdx};
        end
        
        function [transIds, itemIds] = readPrimaryChunked(obj, filepath, fmtStr, transOutIdx, itemOutIdx)
            % Chunked read — streams in blocks to protect memory during parse
            fid = fopen(filepath, 'r');
            fgetl(fid); % skip header
            
            chunks = {};
            chunkIdx = 0;
            totalRows = 0;
            
            while ~feof(fid) %While not end of the file
                data = textscan(fid, fmtStr, obj.ChunkSize, 'Delimiter', ','); %Read at most obj.ChunkSize
                if isempty(data{transOutIdx}), break; end
                
                chunkIdx = chunkIdx + 1;
                chunks{chunkIdx} = [data{transOutIdx}, data{itemOutIdx}]; %#ok<AGROW>
                totalRows = totalRows + numel(data{transOutIdx});
                
                if mod(chunkIdx, 10) == 0
                    fprintf('[TransactionalReader]   ... streamed %d chunks (%d rows)\n', ...
                        chunkIdx, totalRows);
                end
            end
            fclose(fid);
            
            % Concatenate all chunks
            allData  = vertcat(chunks{:});
            transIds = allData(:, 1);
            itemIds  = allData(:, 2);
            
            fprintf('[TransactionalReader]   Streaming complete: %d chunks, %d rows\n', ...
                chunkIdx, totalRows);
        end
        
        function itemIds = enrichItemIds(obj, itemIds)
            % ENRICHITEMIDS  Remap item IDs by joining on an enrichment file.
            %   Uses readtable for the enrichment file (typically small).
            
            enrichPath = fullfile(obj.SourcePath, obj.EnrichFile);
            obj.assertFileExists(enrichPath);
            
            fprintf('[TransactionalReader] Enriching items: %s (col %d -> col %d)\n', ...
                obj.EnrichFile, obj.EnrichFromCol, obj.EnrichToCol);
            
            T = readtable(enrichPath, 'TextType', 'string');
            
            fromIds = uint32(T{:, obj.EnrichFromCol});
            toIds   = uint32(T{:, obj.EnrichToCol});
            
            % Build dense lookup array: index by source ID, value = target ID
            maxId = max(fromIds);
            lookupTable = zeros(maxId, 1, 'uint32');
            lookupTable(fromIds) = toIds;
            
            % Remap
            validMask = itemIds <= maxId & itemIds > 0;
            enriched = zeros(size(itemIds), 'uint32');
            enriched(validMask) = lookupTable(itemIds(validMask));
            
            nUnmapped = sum(enriched == 0);
            if nUnmapped > 0
                fprintf('[TransactionalReader]   WARNING: %d items could not be remapped (dropped)\n', nUnmapped);
            end
            
            itemIds = enriched;
        end
        
        function baskets = groupByTransaction(~, transIds, itemIds)
            % GROUPBYTRANSACTION  Group item IDs by transaction ID into baskets.
            
            fprintf('[TransactionalReader] Grouping by transaction ID...\n');
            
            % Remove unmapped items (itemId == 0)
            validMask = itemIds > 0;
            transIds  = transIds(validMask);
            itemIds   = itemIds(validMask);
            
            % Sort by transaction ID for efficient grouping
            [transIds, sortIdx] = sort(transIds);
            itemIds = itemIds(sortIdx);
            
            % Find unique transactions and their boundaries
            [~, firstIdx, ~] = unique(transIds, 'first');
            [uniqueTrans, lastIdx, ~] = unique(transIds, 'last');
            
            N = numel(uniqueTrans);
            baskets = cell(N, 1);
            
            for i = 1:N
                rawItems = itemIds(firstIdx(i):lastIdx(i)); %Grab all items in this group
                % Remove duplicates within a basket
                baskets{i} = unique(rawItems)';  % row vector
            end
            
            fprintf('[TransactionalReader]   -> %d baskets formed\n', N);
        end
        
        function vocabulary = buildVocabulary(obj, itemIds)
            % BUILDVOCABULARY  Build ID -> human-readable name mapping.
            %   If LookupFile is specified, read it.
            %   Otherwise, auto-generate labels from item IDs.
            
            vocabulary = containers.Map('KeyType', 'uint32', 'ValueType', 'char');
            
            if obj.LookupFile ~= ""
                lookupPath = fullfile(obj.SourcePath, obj.LookupFile);
                obj.assertFileExists(lookupPath);
                
                fprintf('[TransactionalReader] Building vocabulary from: %s\n', obj.LookupFile);
                
                T = readtable(lookupPath, 'TextType', 'string');
                
                ids = uint32(T{:, obj.LookupIdCol});
                labels = T{:, obj.LookupLabelCol};
                
                % Handle both string and cell array labels
                if isstring(labels)
                    labels = cellstr(labels);
                end
                
                for i = 1:numel(ids)
                    vocabulary(ids(i)) = char(labels{i});
                end
            else
                % Auto-generate: "Item 1", "Item 2", ...
                uniqueIds = unique(itemIds(itemIds > 0));
                for i = 1:numel(uniqueIds)
                    vocabulary(uniqueIds(i)) = sprintf('Item %d', uniqueIds(i));
                end
            end
            
            fprintf('[TransactionalReader] Vocabulary built: %d entries\n', vocabulary.Count);
        end
    end
    
    %% ---- FLAT MODE (one-line-per-basket) ----
    methods (Access = private)
        function [baskets, vocabulary] = readFlat(obj)
            % READFLAT  Read flat transactional file (one basket per line).
            
            obj.assertFileExists(obj.SourcePath);
            
            fid = fopen(obj.SourcePath, 'r');
            rawLines = textscan(fid, '%s', 'Delimiter', '\n');
            fclose(fid);
            rawLines = rawLines{1};
            
            % Build vocabulary from unique item strings
            allItems = {};
            N = numel(rawLines);
            splitLines = cell(N, 1);
            
            for i = 1:N
                tokens = strsplit(strtrim(rawLines{i}), obj.Delimiter);
                tokens = strtrim(tokens);
                tokens = tokens(~cellfun('isempty', tokens));
                splitLines{i} = tokens;
                allItems = [allItems, tokens]; %#ok<AGROW>
            end
            
            % Assign integer IDs to unique items
            uniqueItems = unique(allItems);
            itemToId = containers.Map(uniqueItems, num2cell(uint32(1:numel(uniqueItems))));
            
            % Build reverse vocabulary: ID -> string
            vocabulary = containers.Map('KeyType', 'uint32', 'ValueType', 'char');
            for i = 1:numel(uniqueItems)
                vocabulary(uint32(i)) = char(uniqueItems{i});
            end
            
            % Convert lines to integer baskets
            baskets = cell(N, 1);
            for i = 1:N
                tokens = splitLines{i};
                ids = zeros(1, numel(tokens), 'uint32');
                for j = 1:numel(tokens)
                    ids(j) = itemToId(tokens{j});
                end
                baskets{i} = ids;
            end
            
            fprintf('[TransactionalReader] Flat read complete: %d baskets, %d unique items\n', ...
                N, vocabulary.Count);
        end
    end
    
    %% ---- PROFILING ----
    methods (Access = private)
        function prof = computeProfile(~, baskets, vocabSize)
            % COMPUTEPROFILE  Compute dimensional profile metrics.
            %   vocabSize = total items in catalog (from vocabulary file)
            %   nActiveItems = items that actually appear in baskets
            
            N = numel(baskets);
            p = double(vocabSize);  % Full universe (including 0-sales products)
            
            % Count items that actually appear in baskets
            allItems = cellfun(@(b) b(:), baskets, 'UniformOutput', false);
            allItems = cell2mat(allItems);
            nActive = double(numel(unique(allItems)));
            
            basketSizes = cellfun('length', baskets);
            totalItems  = sum(basketSizes);
            
            prof.nTransactions     = N;
            prof.nUniqueItems      = p;        % catalog size (full universe)
            prof.nActiveItems      = nActive;  % items with >= 1 sale
            prof.avgBasketSize     = totalItems / N;
            prof.medianBasketSize  = median(basketSizes);
            prof.maxBasketSize     = max(basketSizes);
            prof.totalNonzeros     = totalItems;
            prof.sparsityRatio     = 1 - totalItems / (N * p);
            prof.memoryEstimate    = N * p;  % bytes (1 byte per logical)
            prof.sparseRecommended = prof.sparsityRatio > 0.90;
            
            % Static helper alias
            fmt  = @OceanPipeline.Ingestion.TransactionalReader.formatNumber;
            fmtB = @OceanPipeline.Ingestion.TransactionalReader.formatBytes;
            
            % Print summary
            fprintf('\n');
            fprintf('  ============================================\n');
            fprintf('          DIMENSIONAL PROFILE\n');
            fprintf('  ============================================\n');
            fprintf('  Transactions (N)    : %s\n', fmt(N));
            fprintf('  Unique Items (p)    : %s\n', fmt(p));
            fprintf('  Avg Basket Size     : %.1f items\n', prof.avgBasketSize);
            fprintf('  Median Basket Size  : %.0f items\n', prof.medianBasketSize);
            fprintf('  Max Basket Size     : %d items\n', prof.maxBasketSize);
            fprintf('  Total Nonzeros      : %s\n', fmt(totalItems));
            fprintf('  Sparsity Ratio      : %.4f%%\n', prof.sparsityRatio * 100);
            fprintf('  Dense Memory Est.   : %s\n', fmtB(prof.memoryEstimate));
            fprintf('  Sparse Recommended  : %s\n', string(prof.sparseRecommended));
            fprintf('  ============================================\n\n');
        end
        
        function meta = buildTransactionalMeta(obj, profile)
            meta.source      = char(obj.SourcePath);
            meta.timestamp   = datetime('now');
            meta.format      = 'transactional';
            meta.mode        = char(obj.Mode);
            meta.nSamples    = profile.nTransactions;
            meta.nFeatures   = profile.nUniqueItems;
            meta.header      = {};
        end
    end
    
    %% ---- UTILITIES ----
    methods (Static)
        function s = formatNumber(n)
            % FORMATNUMBER  Format number with comma separators.
            s = char(string(n));
            if n >= 1000
                s = regexprep(fliplr(s), '(\d{3})(?=\d)', '$1,');
                s = fliplr(s);
            end
        end
        
        function s = formatBytes(b)
            % FORMATBYTES  Format byte count as human-readable string.
            if b < 1024
                s = sprintf('%d B', b);
            elseif b < 1024^2
                s = sprintf('%.1f KB', b / 1024);
            elseif b < 1024^3
                s = sprintf('%.1f MB', b / 1024^2);
            else
                s = sprintf('%.1f GB', b / 1024^3);
            end
        end
    end
    
    methods (Access = private, Static)
        function assertFileExists(filepath)
            if ~isfile(filepath)
                error('OceanPipeline:Ingestion:TransactionalReader:FileNotFound', ...
                    'Required file not found: %s', filepath);
            end
        end
        
        function [fmtStr, transOutIdx, itemOutIdx] = buildTextscanFormat(nCols, transCol, itemCol)
            % BUILDTEXTSCANFORMAT  Build a textscan format string that reads
            %   only the transaction and item columns, skipping all others.
            %
            %   Returns the format string plus the output indices for the
            %   two columns in the textscan result cell array.
            
            fmt = repmat({'%*u32'}, 1, nCols);
            outIdx = 0;
            transOutIdx = 0;
            itemOutIdx = 0;
            
            for c = 1:nCols
                if c == transCol || c == itemCol
                    fmt{c} = '%u32';
                    outIdx = outIdx + 1;
                    if c == transCol
                        transOutIdx = outIdx;
                    else
                        itemOutIdx = outIdx;
                    end
                end
            end
            
            fmtStr = [fmt{:}];  % concatenate without separators
        end
    end
end
