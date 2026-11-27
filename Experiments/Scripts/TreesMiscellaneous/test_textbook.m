function test_textbook()
    try
        % Find data robustly
        targetFile = 'spambase.data';
        foundPath = '';
        
        searchBase = pwd;
        for i = 1:5
            p = fullfile(searchBase, 'Experiments', 'Scripts', 'Data', 'spambase', targetFile);
            if isfile(p), foundPath = p; break; end
            p = fullfile(searchBase, 'Data', 'spambase', targetFile);
            if isfile(p), foundPath = p; break; end
            p = fullfile(searchBase, targetFile);
            if isfile(p), foundPath = p; break; end
            
            searchBase = fileparts(searchBase);
            if isempty(searchBase), break; end
        end
        
        if isempty(foundPath), error('spambase.data not found in search path.'); end
        dataPath = foundPath;
        dataDir = fileparts(foundPath);
        
        raw = load(dataPath);
        X = raw(:, 1:end-1);
        Y = raw(:, end);
        fprintf('Loaded %d samples from %s\n', length(Y), dataPath);
        fprintf('Global Purity: %.4f\n', mean(Y));
        
        % Feature names for mapping
        namesFile = fullfile(dataDir, 'spambase.names');
        fileID = fopen(namesFile, 'r');
        namesRaw = textscan(fileID, '%s %*s', 'Delimiter', ':', 'CommentStyle', '|', 'HeaderLines', 32);
        fclose(fileID);
        featNames = namesRaw{1};
        featNames = featNames(end-56:end); 
        
        % Rule 1 from ESL Table 9.4
        f_ch_excl = find(strcmp(featNames, 'char_freq_!'));
        f_capave  = find(strcmp(featNames, 'capital_run_length_average'));
        f_your    = find(strcmp(featNames, 'word_freq_your'));
        f_1999    = find(strcmp(featNames, 'word_freq_1999'));
        f_captot  = find(strcmp(featNames, 'capital_run_length_total'));
        f_edu     = find(strcmp(featNames, 'word_freq_edu'));
        f_re      = find(strcmp(featNames, 'word_freq_re'));
        f_ch_semi = find(strcmp(featNames, 'char_freq_;'));
        
        mask = (X(:, f_ch_excl) > 0.029) & ...
               (X(:, f_capave) > 2.331) & ...
               (X(:, f_your) > 0.705) & ...
               (X(:, f_1999) < 0.040) & ...
               (X(:, f_captot) > 79.50) & ...
               (X(:, f_edu) < 0.070) & ...
               (X(:, f_re) < 0.535) & ...
               (X(:, f_ch_semi) < 0.030);
               
        fprintf('Textbook Rule 1 Purity: %.4f\n', mean(Y(mask)));
        fprintf('Textbook Rule 1 Support: %.4f\n', mean(mask));
        fprintf('Textbook Rule 1 n: %d\n', sum(mask));
        
        % Check Rule 2
        f_remove = find(strcmp(featNames, 'word_freq_remove'));
        f_george = find(strcmp(featNames, 'word_freq_george'));
        
        mask2 = (X(:, f_remove) > 0.010) & (X(:, f_george) < 0.110);
        
        fprintf('Textbook Rule 2 Purity (on whole data): %.4f\n', mean(Y(mask2)));
        fprintf('Textbook Rule 2 Support: %.4f\n', mean(mask2));
        
    catch ME
        fprintf('Error: %s\n', ME.message);
    end
end
