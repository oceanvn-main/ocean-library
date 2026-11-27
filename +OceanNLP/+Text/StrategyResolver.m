classdef StrategyResolver
    % STRATEGYRESOLVER  Finds strategy classes by NAME CONVENTION so factories
    % never need a hand-edited switch: adding an algorithm = adding one file.
    %
    %   A strategy named "porter" of suffix "Stemmer" in package
    %   OceanNLP.Text.Stem is class OceanNLP.Text.Stem.PorterStemmer.
    %   Matching is case-insensitive and ignores '_' and '-'.
    %
    % Classes that live outside the scanned package can be registered
    % explicitly (register); registrations win over the scan.
    %
    %   ctor  = StrategyResolver.resolve("OceanNLP.Tokenize", "Tokenizer", "regex")
    %   names = StrategyResolver.names("OceanNLP.Tokenize", "Tokenizer")   % {'regex','whitespace'}
    %
    % See also: OceanNLP.Text.Factory

    methods (Static)
        function ctor = resolve(pkg, suffix, name, baseClass)
            % RESOLVE  ctor handle for strategy "name". baseClass (optional): the interface the class
            % must derive from; a scanned class that does not is rejected with a clear error instead
            % of failing later inside a consumer.
            arguments
                pkg (1,1) string
                suffix (1,1) string
                name (1,1) string
                baseClass (1,1) string = ""
            end
            key = OceanNLP.Text.StrategyResolver.normalizeName_(name);
            reg = OceanNLP.Text.StrategyResolver.registry_();
            rk = pkg + "/" + key;
            if isKey(reg, char(rk))
                ctor = reg(char(rk));
                return
            end
            [nm, cls] = OceanNLP.Text.StrategyResolver.scan_(pkg, suffix);
            idx = find(nm == key, 1);
            if isempty(idx)
                error('OceanNLP:Text:StrategyResolver:Unknown', ...
                    'Unknown %s "%s" in %s. Valid: %s.', suffix, name, pkg, ...
                    strjoin(OceanNLP.Text.StrategyResolver.names(pkg, suffix), ', '));
            end
            if baseClass ~= "" && ~any(string(superclasses(char(cls(idx)))) == baseClass)
                error('OceanNLP:Text:StrategyResolver:NotAStrategy', ...
                    '%s matches the name "%s" but does not derive from %s.', cls(idx), name, baseClass);
            end
            ctor = str2func(cls(idx));
        end

        function n = names(pkg, suffix)
            % NAMES  All available strategy names (scanned + registered), sorted.
            arguments
                pkg (1,1) string
                suffix (1,1) string
            end
            [nm, ~] = OceanNLP.Text.StrategyResolver.scan_(pkg, suffix);
            reg = OceanNLP.Text.StrategyResolver.registry_();
            regKeys = string(keys(reg));
            mine = regKeys(startsWith(regKeys, pkg + "/"));
            nm = unique([nm, extractAfter(mine, strlength(pkg) + 1)']);
            n = cellstr(nm);
        end

        function register(pkg, name, ctor)
            % REGISTER  Make a class from ANY package available under (pkg, name).
            arguments
                pkg (1,1) string
                name (1,1) string
                ctor (1,1) function_handle
            end
            reg = OceanNLP.Text.StrategyResolver.registry_();
            reg(char(pkg + "/" + OceanNLP.Text.StrategyResolver.normalizeName_(name))) = ctor;
        end

        function unregisterAll()
            % UNREGISTERALL  Clear explicit registrations (test hygiene).
            reg = OceanNLP.Text.StrategyResolver.registry_();
            remove(reg, keys(reg));
        end
    end

    methods (Static, Access = private)
        function [nm, cls] = scan_(pkg, suffix)
            p = meta.package.fromName(char(pkg));
            nm = strings(1, 0);  cls = strings(1, 0);
            if isempty(p), return; end
            for i = 1:numel(p.ClassList)
                c = p.ClassList(i);
                short = string(extractAfter(c.Name, strlength(pkg) + 1));
                if c.Abstract || ~endsWith(short, suffix) || strlength(short) == strlength(suffix)
                    continue
                end
                nm(end+1) = OceanNLP.Text.StrategyResolver.normalizeName_(extractBefore(short, strlength(short) - strlength(suffix) + 1)); %#ok<AGROW>
                cls(end+1) = string(c.Name); %#ok<AGROW>
            end
            [nm, order] = sort(nm);
            cls = cls(order);
            [~, firstIdx] = unique(nm, 'stable');
            if numel(firstIdx) ~= numel(nm)
                dup = nm(setdiff(1:numel(nm), firstIdx));
                error('OceanNLP:Text:StrategyResolver:AmbiguousName', ...
                    'Strategy names collide after normalization in %s: %s (case, "_" and "-" are ignored).', pkg, strjoin(unique(dup), ', '));
            end
        end

        function s = normalizeName_(name)
            s = lower(regexprep(char(name), '[_\-\s]', ''));
            s = string(s);
        end

        function reg = registry_()
            persistent r
            if isempty(r)
                r = containers.Map('KeyType', 'char', 'ValueType', 'any');
            end
            reg = r;
        end
    end
end
