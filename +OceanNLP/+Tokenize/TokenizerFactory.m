classdef TokenizerFactory
    % TOKENIZERFACTORY  Create ITokenizer objects by name.
    %
    %   Centralizes tokenizer construction so callers never hard-code class
    %   names. Shape: Static create(name, ...), Static validNames().
    %
    %   CONVENTION-BASED (extension rule): a tokenizer named "foo" is the class
    %   OceanNLP.Tokenize.FooTokenizer. Adding a tokenizer is therefore adding
    %   ONE FILE to this package; this factory (and its name list) is never
    %   edited. Currently discovered: 'regex', 'whitespace'. Classes living in
    %   other packages can be added with
    %   OceanNLP.Text.StrategyResolver.register("OceanNLP.Tokenize", name, @ctor).
    %   Names are case-insensitive.
    %
    %   NAME-VALUE PASSTHROUGH: trailing name-value pairs given to create() are
    %   forwarded verbatim to the chosen tokenizer's constructor (e.g. 'Pattern'
    %   for 'regex', 'CaseSensitive' for either).
    %
    %   Usage:
    %     tok = OceanNLP.Tokenize.TokenizerFactory.create('whitespace');
    %     tok = OceanNLP.Tokenize.TokenizerFactory.create('regex', 'Pattern', "[\w']+");
    %     names = OceanNLP.Tokenize.TokenizerFactory.validNames();
    %
    %   See also: OceanNLP.Tokenize.ITokenizer, OceanNLP.Text.Factory

    methods (Static)
        function tok = create(name, options)
            arguments
                name (1,1) string
            end
            arguments (Repeating)
                options
            end
            try
                ctor = OceanNLP.Text.StrategyResolver.resolve("OceanNLP.Tokenize", "Tokenizer", name, "OceanNLP.Tokenize.ITokenizer");
            catch err
                if strcmp(err.identifier, 'OceanNLP:Text:StrategyResolver:Unknown')
                    error('OceanNLP:Tokenize:TokenizerFactory:UnknownTokenizer', ...
                        'Unknown tokenizer "%s". Valid: %s.', ...
                        name, strjoin(OceanNLP.Tokenize.TokenizerFactory.validNames(), ', '));
                end
                rethrow(err);
            end
            tok = ctor(options{:});
        end

        function names = validNames()
            names = OceanNLP.Text.StrategyResolver.names("OceanNLP.Tokenize", "Tokenizer");
        end
    end
end
