"""Tests for active_matcher.tokenizer.tokenizer module.

This module defines tokenizer variants for string preprocessing.
"""
from active_matcher.tokenizer.tokenizer import (
    StrippedWhiteSpaceTokenizer,
    ShingleTokenizer,
    WhiteSpaceTokenizer,
    NumericTokenizer,
    AlphaNumericTokenizer,
    QGramTokenizer,
    StrippedQGramTokenizer,
)


class TestTokenizerBase:
    """Tests for Tokenizer abstract base class."""

    def test_out_col_name(self):
        """Test out_col_name derives output names from column ids."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        result = tokenizer.out_col_name('name')
        expected = f'{str(tokenizer)}(name)'
        assert result == expected

    def test_str(self):
        """Test __str__ returns NAME."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        assert str(tokenizer) == 'stripped_whitespace_tokens'

    def test_tokenize_set(self):
        """Test tokenize_set converts tokenize result to set."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        result = tokenizer.tokenize_set('hello world')
        assert isinstance(result, set)
        assert result == {'hello', 'world'}

    def test_tokenize_set_none(self):
        """Test tokenize_set returns None when tokenize returns None."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        result = tokenizer.tokenize_set(None)
        assert result is None

    def test_eq(self):
        """Test __eq__ method."""
        tokenizer1 = StrippedWhiteSpaceTokenizer()
        tokenizer2 = StrippedWhiteSpaceTokenizer()
        tokenizer3 = WhiteSpaceTokenizer()
        assert tokenizer1 == tokenizer2
        assert tokenizer1 != tokenizer3
        assert tokenizer1 != "not a tokenizer"

    def test_tokenize_spark(self, spark_session):
        """Test tokenize_spark creates Spark UDF."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        df = spark_session.createDataFrame([
            {"text": "hello world"},
            {"text": "test code"},
        ])
        result = df.select(
            tokenizer.tokenize_spark('text').alias('tokens')
        ).collect()
        assert len(result) == 2
        assert isinstance(result[0]['tokens'], list)
        assert isinstance(result[1]['tokens'], list)


class TestWhitespaceTokenizers:
    """Tests for whitespace-based tokenizers."""

    def test_whitespace_tokenizer(self):
        """Test WhiteSpaceTokenizer splits tokens correctly."""
        tokenizer = WhiteSpaceTokenizer()
        result = tokenizer.tokenize('Hello World Test')
        assert result == ['hello', 'world', 'test']
        assert tokenizer.tokenize('  Multiple   Spaces  ') == [
            'multiple', 'spaces'
        ]

    def test_whitespace_tokenizer_none(self):
        """Test WhiteSpaceTokenizer handles None input."""
        tokenizer = WhiteSpaceTokenizer()
        assert tokenizer.tokenize(None) is None
        assert tokenizer.tokenize(123) is None

    def test_stripped_whitespace_tokenizer(self):
        """Test StrippedWhiteSpaceTokenizer trims punctuation."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        result = tokenizer.tokenize('Hello, World! Test.')
        assert result == ['hello', 'world', 'test']
        assert tokenizer.tokenize('Test@123#Code') == ['test123code']

    def test_stripped_whitespace_tokenizer_whitespace(self):
        """Test StrippedWhiteSpaceTokenizer normalizes whitespace."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        result = tokenizer.tokenize('  Multiple   Spaces  ')
        assert result == ['multiple', 'spaces']

    def test_stripped_whitespace_tokenizer_none(self):
        """Test StrippedWhiteSpaceTokenizer handles None input."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        assert tokenizer.tokenize(None) is None
        assert tokenizer.tokenize(123) is None

    def test_shingle_tokenizer(self):
        """Test ShingleTokenizer generates shingles correctly."""
        tokenizer = ShingleTokenizer(2)
        result = tokenizer.tokenize('hello world python')
        assert len(result) > 0
        assert all(isinstance(tok, str) for tok in result)

    def test_shingle_tokenizer_short_string(self):
        """Test ShingleTokenizer with short string."""
        tokenizer = ShingleTokenizer(3)
        result = tokenizer.tokenize('hi')
        assert result == []

    def test_shingle_tokenizer_none(self):
        """Test ShingleTokenizer handles None input."""
        tokenizer = ShingleTokenizer(2)
        assert tokenizer.tokenize(None) is None

    def test_shingle_tokenizer_different_n(self):
        """Test ShingleTokenizer with different n values."""
        tokenizer2 = ShingleTokenizer(2)
        tokenizer3 = ShingleTokenizer(3)
        text = 'hello world python code'
        result2 = tokenizer2.tokenize(text)
        result3 = tokenizer3.tokenize(text)
        assert len(result3) < len(result2)


class TestQGramTokenizers:
    """Tests for q-gram tokenizers."""

    def test_qgram_tokenizer(self):
        """Test QGramTokenizer generates expected n-grams."""
        tokenizer = QGramTokenizer(3)
        result = tokenizer.tokenize('hello')
        assert result == ['hel', 'ell', 'llo']
        assert tokenizer.tokenize('test') == ['tes', 'est']

    def test_qgram_tokenizer_case(self):
        """Test QGramTokenizer lowercases input."""
        tokenizer = QGramTokenizer(2)
        result = tokenizer.tokenize('HELLO')
        assert result == ['he', 'el', 'll', 'lo']

    def test_qgram_tokenizer_short_string(self):
        """Test QGramTokenizer with short string."""
        tokenizer = QGramTokenizer(5)
        result = tokenizer.tokenize('hi')
        assert result == []

    def test_qgram_tokenizer_none(self):
        """Test QGramTokenizer handles None input."""
        tokenizer = QGramTokenizer(3)
        assert tokenizer.tokenize(None) is None
        assert tokenizer.tokenize(123) is None

    def test_qgram_tokenizer_different_n(self):
        """Test QGramTokenizer with different n values."""
        tokenizer2 = QGramTokenizer(2)
        tokenizer3 = QGramTokenizer(3)
        text = 'hello'
        result2 = tokenizer2.tokenize(text)
        result3 = tokenizer3.tokenize(text)
        assert len(result2) > len(result3)

    def test_stripped_qgram_tokenizer(self):
        """Test StrippedQGramTokenizer removes non-alphanumerics."""
        tokenizer = StrippedQGramTokenizer(3)
        result = tokenizer.tokenize('hello-world!')
        assert 'hel' in result
        assert 'world' not in result
        assert '-' not in ''.join(result)
        assert '!' not in ''.join(result)

    def test_stripped_qgram_tokenizer_preproc(self):
        """Test StrippedQGramTokenizer _preproc method."""
        tokenizer = StrippedQGramTokenizer(2)
        result = tokenizer._preproc('test@123#code')
        assert result == 'test123code'
        assert '@' not in result
        assert '#' not in result

    def test_stripped_qgram_tokenizer_short_string(self):
        """Test StrippedQGramTokenizer with short string."""
        tokenizer = StrippedQGramTokenizer(5)
        result = tokenizer.tokenize('hi')
        assert result == []

    def test_stripped_qgram_tokenizer_none(self):
        """Test StrippedQGramTokenizer handles None input."""
        tokenizer = StrippedQGramTokenizer(3)
        assert tokenizer.tokenize(None) is None
        assert tokenizer.tokenize(123) is None

    def test_stripped_qgram_tokenizer_different_n(self):
        """Test StrippedQGramTokenizer with different n values."""
        tokenizer2 = StrippedQGramTokenizer(2)
        tokenizer3 = StrippedQGramTokenizer(3)
        text = 'hello-world'
        result2 = tokenizer2.tokenize(text)
        result3 = tokenizer3.tokenize(text)
        assert len(result2) > len(result3)


class TestNumericTokenizers:
    """Tests for numeric-aware tokenizers."""

    def test_numeric_tokenizer(self):
        """Test NumericTokenizer filters to numeric tokens."""
        tokenizer = NumericTokenizer()
        result = tokenizer.tokenize('test123code456')
        assert result == ['123', '456']
        assert tokenizer.tokenize('abc 123 def 456') == ['123', '456']

    def test_numeric_tokenizer_only_numbers(self):
        """Test NumericTokenizer with only numbers."""
        tokenizer = NumericTokenizer()
        result = tokenizer.tokenize('123456')
        assert result == ['123456']

    def test_numeric_tokenizer_no_numbers(self):
        """Test NumericTokenizer with no numbers."""
        tokenizer = NumericTokenizer()
        result = tokenizer.tokenize('hello world')
        assert result == []

    def test_numeric_tokenizer_none(self):
        """Test NumericTokenizer handles None input."""
        tokenizer = NumericTokenizer()
        assert tokenizer.tokenize(None) is None
        assert tokenizer.tokenize(123) is None

    def test_alpha_numeric_tokenizer(self):
        """Test AlphaNumericTokenizer keeps alphanumeric tokens."""
        tokenizer = AlphaNumericTokenizer()
        result = tokenizer.tokenize('test123code456')
        assert 'test123code456' in result
        assert len(result) == 1

    def test_alpha_numeric_tokenizer_case(self):
        """Test AlphaNumericTokenizer lowercases input."""
        tokenizer = AlphaNumericTokenizer()
        result = tokenizer.tokenize('TEST123CODE')
        assert 'test123code' in result
        assert len(result) == 1

    def test_alpha_numeric_tokenizer_punctuation(self):
        """Test AlphaNumericTokenizer removes punctuation."""
        tokenizer = AlphaNumericTokenizer()
        result = tokenizer.tokenize('test@123#code')
        assert 'test' in result
        assert '123' in result
        assert 'code' in result
        assert '@' not in ''.join(result)
        assert '#' not in ''.join(result)

    def test_alpha_numeric_tokenizer_none(self):
        """Test AlphaNumericTokenizer handles None input."""
        tokenizer = AlphaNumericTokenizer()
        assert tokenizer.tokenize(None) is None
        assert tokenizer.tokenize(123) is None

    def test_alpha_numeric_tokenizer_mixed(self):
        """Test AlphaNumericTokenizer with mixed content."""
        tokenizer = AlphaNumericTokenizer()
        result = tokenizer.tokenize('hello123 world456 test')
        assert len(result) == 3
        assert 'hello123' in result
        assert 'world456' in result
        assert 'test' in result
