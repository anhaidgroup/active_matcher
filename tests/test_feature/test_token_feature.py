"""Tests for active_matcher.feature.token_ similarity feature module.

This module defines token-based similarity features.
"""
from active_matcher.feature.token_feature import (
    JaccardFeature,
    OverlapCoeffFeature,
    CosineFeature,
    MongeElkanFeature,
)

import pandas as pd
import numpy as np
import pytest


class TestJaccardFeature:

    def test_init(self, tokenizer):
        """Test initialization of JaccardFeature."""
        jaccard_feature = JaccardFeature('a_attr', 'b_attr', tokenizer)
        assert jaccard_feature.a_attr == 'a_attr'
        assert jaccard_feature.b_attr == 'b_attr'
        assert jaccard_feature._tokenizer == tokenizer
        assert jaccard_feature._a_toks_col == tokenizer.out_col_name('a_attr')
        assert jaccard_feature._b_toks_col == tokenizer.out_col_name('b_attr')

    def test_sim_func(self, tokenizer):
        """Test the sim_func method of JaccardFeature."""
        jaccard_feature = JaccardFeature('a_attr', 'b_attr', tokenizer)
        assert np.isnan(jaccard_feature.sim_func(None, None))
        assert jaccard_feature.sim_func('', '') == 0.0
        assert jaccard_feature.sim_func('aple', 'apple') == 4.0/5.0
        assert jaccard_feature.sim_func('a', 'a') == 1.0


    def test_str(self, tokenizer):
        """Test the __str__ method of JaccardFeature."""
        jaccard_feature = JaccardFeature('a_attr', 'b_attr', tokenizer)
        assert str(jaccard_feature) == f'jaccard({tokenizer.out_col_name("a_attr")}, {tokenizer.out_col_name("b_attr")})'

    def test_call(self, a_df, b_df, tokenizer):
        """Test the __call__ method of JaccardFeature."""
        jaccard_feature = JaccardFeature('a_attr', 'b_attr', tokenizer)
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()
        a_pdf = jaccard_feature.preprocess(a_pdf, True)
        b_pdf = jaccard_feature.preprocess(b_pdf, False)
        result = jaccard_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))


class TestOverlapCoeffFeature:
    def test_init(self, tokenizer):
        """Test initialization of OverlapCoeffFeature."""
        overlap_coeff_feature = OverlapCoeffFeature('a_attr', 'b_attr', tokenizer)
        assert overlap_coeff_feature.a_attr == 'a_attr'
        assert overlap_coeff_feature.b_attr == 'b_attr'
        assert overlap_coeff_feature._tokenizer == tokenizer
        assert overlap_coeff_feature._a_toks_col == tokenizer.out_col_name('a_attr')
        assert overlap_coeff_feature._b_toks_col == tokenizer.out_col_name('b_attr')

    def test_sim_func(self, tokenizer):
        """Test the sim_func method of OverlapCoeffFeature."""
        overlap_coeff_feature = OverlapCoeffFeature('a_attr', 'b_attr', tokenizer)
        assert np.isnan(overlap_coeff_feature.sim_func(None, None))
        assert overlap_coeff_feature.sim_func('', '') == 0.0
        assert overlap_coeff_feature.sim_func('aple', 'apple') == 1.0
        assert overlap_coeff_feature.sim_func('a', 'a') == 1.0

    def test_str(self, tokenizer):
        """Test the __str__ method of OverlapCoeffFeature."""
        overlap_coeff_feature = OverlapCoeffFeature('a_attr', 'b_attr', tokenizer)
        assert str(overlap_coeff_feature) == f'overlap_coeff({tokenizer.out_col_name("a_attr")}, {tokenizer.out_col_name("b_attr")})'

    def test_call(self, a_df, b_df, tokenizer):
        """Test the __call__ method of OverlapCoeffFeature."""
        overlap_coeff_feature = OverlapCoeffFeature('a_attr', 'b_attr', tokenizer)
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()
        a_pdf = overlap_coeff_feature.preprocess(a_pdf, True)
        b_pdf = overlap_coeff_feature.preprocess(b_pdf, False)
        result = overlap_coeff_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))


class TestCosineFeature:
    def test_init(self, tokenizer):
        """Test initialization of CosineFeature."""
        cosine_feature = CosineFeature('a_attr', 'b_attr', tokenizer)
        assert cosine_feature.a_attr == 'a_attr'
        assert cosine_feature.b_attr == 'b_attr'
        assert cosine_feature._tokenizer == tokenizer
        assert cosine_feature._a_toks_col == tokenizer.out_col_name('a_attr')
        assert cosine_feature._b_toks_col == tokenizer.out_col_name('b_attr')

    def test_sim_func(self, tokenizer):
        """Test the sim_func method of CosineFeature."""
        cosine_feature = CosineFeature('a_attr', 'b_attr', tokenizer)
        assert np.isnan(cosine_feature.sim_func(None, None))
        assert cosine_feature.sim_func('', '') == 0.0
        assert np.isclose(cosine_feature.sim_func('aple', 'apple'), 4.0/np.sqrt(5*4))
        assert cosine_feature.sim_func('a', 'a') == 1.0

    def test_str(self, tokenizer):
        """Test the __str__ method of CosineFeature."""
        cosine_feature = CosineFeature('a_attr', 'b_attr', tokenizer)
        assert str(cosine_feature) == f'cosine({tokenizer.out_col_name("a_attr")}, {tokenizer.out_col_name("b_attr")})' 

    def test_call(self, a_df, b_df, tokenizer):
        """Test the __call__ method of CosineFeature."""
        cosine_feature = CosineFeature('a_attr', 'b_attr', tokenizer)
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()
        a_pdf = cosine_feature.preprocess(a_pdf, True)
        b_pdf = cosine_feature.preprocess(b_pdf, False)
        result = cosine_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))


class TestMongeElkanFeature:
    def test_init(self, tokenizer):
        """Test initialization of MongeElkanFeature."""
        monge_elkan_feature = MongeElkanFeature('a_attr', 'b_attr', tokenizer)
        assert monge_elkan_feature.a_attr == 'a_attr'
        assert monge_elkan_feature.b_attr == 'b_attr'
        assert monge_elkan_feature._tokenizer == tokenizer
        assert monge_elkan_feature._a_toks_col == tokenizer.out_col_name('a_attr')
        assert monge_elkan_feature._b_toks_col == tokenizer.out_col_name('b_attr')

    def test_sim_func(self, tokenizer):
        """Test the sim_func method of MongeElkanFeature."""
        monge_elkan_feature = MongeElkanFeature('a_attr', 'b_attr', tokenizer)
        with pytest.raises(TypeError):
            monge_elkan_feature.sim_func(None, '')
        assert np.isnan(monge_elkan_feature.sim_func('', None))
        assert np.isclose(monge_elkan_feature.sim_func(['aple'], ['apple']), 0.9466, rtol=1e-4)
        assert monge_elkan_feature.sim_func(['a'], ['a']) == 1.0

    def test_str(self, tokenizer):
        """Test the __str__ method of MongeElkanFeature."""
        monge_elkan_feature = MongeElkanFeature('a_attr', 'b_attr', tokenizer)
        assert str(monge_elkan_feature) == f'monge_elkan_jw({"a_attr"}, {"b_attr"})'

    def test_call(self, a_df, b_df, tokenizer):
        """Test the __call__ method of MongeElkanFeature."""
        monge_elkan_feature = MongeElkanFeature('a_attr', 'b_attr', tokenizer)
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()
        a_pdf = monge_elkan_feature.preprocess(a_pdf, True)
        b_pdf = monge_elkan_feature.preprocess(b_pdf, False)
        result = monge_elkan_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))
