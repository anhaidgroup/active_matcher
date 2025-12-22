"""Tests for active_matcher.feature.feature module.

This module defines base Feature classes and string similarity features.
"""
from active_matcher.feature.feature import (
    ExactMatchFeature,
    RelDiffFeature,
    EditDistanceFeature,
    NeedlemanWunschFeature,
    SmithWatermanFeature,
)

import pandas as pd
import numpy as np


class TestExactMatchFeature:
    """Tests for ExactMatchFeature and exercising the Feature base class."""
    em_feature = ExactMatchFeature('a_attr', 'b_attr')

    def test_a_attr(self):
        """Ensure a_attr is set correctly."""
        assert self.em_feature.a_attr == 'a_attr'

    def test_b_attr(self):
        """Ensure b_attr is set correctly."""
        assert self.em_feature.b_attr == 'b_attr'

    def test_preprocess_output_column(self):
        """Ensure preprocess_output_column is set correctly."""
        assert self.em_feature.preprocess_output_column(True) is None
        assert self.em_feature.preprocess_output_column(False) is None

    def test_preprocess_a(self, a_df):
        """Ensure preprocess is set correctly."""
        assert self.em_feature.preprocess(a_df, True) == a_df

    def test_preprocess_b(self, b_df):
        """Ensure preprocess is set correctly."""
        assert self.em_feature.preprocess(b_df, False) == b_df

    def test_call(self, a_df, b_df):
        """Ensure __call__ returns the feature score."""
        self.em_feature.preprocess(a_df, True)
        self.em_feature.preprocess(b_df, False)
        b_pdf = b_df.toPandas()
        a_pdf = a_df.toPandas()
        result = self.em_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))

    def test_str(self):
        """Ensure __str__ returns the feature name."""
        assert str(self.em_feature) == 'exact_match(a_attr, b_attr)'


class TestRelDiffFeature:
    """Tests for RelDiffFeature."""

    rel_diff_feature = RelDiffFeature('a_num', 'b_num')

    def test_a_attr(self):
        """Ensure a_attr is set correctly."""
        assert self.rel_diff_feature.a_attr == 'a_num'

    def test_b_attr(self):
        """Ensure b_attr is set correctly."""
        assert self.rel_diff_feature.b_attr == 'b_num'

    def test_preprocess_output_column(self):
        """Ensure preprocess_output_column is set correctly."""
        assert self.rel_diff_feature.preprocess_output_column(True) == 'float(a_num)'
        assert self.rel_diff_feature.preprocess_output_column(False) == 'float(b_num)'

    def test_preprocess_a(self, a_df):
        """Ensure preprocess is set correctly."""
        a_pdf = a_df.toPandas()
        result_df = self.rel_diff_feature.preprocess(a_pdf, True)
        assert 'float(a_num)' in result_df.columns
        expected = pd.Series([1.0, 2.0, 3.0], name='float(a_num)')
        assert result_df['float(a_num)'].equals(expected)

    def test_preprocess_b(self, b_df):
        """Ensure preprocess is set correctly."""
        b_pdf = b_df.toPandas()
        result_df = self.rel_diff_feature.preprocess(b_pdf, False)
        assert 'float(b_num)' in result_df.columns
        expected = pd.Series([1.0, 2.0, 3.0], name='float(b_num)')
        assert result_df['float(b_num)'].equals(expected)

    def test_call(self, a_df, b_df):
        """Ensure __call__ returns the feature score."""
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()
        self.rel_diff_feature.preprocess(a_pdf, True)
        self.rel_diff_feature.preprocess(b_pdf, False)
        result = self.rel_diff_feature(b_pdf.iloc[0], a_pdf)
        expected = pd.Series([0.0, 0.5, 2.0/3.0])
        pd.testing.assert_series_equal(result, expected, rtol=1e-6)

    def test_str(self):
        """Ensure __str__ returns the feature name."""
        assert str(self.rel_diff_feature) == 'rel_diff(a_num, b_num)'


class TestEditDistanceFeature:
    """Tests for EditDistanceFeature."""

    edit_distance_feature = EditDistanceFeature('a_attr', 'b_attr')

    def test_a_attr(self):
        """Ensure a_attr is set correctly."""
        assert self.edit_distance_feature.a_attr == 'a_attr'

    def test_b_attr(self):
        """Ensure b_attr is set correctly."""
        assert self.edit_distance_feature.b_attr == 'b_attr'

    def test_preprocess_output_column(self):
        """Ensure preprocess_output_column is set correctly."""
        assert self.edit_distance_feature.preprocess_output_column(True) is None
        assert self.edit_distance_feature.preprocess_output_column(False) is None

    def test_preprocess_a(self, a_df):
        """Ensure preprocess is set correctly."""
        assert self.edit_distance_feature.preprocess(a_df, True) == a_df

    def test_preprocess_b(self, b_df):
        """Ensure preprocess is set correctly."""
        assert self.edit_distance_feature.preprocess(b_df, False) == b_df

    def test_call(self, a_df, b_df):
        """Ensure __call__ returns the feature score."""
        self.edit_distance_feature.preprocess(a_df, True)
        self.edit_distance_feature.preprocess(b_df, False)
        b_pdf = b_df.toPandas()
        a_pdf = a_df.toPandas()
        result = self.edit_distance_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))

    def test_str(self):
        """Ensure __str__ returns the feature name."""
        assert str(self.edit_distance_feature) == 'edit_distance(a_attr, b_attr)'


class TestNeedlemanWunschFeature:
    """Tests for NeedlemanWunschFeature."""

    needleman_wunsch_feature = NeedlemanWunschFeature('a_attr', 'b_attr')

    def test_a_attr(self):
        """Ensure a_attr is set correctly."""
        assert self.needleman_wunsch_feature.a_attr == 'a_attr'

    def test_b_attr(self):
        """Ensure b_attr is set correctly."""
        assert self.needleman_wunsch_feature.b_attr == 'b_attr'

    def test_preprocess_output_column(self):
        """Ensure preprocess_output_column is set correctly."""
        assert self.needleman_wunsch_feature.preprocess_output_column(True) is None
        assert self.needleman_wunsch_feature.preprocess_output_column(False) is None

    def test_preprocess_a(self, a_df):
        """Ensure preprocess is set correctly."""
        assert self.needleman_wunsch_feature.preprocess(a_df, True) == a_df

    def test_preprocess_b(self, b_df):
        """Ensure preprocess is set correctly."""
        assert self.needleman_wunsch_feature.preprocess(b_df, False) == b_df

    def test_call(self, a_df, b_df):
        """Ensure __call__ returns the feature score."""
        self.needleman_wunsch_feature.preprocess(a_df, True)
        self.needleman_wunsch_feature.preprocess(b_df, False)
        b_pdf = b_df.toPandas()
        a_pdf = a_df.toPandas()
        result = self.needleman_wunsch_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))

    def test_str(self):
        """Ensure __str__ returns the feature name."""
        assert str(self.needleman_wunsch_feature) == 'needleman_wunch(a_attr, b_attr)'


class TestSmithWatermanFeature:
    """Tests for SmithWatermanFeature."""

    smith_waterman_feature = SmithWatermanFeature('a_attr', 'b_attr')

    def test_a_attr(self):
        """Ensure a_attr is set correctly."""
        assert self.smith_waterman_feature.a_attr == 'a_attr'

    def test_b_attr(self):
        """Ensure b_attr is set correctly."""
        assert self.smith_waterman_feature.b_attr == 'b_attr'

    def test_preprocess_output_column(self):
        """Ensure preprocess_output_column is set correctly."""
        assert self.smith_waterman_feature.preprocess_output_column(True) is None
        assert self.smith_waterman_feature.preprocess_output_column(False) is None

    def test_preprocess_a(self, a_df):
        """Ensure preprocess is set correctly."""
        assert self.smith_waterman_feature.preprocess(a_df, True) == a_df

    def test_preprocess_b(self, b_df):
        """Ensure preprocess is set correctly."""
        assert self.smith_waterman_feature.preprocess(b_df, False) == b_df

    def test_call(self, a_df, b_df):
        """Ensure __call__ returns the feature score."""
        self.smith_waterman_feature.preprocess(a_df, True)
        self.smith_waterman_feature.preprocess(b_df, False)
        b_pdf = b_df.toPandas()
        a_pdf = a_df.toPandas()
        result = self.smith_waterman_feature(b_pdf.iloc[0], a_pdf)
        assert result.equals(pd.Series([1.0, 0.0, 0.0]))

    def test_str(self):
        """Ensure __str__ returns the feature name."""
        assert str(self.smith_waterman_feature) == 'smith_waterman(a_attr, b_attr)'
