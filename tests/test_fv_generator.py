"""Tests for active_matcher.fv_generator module.

This module defines BuildCache and FVGenerator for feature vector creation.
"""
import pytest
import numpy as np
import pandas as pd

from active_matcher.fv_generator import BuildCache, FVGenerator
from active_matcher.feature import RelDiffFeature, ExactMatchFeature


class TestBuildCache:
    """Tests for BuildCache behavior."""

    def test_add_or_get(self):
        """Ensure add_or_get returns existing builder or stores new one."""
        cache = BuildCache()
        builder1 = object()
        builder2 = object()
        builder3 = object()
        result1 = cache.add_or_get(builder1)
        assert result1 is builder1
        result2 = cache.add_or_get(builder1)
        assert result2 is builder1
        result3 = cache.add_or_get(builder2)
        assert result3 is builder2
        result4 = cache.add_or_get(builder1)
        assert result4 is builder1
        result5 = cache.add_or_get(builder3)
        assert result5 is builder3

    def test_clear(self):
        """Verify clear removes cached builders and resets cache state."""
        cache = BuildCache()
        builder1 = object()
        builder2 = object()

        result1a = cache.add_or_get(builder1)
        result2a = cache.add_or_get(builder2)
        assert result1a is builder1
        assert result2a is builder2

        result1b = cache.add_or_get(builder1)
        result2b = cache.add_or_get(builder2)
        assert result1b is builder1
        assert result2b is builder2

        cache.clear()
        builder_new = object()
        assert cache.add_or_get(builder_new) is builder_new

        builder1_again = object()
        assert cache.add_or_get(builder1_again) is builder1_again

        assert cache.add_or_get(builder1_again) is builder1_again


class TestFVGenerator:
    """Tests for FVGenerator workflows."""

    def test_properties(self):
        """Confirm features and feature_names properties reflect inputs."""
        feature1 = RelDiffFeature('num', 'num')
        feature2 = ExactMatchFeature('attr', 'attr')
        features = [feature1, feature2]

        fv_gen = FVGenerator(features)

        assert fv_gen.features == features
        assert len(fv_gen.feature_names) == 2
        assert 'rel_diff(num, num)' in fv_gen.feature_names
        assert 'exact_match(attr, attr)' in fv_gen.feature_names

    def test_build(self, spark_session):
        """Validate build preprocesses tables and initializes caches."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
            {"_id": 12, "attr": "c", "num": 3.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
            {"_id": 21, "attr": "b", "num": 2.0},
            {"_id": 22, "attr": "c", "num": 3.0},
        ])

        feature1 = RelDiffFeature('num', 'num')
        feature2 = ExactMatchFeature('attr', 'attr')
        features = [feature1, feature2]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, b_df)

        assert fv_gen._table_a_preproc is not None
        assert fv_gen._table_b_preproc is not None

        fv_gen.release_resources()

    def test_build_single_table(self, spark_session):
        """Validate build works with only table A."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
        ])

        feature1 = RelDiffFeature('num', 'num')
        features = [feature1]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, None)

        assert fv_gen._table_a_preproc is not None
        assert fv_gen._table_b_preproc is not None
        assert fv_gen._table_a_preproc is fv_gen._table_b_preproc

        fv_gen.release_resources()

    def test_generate_fvs(self, spark_session):
        """Ensure generate_fvs produces feature vectors with _id."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
            {"_id": 12, "attr": "c", "num": 3.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
            {"_id": 21, "attr": "b", "num": 2.0},
            {"_id": 22, "attr": "c", "num": 3.0},
        ])

        feature1 = RelDiffFeature('num', 'num')
        feature2 = ExactMatchFeature('attr', 'attr')
        features = [feature1, feature2]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, b_df)

        pairs = spark_session.createDataFrame([
            {"id2": 20, "id1_list": [10]},
            {"id2": 21, "id1_list": [11]},
            {"id2": 22, "id1_list": [12]},
        ])

        fvs = fv_gen.generate_fvs(pairs)

        columns = fvs.columns
        assert '_id' in columns
        assert 'id1' in columns
        assert 'id2' in columns
        assert 'features' in columns

        fvs_pdf = fvs.toPandas()

        assert len(fvs_pdf) == 3

        assert all(isinstance(fv, list) for fv in fvs_pdf['features'])

        assert all(len(fv) == 2 for fv in fvs_pdf['features'])

        first_row = fvs_pdf[fvs_pdf['id2'] == 20].iloc[0]
        assert first_row['id1'] == 10
        features_first = np.array(first_row['features'], dtype=np.float32)
        np.testing.assert_allclose(features_first[0], 0.0, rtol=1e-5)
        np.testing.assert_allclose(features_first[1], 1.0, rtol=1e-5)

        second_row = fvs_pdf[fvs_pdf['id2'] == 21].iloc[0]
        assert second_row['id1'] == 11
        features_second = np.array(second_row['features'], dtype=np.float32)
        np.testing.assert_allclose(features_second[0], 0.0, rtol=1e-5)
        np.testing.assert_allclose(features_second[1], 1.0, rtol=1e-5)

        third_row = fvs_pdf[fvs_pdf['id2'] == 22].iloc[0]
        assert third_row['id1'] == 12
        features_third = np.array(third_row['features'], dtype=np.float32)
        np.testing.assert_allclose(features_third[0], 0.0, rtol=1e-5)
        np.testing.assert_allclose(features_third[1], 1.0, rtol=1e-5)

        fv_gen.release_resources()

    def test_generate_fvs_multiple_id1(self, spark_session):
        """Test generate_fvs with multiple id1 values for same id2."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
        ])

        feature1 = RelDiffFeature('num', 'num')
        features = [feature1]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, b_df)

        pairs = spark_session.createDataFrame([
            {"id2": 20, "id1_list": [10, 11]},
        ])

        fvs = fv_gen.generate_fvs(pairs)
        fvs_pdf = fvs.toPandas()

        assert len(fvs_pdf) == 2

        row1 = fvs_pdf[fvs_pdf['id1'] == 10].iloc[0]
        features1 = np.array(row1['features'], dtype=np.float32)
        np.testing.assert_allclose(features1[0], 0.0, rtol=1e-5)
        row2 = fvs_pdf[fvs_pdf['id1'] == 11].iloc[0]
        features2 = np.array(row2['features'], dtype=np.float32)
        np.testing.assert_allclose(features2[0], 0.5, rtol=1e-5)

        fv_gen.release_resources()

    def test_generate_fvs_fill_na(self, spark_session):
        """Test generate_fvs with fill_na parameter."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
        ])

        feature1 = RelDiffFeature('num', 'num')
        features = [feature1]

        fv_gen = FVGenerator(features, fill_na=999.0)
        fv_gen.build(a_df, b_df)

        pairs = spark_session.createDataFrame([
            {"id2": 20, "id1_list": [10]},
        ])

        fv_gen.generate_fvs(pairs)

        assert fv_gen._fill_na == 999.0

        fv_gen.release_resources()

    def test_generate_fvs_not_built(self, spark_session):
        """Test that generate_fvs raises error if not built."""
        feature1 = RelDiffFeature('a_num', 'b_num')
        features = [feature1]

        fv_gen = FVGenerator(features)

        pairs = spark_session.createDataFrame([
            {"id2": 20, "id1_list": [10]},
        ])

        with pytest.raises(RuntimeError, match='must be built'):
            fv_gen.generate_fvs(pairs)

    def test_generate_and_score_fvs(self, spark_session):
        """Confirm generate_and_score_fvs adds score column."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
            {"_id": 12, "attr": "c", "num": 3.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
            {"_id": 21, "attr": "b", "num": 2.0},
            {"_id": 22, "attr": "c", "num": 3.0},
        ])

        feature1 = ExactMatchFeature('attr', 'attr')
        features = [feature1]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, b_df)

        pairs = spark_session.createDataFrame([
            {"id2": 20, "id1_list": [10]},
            {"id2": 21, "id1_list": [11]},
            {"id2": 22, "id1_list": [12]},
        ])

        fvs = fv_gen.generate_and_score_fvs(pairs)

        columns = fvs.columns
        assert 'score' in columns
        assert 'features' in columns
        assert '_id' in columns

        fvs_pdf = fvs.toPandas()

        assert len(fvs_pdf) == 3

        assert all(pd.notna(score) for score in fvs_pdf['score'])
        assert all(
            isinstance(score, (int, float)) for score in fvs_pdf['score']
        )

        for _, row in fvs_pdf.iterrows():
            features_arr = np.array(row['features'], dtype=np.float32)
            expected_score = float(np.sum(features_arr))
            np.testing.assert_allclose(row['score'], expected_score, rtol=1e-5)

        fv_gen.release_resources()

    def test_generate_and_score_fvs_mixed_features(
        self, spark_session
    ):
        """Test generate_and_score_fvs with mixed feature types."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
        ])

        feature1 = ExactMatchFeature('attr', 'attr')
        feature2 = RelDiffFeature('num', 'num')
        features = [feature1, feature2]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, b_df)

        pairs = spark_session.createDataFrame([
            {"id2": 20, "id1_list": [10]},
        ])

        fvs = fv_gen.generate_and_score_fvs(pairs)
        fvs_pdf = fvs.toPandas()

        row = fvs_pdf.iloc[0]
        features_arr = np.array(row['features'], dtype=np.float32)
        pos_cor = [1, 0]
        expected_score = sum(f * c for f, c in zip(features_arr, pos_cor))
        np.testing.assert_allclose(row['score'], expected_score, rtol=1e-5)

        fv_gen.release_resources()

    def test_release_resources(self, spark_session):
        """Verify release_resources cleans up memmaps."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "attr": "a", "num": 1.0},
        ])

        feature1 = RelDiffFeature('num', 'num')
        features = [feature1]

        fv_gen = FVGenerator(features)
        fv_gen.build(a_df, b_df)

        assert fv_gen._table_a_preproc is not None
        assert fv_gen._table_b_preproc is not None

        fv_gen.release_resources()

        assert fv_gen._table_a_preproc is None
        assert fv_gen._table_b_preproc is None

    def test_release_resources_not_built(self):
        """Test release_resources when not built."""
        feature1 = RelDiffFeature('a_num', 'b_num')
        features = [feature1]

        fv_gen = FVGenerator(features)

        fv_gen.release_resources()

        assert fv_gen._table_a_preproc is None
        assert fv_gen._table_b_preproc is None
