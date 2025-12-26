"""Tests for active_matcher.algorithms module.

This module exposes helper functions for active learning selection.
"""
from pathlib import Path

from active_matcher.algorithms import down_sample, select_seeds


class TestAlgorithms:
    """Tests for down_sample and select_seeds functions."""

    def test_down_sample(self, fvs_df):
        """Test down_sample reduces the pool appropriately."""
        down_sampled_fvs = down_sample(fvs_df, 0.5)
        assert down_sampled_fvs.count() == fvs_df.count() * 0.5

    def test_select_seeds_sequence(self, fvs_df, labeler, temp_dir: Path):
        """Test select_seeds sequence: initial, enough existing, not enough."""
        parquet_path = str(
            temp_dir / "test-active-matcher-training-data.parquet"
        )

        seeds = select_seeds(fvs_df, 4, labeler, 'score', parquet_path)
        assert len(seeds) == 4
        assert set(seeds['_id'].tolist()) == set([0, 1, 4, 5])

        seeds = select_seeds(fvs_df, 4, labeler, 'score', parquet_path)
        assert len(seeds) == 4
        assert set(seeds['_id'].tolist()) == set([0, 1, 4, 5])

        seeds = select_seeds(fvs_df, 6, labeler, 'score', parquet_path)
        assert len(seeds) == 6
        assert set(seeds['_id'].tolist()) == set([0, 1, 2, 3, 4, 5])
