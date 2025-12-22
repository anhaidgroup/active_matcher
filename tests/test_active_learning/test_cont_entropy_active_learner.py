"""Tests for active_matcher.active_learning.cont_entropy_active_learner module.

This module provides ContinuousEntropyActiveLearner and helpers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from active_matcher.active_learning.cont_entropy_active_learner import (
    ContinuousEntropyActiveLearner,
    PQueueItem,
)
from active_matcher.labeler import CustomLabeler, GoldLabeler
from active_matcher.utils import save_training_data_streaming


class StopLabeler(CustomLabeler):
    def label_pair(self, row1, row2):
        if row1["_id"] == 10 and row2["_id"] == 20:
            return -1.0
        return 1.0 if (row1["_id"] + row2["_id"]) % 2 == 0 else 0.0


class TestPQueueItem:
    """Tests for PQueueItem ordering and construction."""

    def test_init(self):
        """Ensure PQueueItem stores entropy and item as provided."""
        item = PQueueItem(0.3, {"_id": 1})
        assert item.entropy == 0.3
        assert item.item["_id"] == 1


class TestContinuousEntropyActiveLearner:
    """Tests for ContinuousEntropyActiveLearner behavior."""

    def test_init(self, spark_session, default_model):
        """Validate __init__ parameter checks and defaults."""
        labeler = GoldLabeler({(1, 2)})

        learner = ContinuousEntropyActiveLearner(default_model, labeler, queue_size=3, max_labeled=4)
        assert learner._queue_size == 3
        assert learner._max_labeled == 4

        with pytest.raises(ValueError):
            ContinuousEntropyActiveLearner(default_model, labeler, queue_size=0)
        with pytest.raises(ValueError):
            ContinuousEntropyActiveLearner(default_model, labeler, max_labeled=0)

    def test_label_everything(self, spark_session, default_model, fvs_df):
        """Exercise _label_everything for full labeling."""
        labeler = GoldLabeler({(10, 20), (12, 22)})
        learner = ContinuousEntropyActiveLearner(default_model, labeler, queue_size=3, max_labeled=3)

        labeled_model = learner._label_everything(fvs_df)
        assert labeled_model is not None
        assert learner.local_training_fvs_ is not None
        assert "labeled_in_iteration" in learner.local_training_fvs_.columns

    def test_train_basic(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Verify train returns a trained model copy and updates state."""
        labeler = GoldLabeler({(10, 20), (12, 22), (13, 23)})
        parquet_path = temp_dir / "cont_training.parquet"

        learner = ContinuousEntropyActiveLearner(
            default_model, labeler, queue_size=3, max_labeled=3, on_demand_stop=False, parquet_file_path=str(parquet_path)
        )

        filtered = learner._select_training_vectors(fvs_df, [0])
        assert filtered.count() == 1
        pos, neg = learner._get_pos_negative(seed_df)
        assert pos == 1.0
        assert neg == 1.0

        trained = learner.train(fvs_df, seed_df)
        assert trained is not None
        assert trained._trained_model is not None
        assert learner.local_training_fvs_ is not None
        assert "labeled_in_iteration" in learner.local_training_fvs_.columns
        if parquet_path.exists():
            parquet_df = pd.read_parquet(parquet_path)
            assert set(["_id", "id1", "id2", "features", "label"]).issubset(parquet_df.columns)
            assert parquet_df["label"].isin([0.0, 1.0]).all()

    def test_train_existing_data(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Ensure existing parquet data path is used."""
        labeler = GoldLabeler({(10, 20), (12, 22), (13, 23)})
        existing_path = temp_dir / "cont_existing.parquet"

        save_training_data_streaming(seed_df, str(existing_path))

        learner_existing = ContinuousEntropyActiveLearner(
            default_model, labeler, queue_size=3, max_labeled=3, on_demand_stop=False, parquet_file_path=str(existing_path)
        )
        trained_existing = learner_existing.train(fvs_df, seed_df)
        assert trained_existing is not None
        assert trained_existing._trained_model is not None
        assert learner_existing.local_training_fvs_ is not None
        assert "labeled_in_iteration" in learner_existing.local_training_fvs_.columns

    def test_train_label_everything(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Exercise label-everything path."""
        labeler = GoldLabeler({(10, 20), (12, 22)})
        # Use fvs_df fixture and limit to 2 rows
        fvs_small = fvs_df.limit(2)
        learner_label_all = ContinuousEntropyActiveLearner(
            default_model, labeler, queue_size=3, max_labeled=1, on_demand_stop=False, parquet_file_path=str(temp_dir / "cont_label_all.parquet")
        )
        learner_label_all._terminate_if_label_everything = True
        label_all = learner_label_all.train(fvs_small, seed_df)
        assert label_all is not None

    def test_train_user_stop(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df, id_df_factory):
        """Ensure user stop is handled."""
        # Include all id1 and id2 values from fvs_df fixture
        a_df = id_df_factory([10, 11, 12, 13, 14, 15])
        b_df = id_df_factory([20, 21, 22, 23, 24, 25])
        stop_labeler = StopLabeler(a_df, b_df)

        learner_stop = ContinuousEntropyActiveLearner(
            default_model, stop_labeler, queue_size=3, max_labeled=3, on_demand_stop=True, parquet_file_path=str(temp_dir / "cont_stop.parquet")
        )
        stopped = learner_stop.train(fvs_df, seed_df)
        assert stopped is not None

    def test_train_queue_empty_returns_df(self, spark_session, temp_dir: Path, default_model, seed_df, fvs_df):
        """Ensure queue-empty branch returns labeled DataFrame."""
        labeler = GoldLabeler({(10, 20), (12, 22)})
        # Use fvs_df fixture and limit to 2 rows
        fvs_small = fvs_df.limit(2)
        # Use seed_df fixture but make all labels positive
        seed_all_positive = seed_df.copy()
        seed_all_positive["label"] = 1.0
        learner_empty = ContinuousEntropyActiveLearner(
            default_model, labeler, queue_size=3, max_labeled=3, on_demand_stop=True, parquet_file_path=str(temp_dir / "cont_empty.parquet")
        )
        labeled_data = learner_empty.train(fvs_small, seed_all_positive)
        assert isinstance(labeled_data, pd.DataFrame)
