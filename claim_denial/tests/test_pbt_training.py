# Feature: claim-denial-prediction, Property 1: Temporal Train/Test Split Preserves Boundary Date Assignment
# Feature: claim-denial-prediction, Property 2: Class Weight Balancing Does Not Alter Training Data Volume
"""
Property-based tests for TrainingDatasetBuilder temporal split and class-weight balancing.

**Validates: Requirements 8.3, 8.5**

Property 1: Temporal Train/Test Split Preserves Boundary Date Assignment
  For any labeled dataset with ≥1,000 records, applying the temporal 80/20 split SHALL
  place ALL claims whose submission_date equals the boundary cutoff date into the test
  set, and the test set SHALL contain exactly the most-recent 20% (rounded up at ties)
  of distinct submission dates.

Property 2: Class Weight Balancing Does Not Alter Training Data Volume
  For any labeled training dataset where the denied class constitutes fewer than 5% of
  records, applying class-weight balancing (scale_pos_weight / class_weight) SHALL leave
  the total number of training records unchanged.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from claim_denial.training.dataset_builder import (
    TrainingDataset,
    TrainingDatasetBuilder,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2023, 1, 1)
_OUTCOME_DENIED = "denied"
_OUTCOME_PAID = "paid"


def _make_record(outcome: str, submission_date: date) -> dict:
    """Minimal feature-store record for labeling/splitting tests."""
    return {
        "claim_id": f"CLM-{submission_date.isoformat()}",
        "submission_date": submission_date,
        "adjudication_outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def labeled_dataset_with_ties(draw) -> TrainingDataset:
    """
    Generate a TrainingDataset with ≥1,000 records distributed across
    a configurable number of distinct submission dates, with deliberate
    tie scenarios on boundary dates.

    Strategy (optimised to avoid HealthCheck.too_slow):
    - Choose a spread of 10–50 distinct dates.
    - Distribute 1,000–1,500 records across those dates (uneven buckets
      so ties on boundary dates occur naturally).
    - Build the lists directly from compact integer arrays (no per-item
      strategy draws) to keep generation fast.
    """
    n_dates = draw(st.integers(min_value=10, max_value=50))
    n_records = draw(st.integers(min_value=1_000, max_value=1_500))

    # Fixed anchor date for reproducibility under shrinking
    start_day_offset = draw(st.integers(min_value=0, max_value=180))
    base = _BASE_DATE + timedelta(days=start_day_offset)

    # Single compact draw: list of (date_index, is_denied) packed as integers
    # date_index ∈ [0, n_dates-1], is_denied ∈ {0,1}
    # Encode as date_index * 2 + is_denied  →  values in [0, 2*n_dates - 1]
    max_val = 2 * n_dates - 1
    packed = draw(
        st.lists(
            st.integers(min_value=0, max_value=max_val),
            min_size=n_records,
            max_size=n_records,
        )
    )

    # Pre-build date and record templates to avoid repeated timedelta construction
    date_cache = [base + timedelta(days=i) for i in range(n_dates)]

    records: List[dict] = []
    labels: List[int] = []
    dates: List[date] = []

    for p in packed:
        date_idx = p // 2
        denied = p % 2
        sub_date = date_cache[date_idx]
        outcome = _OUTCOME_DENIED if denied else _OUTCOME_PAID
        records.append({"claim_id": f"C{p}", "submission_date": sub_date,
                         "adjudication_outcome": outcome})
        labels.append(denied)
        dates.append(sub_date)

    return TrainingDataset(records=records, labels=labels, submission_dates=dates)


@st.composite
def imbalanced_dataset_denied_under_5pct(draw) -> TrainingDataset:
    """
    Generate a TrainingDataset where the denied class is strictly <5%.

    Strategy:
    - Total records: 1,000–5,000
    - Denied records: between 1 and floor(total * 0.049) so ratio < 5%
    - All records share a single submission date (date does not matter for
      the class-weight property, but the dataset must be valid).
    """
    total = draw(st.integers(min_value=1_000, max_value=5_000))
    max_denied = max(1, math.floor(total * 0.049))
    denied_count = draw(st.integers(min_value=1, max_value=max_denied))

    sub_date = _BASE_DATE

    records: List[dict] = []
    labels: List[int] = []
    dates: List[date] = []

    for i in range(total):
        if i < denied_count:
            outcome = _OUTCOME_DENIED
            label = 1
        else:
            outcome = _OUTCOME_PAID
            label = 0
        records.append(_make_record(outcome, sub_date))
        labels.append(label)
        dates.append(sub_date)

    return TrainingDataset(records=records, labels=labels, submission_dates=dates)


# ---------------------------------------------------------------------------
# Property 1: Temporal Train/Test Split Preserves Boundary Date Assignment
#
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(dataset=labeled_dataset_with_ties())
def test_temporal_split_boundary_date_in_test_set(dataset: TrainingDataset) -> None:
    """**Validates: Requirements 8.3**

    # Feature: claim-denial-prediction, Property 1: Temporal Train/Test Split Preserves Boundary Date Assignment

    For any labeled dataset with ≥1,000 records:
    1. ALL claims whose submission_date == boundary cutoff date SHALL be in the test set.
    2. The test set SHALL contain exactly the most-recent 20% (rounded up at ties) of
       distinct submission dates.
    """
    builder = TrainingDatasetBuilder()
    train_ds, test_ds = builder.temporal_split(dataset)

    n = len(dataset)
    assert n >= 1_000, "Generator invariant violated: dataset must have ≥1,000 records"

    # ── 1. Every claim in the test set must have a date ≥ every train date ─────
    if train_ds.submission_dates and test_ds.submission_dates:
        max_train_date = max(train_ds.submission_dates)
        min_test_date = min(test_ds.submission_dates)
        assert min_test_date >= max_train_date, (
            f"Test set contains a date ({min_test_date}) earlier than "
            f"the latest train date ({max_train_date})"
        )

    # ── 2. Identify the boundary (cutoff) date ────────────────────────────────
    # The boundary date is the smallest submission_date in the test set.
    if not test_ds.submission_dates:
        # Edge case: empty test set is only valid if dataset is all-same-date
        all_same = len(set(dataset.submission_dates)) == 1
        assert all_same, "Empty test set but dataset has multiple distinct dates"
        return

    boundary_date = min(test_ds.submission_dates)

    # ── 3. ALL claims on the boundary date must be in the test set ────────────
    for i, d in enumerate(dataset.submission_dates):
        if d == boundary_date:
            # That record must be in test_ds, not train_ds
            assert d not in train_ds.submission_dates or all(
                td != boundary_date for td in train_ds.submission_dates
            ), (
                f"Claim on boundary_date={boundary_date} found in TRAIN set. "
                "All boundary-date claims must be in the TEST set."
            )

    # ── 4. No train record has the boundary date ──────────────────────────────
    for td in train_ds.submission_dates:
        assert td != boundary_date, (
            f"Train set contains a record with boundary_date={boundary_date}. "
            "All such claims must be assigned to the test set."
        )

    # ── 5. Test set size ≥ ceil(20% of total) ────────────────────────────────
    target_test_size = math.ceil(n * 0.20)
    assert len(test_ds) >= target_test_size, (
        f"Test set size {len(test_ds)} < ceil(20% * {n}) = {target_test_size}"
    )

    # ── 6. Test set contains exactly the most-recent distinct dates ───────────
    # Determine how many distinct dates should be in the test set:
    # sort all distinct dates, then the test set should cover the most-recent
    # dates such that the total record count in those dates >= target_test_size
    # and adding one more date would exceed the split point.
    all_dates_sorted = sorted(set(dataset.submission_dates))
    # Build date → record count mapping
    date_count: dict = {}
    for d in dataset.submission_dates:
        date_count[d] = date_count.get(d, 0) + 1

    # Walk from the most-recent date backward; accumulate until we hit target_test_size
    accumulated = 0
    test_dates_expected = []
    for d in reversed(all_dates_sorted):
        test_dates_expected.append(d)
        accumulated += date_count[d]
        if accumulated >= target_test_size:
            break

    expected_test_date_set = set(test_dates_expected)
    actual_test_date_set = set(test_ds.submission_dates)

    assert actual_test_date_set == expected_test_date_set, (
        f"Test set dates {sorted(actual_test_date_set)} != "
        f"expected {sorted(expected_test_date_set)}"
    )


# ---------------------------------------------------------------------------
# Property 2: Class Weight Balancing Does Not Alter Training Data Volume
#
# Validates: Requirements 8.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(dataset=imbalanced_dataset_denied_under_5pct())
def test_class_weight_balancing_does_not_alter_record_count(
    dataset: TrainingDataset,
) -> None:
    """**Validates: Requirements 8.5**

    # Feature: claim-denial-prediction, Property 2: Class Weight Balancing Does Not Alter Training Data Volume

    For any labeled training dataset where the denied class constitutes <5% of records,
    applying class-weight balancing (check_class_imbalance) SHALL leave the total number
    of training records unchanged.
    """
    builder = TrainingDatasetBuilder()

    # Verify the generator invariant: denied class must be < 5%
    n = len(dataset)
    denied_count = sum(1 for lbl in dataset.labels if lbl == 1)
    denied_pct = denied_count / n
    assert denied_pct < 0.05, (
        f"Generator invariant violated: denied_pct={denied_pct:.4f} is not < 5%"
    )

    record_count_before = len(dataset)

    # Apply class-weight balancing (returns a ClassWeightConfig, never modifies dataset)
    weight_config = builder.check_class_imbalance(dataset)

    record_count_after = len(dataset)

    # ── Core property: total record count is unchanged ────────────────────────
    assert record_count_before == record_count_after, (
        f"Record count changed from {record_count_before} to {record_count_after} "
        "after class-weight balancing. Balancing must NOT modify training data volume."
    )

    # ── Sanity: returned weights reflect the imbalance (scale_pos_weight > 1) ─
    paid_count = n - denied_count
    expected_spw = paid_count / denied_count
    assert weight_config.scale_pos_weight == pytest.approx(expected_spw, rel=1e-6), (
        f"scale_pos_weight={weight_config.scale_pos_weight} != "
        f"paid/denied={expected_spw:.4f}"
    )
    assert weight_config.scale_pos_weight > 1.0, (
        f"scale_pos_weight={weight_config.scale_pos_weight} should be > 1.0 "
        "when denied class < 5%"
    )

    # ── class_weight dict must be consistent with scale_pos_weight ────────────
    assert weight_config.class_weight.get(0) == pytest.approx(1.0), (
        f"class_weight[0]={weight_config.class_weight.get(0)} should be 1.0"
    )
    assert weight_config.class_weight.get(1) == pytest.approx(expected_spw, rel=1e-6), (
        f"class_weight[1]={weight_config.class_weight.get(1)} != "
        f"paid/denied={expected_spw:.4f}"
    )


# need pytest.approx
import pytest  # noqa: E402 — imported after to keep the top-level header clean
