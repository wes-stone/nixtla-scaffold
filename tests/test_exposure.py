from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from nixtla_scaffold.exposure import consume_confirmation, record_exposure, reserve_confirmation
from nixtla_scaffold.profile import profile_dataset
from nixtla_scaffold.schema import ForecastSpec


def data():
    return pd.DataFrame({"unique_id": ["001"] * 8, "ds": pd.date_range("2024-01-01", periods=8, freq="MS"),
                         "y": [float(i) for i in range(8)]})


def test_profiling_prevents_relabeling_known_targets_as_untouched():
    frame, spec = data(), ForecastSpec()
    profile_dataset(frame, spec)
    receipt = reserve_confirmation(frame.tail(3), spec)
    assert receipt["status"] == "previously_exposed_or_reserved"
    assert not consume_confirmation(receipt, frame.tail(3), spec)["passed"]


def test_reservation_is_one_use_even_after_interruptions():
    frame, spec = data(), ForecastSpec()
    receipt = reserve_confirmation(frame, spec)
    assert consume_confirmation(receipt, frame, spec)["status"] == "consumed_for_frozen_confirmation"
    assert consume_confirmation(receipt, frame, spec)["status"] == "confirmation_consumed"
    assert reserve_confirmation(frame.copy(), spec)["status"] == "previously_exposed_or_reserved"


def test_profiling_after_reservation_invalidates_confirmation():
    frame, spec = data(), ForecastSpec()
    receipt = reserve_confirmation(frame, spec)
    record_exposure(frame.tail(1), spec)
    assert consume_confirmation(receipt, frame, spec)["status"] == "exposed_after_reservation"


def test_later_declared_external_exposure_invalidates_reservation():
    frame, spec = data(), ForecastSpec()
    receipt = reserve_confirmation(frame, spec)
    spec = replace(spec, evaluation=replace(spec.evaluation, known_target_exposure="exposed"))
    assert consume_confirmation(receipt, frame, spec)["status"] == "exposed_after_reservation"


def test_changed_vintage_cannot_use_an_existing_reservation():
    frame, spec = data(), ForecastSpec()
    receipt = reserve_confirmation(frame, spec)
    with pytest.raises(ValueError, match="snapshot changed"):
        consume_confirmation(receipt, frame.assign(y=frame.y + 1), spec)


def test_concurrent_reservations_have_only_one_owner():
    frame, spec = data(), ForecastSpec()
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: reserve_confirmation(frame, spec), range(2)))
    assert sorted(row["status"] for row in receipts) == ["previously_exposed_or_reserved", "reserved"]


def test_known_exposure_cannot_be_reset_by_a_new_scope(tmp_path):
    spec = ForecastSpec()
    spec = replace(spec, evaluation=replace(spec.evaluation, evidence_directory=str(tmp_path / "fresh"),
                                           known_target_exposure="exposed"))
    assert reserve_confirmation(data(), spec)["status"] == "previously_exposed_or_reserved"


def test_exposure_survives_process_restart_without_reading_target_values(tmp_path):
    frame, spec = data(), ForecastSpec()
    record_exposure(frame, spec)
    script = (
        "import pandas as pd; from nixtla_scaffold.schema import ForecastSpec; "
        "from nixtla_scaffold.exposure import reserve_confirmation; "
        "f=pd.DataFrame({'unique_id':['001'],'ds':[pd.Timestamp('2024-08-01')],'y':[999.]}); "
        "assert reserve_confirmation(f,ForecastSpec())['status']=='previously_exposed_or_reserved'"
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())
    database = Path(os.environ["NIXTLA_EVIDENCE_DIRECTORY"]) / "exposure.sqlite3"
    assert database.exists()


def test_structural_calendar_profile_does_not_expose_reserved_targets():
    frame, spec = data(), ForecastSpec()
    profile_dataset(frame[["unique_id", "ds"]].assign(y=float("nan")), spec)
    assert reserve_confirmation(frame, spec)["status"] == "reserved"


def test_failed_target_transform_still_records_inspected_history():
    from nixtla_scaffold import TransformSpec, run_forecast
    frame = data().assign(y=-2.)
    spec = ForecastSpec(transform=TransformSpec(target="log"))
    with pytest.raises(ValueError):
        run_forecast(frame, spec)
    assert reserve_confirmation(frame, spec)["status"] == "previously_exposed_or_reserved"
