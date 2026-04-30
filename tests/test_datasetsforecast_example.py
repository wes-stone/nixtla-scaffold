from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_example():
    path = REPO_ROOT / "examples" / "datasetsforecast_tourism_small" / "forecast_tourism_small.py"
    spec = importlib.util.spec_from_file_location("forecast_tourism_small_example", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tourism_small_adapter_builds_scaffold_hierarchy_and_matches_total() -> None:
    module = _load_example()
    dates = pd.to_datetime(["2020-03-31", "2020-06-30"])
    values = {
        "nsw-hol-city": [10.0, 11.0],
        "nsw-hol-noncity": [20.0, 21.0],
        "vic-hol-city": [30.0, 31.0],
        "vic-hol-noncity": [40.0, 41.0],
    }
    rows = []
    for unique_id, series_values in values.items():
        for ds, y in zip(dates, series_values):
            rows.append({"unique_id": unique_id, "ds": ds, "y": y})
    for index, ds in enumerate(dates):
        rows.append({"unique_id": "total", "ds": ds, "y": sum(series[index] for series in values.values())})
    source = pd.DataFrame(rows)
    tags = {module.BOTTOM_TAG: list(values)}
    expected_node_count = 1 + 1 + 2 + 4

    nodes = module.build_tourism_small_hierarchy(source, tags, s_df=pd.DataFrame(index=range(expected_node_count)))

    assert nodes["unique_id"].nunique() == expected_node_count
    assert set(nodes["hierarchy_level"]) == {"total", "purpose", "purpose/state", "purpose/state/city_noncity"}
    total = nodes.loc[nodes["unique_id"] == "Total"].sort_values("ds")
    assert total["y"].tolist() == [100.0, 104.0]
    assert "purpose=hol|state=nsw|city_noncity=city" in set(nodes["unique_id"])


def test_tourism_small_adapter_rejects_duplicate_bottom_rows() -> None:
    module = _load_example()
    source = pd.DataFrame(
        {
            "unique_id": ["nsw-hol-city", "nsw-hol-city"],
            "ds": ["2020-03-31", "2020-03-31"],
            "y": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate source_unique_id/ds"):
        module.tourism_small_leaf_frame(source, {module.BOTTOM_TAG: ["nsw-hol-city"]})


def test_tourism_small_adapter_requires_source_total_for_validation() -> None:
    module = _load_example()
    source = pd.DataFrame(
        {
            "unique_id": ["nsw-hol-city", "nsw-hol-city"],
            "ds": ["2020-03-31", "2020-06-30"],
            "y": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="must include a 'total' series"):
        module.build_tourism_small_hierarchy(source, {module.BOTTOM_TAG: ["nsw-hol-city"]})


def test_tourism_small_loader_requires_explicit_download_when_cache_missing(tmp_path, monkeypatch) -> None:
    module = _load_example()

    class FakeHierarchicalData:
        @staticmethod
        def load(**kwargs):  # pragma: no cover - should fail before load
            raise AssertionError("load should not run without --allow-download when cache is missing")

    monkeypatch.setattr(module, "_load_hierarchical_data_class", lambda: FakeHierarchicalData)

    with pytest.raises(RuntimeError, match="--allow-download"):
        module.load_tourism_small_dataset(tmp_path, allow_download=False)


def test_tourism_small_loader_has_actionable_missing_extra_message(monkeypatch) -> None:
    module = _load_example()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("datasetsforecast"):
            raise ModuleNotFoundError("No module named 'datasetsforecast'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uv run --extra datasets"):
        module._load_hierarchical_data_class()


@pytest.mark.skipif(os.environ.get("NIXTLA_SCAFFOLD_REALDATA") != "1", reason="real-data smoke is opt-in")
def test_tourism_small_real_data_example_smoke(tmp_path) -> None:
    module = _load_example()

    output_dir = module.run_example(tmp_path / "tourism_run", cache_dir=tmp_path / "cache", allow_download=True)

    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "hierarchy_reconciliation.csv").exists()
    forecast = pd.read_csv(output_dir / "forecast.csv")
    assert len(forecast) == module.EXPECTED_SOURCE_SERIES * module.HORIZON
    assert forecast["unique_id"].nunique() == module.EXPECTED_SOURCE_SERIES
