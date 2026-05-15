# Air + Tourism forecast demo

This demo gives a screenshot-friendly path from a fast AirPassengers baseline to a richer run with a log transform, synthetic scenario overlay, known-future regressor audit, and the generated Streamlit Control Pane. TourismSmall remains optional because it can download public data through `datasetsforecast`.

Run the AirPassengers demo:

```powershell
uv run python examples\air_tourism_demo\forecast_air_tourism_demo.py --output runs\air_tourism_demo
```

Open either generated workbench:

```powershell
Set-Location runs\air_tourism_demo\air_passengers_base
.\run_streamlit.ps1

Set-Location ..\air_passengers_full
.\run_streamlit.ps1
```

Include the public TourismSmall hierarchy smoke only when you explicitly allow the dataset download:

```powershell
uv run --extra datasets python examples\air_tourism_demo\forecast_air_tourism_demo.py --include-tourism --allow-download --output runs\air_tourism_demo_full
```

The AirPassengers full run uses synthetic future capacity and a synthetic 1961 event only to demonstrate the audit trail. Treat those assumptions as illustrative, not as real aviation domain evidence.
