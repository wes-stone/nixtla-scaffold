from __future__ import annotations

FPPY_CITATION = (
    "Hyndman, R.J., Athanasopoulos, G., Garza, A., Challu, C., "
    "Mergenthaler, M., & Olivares, K.G. (2025). Forecasting: "
    "Principles and Practice, the Pythonic Way. OTexts: Melbourne, "
    "Australia. Available at: OTexts.com/fpppy. Accessed on 28 April 2026."
)


def fppy_source(note: str, url: str) -> str:
    return f"{FPPY_CITATION} {note}: {url}"
