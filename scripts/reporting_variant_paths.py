from __future__ import annotations

from pathlib import Path


VALID_VARIANTS = {"dynamic", "static"}


def validate_variant(variant: str) -> str:
    normalized = str(variant).strip().lower()
    if normalized not in VALID_VARIANTS:
        raise ValueError(
            f"Unsupported variant {variant!r}. Expected one of {sorted(VALID_VARIANTS)}."
        )
    return normalized


def variant_suffix(variant: str) -> str:
    normalized = validate_variant(variant)
    return "" if normalized == "dynamic" else "_static"


def with_variant_suffix(filename: str, variant: str) -> str:
    suffix = variant_suffix(variant)
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix}"


def model_report_path(root: Path, report_kind: str, variant: str) -> Path:
    return (
        root
        / "slidedeck/data"
        / with_variant_suffix(f"{report_kind}_model_report.xlsx", variant)
    )


def data_path(root: Path, filename: str, variant: str) -> Path:
    return root / "slidedeck/data" / with_variant_suffix(filename, variant)


def asset_path(root: Path, filename: str, variant: str) -> Path:
    return root / "slidedeck/assets" / with_variant_suffix(filename, variant)
