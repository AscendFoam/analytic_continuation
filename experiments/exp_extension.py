"""Extension-domain experiment for sequence types A/B/C."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from math import gamma, isfinite, isclose
import os
from pathlib import Path
import sys
from tempfile import gettempdir
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "analytic_continuation_mpl"))

from analytic_continuation.core.sequence import (  # noqa: E402
    FactorialType,
    FixedBaseTetration,
    RecurrenceSequence,
    VariableBaseTetration,
)
from analytic_continuation.evaluation.metrics import absolute_error  # noqa: E402
from analytic_continuation.evaluation.visualization import plot_series  # noqa: E402
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult  # noqa: E402
from analytic_continuation.methods.chebyshev import ChebyshevMethod  # noqa: E402
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod  # noqa: E402
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod  # noqa: E402


EXTENSION_POINTS = [-2.0, -1.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def _build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "exp_extension" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def _sequence_suite() -> list[tuple[str, RecurrenceSequence]]:
    return [
        ("A", VariableBaseTetration()),
        ("B", FixedBaseTetration(base=1.3)),
        ("C", FactorialType()),
    ]


def _method_suite() -> list[ContinuationMethod]:
    return [
        HermiteCubicMethod(),
        HermiteQuinticMethod(),
        ChebyshevMethod(),
    ]


def _region_label(seq: RecurrenceSequence, z: float) -> str:
    left, right = seq.base_interval
    if isinstance(seq, VariableBaseTetration) and isclose(z, seq.n0 - 1.0, abs_tol=1e-12):
        return "type_a_z0_limit"
    if isinstance(seq, VariableBaseTetration) and isclose(z, seq.n0 - 2.0, abs_tol=1e-12):
        return "type_a_z_minus_1_limit"
    if left <= z <= right:
        return "base_interval"
    if z > right:
        return "forward_extension"
    return "backward_extension"


def _factorial_truth(seq: RecurrenceSequence, z: float) -> tuple[float | str, float | str]:
    if not isinstance(seq, FactorialType):
        return "", ""
    try:
        truth = gamma(z + 1.0)
    except ValueError:
        return "", ""
    if not isfinite(truth):
        return "", ""
    return truth, ""


def _safe_recurrence_residual(
    seq: RecurrenceSequence,
    method: ContinuationMethod,
    result: ContinuationResult,
    z: float,
) -> tuple[float | str, str]:
    if isinstance(seq, VariableBaseTetration) and z <= 0.0:
        return "", "skipped_variable_base_singularity"
    try:
        left_value = method.evaluate(z, seq, result)
        previous_value = method.evaluate(z - 1.0, seq, result)
        residual = abs(left_value - seq.g(z, previous_value))
    except Exception as exc:  # noqa: BLE001
        return "", f"failed:{type(exc).__name__}"
    return float(residual), "ok"


def _endpoint_residual_summary(
    seq: RecurrenceSequence,
    result: ContinuationResult,
    *,
    max_order: int = 4,
) -> tuple[str, float | str]:
    n0, n1 = seq.base_interval
    left_derivatives = {
        order: result.base_solution.derivative(n0, order=order)
        for order in range(1, max_order + 1)
    }
    right_derivatives = {
        order: result.base_solution.derivative(n1, order=order)
        for order in range(1, max_order + 1)
    }
    residuals = seq.derivative_constraint_residuals(
        max_order,
        left_derivatives=left_derivatives,
        right_derivatives=right_derivatives,
    )
    max_residual = max((abs(value) for _, value in residuals), default=0.0)
    return _json({order: value for order, value in residuals}), float(max_residual)


def _value_rows_for_result(
    sequence_code: str,
    seq: RecurrenceSequence,
    method: ContinuationMethod,
    result: ContinuationResult,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for point in EXTENSION_POINTS:
        row: dict[str, float | int | str] = {
            "sequence_code": sequence_code,
            "sequence": seq.name,
            "method": method.name,
            "z": point,
            "region": _region_label(seq, point),
            "value": "",
            "truth": "",
            "abs_error": "",
            "recurrence_residual": "",
            "recurrence_status": "",
            "status": "ok",
            "error_message": "",
        }
        try:
            value = method.evaluate(point, seq, result)
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"failed:{type(exc).__name__}"
            row["error_message"] = str(exc)
            rows.append(row)
            continue

        row["value"] = value
        truth, _ = _factorial_truth(seq, point)
        row["truth"] = truth
        if truth != "":
            row["abs_error"] = absolute_error(float(value), float(truth))
        residual, residual_status = _safe_recurrence_residual(seq, method, result, point)
        row["recurrence_residual"] = residual
        row["recurrence_status"] = residual_status
        rows.append(row)
    return rows


def _run_extension() -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    summary_rows: list[dict[str, float | int | str]] = []
    value_rows: list[dict[str, float | int | str]] = []

    for sequence_code, seq in _sequence_suite():
        for method in _method_suite():
            started = perf_counter()
            try:
                result = method.solve(seq, target_points=[])
                elapsed_ms = (perf_counter() - started) * 1000.0
                endpoint_residuals, max_endpoint_residual = _endpoint_residual_summary(seq, result)
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (perf_counter() - started) * 1000.0
                summary_rows.append(
                    {
                        "sequence_code": sequence_code,
                        "sequence": seq.name,
                        "method": method.name,
                        "solve_status": f"failed:{type(exc).__name__}",
                        "strain_energy": "",
                        "left_derivative": "",
                        "left_second_derivative": "",
                        "z0_value": "",
                        "zminus1_value": "",
                        "max_endpoint_residual": "",
                        "endpoint_residuals_json": "",
                        "elapsed_ms": elapsed_ms,
                        "metadata_json": "",
                        "error_message": str(exc),
                    }
                )
                continue

            rows = _value_rows_for_result(sequence_code, seq, method, result)
            value_rows.extend(rows)
            z0_rows = [row for row in rows if row["z"] == 0.0 and row["status"] == "ok"]
            zminus1_rows = [row for row in rows if row["z"] == -1.0 and row["status"] == "ok"]
            summary_rows.append(
                {
                    "sequence_code": sequence_code,
                    "sequence": seq.name,
                    "method": method.name,
                    "solve_status": "ok",
                    "strain_energy": result.strain_energy,
                    "left_derivative": result.optimal_params.get("left_derivative", ""),
                    "left_second_derivative": result.optimal_params.get("left_second_derivative", ""),
                    "z0_value": z0_rows[0]["value"] if z0_rows else "",
                    "zminus1_value": zminus1_rows[0]["value"] if zminus1_rows else "",
                    "max_endpoint_residual": max_endpoint_residual,
                    "endpoint_residuals_json": endpoint_residuals,
                    "elapsed_ms": elapsed_ms,
                    "metadata_json": _json(result.metadata),
                    "error_message": "",
                }
            )

    return summary_rows, value_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, float | int | str]],
    value_rows: list[dict[str, float | int | str]],
) -> None:
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "extension_points": EXTENSION_POINTS,
                "sequences": [seq.name for _, seq in _sequence_suite()],
                "methods": [method.name for method in _method_suite()],
                "notes": "Type A z=0 and z=-1 use sequence-specific limiting values when available.",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    _write_csv(
        output_dir / "extension_summary.csv",
        summary_rows,
        [
            "sequence_code",
            "sequence",
            "method",
            "solve_status",
            "strain_energy",
            "left_derivative",
            "left_second_derivative",
            "z0_value",
            "zminus1_value",
            "max_endpoint_residual",
            "endpoint_residuals_json",
            "elapsed_ms",
            "metadata_json",
            "error_message",
        ],
    )
    _write_csv(
        output_dir / "extension_values.csv",
        value_rows,
        [
            "sequence_code",
            "sequence",
            "method",
            "z",
            "region",
            "value",
            "truth",
            "abs_error",
            "recurrence_residual",
            "recurrence_status",
            "status",
            "error_message",
        ],
    )

    for sequence_code in ["A", "B", "C"]:
        series: dict[str, tuple[list[float], list[float]]] = {}
        for method_name in ["hermite_cubic", "hermite_quintic", "chebyshev"]:
            rows = [
                row for row in value_rows
                if row["sequence_code"] == sequence_code
                and row["method"] == method_name
                and row["status"] == "ok"
                and row["value"] != ""
            ]
            if not rows:
                continue
            series[method_name] = (
                [float(row["z"]) for row in rows],
                [float(row["value"]) for row in rows],
            )
        if series:
            plot_series(
                series,
                title=f"Extension Values for Sequence {sequence_code}",
                x_label="z",
                y_label="f(z)",
                output_path=output_dir / "figures" / f"extension_values_{sequence_code}.png",
            )

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("Extension experiment completed for sequence types A/B/C.\n")
        file.write("A=variable-base tetration, B=fixed-base tetration, C=factorial type.\n")
        type_a_rows = [
            row for row in summary_rows
            if row["sequence_code"] == "A" and row["solve_status"] == "ok"
        ]
        if type_a_rows:
            file.write("Type A limiting values:\n")
            for row in type_a_rows:
                file.write(
                    f"  {row['method']}: z=0 -> {row['z0_value']}, "
                    f"z=-1 -> {row['zminus1_value']}\n"
                )
        failed_values = [row for row in value_rows if row["status"] != "ok"]
        file.write(f"Failed point evaluations: {len(failed_values)}\n")
        if failed_values:
            for row in failed_values[:20]:
                file.write(
                    f"  {row['sequence_code']}/{row['method']} z={row['z']}: "
                    f"{row['status']} {row['error_message']}\n"
                )


def main() -> None:
    output_dir = _build_output_dir()
    summary_rows, value_rows = _run_extension()
    _write_outputs(output_dir, summary_rows, value_rows)
    print(f"Wrote extension results to {output_dir}")


if __name__ == "__main__":
    main()
