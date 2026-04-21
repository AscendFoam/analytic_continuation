"""Scan the M5 regularized residual-minimization prototype."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from math import gamma
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
from analytic_continuation.methods.regularized_iter import RegularizedIterationMethod  # noqa: E402


EVAL_POINTS = [0.5, 1.5, 2.5]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degrees",
        default="6,8,10,12",
        help="Comma-separated Chebyshev-Lobatto degrees.",
    )
    parser.add_argument(
        "--constraint-orders",
        default="4",
        help="Comma-separated endpoint derivative residual orders.",
    )
    parser.add_argument(
        "--lambda-energies",
        default="1",
        help="Comma-separated curvature-energy weights.",
    )
    parser.add_argument(
        "--lambda-residuals",
        default="0.1,1,10,100",
        help="Comma-separated endpoint-residual weights.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Maximum optimizer iterations per configuration.",
    )
    parser.add_argument(
        "--initial-method",
        default="hermite_quintic",
        choices=["linear", "hermite_cubic", "hermite_quintic"],
        help="Initial curve used for the M5 optimizer.",
    )
    parser.add_argument(
        "--optimizer-backend",
        default="least_squares",
        choices=["least_squares", "minimize"],
        help="Optimization backend used by RegularizedIterationMethod.",
    )
    parser.add_argument(
        "--optimizer-method",
        default="trf",
        help="Backend-specific optimizer method, e.g. trf for least_squares or L-BFGS-B for minimize.",
    )
    parser.add_argument(
        "--residual-scale-strategy",
        default="order_weighted",
        choices=["absolute", "relative", "order_weighted"],
        help="Scaling applied to endpoint derivative residuals.",
    )
    parser.add_argument(
        "--residual-order-weight",
        type=float,
        default=2.0,
        help="Power used by order_weighted residual scaling.",
    )
    return parser


def _build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "exp_regularized_iter" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def _sequence_suite() -> list[tuple[str, RecurrenceSequence]]:
    return [
        ("A", VariableBaseTetration()),
        ("B", FixedBaseTetration(base=1.3)),
        ("C", FactorialType()),
    ]


def _run_scan(
    degrees: list[int],
    constraint_orders: list[int],
    lambda_energies: list[float],
    lambda_residuals: list[float],
    *,
    maxiter: int,
    initial_method: str,
    optimizer_backend: str,
    optimizer_method: str,
    residual_scale_strategy: str,
    residual_order_weight: float,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    summary_rows: list[dict[str, float | int | str]] = []
    value_rows: list[dict[str, float | int | str]] = []

    for sequence_code, seq in _sequence_suite():
        for degree in degrees:
            for constraint_order in constraint_orders:
                if constraint_order > degree:
                    continue
                for lambda_energy in lambda_energies:
                    for lambda_residual in lambda_residuals:
                        run_id = (
                            f"{sequence_code}_deg{degree}_ord{constraint_order}_"
                            f"le{lambda_energy:g}_lr{lambda_residual:g}"
                        )
                        method = RegularizedIterationMethod(
                            degree=degree,
                            constraint_order=constraint_order,
                            lambda_energy=lambda_energy,
                            lambda_residual=lambda_residual,
                            maxiter=maxiter,
                            initial_method=initial_method,
                            optimizer_backend=optimizer_backend,
                            optimizer_method=optimizer_method,
                            residual_scale_strategy=residual_scale_strategy,
                            residual_order_weight=residual_order_weight,
                        )
                        started = perf_counter()
                        try:
                            result = method.solve(seq, target_points=EVAL_POINTS)
                            elapsed_ms = (perf_counter() - started) * 1000.0
                        except Exception as exc:  # noqa: BLE001
                            elapsed_ms = (perf_counter() - started) * 1000.0
                            summary_rows.append(
                                {
                                    "run_id": run_id,
                                    "sequence_code": sequence_code,
                                    "sequence": seq.name,
                                    "status": f"failed:{type(exc).__name__}",
                                    "degree": degree,
                                    "constraint_order": constraint_order,
                                    "lambda_energy": lambda_energy,
                                    "lambda_residual": lambda_residual,
                                    "strain_energy": "",
                                    "objective_value": "",
                                    "residual_penalty": "",
                                    "endpoint_residual_norm": "",
                                    "scaled_endpoint_residual_norm": "",
                                    "max_abs_error": "",
                                    "mean_abs_error": "",
                                    "optimizer_success": "",
                                    "optimizer_iterations": "",
                                    "elapsed_ms": elapsed_ms,
                                    "optimal_params_json": "",
                                    "endpoint_residuals_json": "",
                                    "metadata_json": "",
                                    "error_message": str(exc),
                                }
                            )
                            continue

                        errors: list[float] = []
                        for point in EVAL_POINTS:
                            value = result.eval_at[point]
                            truth: float | str = ""
                            abs_err: float | str = ""
                            if isinstance(seq, FactorialType):
                                truth = gamma(point + 1.0)
                                abs_err = absolute_error(value, truth)
                                errors.append(abs_err)
                            value_rows.append(
                                {
                                    "run_id": run_id,
                                    "sequence_code": sequence_code,
                                    "sequence": seq.name,
                                    "degree": degree,
                                    "constraint_order": constraint_order,
                                    "lambda_energy": lambda_energy,
                                    "lambda_residual": lambda_residual,
                                    "z": point,
                                    "value": value,
                                    "truth": truth,
                                    "abs_error": abs_err,
                                    "status": "ok",
                                }
                            )

                        metadata = result.metadata
                        summary_rows.append(
                            {
                                "run_id": run_id,
                                "sequence_code": sequence_code,
                                "sequence": seq.name,
                                "status": "ok",
                                "degree": degree,
                                "constraint_order": constraint_order,
                                "lambda_energy": lambda_energy,
                                "lambda_residual": lambda_residual,
                                "strain_energy": result.strain_energy,
                                "objective_value": metadata.get("objective_value", ""),
                                "residual_penalty": metadata.get("residual_penalty", ""),
                                "endpoint_residual_norm": metadata.get("endpoint_residual_norm", ""),
                                "scaled_endpoint_residual_norm": metadata.get("scaled_endpoint_residual_norm", ""),
                                "max_abs_error": max(errors) if errors else "",
                                "mean_abs_error": sum(errors) / len(errors) if errors else "",
                                "optimizer_success": metadata.get("optimizer_success", ""),
                                "optimizer_iterations": metadata.get("optimizer_iterations", ""),
                                "elapsed_ms": elapsed_ms,
                                "optimal_params_json": _json(result.optimal_params),
                                "endpoint_residuals_json": _json(metadata.get("endpoint_residuals", {})),
                                "scaled_endpoint_residuals_json": _json(metadata.get("scaled_endpoint_residuals", {})),
                                "endpoint_residual_scales_json": _json(metadata.get("endpoint_residual_scales", {})),
                                "metadata_json": _json(metadata),
                                "error_message": "",
                            }
                        )

    return summary_rows, value_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_dir: Path,
    summary_rows: list[dict[str, float | int | str]],
    degrees: list[int],
) -> None:
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    factorial_rows = [row for row in ok_rows if row["sequence_code"] == "C" and row["max_abs_error"] != ""]
    if factorial_rows:
        error_by_degree: dict[str, tuple[list[float], list[float]]] = {}
        residual_by_degree: dict[str, tuple[list[float], list[float]]] = {}
        for degree in degrees:
            rows = [row for row in factorial_rows if row["degree"] == degree]
            rows.sort(key=lambda row: float(row["lambda_residual"]))
            if rows:
                error_by_degree[f"degree={degree}"] = (
                    [float(row["lambda_residual"]) for row in rows],
                    [float(row["max_abs_error"]) for row in rows],
                )
                residual_by_degree[f"degree={degree}"] = (
                    [float(row["lambda_residual"]) for row in rows],
                    [float(row["scaled_endpoint_residual_norm"]) for row in rows],
                )

        plot_series(
            error_by_degree,
            title="M5 Factorial Gamma Error Scan",
            x_label="lambda_residual",
            y_label="max abs error on z=0.5,1.5,2.5",
            output_path=output_dir / "figures" / "factorial_error_vs_lambda_residual.png",
            logx=True,
            logy=True,
        )
        plot_series(
            residual_by_degree,
            title="M5 Factorial Endpoint Residual Scan",
            x_label="lambda_residual",
            y_label="scaled endpoint residual norm",
            output_path=output_dir / "figures" / "factorial_endpoint_residual_vs_lambda_residual.png",
            logx=True,
            logy=True,
        )

    residual_by_sequence: dict[str, tuple[list[float], list[float]]] = {}
    for sequence_code in ["A", "B", "C"]:
        rows = [row for row in ok_rows if row["sequence_code"] == sequence_code]
        rows.sort(key=lambda row: (float(row["lambda_residual"]), int(row["degree"])))
        if rows:
            residual_by_sequence[sequence_code] = (
                [float(row["lambda_residual"]) for row in rows],
                [float(row["scaled_endpoint_residual_norm"]) for row in rows],
            )
    if residual_by_sequence:
        plot_series(
            residual_by_sequence,
            title="M5 Endpoint Residual by Sequence",
            x_label="lambda_residual",
            y_label="scaled endpoint residual norm",
            output_path=output_dir / "figures" / "endpoint_residual_by_sequence.png",
            logx=True,
            logy=True,
        )


def _write_summary(output_dir: Path, summary_rows: list[dict[str, float | int | str]]) -> None:
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    factorial_rows = [row for row in ok_rows if row["sequence_code"] == "C" and row["max_abs_error"] != ""]

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("M5 regularized iteration scan completed.\n")
        if factorial_rows:
            best_error = min(factorial_rows, key=lambda row: float(row["max_abs_error"]))
            best_residual = min(factorial_rows, key=lambda row: float(row["scaled_endpoint_residual_norm"]))
            file.write(
                "Best FactorialType gamma error: "
                f"run_id={best_error['run_id']}, "
                f"max_abs_error={float(best_error['max_abs_error']):.6e}, "
                f"endpoint_residual_norm={float(best_error['endpoint_residual_norm']):.6e}, "
                f"scaled_endpoint_residual_norm={float(best_error['scaled_endpoint_residual_norm']):.6e}, "
                f"energy={float(best_error['strain_energy']):.6e}\n"
            )
            file.write(
                "Best FactorialType endpoint residual: "
                f"run_id={best_residual['run_id']}, "
                f"endpoint_residual_norm={float(best_residual['endpoint_residual_norm']):.6e}, "
                f"scaled_endpoint_residual_norm={float(best_residual['scaled_endpoint_residual_norm']):.6e}, "
                f"max_abs_error={float(best_residual['max_abs_error']):.6e}, "
                f"energy={float(best_residual['strain_energy']):.6e}\n"
            )

        for sequence_code in ["A", "B", "C"]:
            rows = [row for row in ok_rows if row["sequence_code"] == sequence_code]
            if not rows:
                continue
            best = min(rows, key=lambda row: float(row["scaled_endpoint_residual_norm"]))
            file.write(
                f"Best endpoint residual for {sequence_code}: "
                f"run_id={best['run_id']}, "
                f"endpoint_residual_norm={float(best['endpoint_residual_norm']):.6e}, "
                f"scaled_endpoint_residual_norm={float(best['scaled_endpoint_residual_norm']):.6e}, "
                f"energy={float(best['strain_energy']):.6e}\n"
            )

        failed_rows = [row for row in summary_rows if row["status"] != "ok"]
        file.write(f"Failed configurations: {len(failed_rows)}\n")
        for row in failed_rows[:20]:
            file.write(f"  {row['run_id']}: {row['status']} {row['error_message']}\n")

        nonconverged_rows = [
            row for row in ok_rows
            if str(row["optimizer_success"]) != "True"
        ]
        file.write(f"Non-converged optimizer configurations: {len(nonconverged_rows)}\n")
        for row in nonconverged_rows[:20]:
            file.write(
                f"  {row['run_id']}: iterations={row['optimizer_iterations']}, "
                f"objective={row['objective_value']}\n"
            )


def _write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, float | int | str]],
    value_rows: list[dict[str, float | int | str]],
    *,
    degrees: list[int],
    constraint_orders: list[int],
    lambda_energies: list[float],
    lambda_residuals: list[float],
    maxiter: int,
    initial_method: str,
    optimizer_backend: str,
    optimizer_method: str,
    residual_scale_strategy: str,
    residual_order_weight: float,
) -> None:
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "eval_points": EVAL_POINTS,
                "degrees": degrees,
                "constraint_orders": constraint_orders,
                "lambda_energies": lambda_energies,
                "lambda_residuals": lambda_residuals,
                "maxiter": maxiter,
                "initial_method": initial_method,
                "optimizer_backend": optimizer_backend,
                "optimizer_method": optimizer_method,
                "residual_scale_strategy": residual_scale_strategy,
                "residual_order_weight": residual_order_weight,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    _write_csv(
        output_dir / "m5_summary.csv",
        summary_rows,
        [
            "run_id",
            "sequence_code",
            "sequence",
            "status",
            "degree",
            "constraint_order",
            "lambda_energy",
            "lambda_residual",
            "strain_energy",
            "objective_value",
            "residual_penalty",
            "endpoint_residual_norm",
            "scaled_endpoint_residual_norm",
            "max_abs_error",
            "mean_abs_error",
            "optimizer_success",
            "optimizer_iterations",
            "elapsed_ms",
            "optimal_params_json",
            "endpoint_residuals_json",
            "scaled_endpoint_residuals_json",
            "endpoint_residual_scales_json",
            "metadata_json",
            "error_message",
        ],
    )
    _write_csv(
        output_dir / "m5_values.csv",
        value_rows,
        [
            "run_id",
            "sequence_code",
            "sequence",
            "degree",
            "constraint_order",
            "lambda_energy",
            "lambda_residual",
            "z",
            "value",
            "truth",
            "abs_error",
            "status",
        ],
    )
    _write_plots(output_dir, summary_rows, degrees)
    _write_summary(output_dir, summary_rows)


def main() -> None:
    args = _build_parser().parse_args()
    degrees = _parse_csv_ints(args.degrees)
    constraint_orders = _parse_csv_ints(args.constraint_orders)
    lambda_energies = _parse_csv_floats(args.lambda_energies)
    lambda_residuals = _parse_csv_floats(args.lambda_residuals)

    output_dir = _build_output_dir()
    summary_rows, value_rows = _run_scan(
        degrees,
        constraint_orders,
        lambda_energies,
        lambda_residuals,
        maxiter=args.maxiter,
        initial_method=args.initial_method,
        optimizer_backend=args.optimizer_backend,
        optimizer_method=args.optimizer_method,
        residual_scale_strategy=args.residual_scale_strategy,
        residual_order_weight=args.residual_order_weight,
    )
    _write_outputs(
        output_dir,
        summary_rows,
        value_rows,
        degrees=degrees,
        constraint_orders=constraint_orders,
        lambda_energies=lambda_energies,
        lambda_residuals=lambda_residuals,
        maxiter=args.maxiter,
        initial_method=args.initial_method,
        optimizer_backend=args.optimizer_backend,
        optimizer_method=args.optimizer_method,
        residual_scale_strategy=args.residual_scale_strategy,
        residual_order_weight=args.residual_order_weight,
    )
    print(f"Wrote M5 regularized iteration results to {output_dir}")


if __name__ == "__main__":
    main()
