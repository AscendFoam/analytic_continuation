"""Validate the cubic baseline against Gamma for the factorial sequence."""

from __future__ import annotations

import argparse
from math import gamma
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analytic_continuation.methods.chebyshev import ChebyshevMethod
from analytic_continuation.core.sequence import FactorialType
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autotune-chebyshev",
        action="store_true",
        help="Tune Chebyshev hyperparameters against Gamma before evaluation.",
    )
    parser.add_argument(
        "--autotune-points",
        default="0.5,1.5",
        help="Comma-separated validation points used by autotune.",
    )
    parser.add_argument(
        "--candidate-degrees",
        default="6,10,12,16,20,24",
        help="Comma-separated Chebyshev degrees considered by autotune.",
    )
    parser.add_argument(
        "--candidate-orders",
        default="3,4,5,6",
        help="Comma-separated constraint orders considered by autotune.",
    )
    parser.add_argument(
        "--candidate-regularizations",
        default="1e-8,1e-6",
        help="Comma-separated regularization values considered by autotune.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sequence = FactorialType()
    points = [0.5, 1.5, 2.5]
    chebyshev_method = ChebyshevMethod()
    tuning = None
    if args.autotune_chebyshev:
        tuning = ChebyshevMethod.autotune(
            sequence,
            validation_points=_parse_csv_floats(args.autotune_points),
            reference_fn=lambda z: gamma(z + 1.0),
            candidate_degrees=_parse_csv_ints(args.candidate_degrees),
            candidate_constraint_orders=_parse_csv_ints(args.candidate_orders),
            candidate_regularizations=_parse_csv_floats(args.candidate_regularizations),
        )
        chebyshev_method = tuning.build_method()

    methods = [
        HermiteCubicMethod(),
        HermiteQuinticMethod(),
        chebyshev_method,
    ]

    if tuning is not None:
        print(
            "[chebyshev_autotune] "
            f"degree={tuning.degree} "
            f"constraint_order={tuning.constraint_order} "
            f"regularization={tuning.regularization:.0e} "
            f"max_abs_error={tuning.max_abs_error:.3e} "
            f"schur_condition_number={tuning.schur_condition_number:.3e}"
        )
        print()

    for method in methods:
        result = method.solve(sequence, target_points=points)
        label = method.name
        if tuning is not None and method is chebyshev_method:
            label = f"{method.name}_autotuned"
        print(f"[{label}]")
        for point in points:
            estimate = result.eval_at[point]
            truth = gamma(point + 1.0)
            print(
                f"z={point:.1f} estimate={estimate:.10f} "
                f"gamma={truth:.10f} abs_error={abs(estimate - truth):.3e}"
            )
        print()


if __name__ == "__main__":
    main()
