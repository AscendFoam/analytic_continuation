"""Baseline experiment for the currently implemented methods."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analytic_continuation.core.sequence import FactorialType, FixedBaseTetration, VariableBaseTetration
from analytic_continuation.methods.chebyshev import ChebyshevMethod
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "exp_baseline" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    target_points = [0.5, 1.5, 2.5, 3.5]
    sequences = [
        VariableBaseTetration(),
        FixedBaseTetration(base=1.3),
        FactorialType(),
    ]
    methods = [
        HermiteCubicMethod(),
        HermiteQuinticMethod(),
        ChebyshevMethod(),
    ]

    rows: list[dict[str, float | str]] = []
    for seq in sequences:
        for method in methods:
            result = method.solve(seq, target_points=target_points)
            for point, value in result.eval_at.items():
                rows.append(
                    {
                        "sequence": seq.name,
                        "method": method.name,
                        "z": point,
                        "value": value,
                        "strain_energy": result.strain_energy,
                        "optimal_params": json.dumps(result.optimal_params, ensure_ascii=False, sort_keys=True),
                    }
                )

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "target_points": target_points,
                "methods": [method.name for method in methods],
                "sequences": [seq.name for seq in sequences],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sequence", "method", "z", "value", "strain_energy", "optimal_params"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("Baseline experiment completed with the cubic Hermite method.\n")
        file.write(f"Rows written: {len(rows)}\n")


if __name__ == "__main__":
    main()
