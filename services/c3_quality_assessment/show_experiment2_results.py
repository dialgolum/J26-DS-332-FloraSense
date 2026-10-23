"""Display saved Experiment 2 metrics without training, inference or file writes."""

import argparse
import json
from pathlib import Path


DEFAULT_RESULTS = (
    Path(__file__).resolve().parents[2]
    / "models/checkpoints/c3_features/experiment2_v2"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = []
    try:
        for task in ("root", "leaf"):
            for variant in ("A", "B"):
                path = args.results_dir / variant / task / "evaluation.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                metrics = report["oof_metrics"]
                matrix = metrics["confusion_matrix"]
                if (len(matrix) != 2 or any(len(row) != 2 for row in matrix)
                        or any(type(value) is not int or value < 0
                               for row in matrix for value in row)):
                    raise ValueError(f"Invalid two-class confusion matrix: {path}")
                total = sum(sum(row) for row in matrix)
                if total == 0:
                    raise ValueError(f"Empty confusion matrix: {path}")
                correct = matrix[0][0] + matrix[1][1]
                rows.append((f"{task.title()} {variant}", total, correct,
                             correct / total, metrics))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Could not read saved results: {exc}\n")

    print(f"Saved results: {args.results_dir.resolve()}")
    print("Pooled out-of-fold evaluation; no training or inference is performed.\n")
    print(f"{'Model':<9} {'Correct/Total':>14} {'Accuracy':>10} {'Macro F1':>10} "
          f"{'Balanced Acc':>13} {'Minority Prec':>14} {'Minority Recall':>16}")

    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"

    for name, total, correct, accuracy, metrics in rows:
        print(f"{name:<9} {f'{correct}/{total}':>14} {percent(accuracy):>10} "
              f"{percent(metrics['macro_f1']):>10} "
              f"{percent(metrics['balanced_accuracy']):>13} "
              f"{percent(metrics['minority_precision']):>14} "
              f"{percent(metrics['minority_recall']):>16}")
    print("\nAccuracy = correct predictions / total predictions.")
    print("Minority classes: root = Not Acceptable; leaf = One Side.")
    print("These scores evaluate saved outer-fold predictions, not a final deployment model.")


if __name__ == "__main__":
    main()
