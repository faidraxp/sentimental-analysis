import csv
from pathlib import Path


# RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# results-first_run
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results-first_run"
REPORT_PREFIX = "classification_results_"

def main():
    report_paths = sorted(RESULTS_DIR.glob(f"{REPORT_PREFIX}*.csv"))
    if not report_paths:
        raise SystemExit(f"No classification reports found in {RESULTS_DIR}")

    rows = []
    for report_path in report_paths:
        with report_path.open(encoding="utf-8-sig", newline="") as report_file:
            report = {row["class"]: row for row in csv.DictReader(report_file)}

        summary = report["summary"]
        macro = report["macro avg"]
        rows.append([
            report_path.stem.removeprefix(REPORT_PREFIX),
            f"{float(summary['WA']):.2f}",
            f"{100 * float(macro['precision']):.2f}",
            f"{float(summary['UA']):.2f}",
            f"{float(summary['Macro-F1']):.2f}",
            f"{100 * float(report['weighted avg']['f1-score']):.2f}",
            f"{float(summary['Invalid %']):.2f}",
            int(float(macro["support"])),
        ])

    output_path = RESULTS_DIR / "compare_results.csv"

    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "emotion", "accuracy_pct", "macro_precision_pct", "macro_recall_pct",
            "macro_f1_pct", "weighted_f1_pct", "invalid_pct", "clips",
        ])
        writer.writerows(rows)
    print(f"Saved {len(rows)} reports to {output_path}")


if __name__ == "__main__":
    main()
