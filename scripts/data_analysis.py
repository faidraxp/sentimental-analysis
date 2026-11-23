import re
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import json
import os
from collections import Counter
import csv

from pathlib import Path
from argparse import ArgumentParser

def parse_output(value):
    try:
        data = json.loads(value)
        emotion = data.get("emotion")
        return pd.Series({
            "reasoning": data.get("reasoning"),
            "pred": emotion if isinstance(emotion, str) and emotion.strip() else "INVALID"
        })
    except (json.JSONDecodeError, TypeError, AttributeError):
        return pd.Series({
            "reasoning": value,
            "pred": "INVALID"
        })
def parse_label(text):
    m = LEAD.match(text)
    return m.group(1).lower() if m else "INVALID"

def format_pattern(text):
    m = LEAD.match(text)
    if not m:
        return "no label"
    tail = m.group(2).strip()
    if not tail:
        return "<Label> only (as instructed)"
    if tail.startswith("What do you think"):
        return "<Label>. What do you think ...?"
    if tail.startswith("If you have any other"):
        return "<Label>. If you have any other ..."
    return "<Label>. <other>"
def metrics(df):
    return pd.Series({
        "n": len(df),
        "WA": 100 * accuracy_score(df.gold, df.pred),
        "UA": 100 * recall_score(df.gold, df.pred, labels=LABELS, average="macro", zero_division=0),
        "Macro-F1": 100 * f1_score(df.gold, df.pred, labels=LABELS, average="macro", zero_division=0),
        "Invalid %": 100 * (df.pred == "INVALID").mean(),
    })

LABELS = ["anger", "disgust", "fear", "happiness", "sadness"]

BASE_DIR = Path(__file__).resolve().parent.parent
parser = ArgumentParser(description="Evaluate and export AESDD predictions.")
parser.add_argument("input", nargs="?", type=Path)
input_path = parser.parse_args().input

for lab in LABELS if input_path is None else [None]:
    default_input = BASE_DIR / "results_qwen" / f"aesdd_qwen25omni_native_all_first_{lab}_same_order.jsonl"
    file_path = input_path.resolve() if input_path is not None else default_input
    if not file_path.is_file():
        if input_path is not None:
            parser.error(f"Input file not found: {file_path}")
        print(f"Skipping missing input: {file_path}")
        continue

    print("Reading:", file_path)

    res = pd.read_json(file_path, lines=True)


    res[["reasoning", "pred"]] = res["output"].apply(parse_output)

    print(res.head())
    # The prompt asks for one word, but the model answers "<Label>. <chat filler>".
    # Parse rule: the first word must be one of the labels; anything else is INVALID (counted as wrong).
    LEAD = re.compile(r"^\s*(" + "|".join(LABELS) + r")\b[.!]?\s*(.*)$", re.IGNORECASE | re.DOTALL)


    # res["pred"] = res["output"].map(parse_label)
    # res["pattern"] = res["output"].map(format_pattern)

    print(f"{len(res)} outputs\n")
    # print(res["pattern"].value_counts().to_string())
    print("\nOutputs with no label:")
    # print(res.loc[res.pred == "INVALID", ["file", "output"]].to_string(index=False))


    # display(pd.crosstab(res.gold, res.pred, rownames=["gold"], colnames=["pred"], margins=True))

    #res.to_csv(os.path.join(OUT_DIR, "aesdd_qwen25omni_native_parsed.csv"), index=False)
    # summary.to_csv(os.path.join(OUT_DIR, "aesdd_qwen25omni_native_metrics.csv"))



    summary = metrics(res).to_frame("Qwen2.5-Omni-7B native").T
    summary["n"] = summary["n"].astype(int)
    print("AESDD, all clips; chance = 20%")
    # display(summary.round(2))

    report = classification_report(
        res.gold, res.pred, labels=LABELS, output_dict=True, zero_division=0
    )
    if "accuracy" in report:
        report["accuracy"] = {"f1-score": report["accuracy"], "support": len(res)}
    report_table = pd.DataFrame.from_dict(report, orient="index")
    report_table["support"] = report_table["support"].astype(int)
    print(report_table.round(3).to_string())
    prediction_distribution = res.pred.value_counts().to_dict()
    print("Prediction distribution:", prediction_distribution)

    summary_columns = ["WA", "UA", "Macro-F1", "Invalid %"]
    report_table = report_table.astype(object).fillna("")
    report_table[summary_columns] = ""
    report_table.loc["summary", summary_columns] = summary.iloc[0][summary_columns]
    report_table = report_table.fillna("")
    predictions = res[["file", "gold", "pred", "reasoning", "output"]].rename(
        columns={
            "file": "clip_id",
            "gold": "true_label",
            "pred": "predicted_label",
            "output": "raw_response",
        }
    )

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    label_match = re.search(r"first_?(" + "|".join(LABELS) + r")(?:_|$)", file_path.stem)
    output_label = lab or (label_match.group(1) if label_match else file_path.stem)
    report_path = results_dir / f"classification_results_{output_label}.csv"
    report_table.to_csv(report_path, index_label="class", encoding="utf-8-sig")
    predictions.to_csv(
        results_dir / f"predictions_{output_label}.csv", index=False, encoding="utf-8-sig"
    )
    print("Saved evaluation to:", report_path)
for lab in LABELS if input_path is None else [None]:
    default_input = BASE_DIR / "results_qwen" / f"aesdd_qwen25omni_native_all_first{lab}.jsonl"
    file_path = input_path.resolve() if input_path is not None else default_input
    if not file_path.is_file():
        if input_path is not None:
            parser.error(f"Input file not found: {file_path}")
        print(f"Skipping missing input: {file_path}")
        continue

    print("Reading:", file_path)

    res = pd.read_json(file_path, lines=True)


    res[["reasoning", "pred"]] = res["output"].apply(parse_output)

    print(res.head())
    # The prompt asks for one word, but the model answers "<Label>. <chat filler>".
    # Parse rule: the first word must be one of the labels; anything else is INVALID (counted as wrong).
    LEAD = re.compile(r"^\s*(" + "|".join(LABELS) + r")\b[.!]?\s*(.*)$", re.IGNORECASE | re.DOTALL)


    # res["pred"] = res["output"].map(parse_label)
    # res["pattern"] = res["output"].map(format_pattern)

    print(f"{len(res)} outputs\n")
    # print(res["pattern"].value_counts().to_string())
    print("\nOutputs with no label:")
    # print(res.loc[res.pred == "INVALID", ["file", "output"]].to_string(index=False))


    # display(pd.crosstab(res.gold, res.pred, rownames=["gold"], colnames=["pred"], margins=True))

    #res.to_csv(os.path.join(OUT_DIR, "aesdd_qwen25omni_native_parsed.csv"), index=False)
    # summary.to_csv(os.path.join(OUT_DIR, "aesdd_qwen25omni_native_metrics.csv"))



    summary = metrics(res).to_frame("Qwen2.5-Omni-7B native").T
    summary["n"] = summary["n"].astype(int)
    print("AESDD, all clips; chance = 20%")
    # display(summary.round(2))

    report = classification_report(
        res.gold, res.pred, labels=LABELS, output_dict=True, zero_division=0
    )
    if "accuracy" in report:
        report["accuracy"] = {"f1-score": report["accuracy"], "support": len(res)}
    report_table = pd.DataFrame.from_dict(report, orient="index")
    report_table["support"] = report_table["support"].astype(int)
    print(report_table.round(3).to_string())
    prediction_distribution = res.pred.value_counts().to_dict()
    print("Prediction distribution:", prediction_distribution)

    summary_columns = ["WA", "UA", "Macro-F1", "Invalid %"]
    report_table = report_table.astype(object).fillna("")
    report_table[summary_columns] = ""
    report_table.loc["summary", summary_columns] = summary.iloc[0][summary_columns]
    report_table = report_table.fillna("")
    predictions = res[["file", "gold", "pred", "reasoning", "output"]].rename(
        columns={
            "file": "clip_id",
            "gold": "true_label",
            "pred": "predicted_label",
            "output": "raw_response",
        }
    )

    results_dir = BASE_DIR / "results-first_run"
    results_dir.mkdir(parents=True, exist_ok=True)
    label_match = re.search(r"first_?(" + "|".join(LABELS) + r")(?:_|$)", file_path.stem)
    output_label = lab or (label_match.group(1) if label_match else file_path.stem)
    report_path = results_dir / f"classification_results_{output_label}.csv"
    report_table.to_csv(report_path, index_label="class", encoding="utf-8-sig")
    predictions.to_csv(
        results_dir / f"predictions_{output_label}.csv", index=False, encoding="utf-8-sig"
    )
    print("Saved evaluation to:", report_path)