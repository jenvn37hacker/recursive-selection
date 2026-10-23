"""Consistent display formatting; no changes to stored numeric values."""


def style_metrics(table):
    formats = {}
    for column in table.columns:
        if "(pp)" in column:
            formats[column] = "{:+.3f}"
        elif column in ("Samples", "Step", "Round", "Calls", "Features", "Sensors", "Groups"):
            formats[column] = "{:.0f}"
        elif column == "Log loss":
            formats[column] = "{:.4f}"
        elif any(word in column for word in ("Precision", "Recall", "F1")) or column in (
                "Accuracy", "Full", "Selected", "Absolute minimum", "Full minus allowed drop",
                "Minimum", "Validation minimum"):
            formats[column] = "{:.2%}"
        elif "(s)" in column or "(ms)" in column or "(ms/sample)" in column:
            formats[column] = "{:.4f}"
    return table.style.format(formats, na_rep="Not applicable")
