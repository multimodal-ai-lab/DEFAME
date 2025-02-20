import os
import random
import shutil
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section
import pandas as pd

EXPERIMENT_DIRS = [
    "path/to/experiment/directory",
]

ONLY_SUCCESS_CASES = True

EXAMPLES_PER_EXPERIMENT = 50

IN_DIR = Path("out/")
OUT_DIR = Path("out/user_study/fc_docs")

idx = 0

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

for exp_dir in EXPERIMENT_DIRS:
    fc_dir = Path(exp_dir) / "fact-checks"

    if ONLY_SUCCESS_CASES:
        predictions_path = Path(exp_dir) / "predictions.csv"
        df = pd.read_csv(predictions_path)
        is_success = df["correct"]
        failure_ids = df[is_success]["sample_index"]
        fc_dirs = [fc_dir / f"{idx}" for idx in failure_ids]
    else:
        fc_dirs = os.listdir(fc_dir)

    # Pick samples randomly
    fact_checks = random.sample(fc_dirs, EXAMPLES_PER_EXPERIMENT)

    for fact_check in fact_checks:
        fc_doc = fc_dir / fact_check / "doc.md"

        # Copy the Markdown original
        target_dir = OUT_DIR / f"{idx}"
        shutil.copytree(fc_dir / fact_check, target_dir)

        # Convert the Markdown to PDF
        with fc_doc.open() as f:
            content = f.read()

        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(Section(content, toc=False, root=(fc_dir / fact_check).as_posix()))

        pdf.meta["title"] = "MInFact User Study"
        pdf.save(target_dir / "rendered.pdf")

        idx += 1
