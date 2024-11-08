import os
import random
import shutil
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

EXPERIMENT_DIRS = [
    "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/InFact/out/verite/summary/no_develop/gpt_4o/2024-11-08_12-05",
]

EXAMPLES_PER_EXPERIMENT = 10

IN_DIR = Path("out/")
OUT_DIR = Path("out/user_study/fc_docs")

idx = 0

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

for exp_dir in EXPERIMENT_DIRS:
    # Pick samples randomly
    fc_dir = Path(exp_dir) / "fact-checks"
    all_fc_dirs = os.listdir(fc_dir)
    fact_checks = random.sample(all_fc_dirs, EXAMPLES_PER_EXPERIMENT)

    for fact_check in fact_checks:
        fc_doc = fc_dir / fact_check / "doc.md"
        with fc_doc.open() as f:
            content = f.read()

        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(Section(content, toc=False, root=(fc_dir / fact_check).as_posix()))

        pdf.meta["title"] = "MInFact User Study"
        pdf.save(OUT_DIR / f"{idx}.pdf")
        idx += 1
