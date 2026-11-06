"""Master runner: simulate → train/predict → control → plots."""
from __future__ import annotations
import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

def run(script):
    print(f"\n######## Running {script} ########", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=ROOT, check=False)
    print(f"######## {script} exit={r.returncode} in {time.time()-t0:.1f}s ########", flush=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

if __name__ == "__main__":
    run("simulate.py")
    run("train_predict.py")
    run("control_compare.py")
    run("plot_figures.py")
    print("ALL DONE")
