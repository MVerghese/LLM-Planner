import os
import sys

import numpy as np
from PIL import Image

from run_eval_real_env import VLM_Functions


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def main():
    frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # frame_1/2/3 live in LLM-Planner/e2e/
    e2e_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    f1 = load(os.path.join(e2e_dir, "frame_1.png"))
    f2 = load(os.path.join(e2e_dir, "frame_2.png"))
    f3 = load(os.path.join(e2e_dir, "frame_3.png"))

    # (before, after, action, expected, label)
    cases = [
        (f1, f2, "PickupObject Lemon",      True,  "1->2 pickup lemon (correct)"),
        (f1, f2, "PutObject Lemon Drawer",  False, "1->2 put lemon in drawer (wrong action)"),
        (f2, f3, "PutObject Lemon Drawer",  True,  "2->3 put lemon in drawer (correct)"),
        (f2, f3, "PickupObject Lemon",      False, "2->3 pickup lemon (wrong action)"),
    ]

    vlm = VLM_Functions(client=None, engine="claude-sonnet-4-6")

    results = []
    for before, after, action, expected, label in cases:
        predicted = vlm.get_success_pair(before, after, action)
        ok = predicted == expected
        results.append((label, expected, predicted, ok))
        print(f"  {label}: expected={expected} predicted={predicted} match={ok}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, expected, predicted, ok in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}: expected={expected} predicted={predicted}")
    all_ok = all(r[3] for r in results)
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
