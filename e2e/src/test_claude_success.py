import os
import sys

import numpy as np
from PIL import Image

from run_eval_real_env import VLM_Functions


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cases = [
        ("../frame_2.png", True),   # drawer open -> OpenObject Drawer should be success
        ("../frame_3.png", False),         # drawer closed -> OpenObject Drawer should be failure
    ]

    vlm = VLM_Functions(client=None, engine="claude-sonnet-4-6")
    action = "pickupobject Lemon"
    prompt = "Is the robot currently holding the lemon? Answer with exactly one word: Yes or No."
    describe = "Describe the state of the Lemon in this image in one sentence."

    results = []
    for fname, expected in cases:
        img = np.array(Image.open(os.path.join(here, fname)).convert("RGB"))
        encoded = vlm._image_to_base64(img)
        desc = vlm.vlm.generate(describe, images=[encoded], temperature=0.0, n=1, max_tokens=120)[0]
        raw = vlm.vlm.generate(prompt, images=[encoded], temperature=0.0, n=1)[0]
        probs = vlm.vlm.yes_no_probabilities(prompt, images=[encoded])
        predicted = vlm.get_success(img, action)
        ok = predicted == expected
        results.append((fname, expected, predicted, ok))
        print(f"--- {fname} ---")
        print(f"  describe: {desc}")
        print(f"  raw     : {raw!r}")
        print(f"  probs   : Yes={probs[0]:.6f} No={probs[1]:.6f}")
        print(f"  expected={expected} predicted={predicted} match={ok}")

    all_ok = all(r[3] for r in results)
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
