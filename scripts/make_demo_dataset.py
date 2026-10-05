#!/usr/bin/env python3
"""Generate a synthetic multi-view reference set for testing and demos.

This renders one of the built-in procedural subjects from several viewpoints and
writes reference images (plus optional masks and a ``ground_truth.json``) exactly
the way a user's photo set would arrive, so the whole pipeline can be exercised
without shipping photographs.

Examples
--------
::

    python scripts/make_demo_dataset.py --out ./demo/refs --views 9
    python scripts/make_demo_dataset.py --out ./demo/refs --subject creature --resolution 768
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon3d.engine.reconstruction.dataset import DatasetSpec, generate_reference_set  # noqa: E402

VIEW_PRESETS = {
    4: (("front", 0.0, 0.0), ("right", 90.0, 0.0), ("back", 180.0, 0.0), ("left", 270.0, 0.0)),
    6: (("front", 0.0, 0.0), ("front_right", 60.0, 0.0), ("back_right", 120.0, 0.0),
        ("back", 180.0, 0.0), ("back_left", 240.0, 0.0), ("front_left", 300.0, 0.0)),
    9: (("front", 0.0, 0.0), ("front_right", 45.0, 0.0), ("right", 90.0, 0.0),
        ("back_right", 135.0, 0.0), ("back", 180.0, 0.0), ("back_left", 225.0, 0.0),
        ("left", 270.0, 0.0), ("front_left", 315.0, 0.0), ("top", 0.0, 80.0)),
    12: (("front", 0.0, 0.0), ("front_right", 30.0, 0.0), ("right_front", 60.0, 0.0),
         ("right", 90.0, 0.0), ("right_back", 120.0, 0.0), ("back_right", 150.0, 0.0),
         ("back", 180.0, 0.0), ("back_left", 210.0, 0.0), ("left_back", 240.0, 0.0),
         ("left", 270.0, 0.0), ("left_front", 300.0, 0.0), ("front_left", 330.0, 0.0)),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="directory for the images")
    parser.add_argument("--subject", default="robot",
                        choices=["robot", "creature", "prop", "vehicle", "humanoid"])
    parser.add_argument("--views", type=int, default=9, choices=sorted(VIEW_PRESETS))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fov", type=float, default=38.0, help="ground-truth lens (degrees)")
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma (0-1)")
    parser.add_argument("--jpeg-quality", type=int, default=None,
                       help="write JPEG at this quality instead of PNG")
    parser.add_argument("--masks", action="store_true", help="also write reference masks")
    parser.add_argument("--seed-free", action="store_true",
                       help="omit ground_truth.json (use when you do not want the answer key)")
    args = parser.parse_args(argv)

    spec = DatasetSpec(
        kind=args.subject,
        views=VIEW_PRESETS[args.views],
        resolution=args.resolution,
        fov_deg=args.fov,
        distance=args.distance,
        noise=args.noise,
        jpeg_quality=args.jpeg_quality,
        write_masks=args.masks,
    )
    metadata = generate_reference_set(args.out, spec)
    if args.seed_free:
        (args.out / "ground_truth.json").unlink(missing_ok=True)

    images = sorted(p for p in args.out.iterdir() if p.suffix.lower() in {".png", ".jpg"})
    summary = {
        "output": str(args.out.resolve()),
        "images": [p.name for p in images],
        "views": args.views,
        "resolution": args.resolution,
        "subject": args.subject,
        "ground_truth": str(args.out / "ground_truth.json") if not args.seed_free else None,
        "subject_height": metadata["subject"]["height"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
