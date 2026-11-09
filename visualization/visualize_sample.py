#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize a ReCalib image/JSON sample, with optional ETH-XGaze inference."""

import argparse
import sys
from pathlib import Path

from read_gaze_data import Read_gaze_data
from sample_overlay_plots import overlay_visualization
from utils import checkIfisAValidPNGPair


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_IMAGE = REPO_ROOT / "example" / "07_00_02_img-0039.png"
DEFAULT_SETUP_CONFIG = REPO_ROOT / "docs" / "setup.json"
DEFAULT_SHAPE_PREDICTOR = (
    REPO_ROOT / "evaluation" / "modules" / "shape_predictor_68_face_landmarks.dat"
)
DEFAULT_FACE_MODEL = Path(__file__).resolve().parent / "face_model.txt"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize a ReCalib PNG/JPG sample and its paired JSON annotation."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=DEFAULT_TEST_IMAGE,
        help=f"Input image (default: {DEFAULT_TEST_IMAGE}).",
    )
    parser.add_argument(
        "--setup-config",
        type=Path,
        default=DEFAULT_SETUP_CONFIG,
        help=f"Acquisition setup JSON (default: {DEFAULT_SETUP_CONFIG}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional ETH-XGaze checkpoint. Inference is disabled when omitted.",
    )
    parser.add_argument(
        "--shape-predictor",
        type=Path,
        default=DEFAULT_SHAPE_PREDICTOR,
        help="dlib 68-landmark model used with --checkpoint (image-only landmarks).",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=DEFAULT_FACE_MODEL,
        help="3D face model used with --checkpoint.",
    )
    parser.add_argument(
        "--camera-intrinsics",
        type=Path,
        default=None,
        help="Camera intrinsics NPZ used with --checkpoint (default: docs/camera_intrinsics.npz).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for optional inference: auto, cpu, cuda, etc.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Write the image overlay, 2D screen view and 3D view to this folder instead of displaying them.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Validate and reconstruct the sample without opening interactive windows.",
    )
    return parser


def existing_path(parser, path, label):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        parser.error(f"{label} not found: {resolved}")
    return resolved


def create_estimator(args, parser):
    checkpoint = existing_path(parser, args.checkpoint, "checkpoint")
    shape_predictor = existing_path(parser, args.shape_predictor, "shape predictor")
    face_model = existing_path(parser, args.face_model, "face model")
    camera_intrinsics = None
    if args.camera_intrinsics is not None:
        camera_intrinsics = existing_path(
            parser, args.camera_intrinsics, "camera intrinsics"
        )

    evaluation_dir = REPO_ROOT / "evaluation"
    sys.path.insert(0, str(evaluation_dir))
    try:
        from eth_xGaze_inf import ETHXGazeEstimator
    except ModuleNotFoundError as exc:
        parser.error(
            "ETH-XGaze inference dependencies are missing "
            f"({exc.name}). Install the full requirements.txt file."
        )

    return ETHXGazeEstimator(
        shape_predictor_path=str(shape_predictor),
        face_model_path=str(face_model),
        ckpt_path=str(checkpoint),
        camera_npz_path=str(camera_intrinsics) if camera_intrinsics else None,
        camera_xml_path=None,
        device=args.device,
    )


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.image_path = args.image_path.expanduser().resolve()
    args.setup_config = existing_path(parser, args.setup_config, "setup config")

    is_valid, png_path, json_path = checkIfisAValidPNGPair(str(args.image_path))
    if not is_valid:
        parser.error(
            "input must be an existing .png/.jpg image with a same-named JSON file: "
            f"{args.image_path}"
        )
    args.png_path = Path(png_path)
    args.json_path = Path(json_path)
    return args, parser


def main(argv=None):
    args, parser = parse_args(argv)
    print(f"Processing: {args.png_path}")

    gaze_reader = Read_gaze_data(str(args.png_path), str(args.json_path))
    gaze_reader.loadSetupSpecs(str(args.setup_config))
    gaze_reader.sceneReconstruction()

    gaze_prediction = None
    if args.checkpoint is not None:
        estimator = create_estimator(args, parser)
        gaze_prediction = estimator.predict_gaze_vector(str(args.png_path))
        gaze_reader.addGazePrediction(gaze_prediction)

    if args.image_path == DEFAULT_TEST_IMAGE:
        print("Validated bundled sample.")
    else:
        print("Validated sample.")

    if args.save_dir is not None:
        out = args.save_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        overlay_visualization(str(args.png_path), str(args.json_path), pred_vector=gaze_prediction,
                              save_path=out / "sample_overlay_visualization.png")
        gaze_reader.plot2D(write=out / "2d_sample_visualization.png")
        gaze_reader.plot3D(save_path=out / "3d_sample_visualization.png")
        print(f"Views written to {out}")
        return

    if args.no_display:
        return

    print("Close each interactive view to continue to the next one.")
    overlay_visualization(
        str(args.png_path),
        str(args.json_path),
        pred_vector=gaze_prediction,
    )
    gaze_reader.plot3D()
    gaze_reader.plot2D()


if __name__ == "__main__":
    main()
