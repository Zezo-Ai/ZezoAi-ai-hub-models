# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

from qai_hub_models.models.edgetam.app import EdgeTAMApp, EdgeTAMVideoApp
from qai_hub_models.models.edgetam.model import (
    DEFAULT_MODEL_TYPE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EdgeTAM,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

VIDEO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo.mp4"
)
DEFAULT_MAX_FRAMES = 60


def generate_frames_from_video(video_path: str) -> list[np.ndarray]:
    """
    Extract RGB frames from a video file.

    Parameters
    ----------
    video_path
      Path to the input video file.

    Returns
    -------
    list[np.ndarray]
      Video frames as RGB numpy arrays with shape ``(H, W, C)``.

    Notes
    -----
    Frame extraction stops early if system memory usage exceeds 90 percent.
    """
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if psutil.virtual_memory().percent > 90:
                print(
                    "Memory usage is too high (>90%). "
                    "Please reduce the video resolution or frame rate."
                )
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    cap.release()
    return frames


def generate_video_from_frames(
    frames: list[np.ndarray], output_path: str, fps: int = 30
) -> None:
    """
    Write RGB frames to a video file.

    Parameters
    ----------
    frames
      Frames to write, where each frame is an RGB numpy array.
    output_path
      Output filename written under the ``build`` directory.
    fps
      Output video frame rate. Defaults to 30.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter.fourcc(*"vp09")
    output_dir = os.path.join(Path.cwd(), "build")
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, os.path.basename(output_path))
    print(f"Saving video to {full_path}")
    video = cv2.VideoWriter(full_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


def main(is_test: bool = False, image_path: str | None = None) -> None:
    """
    Run the EdgeTAM video object tracking demo.

    This demo loads a video, parses one or more user-provided point prompts,
    initializes the EdgeTAM tracking pipeline, and tracks the selected object
    across video frames. In on-device mode, exported model components are loaded
    from Hub-backed demo arguments.

    Parameters
    ----------
    is_test
      If True, uses the default model type and limits processing to a few
      frames for automated testing. Defaults to False.
    image_path
      Optional path to a single image file. When provided together with
      ``is_test=True``, the demo exercises the image mode code path instead
      of the default video path. Has no effect when ``is_test`` is False
      (use ``--image`` on the command line instead).
    """
    parser = get_model_cli_parser(EdgeTAM)
    parser = get_on_device_demo_parser(parser)

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image file path for single-frame segmentation. "
        "When provided, uses EdgeTAMApp and saves a PNG output instead of a video.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file path. If not provided, uses the default demo video.",
    )
    parser.add_argument(
        "--point-coordinates",
        type=str,
        default="400,150;430,380",
        help="Comma-separated x,y coordinate. Multiple points separated by `;`."
        " e.g. `x1,y1;x2,y2`. Default: `400,150;430,380`.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "Maximum number of video frames to process. "
            f"Defaults to {DEFAULT_MAX_FRAMES}. Use a larger value or <= 0 to disable the limit."
        ),
    )
    args = parser.parse_args(["--model-type", DEFAULT_MODEL_TYPE] if is_test else None)
    if is_test and image_path is not None:
        args = parser.parse_args(
            ["--model-type", DEFAULT_MODEL_TYPE, "--image", image_path]
        )
    validate_on_device_demo_args(args, MODEL_ID)

    # Parse point coordinates
    coordinates: list[str] = list(filter(None, args.point_coordinates.split(";")))
    input_coords: list[list[int]] = []
    input_labels: list[int] = []
    for coord in coordinates:
        parts = coord.split(",")
        if len(parts) != 2:
            raise RuntimeError(f"Expected comma-separated x,y coordinate. Got: {parts}")
        input_coords.append([int(parts[0]), int(parts[1])])
        input_labels.append(1)

    point_coords = torch.tensor(input_coords, dtype=torch.float32)
    point_labels = torch.tensor(input_labels, dtype=torch.float32)

    image_mode = args.image is not None

    if image_mode:
        # Single-image mode: load image directly
        print(f"\n** Loading image: {args.image} **\n")
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise RuntimeError(
                f"Could not read image: {args.image}. "
                "Check that the file exists and is a valid image."
            )
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"Loaded image ({frame.shape[1]}x{frame.shape[0]})")
    else:
        # Video mode: load frames from video
        video_input = str(VIDEO_ADDRESS.fetch()) if not args.video else str(args.video)
        print(f"\n** Loading video: {video_input} **\n")
        frames = generate_frames_from_video(video_input)
        if len(frames) == 0:
            raise RuntimeError(
                f"No frames could be read from video: {video_input}. "
                "Check that the file exists and is a valid video."
            )
        print(
            f"Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})"
        )

        if args.max_frames > 0 and len(frames) > args.max_frames:
            print(
                f"Warning: video has {len(frames)} frames; truncating to {args.max_frames}. "
                "Pass --max-frames to change or disable this limit."
            )
            frames = frames[: args.max_frames]
        if is_test:
            frames = frames[:3]

    print("\n** Loading EdgeTAM model... **\n")
    model, (encoder, memory_encoder, video_decoder) = (
        demo_model_components_from_cli_args(EdgeTAM, MODEL_ID, args)
    )

    common_kwargs = dict(
        encoder=encoder,
        video_decoder=video_decoder,
        memory_encoder=memory_encoder,
        sam2=model.sam2,
        encoder_input_img_size=model.sam2.image_size,
    )
    common_kwargs["maskmem_pos_enc"] = model.memory_encoder.maskmem_pos_enc

    if image_mode:
        print("\n** Segmenting object in image... **\n")
        app_image = EdgeTAMApp(**common_kwargs)
        painted = app_image.predict(frame, point_coords, point_labels)
        if not is_test:
            output_dir = os.path.join(Path.cwd(), "build")
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "edgetam_segmentation_output.png")
            cv2.imwrite(out_path, cv2.cvtColor(painted, cv2.COLOR_RGB2BGR))
            print(f"Saved segmentation to {out_path}")
    else:
        print("\n** Tracking object across frames... **\n")
        app_video = EdgeTAMVideoApp(**common_kwargs)
        painted_frames = app_video.track(frames, point_coords, point_labels)
        if not is_test:
            generate_video_from_frames(
                painted_frames, output_path="edgetam_tracking_output.mp4", fps=30
            )


if __name__ == "__main__":
    main()
