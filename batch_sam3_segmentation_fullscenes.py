

import os
import sys
from pathlib import Path
import glob
import json
import re

import cv2
import numpy as np
import torch
from PIL import Image

# Add sam3 to path
sam3_root = Path(__file__).parent.parent / "sam3"
sys.path.insert(0, str(sam3_root))

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
    render_masklet_frame,
)

# Configuration
VIDEO_DIR = Path(__file__).parent.parent / "data" / "videos"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sam3_batch_segmentations"
ANNOTATION_DIR = Path(__file__).parent.parent / "data" / "Annotations" / "HyperNeRF"
PRECHECK_FRAMES = 5

# Text prompts for each scene - single object prompts (SAM3 auto-detects multiple instances)
SCENE_PROMPTS = {
    "americano": [
        "glass cup",
        "spoon",
        "hand",
        "table surface",
        "woven placemat",
        "wooden tray",
        "round coaster",
        "cutting board",
        "napkin",
    ],
    "chickchicken": [
        "red container",
        "hands",
        "cutting board",
        "countertop",
        "sleeve",
    ],
    "espresso": [
        "espresso machine",
        "portafilter",
        "coffee scale",
        "glass cup",
        "round coaster",
        "drip tray",
        "countertop",
        "power cable",
    ],
    "keyboard": [
        "keyboard",
        "keycaps",
        "hands",
        "desk mat",
        "laptop",
        "table surface",
    ],
    "split-cookie": [
        "cookie",
        "hands",
        "cutting board",
        "blue checkered paper",
        "round coaster",
        "table surface",
    ],
    "torchocolate": [
        "blowtorch",
        "hand",
        "baking tray",
        "chocolate",
        "parchment paper",
        "cutting board",
        "countertop",
        "shirt",
    ],
}

def normalize_prompts(text_prompt):
    if text_prompt is None:
        return []
    if isinstance(text_prompt, (list, tuple)):
        return [p for p in (p.strip() for p in text_prompt) if p]
    return [p for p in (p.strip() for p in str(text_prompt).split(",")) if p]


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return value


def merge_outputs(combined, outputs_per_frame, id_offset):
    for frame_idx, outputs in outputs_per_frame.items():
        masks = outputs.get("out_binary_masks")
        if masks is None:
            continue
        masks = to_numpy(masks)
        obj_ids = outputs.get("out_obj_ids", [])
        obj_ids = np.array(obj_ids, dtype=np.int32)
        if obj_ids.size:
            obj_ids = obj_ids + int(id_offset)
        probs = outputs.get("out_probs")
        if probs is not None:
            probs = np.array(to_numpy(probs), dtype=np.float32)
        boxes = outputs.get("out_boxes_xywh")
        if boxes is not None:
            boxes = np.array(to_numpy(boxes), dtype=np.float32)

        if frame_idx not in combined:
            combined[frame_idx] = {
                "out_binary_masks": masks,
                "out_obj_ids": obj_ids,
                "out_probs": probs,
                "out_boxes_xywh": boxes,
            }
            continue

        combined[frame_idx]["out_binary_masks"] = np.concatenate(
            [combined[frame_idx]["out_binary_masks"], masks], axis=0
        )
        combined[frame_idx]["out_obj_ids"] = np.concatenate(
            [combined[frame_idx]["out_obj_ids"], obj_ids], axis=0
        )
        if probs is not None and combined[frame_idx]["out_probs"] is not None:
            combined[frame_idx]["out_probs"] = np.concatenate(
                [combined[frame_idx]["out_probs"], probs], axis=0
            )
        if boxes is not None and combined[frame_idx]["out_boxes_xywh"] is not None:
            combined[frame_idx]["out_boxes_xywh"] = np.concatenate(
                [combined[frame_idx]["out_boxes_xywh"], boxes], axis=0
            )
    return combined


def load_annotation_prompts(scene_name):
    annotations_path = ANNOTATION_DIR / scene_name / "video_annotations.json"
    if not annotations_path.exists():
        return []
    try:
        data = json.loads(annotations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: Failed to read annotations for {scene_name}: {exc}")
        return []
    if isinstance(data, dict):
        return [str(k).strip() for k in data.keys() if str(k).strip()]
    return []


def safe_name(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "object"


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def find_prompt_frame(predictor, session_id, prompt, frame_count):
    check_frames = min(PRECHECK_FRAMES, frame_count) if frame_count > 0 else PRECHECK_FRAMES
    print(f"Testing prompt on first {check_frames} frames...")
    for frame_idx in range(check_frames):
        _ = predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                text=prompt,
            )
        )
        out = response["outputs"]
        num_objects = len(out.get("out_obj_ids", []))
        if num_objects > 0:
            return frame_idx, out
    return None, None


def propagate_in_video(predictor, session_id):
    """Propagate segmentation through entire video"""
    outputs_per_frame = {}
    # Use autocast for performance (bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=0,
                max_frame_num_to_track=None,
                propagation_direction="forward",
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def save_masks_npz(outputs_per_frame, output_path):
    """Save masks to compressed NPZ format"""
    masks_dict = {}
    for frame_idx, outputs in outputs_per_frame.items():
        # SAM3 returns 'out_binary_masks', not 'masks'
        if "out_binary_masks" in outputs and outputs["out_binary_masks"] is not None:
            masks = outputs["out_binary_masks"]
            # Convert to numpy array if needed
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            masks_dict[f"frame_{frame_idx:06d}"] = masks

    if masks_dict:
        np.savez_compressed(output_path, **masks_dict)
        print(f"Saved {len(masks_dict)} frames of masks to {output_path}")
    else:
        print(f"Warning: No masks to save!")


def save_metadata(outputs_per_frame, text_prompt, output_path, prompt_object_ids=None):
    """Save segmentation metadata to JSON"""
    metadata = {
        "text_prompt": text_prompt,
        "num_frames": len(outputs_per_frame),
        "frames": {}
    }
    if prompt_object_ids is not None:
        metadata["prompt_object_ids"] = prompt_object_ids

    for frame_idx, outputs in outputs_per_frame.items():
        # SAM3 uses 'out_obj_ids', 'out_probs', 'out_boxes_xywh'
        frame_info = {
            "frame_index": frame_idx,
            "num_objects": len(outputs.get("out_obj_ids", [])),
            "object_ids": [int(x) for x in outputs.get("out_obj_ids", [])],
        }

        # Add scores if available
        if "out_probs" in outputs:
            scores = outputs["out_probs"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy().tolist()
            elif hasattr(scores, 'tolist'):
                scores = scores.tolist()
            frame_info["scores"] = scores

        # Add bounding boxes if available
        if "out_boxes_xywh" in outputs:
            boxes = outputs["out_boxes_xywh"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy().tolist()
            elif hasattr(boxes, 'tolist'):
                boxes = boxes.tolist()
            frame_info["boxes"] = boxes

        metadata["frames"][str(frame_idx)] = frame_info

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_path}")


def save_visualization_video(outputs_per_frame, video_path, output_path, alpha=0.6):
    """Save visualization video with mask overlays"""
    print(f"Creating visualization video...")

    # Open original video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Render masks if available for this frame
        if frame_idx in outputs_per_frame:
            outputs = outputs_per_frame[frame_idx]

            # Convert tensors to numpy if needed
            outputs_np = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    outputs_np[key] = value.cpu().numpy()
                else:
                    outputs_np[key] = value

            # Render masks on frame
            rendered_rgb = render_masklet_frame(frame_rgb, outputs_np, frame_idx=frame_idx, alpha=alpha)
        else:
            rendered_rgb = frame_rgb

        # Convert RGB back to BGR for video writer
        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        out.write(rendered_bgr)

        frame_idx += 1

    cap.release()
    out.release()

    print(f"Saved visualization video to {output_path}")
    print(f"  Processed {frame_idx} frames at {fps} FPS")


def process_video_multi(predictor, video_path, scene_name, prompts, output_dir):
    """Process a single video with multiple prompts and merge outputs."""
    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"Video: {video_path}")
    print(f"Text prompts: {prompts}")
    print(f"{'='*60}\n")

    scene_output_dir = output_dir / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    combined_outputs = {}
    next_offset = 0
    prompt_object_ids = []
    frame_count = get_video_frame_count(video_path)
    for prompt in prompts:
        print("Starting SAM3 session...")
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]
        print(f"Session ID: {session_id}")

        try:
            print(f"Adding text prompt: '{prompt}'")
            frame_idx, out = find_prompt_frame(predictor, session_id, prompt, frame_count)
            if frame_idx is None:
                print(f"No objects detected for '{prompt}' in first {PRECHECK_FRAMES} frames, skipping.")
                continue
            obj_ids = out.get("out_obj_ids", [])
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.cpu().numpy()
            obj_ids = [int(x) for x in obj_ids]
            global_ids = [int(x) + int(next_offset) for x in obj_ids]
            prompt_object_ids.append({"prompt": prompt, "object_ids": global_ids})
            num_objects = len(obj_ids)
            print(f"Detected {num_objects} objects on frame {frame_idx}")
            print(f"  Object IDs: {obj_ids}")
            print(f"  Confidence scores: {out['out_probs']}")

            print("Propagating through video...")
            outputs_per_frame = propagate_in_video(predictor, session_id)
            print(f"Processed {len(outputs_per_frame)} frames")

            print("Saving per-object outputs...")
            prompt_slug = safe_name(prompt)
            save_masks_npz(
                outputs_per_frame,
                scene_output_dir / f"masks_{prompt_slug}.npz",
            )
            save_metadata(
                outputs_per_frame,
                prompt,
                scene_output_dir / f"metadata_{prompt_slug}.json",
                prompt_object_ids=[{"prompt": prompt, "object_ids": obj_ids}],
            )
            save_visualization_video(
                outputs_per_frame,
                video_path,
                scene_output_dir / f"visualization_{prompt_slug}.mp4",
                alpha=0.6,
            )

            combined_outputs = merge_outputs(combined_outputs, outputs_per_frame, next_offset)
            max_id = -1
            for outputs in outputs_per_frame.values():
                obj_ids = outputs.get("out_obj_ids", [])
                if isinstance(obj_ids, torch.Tensor):
                    obj_ids = obj_ids.cpu().numpy()
                if len(obj_ids):
                    max_id = max(max_id, int(np.max(obj_ids)))
            if max_id >= 0:
                next_offset += max_id + 1

        finally:
            print("Closing session...")
            _ = predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

    if not combined_outputs:
        print(f"Warning: No masks to save for {scene_name}")
        return

    print("Saving combined outputs...")
    save_masks_npz(
        combined_outputs,
        scene_output_dir / "masks.npz"
    )
    save_metadata(
        combined_outputs,
        prompts,
        scene_output_dir / "metadata.json",
        prompt_object_ids=prompt_object_ids,
    )
    save_visualization_video(
        combined_outputs,
        video_path,
        scene_output_dir / "visualization.mp4",
        alpha=0.6
    )

    print(f"Successfully processed {scene_name}")

def process_video(predictor, video_path, scene_name, text_prompt, output_dir):
    """Process a single video with SAM3"""
    prompts = normalize_prompts(text_prompt)
    if not prompts:
        print(f"Warning: No prompts for {scene_name}, skipping")
        return
    if len(prompts) > 1:
        process_video_multi(predictor, video_path, scene_name, prompts, output_dir)
        return
    text_prompt = prompts[0]
    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"Video: {video_path}")
    print(f"Text prompt: '{text_prompt}'")
    print(f"{'='*60}\n")

    # Create output directory
    scene_output_dir = output_dir / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = get_video_frame_count(video_path)

    # Start session
    print("Starting SAM3 session...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(video_path),
        )
    )
    session_id = response["session_id"]
    print(f"Session ID: {session_id}")

    try:
        # Add text prompt on frame 0
        print(f"Adding text prompt: '{text_prompt}'")
        frame_idx, out = find_prompt_frame(predictor, session_id, text_prompt, frame_count)
        if frame_idx is None:
            print(f"No objects detected for '{text_prompt}' in first {PRECHECK_FRAMES} frames, skipping.")
            return
        obj_ids = out.get("out_obj_ids", [])
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.cpu().numpy()
        obj_ids = [int(x) for x in obj_ids]
        prompt_object_ids = [{"prompt": text_prompt, "object_ids": obj_ids}]
        num_objects = len(obj_ids)
        print(f"Detected {num_objects} objects on frame {frame_idx}")
        print(f"  Object IDs: {obj_ids}")
        print(f"  Confidence scores: {out['out_probs']}")

        # Propagate through video
        print("Propagating through video...")
        outputs_per_frame = propagate_in_video(predictor, session_id)
        print(f"Processed {len(outputs_per_frame)} frames")

        # Save outputs
        print("Saving outputs...")
        save_masks_npz(
            outputs_per_frame,
            scene_output_dir / "masks.npz"
        )
        save_metadata(
            outputs_per_frame,
            text_prompt,
            scene_output_dir / "metadata.json",
            prompt_object_ids=prompt_object_ids,
        )
        save_visualization_video(
            outputs_per_frame,
            video_path,
            scene_output_dir / "visualization.mp4",
            alpha=0.6
        )

        print(f"Successfully processed {scene_name}")

    finally:
        # Close session to free resources
        print("Closing session...")
        _ = predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )


def main():
    """Process all videos in VIDEO_DIR"""
    print("="*60)
    print("SAM3 Batch Video Segmentation")
    print("="*60)
    print(f"Video directory: {VIDEO_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all videos
    video_files = list(VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    # Initialize predictor
    print("\nInitializing SAM3 predictor...")
    gpus_to_use = range(torch.cuda.device_count())
    print(f"Using GPUs: {list(gpus_to_use)}")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    print("Predictor initialized")

    # Process each video
    successful = []
    failed = []

    for video_path in sorted(video_files):
        # Extract scene name (e.g., "americano_2x.mp4" -> "americano")
        scene_name = video_path.stem.split("_")[0]

        # Prefer explicit prompt list if provided, else fall back to annotations
        text_prompt = SCENE_PROMPTS.get(scene_name)
        if not isinstance(text_prompt, (list, tuple)):
            ann_prompts = load_annotation_prompts(scene_name)
            if ann_prompts:
                text_prompt = ann_prompts
        if not text_prompt:
            print(f"Warning: No text prompt defined for '{scene_name}', skipping")
            failed.append(scene_name)
            continue

        try:
            process_video(
                predictor,
                video_path,
                scene_name,
                text_prompt,
                OUTPUT_DIR
            )
            successful.append(scene_name)
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(scene_name)

    # Shutdown predictor
    print("\nShutting down predictor...")
    predictor.shutdown()

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Successfully processed: {len(successful)} scenes")
    for scene in successful:
        print(f"  OK {scene}")

    if failed:
        print(f"\nFailed to process: {len(failed)} scenes")
        for scene in failed:
            print(f"  FAIL {scene}")

    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
