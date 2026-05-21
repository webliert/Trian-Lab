import argparse
import json
import numpy as np


def clip_motion(input_path, output_path, start_time, end_time, fps):
    with open(input_path) as f:
        data = json.load(f)

    frames = data["Frames"]
    frame_duration = float(data.get("FrameDuration", 1.0 / fps))
    total_frames = len(frames)

    start_frame = max(0, int(start_time / frame_duration))
    end_frame = min(total_frames, int(end_time / frame_duration))

    if end_frame - start_frame < 2:
        raise ValueError(
            f"Clipped segment has {end_frame - start_frame} frames (< 2). "
            f"Total frames: {total_frames}, requested [{start_frame}, {end_frame})."
        )

    data["Frames"] = frames[start_frame:end_frame]

    with open(output_path, "w") as f:
        f.write("{\n")
        for i, (key, value) in enumerate(data.items()):
            if key == "Frames":
                continue
            comma = "," if i < len(data) - 1 else ""
            if isinstance(value, str):
                f.write(f'"{key}": "{value}"{comma}\n')
            elif isinstance(value, bool):
                f.write(f'"{key}": {str(value).lower()}{comma}\n')
            else:
                f.write(f'"{key}": {value}{comma}\n')

        f.write("\n")
        f.write('"Frames":\n[\n')

        frames_data = data["Frames"]
        for j, frame in enumerate(frames_data):
            line = ", ".join(f"{v:.6f}" for v in frame)
            if j == len(frames_data) - 1:
                f.write(f"  [{line}]\n")
            else:
                f.write(f"  [{line}],\n")

        f.write("]\n}")

    clipped_duration = (end_frame - start_frame) * frame_duration
    original_duration = total_frames * frame_duration
    print(
        f"Clipped {original_duration:.1f}s → {clipped_duration:.1f}s "
        f"(frames [{start_frame}, {end_frame}), {end_frame - start_frame} frames)"
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip a motion JSON file by time range.")
    parser.add_argument("--input", type=str, required=True, help="Path to input motion JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output clipped motion JSON file.")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end_time", type=float, required=True, help="End time in seconds.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30.0).")
    args = parser.parse_args()

    clip_motion(args.input, args.output, args.start_time, args.end_time, args.fps)
