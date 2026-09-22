"""Combine report frames into a compact, portable GIF."""
import argparse
from pathlib import Path
from PIL import Image


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--frames-dir", default=str(root / "results/frames"))
parser.add_argument("--output", default=str(root / "results/wcsph_animation.gif"))
parser.add_argument("--fps", type=float, default=11.111111)
args = parser.parse_args()
paths = sorted(Path(args.frames_dir).glob("frame_*.png"))
if not paths:
    raise SystemExit("No frame_*.png files found")
frames = []
for path in paths:
    with Image.open(path) as image:
        frame = image.convert("RGB")
        frame.thumbnail((720, 720), Image.Resampling.LANCZOS)
        frames.append(frame.copy())
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
gif_duration_ms = max(10, round(100.0 / args.fps) * 10)
frames[0].save(output, save_all=True, append_images=frames[1:],
               duration=gif_duration_ms,
               loop=0, optimize=True, disposal=2)
print(f"animation={output}, frames={len(frames)}")
