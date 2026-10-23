import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

import cv2

# Encoder / software strings that hint the file was processed by an editor or re-encoder
EDITOR_HINTS = ["lavf", "lavc", "ffmpeg", "handbrake", "premiere", "adobe",
                "davinci", "final cut", "vegas", "camtasia", "filmora",
                "capcut", "kdenlive", "shotcut", "openshot"]


def _fps_from_ratio(ratio):
    try:
        num, den = ratio.split("/")
        return round(float(num) / float(den), 3) if float(den) else None
    except Exception:
        return None


def _opencv_fallback(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 3),
    }
    cap.release()
    return info


def extract_metadata(video_path):
    """Container + stream metadata via ffprobe (if installed), else OpenCV only."""
    st = os.stat(video_path)
    meta = {
        "file_size_bytes": st.st_size,
        "fs_modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "source": "opencv",
    }
    meta.update(_opencv_fallback(video_path))
    meta["warnings"] = []

    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", video_path],
                capture_output=True, text=True, timeout=60).stdout
            data = json.loads(out)
            fmt = data.get("format", {})
            tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
            v = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), {})
            a = next((s for s in data.get("streams", []) if s["codec_type"] == "audio"), {})
            vtags = {k.lower(): v_ for k, v_ in v.get("tags", {}).items()}

            meta.update({
                "source": "ffprobe",
                "container": fmt.get("format_long_name"),
                "duration_sec": float(fmt["duration"]) if "duration" in fmt else None,
                "bit_rate": fmt.get("bit_rate"),
                "video_codec": v.get("codec_name"),
                "pixel_format": v.get("pix_fmt"),
                "avg_fps": _fps_from_ratio(v.get("avg_frame_rate", "")),
                "audio_codec": a.get("codec_name"),
                "creation_time": tags.get("creation_time") or vtags.get("creation_time"),
                "encoder": tags.get("encoder") or vtags.get("encoder"),
                "handler_name": vtags.get("handler_name"),
                "device_make_model": tags.get("com.apple.quicktime.model") or tags.get("model"),
                "gps": tags.get("location") or tags.get("com.apple.quicktime.location.iso6709"),
            })

            # Simple forensic hints
            sw = " ".join(str(x) for x in [meta["encoder"], meta["handler_name"]] if x).lower()
            if any(h in sw for h in EDITOR_HINTS):
                meta["warnings"].append(
                    f"Encoder/software tag suggests processing by an editor or re-encoder: '{meta['encoder'] or meta['handler_name']}'")
            if not meta["creation_time"]:
                meta["warnings"].append("No creation_time tag (often stripped by editing or re-encoding).")
            if v.get("avg_frame_rate") != v.get("r_frame_rate"):
                meta["warnings"].append(
                    "Average and real frame rate differ (variable frame rate - possible edit or screen recording).")
        except Exception as e:
            meta["warnings"].append(f"ffprobe failed, using OpenCV only: {e}")
    else:
        meta["warnings"].append("ffprobe not installed - only basic OpenCV metadata available.")

    return meta