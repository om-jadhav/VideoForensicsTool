import cv2
import numpy as np


def _robust_z(x):
    """Median/MAD z-score: not thrown off by the very outliers we want to find."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-6
    return 0.6745 * (x - med) / mad


def _group_runs(indices):
    """[5,6,7,20,21] -> [(5,7),(20,21)] so reports show runs, not 100 numbers."""
    runs, start, prev = [], None, None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    if start is not None:
        runs.append((start, prev))
    return runs


def analyze_frames(video_path, z_thresh=4.0, dup_thresh=0.3, progress_cb=None):
    """
    Single pass over the video. Per consecutive frame pair we measure:
      - mean absolute pixel difference (on a small blurred grayscale copy)
      - histogram correlation (low value = abrupt visual change / cut)
      - timestamp delta (irregular gaps = dropped or inserted frames)
    Then flag statistical anomalies instead of 'any difference'.
    Frame numbers are 0-based; entry k of the arrays describes frame k+1 vs k.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    diffs, corrs, times = [], [], []
    prev_gray = prev_hist = None
    n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        if prev_gray is not None:
            diffs.append(float(cv2.absdiff(gray, prev_gray).mean()))
            corrs.append(float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)))
        prev_gray, prev_hist = gray, hist
        n += 1
        if progress_cb and n % 20 == 0:
            progress_cb(min(n / total, 1.0))

    cap.release()
    if progress_cb:
        progress_cb(1.0)

    result = {"total_frames": n, "diffs": [], "scene_cuts": [], "motion_spikes": [],
              "duplicate_runs": [], "timestamp_gaps": [], "flagged_frames": [],
              "notes": []}
    if n < 10:
        result["notes"].append("Video too short for statistical analysis.")
        return result

    diffs, corrs = np.array(diffs), np.array(corrs)
    z = _robust_z(diffs)
    spike_idx = np.where(z > z_thresh)[0]

    # Spike + big histogram change = likely cut; spike without it = sudden motion
    cuts = [int(k + 1) for k in spike_idx if corrs[k] < 0.85]
    spikes = [int(k + 1) for k in spike_idx if corrs[k] >= 0.85]

    # Regularly spaced spikes = normal keyframe (GOP) pulses from the encoder, not edits
    if len(spikes) >= 3:
        gaps_between = np.diff(spikes)
        common = np.bincount(gaps_between).argmax()
        if common > 1 and np.mean(gaps_between == common) >= 0.7:
            result["notes"].append(
                f"Ignored {len(spikes)} evenly spaced spikes (every ~{common} frames): "
                "typical encoder keyframe pattern.")
            spikes = []

    # Duplicate / frozen frames - only meaningful if the video normally has motion
    dups = []
    limit = min(dup_thresh, 0.25 * np.median(diffs))  # must be far below normal motion
    if limit > 0:
        dups = [int(k + 1) for k in np.where(diffs < limit)[0]]
        if len(dups) > 0.08 * len(diffs):
            result["notes"].append(
                f"{len(dups)} near-identical frames ({100 * len(dups) / len(diffs):.0f}% of the video). "
                "That is too many to be tampering: the source is probably low-frame-rate content "
                "converted to a higher rate (e.g. 24/15 fps -> 30 fps) or has long static stretches "
                "(slides, screen recording). Duplicate detection was ignored.")
            dups = []

    # Timestamp irregularities
    gaps = []
    t = np.diff(np.array(times))
    if len(t) > 5 and np.median(t) > 0:
        med_dt = np.median(t)
        gaps = [int(k + 1) for k in np.where((t > 1.8 * med_dt) | (t < 0.3 * med_dt))[0]]

    if len(cuts) > 5 and n > 30:
        result["notes"].append(
            f"{len(cuts)} scene cuts found (~{len(cuts) / max(n / 30, 1) * 60:.1f} per minute at 30 fps). "
            "Edited or produced videos naturally contain many cuts; raise the spike threshold "
            "in the sidebar to see only the most abrupt ones.")

    result.update({
        "diffs": [round(d, 3) for d in diffs.tolist()],
        "scene_cuts": cuts,
        "motion_spikes": spikes,
        "duplicate_runs": [list(r) for r in _group_runs(dups)],
        "timestamp_gaps": gaps,
        "flagged_frames": sorted(set(cuts + spikes + dups + gaps)),
        "mean_diff": round(float(diffs.mean()), 3),
        "median_diff": round(float(np.median(diffs)), 3),
    })
    return result


def risk_level(analysis, metadata):
    """
    Simple, explainable score. Scene cuts alone are NOT strong evidence (normal editing
    produces them), so they add at most 1 point. Stronger signals weigh more.
    """
    score = 0
    reasons = []
    if analysis["scene_cuts"]:
        score += 1
        reasons.append("scene cuts present (+1)")
    runs = len(analysis["duplicate_runs"])
    if runs:
        pts = min(runs, 2)
        score += pts
        reasons.append(f"{runs} duplicate-frame run(s) (+{pts})")
    if analysis["timestamp_gaps"]:
        score += 2
        reasons.append("timestamp irregularities (+2)")
    w = min(len(metadata.get("warnings", [])), 2)
    if w:
        score += w
        reasons.append(f"{w} metadata warning(s) (+{w})")
    level = "LOW" if score <= 2 else "MEDIUM" if score <= 4 else "HIGH"
    return level, score, reasons


def grab_frames(video_path, indices, max_n=6):
    """Fetch a few flagged frames (RGB) for on-screen preview."""
    cap = cv2.VideoCapture(video_path)
    out = []
    for i in indices[:max_n]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret:
            out.append((i, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out