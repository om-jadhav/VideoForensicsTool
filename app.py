import json
import os
import tempfile
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from frame_analyzer import analyze_frames, grab_frames, risk_level
from hash_calculator import calculate_hashes
from metadata import extract_metadata

st.set_page_config(page_title="Video Forensic Analysis Tool", layout="wide")
st.title("Video Forensic Analysis Tool")
st.caption("Detects anomalies that MAY indicate tampering. Results are indicators, not proof.")

# ---------- Sidebar: tunable sensitivity ----------
st.sidebar.header("Sensitivity")
z_thresh = st.sidebar.slider("Spike threshold (robust z-score)", 2.0, 8.0, 4.0, 0.5,
                             help="Lower = more sensitive, more false positives.")
dup_thresh = st.sidebar.slider("Duplicate-frame threshold (mean pixel diff)", 0.05, 1.0, 0.3, 0.05,
                               help="Frames with difference below this are treated as duplicates.")

uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    # Unique temp file (no path traversal, no name clashes), always cleaned up
    suffix = os.path.splitext(os.path.basename(uploaded.name))[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        path = tmp.name

    try:
        with st.spinner("Hashing and reading metadata..."):
            hashes = calculate_hashes(path)
            meta = extract_metadata(path)

        bar = st.progress(0.0, text="Analyzing frames...")
        analysis = analyze_frames(path, z_thresh, dup_thresh,
                                  progress_cb=lambda p: bar.progress(p, text="Analyzing frames..."))
        bar.empty()

        level, score, reasons = risk_level(analysis, meta)
        previews = grab_frames(path, analysis["flagged_frames"]
                               if not analysis["scene_cuts"] else analysis["scene_cuts"])
    finally:
        os.remove(path)

    # ---------- Summary ----------
    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[level]
    st.markdown(f"### Tamper-risk indicator: :{color}[{level}] (score {score})")
    if reasons:
        st.caption("Score based on: " + ", ".join(reasons))
    for note in analysis["notes"]:
        st.info(note)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames analyzed", analysis["total_frames"])
    c2.metric("Scene cuts", len(analysis["scene_cuts"]))
    c3.metric("Duplicate-frame runs", len(analysis["duplicate_runs"]))
    c4.metric("Timestamp gaps", len(analysis["timestamp_gaps"]))

    tab1, tab2, tab3, tab4 = st.tabs(["Frame analysis", "Metadata", "Integrity hashes", "Report"])

    # ---------- Tab 1: chart ----------
    with tab1:
        if analysis["diffs"]:
            df = pd.DataFrame({"frame": range(1, len(analysis["diffs"]) + 1),
                               "difference": analysis["diffs"]})
            df["flag"] = "normal"
            df.loc[df["frame"].isin(analysis["motion_spikes"]), "flag"] = "motion spike"
            df.loc[df["frame"].isin(analysis["scene_cuts"]), "flag"] = "scene cut"
            dup_frames = {f for a, b in analysis["duplicate_runs"] for f in range(a, b + 1)}
            df.loc[df["frame"].isin(dup_frames), "flag"] = "duplicate"
            df.loc[df["frame"].isin(analysis["timestamp_gaps"]), "flag"] = "timestamp gap"

            line = alt.Chart(df).mark_line(color="#888").encode(x="frame:Q", y="difference:Q")
            pts = alt.Chart(df[df["flag"] != "normal"]).mark_circle(size=70).encode(
                x="frame:Q", y="difference:Q",
                color=alt.Color("flag:N", scale=alt.Scale(
                    domain=["scene cut", "motion spike", "duplicate", "timestamp gap"],
                    range=["red", "orange", "blue", "purple"])),
                tooltip=["frame", "difference", "flag"])
            st.altair_chart((line + pts).interactive(), width="stretch")
            st.caption("Grey line = change between each frame and the previous one. "
                       "Coloured dots = anomalies.")

            st.write("**Flagged frames**")
            flagged = df[df["flag"] != "normal"]
            if len(flagged) > 0:
                st.dataframe(flagged, width="stretch", hide_index=True)
            else:
                st.success("No anomalies detected at the current sensitivity.")
        if previews:
            st.write("**Preview of flagged frames**")
            cols = st.columns(len(previews))
            for col, (idx, img) in zip(cols, previews):
                col.image(img, caption=f"Frame {idx}", width="stretch")

    # ---------- Tab 2 ----------
    with tab2:
        for w in meta.get("warnings", []):
            st.warning(w)
        st.json(meta)

    # ---------- Tab 3 ----------
    with tab3:
        st.write("Record these at acquisition time; a later mismatch means the file changed.")
        st.code(f"SHA-256: {hashes['sha256']}\nMD5:     {hashes['md5']}")

    # ---------- Tab 4: report ----------
    report = {
        "tool": "Video Forensic Analysis Tool v2",
        "analysed_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_name": os.path.basename(uploaded.name),
        "hashes": hashes,
        "metadata": meta,
        "risk_level": level,
        "risk_score": score,
        "risk_reasons": reasons,
        "parameters": {"z_thresh": z_thresh, "dup_thresh": dup_thresh},
        "analysis": {k: v for k, v in analysis.items() if k != "diffs"},
        "disclaimer": "Indicators only; not conclusive proof of tampering.",
    }
    report_name = f"report-{os.path.splitext(os.path.basename(uploaded.name))[0]}.json"
    report_json = json.dumps(report, indent=4, default=str)

    # Actually save a copy to disk (the old version only claimed to)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", report_name), "w") as f:
        f.write(report_json)

    with tab4:
        st.success(f"Report saved to reports/{report_name}")
        st.download_button("Download report as JSON", report_json,
                           file_name=report_name, mime="application/json")
        st.json(report, expanded=False)