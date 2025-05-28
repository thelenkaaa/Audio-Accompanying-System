import streamlit as st
import tempfile
import json
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip


# Initialize pipeline
pipeline = FullVideoAudioPipeline(
    GeminiSettings(),
    OpenAISettings(),
    StableAudioSettings(),
    ComposerSettings()
)

# Import the pipeline and settings classes
from your_module import (
    GeminiSettings,
    OpenAISettings,
    StableAudioSettings,
    ComposerSettings,
    FullVideoAudioPipeline,
)

# === Initialize Pipeline ===
gemini_settings   = GeminiSettings()
openai_settings   = OpenAISettings()
audio_settings    = StableAudioSettings()
composer_settings = ComposerSettings()

pipeline = FullVideoAudioPipeline(
    gemini_settings,
    openai_settings,
    audio_settings,
    composer_settings
)

# === Page Style & Config ===
st.set_page_config(page_title="🎥🔊 AI Audio Companion", layout="wide")

st.markdown(
    """
    <style>
    .centered-title { text-align: center; font-size: 2.2em; font-weight: 700; margin-bottom: 0.2em; }
    .centered-sub   { text-align: center; font-size: 1.1em; color: gray; margin-bottom: 2em; }
    .stButton>button { font-size: 16px; padding: 0.6em 1em; width: 100%; }
    .full-width-button button { width: 100%; font-size: 18px; font-weight: 600; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="centered-title">🎬 Audio Accompanying System for Silent Videos</div>', unsafe_allow_html=True)
st.markdown('<div class="centered-sub">Upload your silent video, analyze objects, and generate realistic visual sounds with automatic final video assembly.</div>', unsafe_allow_html=True)

# === Session State Init ===
for key, default in {
    "audio_prompts": {},
    "audio_files": {},
    "durations": {},
    "timings": {},
    "gemini_objects": [],
    "video_path": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === Main Layout ===
left_col, right_col = st.columns([1.2, 2], gap="large")

with left_col:
    st.markdown("### 📤 Upload Silent Video")
    uploaded = st.file_uploader("", type=["mp4","mov","avi"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            st.session_state.video_path = tmp.name
            st.video(tmp.name)

    if st.button("🔊 Analyze Video and Generate Audios"):
        if not st.session_state.video_path:
            st.warning("Please upload a video first.")
        else:
            with st.spinner("🔍 Processing... this may take a while..."):
                # Run full analysis pipeline up to raw objects
                result = pipeline.analyzer.analyze(st.session_state.video_path)
                objects = result.get("objects", [])

                # Filter relevant
                labels     = [o.get("label") for o in objects]
                sound_tags = pipeline.openai.get_sound_relevant_tags(labels)
                relevant   = [o for o in objects if o.get("label") in sound_tags]
                st.session_state.gemini_objects = relevant

                # Prompts and audio
                prompts = pipeline.openai.generate_audio_prompts_from_objects(relevant)
                durations_map = pipeline._extract_durations(relevant)
                timings_map   = pipeline._extract_timings(relevant)
                audio_files = pipeline.audio.generate_audio_for_tags(prompts, durations_map)

                # Store state
                st.session_state.audio_prompts = prompts
                st.session_state.durations    = durations_map
                st.session_state.timings      = timings_map
                st.session_state.audio_files  = audio_files

with right_col:
    if st.session_state.audio_files:
        st.markdown("### 🎧 Generated Audios with Prompts")
        for key, files in st.session_state.audio_files.items():
            st.subheader(f"🎯 Object: {key}")
            pcol, rcol = st.columns([3,1])
            with pcol:
                new_p = st.text_input(f"Prompt for {key}", value=st.session_state.audio_prompts[key], key=f"prompt_{key}")
                st.session_state.audio_prompts[key] = new_p
            with rcol:
                if st.button("🔁 Regenerate", key=f"regen_{key}"):
                    dur = st.session_state.durations.get(key, 3.0)
                    new_files = pipeline.audio.generate_audio_files(key, new_p, dur)
                    st.session_state.audio_files[key] = new_files

            for f in files:
                st.audio(f)

# === Compose & Merge ===
if st.session_state.audio_files:
    st.markdown('<div class="full-width-button">', unsafe_allow_html=True)
    if st.button("🎼 Compose Final Audio & Merge with Video"):
        with st.spinner("🎛️ Composing & merging..."):
            vd = pipeline._get_video_duration(st.session_state.video_path)
            audio_path = pipeline.composer.compose_final_audio(
                st.session_state.audio_files,
                st.session_state.timings,
                vd
            )
            final_video = pipeline.composer.merge_audio_with_video(
                st.session_state.video_path,
                audio_path,
                output_path="final_with_audio.mp4"
            )
            st.success("✅ Final video ready!")
    st.markdown('</div>', unsafe_allow_html=True)

    if 'final_video' in locals():
        col1, _, _ = st.columns([1,2,2])
        with col1:
            st.markdown("### 🎥 Final Output")
            st.video(final_video)
            with open(final_video, "rb") as vid:
                st.download_button(
                    label="⬇️ Download Final Video",
                    data=vid,
                    file_name="final_with_audio.mp4",
                    mime="video/mp4"
                )