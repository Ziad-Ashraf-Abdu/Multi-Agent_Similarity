import streamlit as st
import json
import os
import tempfile
import numpy as np
import librosa
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time

# ====== CONFIGURATION ======
DATABASE_FILE = "melody_database.json"
SONGS_FOLDER = "songs"
# ===========================

# Initialize session state
if 'database' not in st.session_state:
    st.session_state.database = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'duration' not in st.session_state:
    st.session_state.duration = 0.0
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'query_seq' not in st.session_state:
    st.session_state.query_seq = None


@st.cache_resource
def load_database():
    """Load melody database"""
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            return None
    return None


def extract_hum_pitch(y, sr):
    """Extract pitch contour from humming audio"""
    if len(y) == 0 or np.all(y == 0):
        return None

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr
    )

    if f0 is None or len(f0) == 0:
        return None

    pitch_track = f0[voiced_flag]
    if len(pitch_track) < 10:
        return None

    midi_track = librosa.hz_to_midi(pitch_track)
    midi_normalized = midi_track - np.mean(midi_track)

    if len(midi_normalized) > 5:
        q1, q3 = np.percentile(midi_normalized, [25, 75])
        iqr = q3 - q1
        mask = (midi_normalized >= q1 - 1.5 * iqr) & (midi_normalized <= q3 + 1.5 * iqr)
        midi_normalized = midi_normalized[mask]

    return midi_normalized


def get_top_matches_subsequence(hum_path, top_n=2):
    """Find best matching song segments with subsequence DTW"""
    y, sr = librosa.load(hum_path, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < 1000:
        raise Exception("Audio too short after silence removal")

    query_seq = extract_hum_pitch(y, sr)
    if query_seq is None:
        raise Exception("No valid pitch detected (too quiet/noisy/short)")

    # Downsample query for speed
    query_seq = query_seq[::10]
    if len(query_seq) < 5:
        raise Exception("Hum too short for reliable matching (need ≥5 distinct notes)")

    query_seq_reshaped = query_seq.reshape(-1, 1)
    results = []

    for song_name, song_data in st.session_state.database.items():
        ref_seq = np.array(song_data)
        ref_seq_reshaped = ref_seq.reshape(-1, 1)
        min_dist = float('inf')
        best_segment = []

        query_len = len(query_seq)
        min_window = max(5, int(query_len * 0.8))
        step = max(1, min_window // 4)

        # Sliding window to find best subsequence
        for start in range(0, len(ref_seq) - min_window + 1, step):
            end = min(start + min_window, len(ref_seq))
            segment = ref_seq_reshaped[start:end]

            if len(segment) < min_window * 0.7:
                continue

            distance, _ = fastdtw(query_seq_reshaped, segment, dist=euclidean, radius=15)
            normalized_dist = distance / len(query_seq)

            if normalized_dist < min_dist:
                min_dist = normalized_dist
                # Store the actual 1D segment for plotting
                best_segment = ref_seq[start:end]

        clean_name = (song_name.replace("vocals.wav", "")
                      .replace("Vocals.wav", "")
                      .replace(".wav", "")
                      .replace(".mp3", "")
                      .strip("_- "))

        results.append({
            "name": clean_name,
            "score": min_dist,
            "segment": best_segment
        })

    results.sort(key=lambda x: x["score"])
    return query_seq, results[:top_n]


def process_audio_bytes(audio_bytes):
    """Process raw audio bytes"""
    st.session_state.processing = True
    start_time = time.time()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        query_seq, top_matches = get_top_matches_subsequence(tmp_path)
        duration = time.time() - start_time
        return query_seq, top_matches, duration
    except Exception as e:
        raise e
    finally:
        st.session_state.processing = False
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ====== STREAMLIT UI ======
st.set_page_config(page_title="Hum2Song Recognizer", page_icon="🎵", layout="wide")

st.title("🎵 Hum to Song Recognizer")
st.markdown("Hum a tune or upload audio to find matching songs!")

# Load database
if st.session_state.database is None:
    with st.spinner("Loading song database..."):
        st.session_state.database = load_database()

if not st.session_state.database:
    st.error(f"""
    **Database not found!** 1. Run `processor.py` to build `{DATABASE_FILE}`  
    2. Place vocal files in `{SONGS_FOLDER}/` folder
    """)
    st.stop()

# Layout: Input on Left, Tips on Right
col_input, col_tips = st.columns([1, 1])

with col_input:
    st.success(f"✅ Database loaded: {len(st.session_state.database)} songs")

    # Recording section
    st.subheader("1. Input Audio")
    tab_record, tab_upload = st.tabs(["🎤 Record Hum", "📤 Upload File"])

    with tab_record:
        try:
            from streamlit_mic_recorder import mic_recorder

            audio = mic_recorder(
                start_prompt="⏺️ Start Recording",
                stop_prompt="⏹️ Stop Recording",
                format="wav",
                key='my_recorder'
            )

            if audio and audio.get('bytes') and st.session_state.recorded_audio is None:
                st.session_state.recorded_audio = audio['bytes']
                st.rerun()

            if st.session_state.recorded_audio is not None:
                st.audio(st.session_state.recorded_audio, format="audio/wav")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔍 Recognize Hum", type="primary", disabled=st.session_state.processing,
                                 use_container_width=True):
                        with st.spinner("Analyzing melody..."):
                            try:
                                q_seq, results, duration = process_audio_bytes(st.session_state.recorded_audio)
                                st.session_state.query_seq = q_seq
                                st.session_state.results = results
                                st.session_state.duration = duration
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ {str(e)}")
                with c2:
                    if st.button("🗑️ Clear", use_container_width=True):
                        st.session_state.recorded_audio = None
                        st.session_state.results = None
                        st.session_state.query_seq = None
                        st.rerun()

        except ImportError:
            st.error("Microphone library missing. pip install streamlit-mic-recorder")

    with tab_upload:
        uploaded_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"], label_visibility="collapsed")
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            if st.button("🔍 Recognize File", type="primary", disabled=st.session_state.processing,
                         use_container_width=True):
                with st.spinner("Analyzing melody..."):
                    try:
                        q_seq, results, duration = process_audio_bytes(uploaded_file.read())
                        st.session_state.query_seq = q_seq
                        st.session_state.results = results
                        st.session_state.duration = duration
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

with col_tips:
    with st.expander("💡 Tips for Best Results", expanded=True):
        st.markdown("""
        - ✅ Hum **10-30 seconds** of the most recognizable part.
        - ✅ Hum with **clear pitch changes** (avoid monotone).
        - ✅ System is key-invariant (you can be off-key!).
        """)

st.divider()

# ====== VISUALIZATION & RESULTS SECTION ======
if st.session_state.results and st.session_state.query_seq is not None:

    # --- 1. TOP MATCHES SECTION (NOW ON TOP) ---
    st.subheader("2. Top Matches")

    res_col1, res_col2 = st.columns(2)

    for i, match in enumerate(st.session_state.results):
        score = match["score"]
        confidence = max(0, min(100, int(100 - score * 15)))

        if score < 5.0:
            badge = "✅ HIGH MATCH"
            color = "#4CAF50"
        elif score < 8.0:
            badge = "⚠️ POSSIBLE MATCH"
            color = "#FFA726"
        else:
            badge = "❓ LOW CONFIDENCE"
            color = "#EF5350"

        target_col = res_col1 if i == 0 else res_col2

        with target_col:
            st.markdown(f"""
            <div style="background-color: {color}10; padding: 20px; border-radius: 12px; border: 1px solid {color}40; height: 100%;">
                <h2 style="margin: 0; color: {color};">#{i + 1} {match['name']}</h2>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Confidence</span>
                        <span style="font-weight: bold; color: {color};">{confidence}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden;">
                        <div style="width: {confidence}%; height: 100%; background-color: {color};"></div>
                    </div>
                </div>
                <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    DTW Distance: {score:.2f}<br>{badge}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.caption(f"⏱️ Analysis took: {st.session_state.duration:.2f}s")

    st.write("")  # Spacer

    # --- 2. PLOTLY CHART SECTION (FOLDABLE) ---
    with st.expander("📈 View Technical Analysis (Melody Alignment)", expanded=False):

        user_seq = st.session_state.query_seq
        user_seq_norm = user_seq - np.mean(user_seq)

        fig = go.Figure()

        # Trace 1: User's Hum
        fig.add_trace(go.Scatter(
            y=user_seq_norm,
            mode='lines',
            name='Your Hum',
            line=dict(color='#29B5E8', width=5),
            opacity=0.9
        ))

        # Trace 2 & 3: Top Matches
        colors = ['#4CAF50', '#FFA726']  # Green, Orange

        for i, match in enumerate(st.session_state.results[:2]):
            segment = match['segment']
            if len(segment) > 0:
                # Normalize segment for visual comparison
                seg_norm = segment - np.mean(segment)

                fig.add_trace(go.Scatter(
                    y=seg_norm,
                    mode='lines',
                    name=f"Match #{i + 1}: {match['name']}",
                    line=dict(color=colors[i], width=3, dash='dashdot'),
                    opacity=0.8
                ))

        # Layout Customization
        fig.update_layout(
            title="Normalized Pitch Contour Comparison",
            xaxis_title="Time Steps (Normalized)",
            yaxis_title="Relative Pitch (Semitone Offset)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

        st.plotly_chart(fig, use_container_width=True)