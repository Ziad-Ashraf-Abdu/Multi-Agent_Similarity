import streamlit as st
import json
import os
import tempfile
import numpy as np
import librosa
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
        mask = (midi_normalized >= q1 - 1.5*iqr) & (midi_normalized <= q3 + 1.5*iqr)
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
    
    query_seq = query_seq[::10]
    if len(query_seq) < 5:
        raise Exception("Hum too short for reliable matching (need ≥5 distinct notes)")
    
    query_seq = query_seq.reshape(-1, 1)
    results = []
    
    for song_name, song_data in st.session_state.database.items():
        ref_seq = np.array(song_data).reshape(-1, 1)
        min_dist = float('inf')
        
        query_len = len(query_seq)
        min_window = max(5, int(query_len * 0.8))
        step = max(1, min_window // 4)
        
        for start in range(0, len(ref_seq) - min_window + 1, step):
            end = min(start + min_window, len(ref_seq))
            segment = ref_seq[start:end]
            
            if len(segment) < min_window * 0.7:
                continue
                
            distance, _ = fastdtw(query_seq, segment, dist=euclidean, radius=15)
            normalized_dist = distance / len(query_seq)
            
            if normalized_dist < min_dist:
                min_dist = normalized_dist
        
        clean_name = (song_name.replace("vocals.wav", "")
                                .replace("Vocals.wav", "")
                                .replace(".wav", "")
                                .replace(".mp3", "")
                                .strip("_- "))
        results.append({"name": clean_name, "score": min_dist})
    
    results.sort(key=lambda x: x["score"])
    return results[:top_n]

def process_audio_bytes(audio_bytes):
    """Process raw audio bytes"""
    st.session_state.processing = True
    start_time = time.time()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        top_matches = get_top_matches_subsequence(tmp_path)
        duration = time.time() - start_time
        return top_matches, duration
    except Exception as e:
        raise e
    finally:
        st.session_state.processing = False
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ====== STREAMLIT UI ======
st.set_page_config(page_title="Hum2Song Recognizer", page_icon="🎵", layout="centered")
st.title("🎵 Hum to Song Recognizer")
st.markdown("Hum a tune or upload audio to find matching songs!")

# Load database
if st.session_state.database is None:
    with st.spinner("Loading song database..."):
        st.session_state.database = load_database()

if not st.session_state.database:
    st.error(f"""
    **Database not found!**  
    1. Run `processor.py` to build `{DATABASE_FILE}`  
    2. Place vocal files in `{SONGS_FOLDER}/` folder
    """)
    st.stop()

st.success(f"✅ Loaded {len(st.session_state.database)} songs")

# Recording section
st.subheader("🎤 Record Your Hum (10-30 seconds)")

try:
    from streamlit_mic_recorder import mic_recorder
    
    # CRITICAL: format="wav" outputs WAV directly - NO FFmpeg needed!
    audio = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        format="wav",  # <-- This is the key! Outputs WAV natively
        key='my_recorder'
    )
    
    if audio and audio.get('bytes') and st.session_state.recorded_audio is None:
        st.session_state.recorded_audio = audio['bytes']
        st.rerun()
    
    if st.session_state.recorded_audio is not None:
        st.audio(st.session_state.recorded_audio, format="audio/wav")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Recognize", type="primary", disabled=st.session_state.processing, use_container_width=True):
                with st.spinner("Analyzing melody..."):
                    try:
                        results, duration = process_audio_bytes(st.session_state.recorded_audio)
                        st.session_state.results = results
                        st.session_state.duration = duration
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
                        st.error("Tip: Hum clearly for 10+ seconds with pitch variation")
        
        with col2:
            if st.button("🗑️ Record Again", use_container_width=True):
                st.session_state.recorded_audio = None
                st.session_state.results = None
                st.rerun()

except ImportError:
    st.error("""
    **Microphone recorder not installed**  
    Run this command in your terminal:
    ```
    pip install streamlit-mic-recorder==0.2.0
    ```
    """)
    st.code("pip install streamlit-mic-recorder==0.2.0", language="bash")

# Upload section
st.subheader("📤 Or Upload Audio File")
uploaded_file = st.file_uploader("Supported formats: WAV, MP3", type=["wav", "mp3"], label_visibility="collapsed")

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("🔍 Recognize Upload", type="primary", disabled=st.session_state.processing, use_container_width=True):
        with st.spinner("Analyzing melody..."):
            try:
                results, duration = process_audio_bytes(uploaded_file.read())
                st.session_state.results = results
                st.session_state.duration = duration
                st.rerun()
            except Exception as e:
                st.error(f"❌ {str(e)}")

# Results display
if st.session_state.results:
    st.divider()
    st.subheader("🏆 Top Matches")
    
    for i, match in enumerate(st.session_state.results):
        score = match["score"]
        confidence = max(0, min(100, int(100 - score * 15)))
        
        if score < 5.0:
            badge = "✅ HIGH CONFIDENCE"
            color = "#4CAF50"
        elif score < 8.0:
            badge = "⚠️ MEDIUM CONFIDENCE"
            color = "#FFA726"
        else:
            badge = "❓ LOW CONFIDENCE"
            color = "#EF5350"
        
        st.markdown(f"""
        <div style="background-color: {color}15; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 4px solid {color};">
            <h3 style="margin: 0; color: {color};">#{i+1} • {match['name']}</h3>
            <div style="display: flex; align-items: center; margin-top: 10px;">
                <div style="flex: 1; height: 8px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden;">
                    <div style="width: {min(100, confidence)}%; height: 100%; background-color: {color};"></div>
                </div>
                <span style="margin-left: 15px; font-weight: bold; color: {color};">{confidence}%</span>
            </div>
            <div style="margin-top: 5px; font-size: 0.9em; color: #666;">
                Similarity score: {score:.2f} • {badge}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption(f"⏱️ Processing time: {st.session_state.duration:.2f}s")

# Tips
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    - ✅ Hum **10-30 seconds** of the most recognizable part (chorus works best)
    - ✅ Hum with **clear pitch changes** (avoid monotone humming)
    - ✅ Record in a **quiet room** away from background noise
    - ✅ Hum in **any key** – system works even if you're off-key!
    - ❌ Avoid whispering, very short clips (<5 sec), or noisy environments
    """)

st.divider()
st.caption("🎵 Hum2Song Recognizer | No FFmpeg required • Uses WAV recording")