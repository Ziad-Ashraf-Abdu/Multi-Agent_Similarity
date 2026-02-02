import json
import librosa
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time

app = FastAPI()

DATABASE_FILE = "melody_database.json"
TEMP_FOLDER = "temp"

os.makedirs(TEMP_FOLDER, exist_ok=True)

if os.path.exists(DATABASE_FILE):
    print("Loading melody database...")
    with open(DATABASE_FILE, 'r') as f:
        DATABASE = json.load(f)
    print(f"Database loaded with {len(DATABASE)} melodies!")
else:
    DATABASE = {}

def extract_hum_pitch(y, sr):
    # Use tighter range for hums (usually C3 to C6) to avoid noise errors
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
    )
    
    # Filter silence
    pitch_track = f0[voiced_flag]
    if len(pitch_track) == 0: return None
    
    # Hz -> MIDI
    midi_track = librosa.hz_to_midi(pitch_track)
    
    # Normalize (Zero-Center)
    midi_normalized = midi_track - np.mean(midi_track)
    
    return midi_normalized

def get_top_matches(hum_path, top_n=2):
    # 1. Load Hum
    y, sr = librosa.load(hum_path, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20) 
    
    # 2. Extract Melody Shape
    query_seq = extract_hum_pitch(y, sr)
    
    if query_seq is None:
        raise Exception("No pitch detected in hum! (Too quiet?)")
        
    # Downsample
    query_seq = query_seq[::10]
    
    # Reshape to 2D Column Vector
    query_seq = query_seq.reshape(-1, 1)
    
    results = []
    
    # 3. Compare Shapes
    for song_name, song_data in DATABASE.items():
        ref_seq = np.array(song_data)
        ref_seq = ref_seq.reshape(-1, 1)
        
        # Calculate Distance
        distance, path = fastdtw(query_seq, ref_seq, dist=euclidean, radius=15)
        normalized_dist = distance / len(path)
        
        print(f"{song_name}: {normalized_dist:.3f}")
        
        # Clean Name
        clean_name = song_name.replace("vocals.wav", "").replace("Vocals.wav", "")
        results.append({"name": clean_name, "score": normalized_dist})
            
    # 4. Sort by Score (Lowest is Best)
    results.sort(key=lambda x: x["score"])
    
    # Return top N
    return results[:top_n]

@app.post("/recognize")
async def recognize_hum(file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        start_time = time.time()
        
        # Get Top 2 Matches
        top_matches = get_top_matches(temp_path, top_n=2)
        duration = time.time() - start_time
        
        if not top_matches:
            return {"status": "error", "message": "Database is empty."}

        # Best match logic (Score < 5.0)
        best_match = top_matches[0]
        match_found = best_match["score"] < 5.0
        
        return {
            "status": "success",
            "best_match": best_match["name"] if match_found else "Unknown",
            "matches": top_matches,  # Returns list of top 2
            "processing_time": f"{duration:.2f}s"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)