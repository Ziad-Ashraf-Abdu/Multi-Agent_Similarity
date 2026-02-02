import os
import json
import librosa
import numpy as np

# CONFIGURATION
# Point this to the folder where your 'Enemyvocals.wav', etc. are located
SONG_FOLDER = "songs" 
OUTPUT_FILE = "melody_database.json"

def extract_pitch_contour(y, sr):
    # 1. Extract Pitch (F0) using pyin
    # fmin=Note C2, fmax=Note C7 (Covers human vocal range)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    
    # 2. Filter Silence (Keep only voiced parts)
    # We "squeeze" the melody by removing NaNs (silence).
    # This creates a pure sequence of notes: [C, C, G, G, A, A, G]
    pitch_track = f0[voiced_flag]
    
    if len(pitch_track) == 0:
        return None
        
    # 3. Convert Hz to MIDI Note Numbers (Linear Pitch)
    # This makes the math easier: C to C# is always a difference of 1.0
    midi_track = librosa.hz_to_midi(pitch_track)
    
    # 4. Normalize (Zero-Center)
    # Subtract the mean pitch. This makes the melody "Key Invariant".
    # If the song is in D major and you hum in C major, this makes them look identical.
    midi_normalized = midi_track - np.mean(midi_track)
    
    return midi_normalized

def process_songs():
    database = {}
    
    # Get all .wav files (assuming these are your vocal stems)
    files = [f for f in os.listdir(SONG_FOLDER) if f.endswith('.wav')]
    print(f"Found {len(files)} vocal tracks. extracting melody contours...")

    for file in files:
        path = os.path.join(SONG_FOLDER, file)
        print(f"Processing: {file}...")
        
        try:
            # Load Audio (Mono)
            y, sr = librosa.load(path, mono=True)
            
            # Extract the Melody Line
            melody = extract_pitch_contour(y, sr)
            
            if melody is not None:
                # Downsample heavily for storage/speed (keep 1 in 10 points)
                # Pitch tracks are dense, we don't need 100 points per second.
                database[file] = melody[::10].tolist() 
            else:
                print(f"⚠️ Warning: No melody detected in {file}")
            
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(database, f)
    
    print(f"✅ Melody Database built! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_songs()