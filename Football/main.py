from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import glob
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

app = FastAPI(title="Sports Pattern Matching Server")

# ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PLAY_DATABASE = {}
DATA_FOLDER = "processed_datasets"


@app.on_event("startup")
async def load_database():
    print("Loading datasets into memory...")
    if not os.path.exists(DATA_FOLDER):
        print(f"WARNING: Folder '{DATA_FOLDER}' not found.")
        return

    files = glob.glob(os.path.join(DATA_FOLDER, "dataset_length_*.json"))

    if not files:
        print(f"WARNING: No dataset files found.")
        return

    count = 0
    for f_path in files:
        try:
            filename = os.path.basename(f_path)
            length_key = int(filename.split('_')[-1].split('.')[0])
            with open(f_path, 'r') as f:
                data = json.load(f)
                PLAY_DATABASE[length_key] = data
                count += len(data)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")

    print(f"Server Ready. Loaded {count} plays.")


# --- HELPER: PARSE UPLOADED FILE ---
def parse_uploaded_json_content(json_content):
    extracted_plays = []
    try:
        events = json.loads(json_content)
        events.sort(key=lambda x: (
            x.get('sequence') if x.get('sequence') is not None else 0,
            x.get('eventTime') if x.get('eventTime') is not None else 0
        ))

        current_team_id = None
        current_sequence = []
        current_metadata = []
        last_event_time = -999

        for event in events:
            try:
                game_event = event.get('gameEvents', {})
                team_id = game_event.get('teamId')
                event_time = event.get('eventTime', 0) or 0

                # METADATA EXTRACTION
                player_id = game_event.get('playerId')
                player_name = game_event.get('playerName', 'Unknown')
                team_name = game_event.get('teamName', str(team_id))

                game_type = game_event.get('gameEventType', '')
                poss_type = event.get('possessionEvents', {}).get('possessionEventType', '')
            except:
                continue

            time_gap = event_time - last_event_time
            if (team_id != current_team_id) or (time_gap > 30):
                if len(current_sequence) >= 2:
                    extracted_plays.append({
                        'trajectory': current_sequence,
                        'metadata': current_metadata,
                        'length': len(current_sequence),
                        'play_id_local': f"upload_{len(extracted_plays)}"
                    })
                current_sequence = []
                current_metadata = []
                current_team_id = team_id

            if event_time > 0: last_event_time = event_time

            # --- SMART PASS DETECTION ---
            is_pass = False

            if game_type and 'PASS' in game_type.upper():
                is_pass = True
            elif poss_type and ('PA' == poss_type.upper() or 'PASS' in poss_type.upper()):
                is_pass = True

            if is_pass:
                coords = None

                # Standard Home/Away Search
                if 'homePlayers' in event:
                    for p in event['homePlayers']:
                        if p['playerId'] == player_id:
                            coords = [p['x'], p['y']]
                            break
                        if coords is None: coords = [p['x'], p['y']]

                if not coords and 'awayPlayers' in event:
                    for p in event['awayPlayers']:
                        if p['playerId'] == player_id:
                            coords = [p['x'], p['y']]
                            break
                        if coords is None: coords = [p['x'], p['y']]

                if coords:
                    current_sequence.append(coords)
                    current_metadata.append({
                        'player': player_name,
                        'team': team_name
                    })

        # Flush last
        if len(current_sequence) >= 2:
            extracted_plays.append({
                'trajectory': current_sequence,
                'metadata': current_metadata,
                'length': len(current_sequence),
                'play_id_local': f"upload_{len(extracted_plays)}"
            })

    except Exception as e:
        print(f"Parsing Error: {e}")
        return []

    return extracted_plays


@app.post("/analyze_match")
async def analyze_match(
        file: UploadFile = File(...),
        match_mode: str = Form("absolute")  # Default to absolute
):
    try:
        content = await file.read()
        input_plays = parse_uploaded_json_content(content)

        if not input_plays:
            return {"status": "error", "message": "No valid plays found in uploaded file."}

        results = []

        for query_play in input_plays:
            length = query_play['length']
            q_traj = np.array(query_play['trajectory'])

            # Normalization for Relative Matching (Zero-Centering)
            q_comp = q_traj
            if match_mode == "relative":
                q_start = q_traj[0]
                q_comp = q_traj - q_start

            candidates = PLAY_DATABASE.get(length, [])
            best_matches = []

            if candidates:
                play_scores = []
                for db_play in candidates:
                    raw_traj = db_play['trajectory']

                    if len(raw_traj) > 0 and isinstance(raw_traj[0], dict):
                        db_coords = np.array([[p['x'], p['y']] for p in raw_traj])
                    else:
                        db_coords = np.array(raw_traj)

                    # Prepare DB Comparison Trajectory
                    db_comp = db_coords
                    if match_mode == "relative":
                        db_start = db_coords[0]
                        db_comp = db_coords - db_start

                    dist, _ = fastdtw(q_comp, db_comp, dist=euclidean)

                    play_scores.append({
                        "match_play_id": db_play['play_id'],
                        "similarity_score": dist,
                        "match_trajectory": raw_traj,
                        "match_metadata": db_play.get('metadata', [])
                    })

                play_scores.sort(key=lambda x: x['similarity_score'])
                best_matches = play_scores[:3]

            results.append({
                "uploaded_play_id": query_play['play_id_local'],
                "length": length,
                "input_trajectory": query_play['trajectory'],
                "input_metadata": query_play['metadata'],
                "matches": best_matches
            })

        return {
            "status": "success",
            "total_plays_analyzed": len(input_plays),
            "analysis_results": results
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))