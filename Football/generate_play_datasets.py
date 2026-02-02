import os
import glob
import json
import numpy as np

# CONFIGURATION: Update this to your folder path
INPUT_FOLDER = "football dataset" 
OUTPUT_FOLDER = "processed_datasets"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- REUSING THE ROBUST PARSING LOGIC ---
def get_player_coordinates(event_data, player_id):
    found_coords = None
    if 'homePlayers' in event_data:
        for player in event_data['homePlayers']:
            if player['playerId'] == player_id:
                return player['x'], player['y']
            if found_coords is None: found_coords = (player['x'], player['y'])
    if 'awayPlayers' in event_data:
        for player in event_data['awayPlayers']:
            if player['playerId'] == player_id:
                return player['x'], player['y']
            if found_coords is None: found_coords = (player['x'], player['y'])
    return found_coords

def extract_plays_from_file(file_path):
    """
    Extracts plays from a single JSON file using Team ID + Time Gap logic.
    """
    file_plays = []
    filename = os.path.basename(file_path)
    
    with open(file_path, 'r') as f:
        try:
            events = json.load(f)
        except:
            return []

    # Sort to ensure chronological order
    events.sort(key=lambda x: (
        x.get('sequence') if x.get('sequence') is not None else 0, 
        x.get('eventTime') if x.get('eventTime') is not None else 0
    ))

    current_team_id = None
    current_sequence = []
    last_event_time = -999

    for event in events:
        try:
            game_event = event.get('gameEvents', {})
            team_id = game_event.get('teamId')
            event_time = event.get('eventTime', 0)
            if event_time is None: event_time = 0
            
            game_type = game_event.get('gameEventType', '')
            poss_type = event.get('possessionEvents', {}).get('possessionEventType', '')
            player_id = game_event.get('playerId')
        except:
            continue

        # Logic: New chain if Team changes OR Time gap > 30s
        time_gap = event_time - last_event_time
        if (team_id != current_team_id) or (time_gap > 30):
            if len(current_sequence) >= 2:
                file_plays.append({
                    'play_id': f"{filename}_{len(file_plays)}",
                    'trajectory': current_sequence, # Keep as list for JSON serialization
                    'length': len(current_sequence),
                    'team_id': current_team_id
                })
            current_sequence = []
            current_team_id = team_id
        
        if event_time > 0: last_event_time = event_time

        # Check for Pass
        is_pass = False
        if game_type and 'PASS' in game_type.upper(): is_pass = True
        elif poss_type and ('PA' == poss_type.upper() or 'PASS' in poss_type.upper()): is_pass = True

        if is_pass:
            coords = get_player_coordinates(event, player_id)
            if coords:
                current_sequence.append(coords)
    
    # Flush last sequence
    if len(current_sequence) >= 2:
        file_plays.append({
            'play_id': f"{filename}_{len(file_plays)}",
            'trajectory': current_sequence,
            'length': len(current_sequence),
            'team_id': current_team_id
        })
            
    return file_plays

# --- MAIN EXECUTION ---
def main():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    print(f"Processing {len(files)} files...")
    
    # Buckets for plays: { 2: [play, play], 3: [play, play] ... }
    play_buckets = {}
    
    total_plays = 0
    
    for f in files:
        plays = extract_plays_from_file(f)
        for p in plays:
            length = p['length']
            if length not in play_buckets:
                play_buckets[length] = []
            play_buckets[length].append(p)
            total_plays += 1
            
    print(f"Extraction complete. Total plays: {total_plays}")
    print("Saving categorized datasets...")
    
    # Save each bucket to a separate file
    for length, plays in play_buckets.items():
        output_file = os.path.join(OUTPUT_FOLDER, f"dataset_length_{length}.json")
        with open(output_file, 'w') as f:
            json.dump(plays, f)
        print(f"  -> Saved {len(plays)} plays to {output_file}")

if __name__ == "__main__":
    main()