import shutil
import re
import json
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = Path(r'E:\user_personalization_tagged\Mamu')
DRY_RUN = False  # Set to False to apply changes
BLACKLIST_FILE = Path(r'E:\user_personalization_tagged\Mamu\blacklist_orphans.json')
# ---------------------

def standardize_and_validate(root_dir):
    if not root_dir.exists():
        print(f"Error: Path {root_dir} not found.")
        return

    mode = "DRY RUN" if DRY_RUN else "LIVE MODE"
    print(f"--- Standardizing and Validating: {mode} ---\n")
    
    blacklist = []

    # 1. Level 1: User Folders
    user_folders = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    for u_idx, user_folder in enumerate(user_folders):
        u_pad = str(u_idx).zfill(2)
        
        # Get sessions before renaming user folder
        sessions = sorted([d for d in user_folder.iterdir() if d.is_dir()])
        
        for s_idx, session_folder in enumerate(sessions):
            s_pad = str(s_idx).zfill(2)
            
            # --- PART A: Organize Tasks ---
            existing_test_dirs = sorted([
                d for d in session_folder.iterdir() 
                if d.is_dir() and not d.name.startswith("task_")
            ])

            # task_01, task_02, task_03
            for t_idx, test_dir in enumerate(existing_test_dirs, start=1):
                t_pad = str(t_idx).zfill(2)
                task_name = f"task_{u_pad}_{s_pad}_{t_pad}"
                if not DRY_RUN:
                    test_dir.rename(session_folder / task_name)
                else:
                    print(f"    [TASK] {test_dir.name} -> {task_name}")

            # task_00 (Calibration)
            loose_files = [f for f in session_folder.iterdir() if f.is_file()]
            if loose_files:
                task0_name = f"task_{u_pad}_{s_pad}_00"
                task0_dir = session_folder / task0_name
                if not DRY_RUN:
                    task0_dir.mkdir(exist_ok=True)
                    for f in loose_files:
                        shutil.move(str(f), str(task0_dir / f.name))
                else:
                    print(f"    [TASK] Moving loose files to {task0_name}")

            # --- PART B: Rename Files and Check Pairs ---
            current_tasks = [d for d in session_folder.iterdir() if d.is_dir()]
            
            for task_folder in current_tasks:
                # Extract index prefix (e.g., 00_01_02)
                folder_indices = task_folder.name.replace("task_", "")
                
                # Dictionary to track pairs in this task folder
                # Key: 'img-001', Value: [ext1, ext2]
                file_pairs = {}

                # Loop 1: Rename files and collect pairing info
                for file in list(task_folder.iterdir()):
                    if not file.is_file(): continue
                    
                    match = re.search(r"img-(\d+)", file.name)
                    if match:
                        img_id = match.group(1)
                        img_num_padded = img_id.zfill(3)
                        new_name = f"{folder_indices}_img-{img_num_padded}{file.suffix}"
                        
                        # Store for validation
                        if img_num_padded not in file_pairs:
                            file_pairs[img_num_padded] = []
                        file_pairs[img_num_padded].append(file.suffix.lower())

                        if not DRY_RUN:
                            file.rename(task_folder / new_name)
                        else:
                            print(f"      [FILE] {file.name} -> {new_name}")

                # Loop 2: Validate pairs (Check if both .png and .json exist)
                for img_num, extensions in file_pairs.items():
                    if '.png' not in extensions or '.json' not in extensions:
                        orphan_info = {
                            "path": str(task_folder),
                            "image_index": img_num,
                            "found_extensions": extensions,
                            "full_prefix": f"{folder_indices}_img-{img_num}"
                        }
                        blacklist.append(orphan_info)
                        print(f"      [!] BLACKLISTED: {folder_indices}_img-{img_num} (Missing pair)")

            # --- PART C: Rename Session Folder ---
            session_name = f"session_{u_pad}_{s_pad}"
            if not DRY_RUN:
                session_folder.rename(user_folder / session_name)
            else:
                print(f"  [SESSION] {session_folder.name} -> {session_name}")
            
        # --- PART D: Rename User Folder ---
        user_name = f"user_{u_pad}"
        if not DRY_RUN:
            user_folder.rename(root_dir / user_name)
            print(f"DONE: {user_name}")
        else:
            print(f"[USER] {user_folder.name} -> {user_name}")

    # Write Blacklist to file
    if blacklist:
        with open(BLACKLIST_FILE, 'w') as f:
            json.dump(blacklist, f, indent=4)
        print(f"\nFound {len(blacklist)} outliers. Details saved to: {BLACKLIST_FILE}")
    else:
        print("\nNo outliers found. All files have matching pairs!")

if __name__ == "__main__":
    standardize_and_validate(BASE_PATH)