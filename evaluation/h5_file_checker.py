import h5py
import numpy as np
import cv2
from pathlib import Path

def verify_h5_integrity(h5_path):
    print(f"--- Verifying H5 File: {h5_path} ---\n")
    
    try:
        with h5py.File(h5_path, 'r') as h5:
            # 1. Check for expected Top-Level Datasets
            expected_datasets = ['face_patch', 'face_gaze']
            for ds in expected_datasets:
                if ds in h5:
                    shape = h5[ds].shape
                    print(f"[OK] Dataset '{ds}' found. Shape: {shape}")
                else:
                    print(f"[ERROR] Missing dataset: {ds}")

            # 2. Check for Metadata Group
            if 'meta' in h5:
                print("[OK] 'meta' group found.")
                for m_ds in ['user', 'session', 'task', 'image_name']:
                    if m_ds in h5['meta']:
                        sample_val = h5['meta'][m_ds][0].decode('utf-8') if len(h5['meta'][m_ds]) > 0 else "EMPTY"
                        print(f"    - Found '{m_ds}'. Sample value: {sample_val}")
                    else:
                        print(f"    - [ERROR] Missing metadata: {m_ds}")
            else:
                print("[ERROR] 'meta' group missing!")

            # 3. Verify Data Content & Types
            if len(h5['face_patch']) > 0:
                # Check if face_patch is uint8 [0, 255]
                patch_dtype = h5['face_patch'].dtype
                print(f"[OK] 'face_patch' dtype: {patch_dtype}")
                
                # Check if face_gaze has 2 columns (pitch, yaw)
                gaze_cols = h5['face_gaze'].shape[1]
                print(f"[OK] 'face_gaze' has {gaze_cols} columns.")

                # Verify 'pog_px' was converted to integer (if you stored it as a dataset)
                # Note: If 'pog_px' is inside a metadata attributes or separate dataset:
                if 'pog_px' in h5:
                    pos_sample = h5['pog_px'][0]
                    if np.issubdtype(pos_sample.dtype, np.integer):
                        print(f"[OK] 'pog_px' is stored as Integer: {pos_sample}")
                    else:
                        print(f"[WARNING] 'pog_px' is NOT an integer. Dtype: {pos_sample.dtype}")

            print(f"\nTotal Samples Verified: {len(h5['face_patch'])}")

    except Exception as e:
        print(f"[CRITICAL ERROR] Could not read H5 file: {e}")

def visualize_normalization(h5_path, num_samples=5):
    """
    Displays random samples from the H5 file to verify normalization.
    """
    print(f"--- Visualizing {num_samples} samples from: {h5_path} ---")
    
    with h5py.File(h5_path, 'r') as h5:
        total_samples = h5['face_patch'].shape[0]
        if total_samples == 0:
            print("H5 file is empty!")
            return

        # Pick random indices
        indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

        for idx in indices:
            # Load data
            img = h5['face_patch'][idx] # Already uint8 224x224
            gaze = h5['face_gaze'][idx] # [pitch, yaw]
            
            # Metadata
            user = h5['meta/user'][idx].decode('utf-8')
            sess = h5['meta/session'][idx].decode('utf-8')
            task = h5['meta/task'][idx].decode('utf-8')

            # Create a display copy
            # Note: face_patch was saved as BGR in the script
            disp_img = cv2.flip(img.copy(), 0)
            
            # Overlay info
            info_text = f"U:{user} S:{sess} T:{task} | P:{gaze[0]:.2f} Y:{gaze[1]:.2f}"
            cv2.putText(disp_img, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Normalization Check (Press any key)", disp_img)
            key = cv2.waitKey(0)
            if key == 27: # ESC to exit
                break

    cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent
    H5_PATH = ROOT_DIR / 'temp' / 'full_dataset.h5'

    verify_h5_integrity(H5_PATH)
    visualize_normalization(H5_PATH)