import os
import shutil

# Paths for raw VisDrone and the new MOT-formatted output
VISDRONE_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-test-dev"
OUT_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-test-dev"

os.makedirs(OUT_DIR, exist_ok=True)
seqs = [s for s in os.listdir(os.path.join(VISDRONE_DIR, 'sequences')) if not s.startswith('.')]

for seq in seqs:
    seq_path = os.path.join(VISDRONE_DIR, 'sequences', seq)
    ann_path = os.path.join(VISDRONE_DIR, 'annotations', seq + '.txt')

    out_seq_dir = os.path.join(OUT_DIR, seq)
    out_img_dir = os.path.join(out_seq_dir, 'img1')
    out_gt_dir = os.path.join(out_seq_dir, 'gt')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    print(f"Copying images for sequence: {seq}...")
    for img in os.listdir(seq_path):
        if img.endswith('.jpg'):
            shutil.copy(os.path.join(seq_path, img), os.path.join(out_img_dir, img))

    print(f"Converting annotations for sequence: {seq}...")
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            lines = f.readlines()

        with open(os.path.join(out_gt_dir, 'gt.txt'), 'w') as f:
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 8: 
                    continue
                
                # VisDrone format: frame_id, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category
                frame_id = parts[0]
                target_id = parts[1]
                x, y, w, h = parts[2], parts[3], parts[4], parts[5]
                score = parts[6]
                cls_id = parts[7]
                
                # Write to MOT format. Forcing visibility to 1.
                mot_line = f"{frame_id},{target_id},{x},{y},{w},{h},{score},{cls_id},1\n"
                f.write(mot_line)

print("\nSuccess! VisDrone to MOT conversion is complete.")