import os
import json
import cv2

# Define paths
MOT_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-test-dev"
OUT_JSON = os.path.join(MOT_DIR, "test-dev.json")

# Initialize the COCO dictionary structure
dataset = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": str(i)} for i in range(1, 13)] # VisDrone classes
}

seqs = [s for s in os.listdir(MOT_DIR) if os.path.isdir(os.path.join(MOT_DIR, s))]

img_id = 1
ann_id = 1

for seq in seqs:
    print(f"Processing sequence {seq} for COCO JSON...")
    img_dir = os.path.join(MOT_DIR, seq, 'img1')
    gt_path = os.path.join(MOT_DIR, seq, 'gt', 'gt.txt')

    frame_to_img_id = {}
    
    # Process Images
    images = sorted(os.listdir(img_dir))
    for frame_idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        dataset["images"].append({
            "file_name": f"{seq}/img1/{img_name}",
            "id": img_id,
            "frame_id": frame_idx,
            "sequence": seq,
            "width": width,
            "height": height
        })
        frame_to_img_id[frame_idx] = img_id
        img_id += 1

    # Process Annotations
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                score = float(parts[6])
                cls_id = int(parts[7])
                
                if frame_idx not in frame_to_img_id: 
                    continue
                
                dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": frame_to_img_id[frame_idx],
                    "category_id": cls_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "track_id": target_id
                })
                ann_id += 1

# Save the JSON file
with open(OUT_JSON, 'w') as f:
    json.dump(dataset, f)

print(f"\nSuccess! COCO JSON saved to: {OUT_JSON}")