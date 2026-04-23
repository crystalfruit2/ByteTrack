import os
import json

def convert_mot_to_coco(data_root, split='test'):
    out_dir = os.path.join(data_root, 'annotations')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{split}.json')

    out = {
        "videos": [],        # <-- required by ByteTrack
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pedestrian"}]
    }

    image_id = 1
    ann_id = 1
    video_id = 1

    sequences = sorted([
        s for s in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, s)) and s != 'annotations'
    ])

    for seq in sequences:
        print(f"Processing: {seq}")
        seq_path = os.path.join(data_root, seq)
        img_dir = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')

        # Read seqinfo.ini
        ini_path = os.path.join(seq_path, 'seqinfo.ini')
        seq_info = {}
        with open(ini_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    seq_info[k.strip()] = v.strip()

        width = int(seq_info.get('imWidth', 1920))
        height = int(seq_info.get('imHeight', 1080))
        img_ext = seq_info.get('imExt', '.jpg')

        # Add video entry
        out['videos'].append({
            "id": video_id,
            "file_name": seq
        })

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(img_ext)
        ])
        seq_length = len(img_files)

        # Track first image id of this sequence (for first_frame_image_id)
        first_image_id = image_id

        frame_to_image_id = {}

        for img_file in img_files:
            frame_num = int(os.path.splitext(img_file)[0])
            frame_to_image_id[frame_num] = image_id

            out['images'].append({
                "id": image_id,
                "video_id": video_id,           # <-- was missing
                "file_name": f"{seq}/img1/{img_file}",
                "width": width,
                "height": height,
                "frame_id": frame_num,
                "seq_length": seq_length,
                "first_frame_image_id": first_image_id  # <-- was wrong before
            })
            image_id += 1

        # Read GT annotations
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    frame = int(parts[0])
                    tid   = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), \
                                 float(parts[4]), float(parts[5])

                    if w <= 0 or h <= 0:
                        continue
                    if frame not in frame_to_image_id:
                        continue

                    out['annotations'].append({
                        "id": ann_id,
                        "image_id": frame_to_image_id[frame],
                        "video_id": video_id,   # <-- also add here
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": tid,
                        "visibility": 1.0
                    })
                    ann_id += 1

        video_id += 1

    with open(out_path, 'w') as f:
        json.dump(out, f)

    print(f"\nDone! {len(out['videos'])} videos, {len(out['images'])} images, "
          f"{len(out['annotations'])} annotations -> {out_path}")


if __name__ == '__main__':
    DATA_ROOT = r'C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val'
    convert_mot_to_coco(DATA_ROOT, split='test')