import os
import shutil

def convert_visdrone_to_mot(visdrone_dir, output_dir):
    """
    Converts VisDrone MOT dataset to MOTChallenge format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # VisDrone sequences are in the 'sequences' folder, annotations in 'annotations'
    seq_dir = os.path.join(visdrone_dir, 'sequences')
    ann_dir = os.path.join(visdrone_dir, 'annotations')

    sequences = [s for s in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, s))]

    for seq in sequences:
        print(f"Converting sequence: {seq}")
        
        # 1. Create MOT format directory structure
        seq_output_dir = os.path.join(output_dir, seq)
        img1_dir = os.path.join(seq_output_dir, 'img1')
        gt_dir = os.path.join(seq_output_dir, 'gt')
        
        os.makedirs(img1_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        # 2. Copy images to img1/
        src_images = os.path.join(seq_dir, seq)
        for img_file in os.listdir(src_images):
            if img_file.endswith('.jpg'):
                shutil.copy(os.path.join(src_images, img_file), os.path.join(img1_dir, img_file))

        # 3. Read VisDrone annotations and convert to MOT format
        # VisDrone: <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        # MOT: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        visdrone_anno_path = os.path.join(ann_dir, seq + '.txt')
        mot_anno_path = os.path.join(gt_dir, 'gt.txt')

        with open(visdrone_anno_path, 'r') as f_in, open(mot_anno_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                
                frame, tid, x, y, w, h, score, cat = parts[:8]
                
                # Filter out ignored regions (cat 0) or others (cat 11) if desired. 
                # For baseline UAV tracking, keeping classes 1-10 is standard.
                if int(cat) in [0, 11]:
                    continue
                    
                # Write MOT format (ignoring trunc/occ, setting conf to 1, x,y,z to -1)
                f_out.write(f"{frame},{tid},{x},{y},{w},{h},1,1,1\n")

        # 4. Generate seqinfo.ini (Crucial for ByteTrack evaluator)
        img_list = sorted(os.listdir(img1_dir))
        if img_list:
            import cv2
            sample_img = cv2.imread(os.path.join(img1_dir, img_list[0]))
            height, width, _ = sample_img.shape
            seq_length = len(img_list)
            
            ini_content = f"""[Sequence]
name={seq}
imDir=img1
frameRate=30
seqLength={seq_length}
imWidth={width}
imHeight={height}
imExt=.jpg
"""
            with open(os.path.join(seq_output_dir, 'seqinfo.ini'), 'w') as f_ini:
                f_ini.write(ini_content)

if __name__ == '__main__':
    # Use 'r' before the string to treat backslashes as literal characters
    RAW_VISDRONE_PATH = r'C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val' 
    
    # We will save the MOT formatted data to a new folder on your desktop
    OUTPUT_MOT_PATH = r'C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val'
    
    convert_visdrone_to_mot(RAW_VISDRONE_PATH, OUTPUT_MOT_PATH)
    print(f"Conversion Complete! Your files are at: {OUTPUT_MOT_PATH}")