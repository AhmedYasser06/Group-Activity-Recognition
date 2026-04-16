# ============================================================
#  Run this cell ONCE before baseline_b8_kaggle.py.
#  Output: /kaggle/working/annot_all.pkl
# ============================================================

import os
import pickle
from typing import List

# ─────────────────────────────────────────────────────────────
#  !! UPDATE THIS PATH IF NEEDED !!
#  Must be the folder that directly contains:
#    videos/
#    volleyball_tracking_annotation/
# ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_"

VIDEOS_ROOT  = f"{DATASET_ROOT}/videos"
ANNOT_ROOT   = f"{DATASET_ROOT}/volleyball_tracking_annotation"
OUTPUT_PATH  = "/kaggle/working/annot_all.pkl"


# ── BoxInfo  (exact copy of professor's boxinfo.py) ──────────
class BoxInfo:
    def __init__(self, line):
        words          = line.split()
        self.category  = words.pop()                        # last token = action label
        words          = [int(string) for string in words]  # everything else is int
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box       = x1, y1, x2, y2
        self.frame_ID  = frame_ID
        self.lost      = lost
        self.grouping  = grouping
        self.generated = generated

    def __repr__(self):
        return (f"BoxInfo(player={self.player_ID}, frame={self.frame_ID}, "
                f"box={self.box}, category='{self.category}')")


# ── exact copy of professor's load_tracking_annot() ──────────
def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes    = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        # create view from frame → boxes
        # keep the middle 9 frames only  (skip first 5, last 6)
        for player_ID, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []
                frame_boxes_dct[box_info.frame_ID].append(box_info)

    return frame_boxes_dct


# ── exact copy of professor's load_video_annot() ─────────────
def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items    = line.strip().split(' ')[:2]
            if len(items) < 2:
                continue
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

    return clip_category_dct


# ── exact copy of professor's load_volleyball_dataset() ──────
def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot       = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            assert clip_dir in clip_category_dct, \
                f"clip_dir '{clip_dir}' not found in annotations.txt of video '{video_dir}'"

            annot_file      = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)

            clip_annot[clip_dir] = {
                'category'       : clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct,
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


# ── save pkl  (exact copy of professor's create_pkl_version) ─
def create_pkl_version():
    print("=" * 60)
    print("  Building annot_all.pkl")
    print(f"  videos root : {VIDEOS_ROOT}")
    print(f"  annot root  : {ANNOT_ROOT}")
    print(f"  output      : {OUTPUT_PATH}")
    print("=" * 60)

    videos_annot = load_volleyball_dataset(VIDEOS_ROOT, ANNOT_ROOT)

    with open(OUTPUT_PATH, 'wb') as file:
        pickle.dump(videos_annot, file)

    total_clips = sum(len(v) for v in videos_annot.values())
    print()
    print("=" * 60)
    print(f"  ✅  Done!  Saved → {OUTPUT_PATH}")
    print(f"  Videos : {len(videos_annot)}")
    print(f"  Clips  : {total_clips}")
    print("=" * 60)


# ── sanity check  (exact copy of professor's test_pkl_version) 
def test_pkl_version():
    print("\n── Sanity Check ─────────────────────────────────────")
    with open(OUTPUT_PATH, 'rb') as file:
        videos_annot = pickle.load(file)

    # professor's exact test
    boxes: List[BoxInfo] = videos_annot['0']['13456']['frame_boxes_dct'][13454]
    print(f"  videos_annot['0']['13456']['frame_boxes_dct'][13454]")
    print(f"  boxes[0].category = {boxes[0].category}")
    print(f"  boxes[0].box      = {boxes[0].box}")
    print("── End Sanity Check ─────────────────────────────────")


# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    create_pkl_version()
    test_pkl_version()

    print()
    print("  Next: in baseline_b8_kaggle.py set")
    print(f'  ANNOT_PATH = "{OUTPUT_PATH}"')
