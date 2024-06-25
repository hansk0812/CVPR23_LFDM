# Preprocess MHAD dataset to generate cropped video-text pairs
# Using depth map to generate bounding box for cropping videos
import os
import scipy.io
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio


def import_data(data_dir_path, subject_id, descriptions):
    filename = os.path.join(data_dir_path, '%02d__%s.mp4' % (subject_id, descriptions))

    frame_list = []
    if Path(filename).is_file():
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = frame.astype(np.uint8)
                frame = cv2.resize(frame, (256,256))
                frame_list.append(frame[:, :, ::-1])
                if len(frame_list) > 240:
                    break
            else:
                break
        return frame_list
    else:
        return None


def analyse_FFPP():
    data_dir = "/data/hfn5052/text2motion/dataset/MHAD/crop_image"
    video_name_list = os.listdir(data_dir)
    video_name_list.sort()
    video_path_list = [os.path.join(data_dir, x) for x in video_name_list]
    min_num_frame = 1e4
    max_num_frame = -1
    min_video_name = None
    max_video_name = None
    num_frame_list = []
    for video_path in video_path_list:
        frame_name_list = os.listdir(video_path)
        num_frame = len(frame_name_list)
        if num_frame < min_num_frame:
            min_num_frame = num_frame
            min_video_name = video_path
        if num_frame > max_num_frame:
            max_num_frame = num_frame
            max_video_name = video_path
        num_frame_list.append(num_frame)
    num_frame_list = np.array(num_frame_list)
    print(min_num_frame, min_video_name)
    print(max_num_frame, max_video_name)
    print(num_frame_list.min(), num_frame_list.max(), num_frame_list.mean())
    # 32 /data/hfn5052/text2motion/MHAD/crop_image/a15_s2_t4
    # 96 /data/hfn5052/text2motion/MHAD/crop_image/a21_s8_t1


def split_train_test_FFPP():
    male = [8, 9, 10, 11, 13, 14]
    female = [1, 2, 3, 4, 5, 6, 7, 12, 15]
    train_sub = [8, 9, 10, 1, 2, 3, 4]
    test_sub = [11, 13, 14, 5, 6, 7, 12, 15]

    return train_sub, test_sub


if __name__ == "__main__":

    # 16 subjects and 16 actions
    SUBJECT_IDS = list(range(1, 17))
    ACTIONS = [
            "exit_phone_room",
            "hugging_happy",                      
            "kitchen_pan",                        
            "kitchen_still",                      
            "meeting_serious",                    
            "outside_talking_pan_laughing",
            "outside_talking_still_laughing",     
            "podium_speech_happy",                
            "talking_against_wall", 
            "secret_conversation",
            "talking_angry_couch",
            "walk_down_hall_angry",               
            "walking_and_outside_surprised",      
            "walking_down_indoor_hall_disgust",
            "walking_down_street_outside_angry",
            "walking_outside_cafe_disgusted"
            ]

    for subject_id in SUBJECT_IDS:
        for action in ACTIONS:
            print (subject_id, action)
            import_data("/home/hans/CVPR23_LFDM/FF++/real", subject_id, action)

    # the train/test split can be found in:
    split_train_test_FFPP()

    pass
