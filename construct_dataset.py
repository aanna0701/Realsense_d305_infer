from genericpath import exists
import cv2
import numpy as np
import sys
import argparse
from pathlib import Path
import pandas as pd

# =========================================== Arguments ===========================================

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='Construct Depth/RGB dataset for FAS')
    parser.add_argument('--dataset_name', default="suprema_realsense", help='The name of dataset')
    parser.add_argument('--mode', default="train", choices=['train', 'test'],help='Set whether it is train or test')
    parser.add_argument('--video_dir', default='dirs', type=str, help='directory of videos')
    
    args = parser.parse_args()

    return args

# ============================================= main ==============================================

def main(video_paths):
    labels = pd.DataFrame(columns=['NAME', 'CLASS'])
    i = 0
    
    for path in video_paths:
                
        video_path = path # 동영상 파일 경로
        video_name = video_path.name
        
        label = video_name.split('_')[0]
        if not label in ['live', 'replay', 'print']:
            print(f'*** {video_name} cannot be processed, because it has invalid label!!!')
            continue
            
        print(f'{video_name} is being processed !!!')
        
        
        video_complete = video_path / 'complete.avi'
        video_cropped = video_path / 'cropped.avi'

        cap_complete = cv2.VideoCapture(str(video_complete)) 
        cap_cropped = cv2.VideoCapture(str(video_cropped)) 

        for dir in SAVE_DIRS:
            dir.mkdir(parents=True, exist_ok=True)

        while (cap_complete.isOpened() or cap_cropped.isOpened()):
            _, img_com = cap_complete.read()
            _, img_crop = cap_cropped.read()
            
            if img_com is None:
                break
                
            _, W, _ = np.shape(img_com)
            _, w, _ = np.shape(img_crop)

            frame = int(cap_complete.get(1))
            
            if(frame % 10 == 0):    
                cv2.imwrite(str(SAVE_DIRS[0] / f'{video_name}_{frame}.png'), img_com[:,:W//2,:])
                cv2.imwrite(str(SAVE_DIRS[1] / f'{video_name}_{frame}.png'), img_com[:,W//2:,:])
                cv2.imwrite(str(SAVE_DIRS[2] / f'{video_name}_{frame}.png'), img_crop[:,:w//2,:])
                cv2.imwrite(str(SAVE_DIRS[3] / f'{video_name}_{frame}.png'), img_crop[:,w//2:,:])   
                
                # Append labeling information to dataframe
                labels.loc[i] = [f'{video_name}_{frame}', 0 if label == 'live' else 1]
                i += 1
        
        cap_complete.release()
        cap_cropped.release()        
    
    # Convert dataframe to csv file and Save
    labels.to_csv(str(CSV_PATH), mode='w', header=False, index=False)
    print('Save csv file !!!')
    
# ==================================================================================================

if __name__ == "__main__":
    
    ########### Args ########### 
    global args
    args = parse_args()
    ############################
    
    
    ########### Parameter Settig ###########
    ## mode of dataset (train or test)
    MODE = args.mode
    
    ## Save paths
    # Path class ($dataset_name/)
    p = Path(args.dataset_name) 
    
    # csv file path
    CSV_PATH = p / (MODE + '_labels.csv')
    
    # images path
    SAVE_DIRS = [
        p / MODE / f'{MODE}_rgb',
        p / MODE / f'{MODE}_depth',
        p / MODE / f'{MODE}_rgb_cropped',
        p / MODE / f'{MODE}_depth_cropped'
    ]
    
    ## Input video paths
    # Path class ($video_dir/)
    p = Path(args.video_dir)
    # load directories
    video_dirs = [x for x in p.iterdir() if x.is_dir()]
    ########################################    
    
    
    main(video_dirs)