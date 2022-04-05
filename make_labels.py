import cv2
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

# =========================================== Arguments ===========================================

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='Make labels(csv format) for FAS')
    parser.add_argument('--input_dir', default=r'suprema_realsense\train\train_rgb', help='Directory of inputs')
    parser.add_argument('--save_dir', default="suprema_realsense", help='The name of dataset')
    parser.add_argument('--mode', default="train", choices=['train', 'test'],help='Set whether it is train or test')
    
    args = parser.parse_args()

    return args

# ============================================= main ==============================================

def main(video_paths):
    ## data frame object 
    labels = pd.DataFrame(columns=['NAME', 'CLASS'])
    i = 0
    
    for path in video_paths:

        video_name = path.name
        
        label = video_name.split('_')[0]
        if not label in ['live', 'replay', 'print']:
            print(f'*** {video_name} cannot be processed, because it has invalid label!!!')
            continue
            
        print(f'{video_name} is being processed !!!')
        
        video_complete = path / 'complete.avi'
        video_cropped = path / 'cropped.avi'

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
    ## Save paths
    # Path class ($save_dir/)
    p = Path(args.save_dir)
    # Save path
    SAVE_PATH = p / f'{args.mode}_labels.csv'

    ## Input images
    # Path class ($input_dir/)
    p = Path(args.input_dir)     
    # extract image names
    img_names = [x.name for x in p.glob('*.png')]
    ########################################
    
    
    ## data frame object 
    labels = pd.DataFrame(columns=['NAME', 'CLASS'])
    i = 0
    
    for img_name in img_names:
        label = img_name.split('_')[0]
        if not label in ['live', 'replay', 'print']:
            print(f'*** {img_name} cannot be processed, because it has invalid label!!!')
            continue
        
        print(f'{img_name} is being processed !!!')
        
        # Append labeling information to dataframe
        labels.loc[i] = [f'{img_name}', 0 if label == 'live' else 1]
        i += 1
        
    # Convert dataframe to csv file and Save
    labels.to_csv(str(SAVE_PATH), mode='w', header=False, index=False)
    print('Save csv file !!!')