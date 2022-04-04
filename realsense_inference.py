## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

def stream(pipeline, fas_model=None, th=220, alpha=0.03, SAVE_DIR='./results'):
    
    now = datetime.now()
    SAVE_DIR = os.path.join(SAVE_DIR, f'{now.month}_{now.day}_{now.hour}h_{now.minute}m_{now.second}s')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    """
        ** Description **
        Output rgb image and depth image in real time
        
        ** Args **
        pipline     : stream pipeline
        fas_model   : Face-Anit Spoofing Model
        th          : Thresholding parameter, adjusting the acceptance distance for detph images
        alpha       : Depth image scaling parameter, adjusts the degree of differentiation according to depth
    
    """
    
    i = 0
    j = 0
    pred = 0
    TEXT = ['Bonafide', 'Spoofing']
    
    while True:    
                
        # -------------------------------------------
        # Getting raw depth and rgb image
        # -------------------------------------------
        
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # -------------------------------------------
        # Post-processing
        # -------------------------------------------
        
        # cv2.convertScaleAbs(depth_image, alpha=alpha): return gray image of range [0, 255]
        depth_scaled = cv2.convertScaleAbs(depth_image, alpha=alpha)
        
        # histogram equlization
        depth_equal = cv2.equalizeHist(depth_scaled)
        
        # cv2.cvtColor: for visualization, convert 1-d image to 3-d image
        depth_colormap = cv2.cvtColor(depth_equal, cv2.COLOR_GRAY2BGR)
                
        if fas_model is not None:
            # output: spoof label
            pred = 1
        
        # -------------------------------------------
        # Postprocessing using thresholding
        # -------------------------------------------
        # depth_colormap[depth_equal > th] = 0.
        
        
        H, W, _ = np.shape(color_image)
        
        # ROI
        cv2.rectangle(color_image, (W//2-91, H//2-91), (W//2+91, H//2+91), (255, 0, 0), 1)
        
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        # To quick, press 'q' key
        # To save, press 'space bar'
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1)  == ord('q'):
            break
        elif cv2.waitKey(1)  == 32:
            
            i += 1
            
            save_dir = os.path.join(SAVE_DIR, str(i))
            print(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            
            # concat = np.concatenate([color_image, depth_colormap], axis=-1)
            concat = np.concatenate([cv2.putText(color_image, TEXT[pred], (0,0)), depth_colormap], axis=-1)
            concat_crop = concat[H//2-90:H//2+90, W//2-90:W//2+90]
            
            cv2.imwrite(os.path.join(save_dir, f'{i}_full.png'), images)
            cv2.imwrite(os.path.join(save_dir, f'{i}_crop_rgb.png'), concat_crop[:,:,:3])
            cv2.imwrite(os.path.join(save_dir, f'{i}_crop_depth.png'), concat_crop[:,:,3:])
            
            del concat, concat_crop
            
            ### check effectiveness of processing 
            depth_base = cv2.cvtColor(depth_scaled, cv2.COLOR_GRAY2BGR)
            depth_hist = cv2.cvtColor(depth_equal, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(depth_hist, (0, 0), (W, H), (255, 0, 0), 2)
            concat_depths = np.hstack([depth_base, depth_hist])
            cv2.imwrite(os.path.join(save_dir, f'{j}_concat_depths.png'), concat_depths)
            
            del depth_base, depth_hist
            
            print('Capture')

    
def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    # Start streaming
    pipeline.start(config)

    try:
        # -------------------------
        # Main function
        # -------------------------
        stream(pipeline=pipeline)    
        exit()
        
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()