import os
import cv2
from src import CAM_RES, CONFIG_FILE
from src.StereoCam import StereoCam
from src.utils import read_calibration_params, calibrate_stereo_camera, save_logs

import time

DLINES_FLAG  = CONFIG_FILE.get( "DRAW_REF_LINES", True )
RES_W, RES_H = CAM_RES.get( CONFIG_FILE.get( "CAM_RES", "HD" ), (1280, 720) )

#HL, HR = read_homography( r".\homography", False )
mapL, mapR, Q = calibrate_stereo_camera( *read_calibration_params( r".\homography" ) )

cam = StereoCam( mapL, mapR, Q, RES_W=RES_W, RES_H=RES_H, rectify=True )

cv2.namedWindow( "Left" )
cv2.setMouseCallback( "Left", cam.mouseCallback )

record_flag = False
i = 1

if not os.path.exists(r".\results"): os.makedirs(r".\results") 
with open( f".\\results\\results_{time.strftime('%b-%d-%Y_%H%M', time.localtime())}.csv", "w" ) as pFile:
    while True:
        s = time.time()
        FL, FR = cam.next()
        
        if record_flag: save_logs( pFile, cam.tracker.stereo_reconstruct() )

        cv2.putText(FL, str( round( 1 / ( time.time() - s ), 2 ) ), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  1, (250,250,250), 2, cv2.LINE_AA) 

        if DLINES_FLAG:
            cv2.line( FL, (0,25), (RES_W,25), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,25), (RES_W,25), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FL, (0,75), (RES_W,75), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,75), (RES_W,75), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FL, (0,125), (RES_W,125), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,125), (RES_W,125), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FL, (0,175), (RES_W,175), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,175), (RES_W,175), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FL, (0,225), (RES_W,225), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,225), (RES_W,225), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FL, (0,275), (RES_W,275), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,275), (RES_W,275), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FL, (0,325), (RES_W,325), (0,255,0), 1, cv2.LINE_AA )
            cv2.line( FR, (0,325), (RES_W,325), (0,255,0), 1, cv2.LINE_AA )

        cv2.imshow( "Left", FL )
        cv2.imshow( "Right", FR )
        k = cv2.waitKey( 1 )

        if   k == ord( "q" ): break
        elif k == ord( "c" ): cv2.imwrite( f".\media\Left_{i}.png", FL ); cv2.imwrite( f".\media\Right_{i}.png", FR ); i = i + 1
        elif k == ord( "n" ): cam.startCallback(); record_flag = True
        

