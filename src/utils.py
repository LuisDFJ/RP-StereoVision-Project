import os
import struct
import numpy as np
import cv2

from src import CONFIG_FILE, CAM_RES
RES_W, RES_H = CAM_RES.get( CONFIG_FILE.get( "CAM_RES", "HD" ), (1280, 720) )

def read_homography( path : str, debug = False ):
    if not debug:
        with open( os.path.join( path, "HL.bin" ), 'rb') as pfile:
            file = pfile.read()
            fmt = f'@{int(len(file)/8)}d'
            HL = struct.unpack( fmt, file )
        
        with open( os.path.join( path, "HR.bin" ), 'rb') as pfile:
            file = pfile.read()
            fmt = f'@{int(len(file)/8)}d'
            HR = struct.unpack( fmt, file )

        HL = np.array( HL, dtype=np.float64 ).reshape( (3,3) ).T
        HR = np.array( HR, dtype=np.float64 ).reshape( (3,3) ).T
    else:
        HL = np.eye(3)
        HR = np.eye(3)
    return HL, HR

def read_calibration_params( path : str ):
    def _read( var ):
        with open( os.path.join( path, f"{var}.bin" ), 'rb') as pfile:
            file = pfile.read()
            fmt = f'@{int(len(file)/8)}d'
        return struct.unpack( fmt, file )
    def _reshape( var, size, flag = True ):
        if flag: return np.array( var, dtype=np.float64 ).reshape( size ).T
        else: return np.array( var, dtype=np.float64 ).reshape( size )

    
    KL = _read( "KL" ); KR = _read( "KR" )
    dL = _read( "dL" ); dR = _read( "dR" )
    T = _read( "T" ); R = _read( "R" )

    KL = _reshape( KL, (3,3) )
    KR = _reshape( KR, (3,3) )
    R = _reshape( R, (3,3), True )
    T = _reshape( T, (3,1), False )
    dL = _reshape( dL, (1,5), False )
    dR = _reshape( dR, (1,5), False )
    
    return KL, KR, R, T, dL, dR


def calibrate_stereo_camera( KL, KR, R, T, dL, dR ):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify( KL, dL, KR, dR, (RES_W,RES_H), R, T, alpha=0 )

    m1L, m2L = cv2.initUndistortRectifyMap( KL, dL, R1, P1, (RES_W,RES_H), cv2.CV_32FC1 )
    m1R, m2R = cv2.initUndistortRectifyMap( KR, dR, R2, P2, (RES_W,RES_H), cv2.CV_32FC1 )

    return ( m1L, m2L ), ( m1R, m2R ), Q

def save_logs( pFile, POS ):
    line = ""
    for x, y, z in POS:
        line = line + "{:.2f},{:.2f},{:.2f},".format( x, y, z )
    line = line[0:-1] + "\n"
    pFile.write( line )