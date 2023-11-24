import numpy as np
import cv2

from src import CONFIG_FILE
from src.StereoCamTracking import StereoCamTracking

SCALE = CONFIG_FILE.get( "IMAGE_SCALE", 1.0 )

class StereoCam( StereoCamTracking ):
    def __init__(self, HL : np.ndarray, HR : np.ndarray, portL : int = 1, portR : int = 2, RES_W = 1280, RES_H = 720, offline = False ) -> None:
        super().__init__()
        if not offline:
            self.capL = cv2.VideoCapture( portL, cv2.CAP_DSHOW )
            self.capL.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
            self.capL.set(cv2.CAP_PROP_FRAME_HEIGHT,RES_H)

            self.width  = int( self.capL.get( cv2.CAP_PROP_FRAME_WIDTH ) )
            self.height = int( self.capL.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
            self.size   = (self.width,self.height)
            self.rsize  = ( int( self.width * SCALE ), int( self.height * SCALE ) )

            self.capR = cv2.VideoCapture( portR, cv2.CAP_DSHOW )
            self.capR.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
            self.capR.set(cv2.CAP_PROP_FRAME_HEIGHT,RES_H)

        self.HL = HL
        self.HR = HR
        self._gen = self.gen()

    def rectify( self, FL : np.ndarray, FR : np.ndarray ):
        rFL = cv2.warpPerspective( FL, self.HL, self.size, flags=cv2.INTER_LINEAR )
        rFR = cv2.warpPerspective( FR, self.HR, self.size, flags=cv2.INTER_LINEAR )
        return rFL , rFR

    def post_process( self, FL : np.ndarray, FR : np.ndarray ):
        return ( 
            cv2.resize( FL, self.rsize, interpolation=cv2.INTER_AREA ),
            cv2.resize( FR, self.rsize, interpolation=cv2.INTER_AREA )
        )

    def next( self ):
        return next( self._gen )

    def gen( self ):
         while True:
              FL, FR = self.read()
              if not isinstance( FL, type(None) ) and not isinstance( FR, type(None) ):
                yield self.post_process( *self._tracking_routine( *self.rectify( *self.read() ) ) )

    def read( self ) -> tuple:
        ret_l, FL = self.capL.read(); ret_r, FR = self.capR.read()
        if ret_l and ret_r:
            #FL = cv2.rotate( FL, cv2.ROTATE_90_COUNTERCLOCKWISE )
            FR = cv2.rotate( FR, cv2.ROTATE_180 )
            return FL, FR
        return None, None



