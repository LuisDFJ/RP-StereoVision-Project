import cv2
import numpy as np

from src import CONFIG_FILE
from src.vot.CTracker import CTracker

RADIUS    = CONFIG_FILE.get( "MARKER_RADIUS", 10 )

class StereoCamTracking:
    def __init__(self,Q : np.ndarray):
        self.Q = Q
        self.POS = []
        self.tracker = None

    def mouseCallback( self, event, x, y, *_ ):
        if isinstance( self.tracker, type( None ) ):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.POS.append( ( y, x ) )
    
    def startCallback( self ):
        if isinstance( self.tracker, type( None ) ) and len( self.POS ):
            self.tracker = CTracker( self.Q, self.POS )

    def next_tracker( self, FL : np.ndarray, FR : np.ndarray ):
        if isinstance( self.tracker, CTracker ):
            self.tracker.update( FL, FR )
            self._draw( FL, [ roi.pos for roi in self.tracker.nodesL ], True )
            self._draw( FR, [ roi.pos for roi in self.tracker.nodesR ], True )
        else:
            self._draw( FL, self.POS, False )

    def _draw( self, F : np.ndarray, pos : list[ tuple[ int, int ] ], live : bool = False ):
        t = 2 if live else -1
        for x, y in pos:
            cv2.circle( F, (y,x), RADIUS, (200,0,0), t )

    def _tracking_routine( self, FL : np.ndarray, FR : np.ndarray ):
        self.next_tracker( FL, FR )
        return FL, FR
        