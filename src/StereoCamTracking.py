import cv2
import numpy as np

from src import CONFIG_FILE
from src.vot.CTracker import CTracker

RADIUS = CONFIG_FILE.get( "MARKER_RADIUS", 10 )

class StereoCamTracking:
    def __init__(self):
        self.POS = []
        self.tracker = None

    def mouseCallback( self, event, x, y, *_ ):
        if isinstance( self.tracker, type( None ) ):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.POS.append( ( y, x ) )
    
    def startCallback( self ):
        if isinstance( self.tracker, type( None ) ) and len( self.POS ):
            self.tracker = CTracker( self.POS )

    def next_tracker( self, F : np.ndarray ):
        if isinstance( self.tracker, CTracker ):
            self.tracker.update( F )
            self._draw( F, [ roi.pos for roi in self.tracker.nodes ], True )
        else:
            self._draw( F, self.POS, False )

    def _draw( self, F : np.ndarray, pos : list[ tuple[ int, int ] ], live : bool = False ):
        t = 2 if live else -1
        for x, y in pos:
            cv2.circle( F, (y,x), RADIUS, (200,0,0), t )

    def _tracking_routine( self, FL : np.ndarray, FR : np.ndarray ):
        self.next_tracker( FL )
        return FL, FR

#    def window_crop( self, F : np.ndarray, pos : tuple[int,int] ):
#        n, m = StereoCam.WINDOW_SIZE
#        h, w, _ = F.shape
#        
#        cw1, ch1 = None, None
#
#        if pos[1] + m >= w: cw1 = w - 2*m - 1; cw2 = w - 1
#        if pos[0] + n >= h: ch1 = h - 2*n - 1; ch2 = h - 1
#        if pos[1] - m < 0: cw1 = 0; cw2 = 2*m
#        if pos[0] - n < 0: ch1 = 0; ch2 = 2*n
#
#        if isinstance( cw1, type(None) ): cw1 = pos[1] - m; cw2 = pos[1] + m
#        if isinstance( ch1, type(None) ): ch1 = pos[0] - n; ch2 = pos[0] + n
#
#        return F[ ch1:ch2, cw1:cw2, : ]

#    def NCC( self, f, t ):
#        return np.sum( f*t ) / ( np.sqrt(np.sum(f**2))*np.sqrt(np.sum(t**2)) )

            
#    def stereo_match( self, F : np.ndarray, T : np.ndarray, roi : tuple[int,int] ):
#        pass
#        #t = self.window_crop( T, roi )
#        #_, m = StereoCam.WINDOW_SIZE
#        #_, w, _ = F.shape
#        #l = []
#        #for i in range( m, w - m ):
#        #    f = self.window_crop( F, (roi[0], i) )
#        #    l.append( self.NCC( f, t ) )
#        #l = np.array( l )
#        #return l.argmax() + m
        