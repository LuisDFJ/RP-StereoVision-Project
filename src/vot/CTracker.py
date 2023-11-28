import numpy as np
import cv2

from src import CONFIG_FILE
from src.vot.CKCF import CKCF
from src.vot.CROI import CROI

RADIUS    = CONFIG_FILE.get( "MARKER_RADIUS", 10 )
XC_RADIUS = CONFIG_FILE.get( "XCORR_RADIUS", 10 )

class CTracker:
    def __init__(self, Q : np.ndarray, POS : list[ tuple[ int, int ] ]) -> None:
        self.Q         = Q
        self.trackers  = None
        self.nodesL    = [ CROI( x=x, y=y, w=RADIUS, h=RADIUS ) for x, y in POS ]
        self.nodesR    = [ CROI( x=x, y=y, w=RADIUS, h=RADIUS ) for x, y in POS ]
    def init(self, img : np.ndarray) -> None:
        self.trackers = [ CKCF( img, roi.roi ) for roi in self.nodesL ]
    def update( self, imgL : np.ndarray, imgR : np.ndarray ) -> None:
        if  isinstance( self.trackers, type( None ) ):
            self.init( imgL )
        else:
            for i, kcf in enumerate( self.trackers ):
                kcf.update( imgL )
                self.nodesL[i].pos = kcf.roi.pos
            
            for i in range( len( self.nodesR ) ):
                x, y = self.nodesL[i].pos
                y = self.stereo_match( imgR, imgL, (x,y)  )
                self.nodesR[i].pos = ( x, y )

    def window_crop( self, F : np.ndarray, pos : tuple[int,int] ):
        n = m = XC_RADIUS
        h, w, _ = F.shape
        
        cw1, ch1 = None, None

        if pos[1] + m >= w: cw1 = w - 2*m - 1; cw2 = w - 1
        if pos[0] + n >= h: ch1 = h - 2*n - 1; ch2 = h - 1
        if pos[1] - m < 0: cw1 = 0; cw2 = 2*m
        if pos[0] - n < 0: ch1 = 0; ch2 = 2*n

        if isinstance( cw1, type(None) ): cw1 = pos[1] - m; cw2 = pos[1] + m
        if isinstance( ch1, type(None) ): ch1 = pos[0] - n; ch2 = pos[0] + n

        return F[ ch1:ch2, cw1:cw2, : ]

    def NCC( self, f, t ):
        return np.sum( (f - np.mean(f))*(t - np.mean(t)) ) / ( np.std( f ) * np.std( t ) )

    def stereo_match( self, F : np.ndarray, T : np.ndarray, roi : tuple[int,int] ):        
        t = cv2.cvtColor( self.window_crop( T, roi ), cv2.COLOR_BGR2GRAY )
        m = XC_RADIUS
        _, w, _ = F.shape
        l = []
        for i in range( m, w - m ):
            f = cv2.cvtColor( self.window_crop( F, (roi[0], i ) ), cv2.COLOR_BGR2GRAY )
            l.append( self.NCC( f, t ) )
        l = np.array( l )
        return l.argmax() + m
    
    def stereo_reconstruct( self ):
        P = []
        for i in range( len( self.nodesL ) ):
            v,u = self.nodesL[i].pos
            _,d = self.nodesR[i].pos
            X = self.Q @ np.c_[ [ u,v,abs(d-u),1 ] ]
            if X[-1] != 0:
                x = X[0:-1] / X[-1]
                P.append( tuple( x.T.tolist()[0] ) )
            else:
                P.append( (0,0,0) )
        return P
