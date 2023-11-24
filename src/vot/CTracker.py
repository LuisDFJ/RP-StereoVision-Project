import numpy as np

from src import CONFIG_FILE
from src.vot.CKCF import CKCF
from src.vot.CROI import CROI

RADIUS         = CONFIG_FILE.get( "MARKER_RADIUS", 10 )

class CTracker:
    def __init__(self, POS : list[ tuple[ int, int ] ]) -> None:
        self.trackers  = None
        self.nodes     = [ CROI( x=x, y=y, w=RADIUS, h=RADIUS ) for x, y in POS ]
    def init(self, img : np.ndarray) -> None:
        self.trackers = [ CKCF( img, roi.roi ) for roi in self.nodes ]
    def update( self, img : np.ndarray ) -> None:
        if  isinstance( self.trackers, type( None ) ):
            self.init( img )
        else:
            for i, kcf in enumerate( self.trackers ):
                kcf.update( img )
                self.nodes[i].pos = kcf.roi.pos