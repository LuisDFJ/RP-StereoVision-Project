from typing import Tuple

class CROI:
    def __init__( self, x : int = 0, y : int = 0 , w : int = 0, h : int = 0, lr : float = 1 ) -> None:
        self.x = x; self.y = y
        self.w = w; self.h = h
        self.lr = lr
    def delta( self, x : int, y : int ) -> None:
        if x > self.w: x = x - 2 * self.w
        if y > self.h: y = y - 2 * self.h
        self.pos = ( int( self.x - self.lr*x ),
                     int( self.y - self.lr*y ) )
    
    @property
    def pos( self ):
        return self.x, self.y
    @pos.setter
    def pos( self, pos : Tuple[ int, int ] ):
        def sset( x, y ):
            self.x = x; self.y = y
        sset( *pos )
    
    @property
    def dim( self ):
        return self.w, self.h
    @dim.setter
    def dim( self, dim : Tuple[ int, int ] ):
        def sset( w, h ):
            self.w = w; self.h = h
        sset( *dim )

    @property
    def roi( self ):
        return self.x, self.y, self.w, self.h
    @roi.setter
    def roi( self, roi : Tuple[ int, int, int, int ] ):
        def sset( x, y, w, h ):
            self.pos = (x,y)
            self.dim = (w,h)    
        sset( *roi )
    
    @property
    def pt1( self ):
        return self.y - self.h, self.x - self.w
    @property
    def pt2( self ):
        return self.y + self.h, self.x + self.w