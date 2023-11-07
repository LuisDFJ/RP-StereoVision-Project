import cv2
import numpy as np

class StereoCamCAL:
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    CHESSBOARD = (6,9)
    MAXCOUNT = 180
    SIZE = 20
    FLAGS = 0 | cv2.CALIB_FIX_INTRINSIC

    def __init__(self, size : tuple[int,int] ) -> None:
        self.size = size
        self.cal_flag = False
        self.corners_L = []
        self.corners_R = []
        self.corners   = []

        self.count = 0

        self.LS_Map = None
        self.RS_Map = None

        self.P = np.zeros((StereoCamCAL.CHESSBOARD[0] * StereoCamCAL.CHESSBOARD[1], 3), np.float32)
        self.P[:,:2] = np.mgrid[0:StereoCamCAL.CHESSBOARD[0], 0:StereoCamCAL.CHESSBOARD[1]].T.reshape(-1, 2) * StereoCamCAL.SIZE

    def sort_corners( self, P : np.ndarray ) -> np.ndarray:
        PYSorted = P[ P[:,1].argsort()[::-1] ]
        N,M = StereoCamCAL.CHESSBOARD
        PSorted = np.zeros( (M*N,2), dtype=np.float32 )
        for i in range( M ):
            PXSorted = PYSorted[i*N:(i+1)*N,:]
            PSorted[i*N:(i+1)*N,:] = PXSorted[ PXSorted[:,0].argsort() ]
        return PSorted

    def _state_calibration( self, FL, FR ):
        if self.cal_flag:
            GL = cv2.cvtColor(FL, cv2.COLOR_BGR2GRAY)
            GR = cv2.cvtColor(FR, cv2.COLOR_BGR2GRAY)
            ret_L, cor_L = cv2.findChessboardCorners( GL, StereoCamCAL.CHESSBOARD, None )
            ret_R, cor_R = cv2.findChessboardCorners( GR, StereoCamCAL.CHESSBOARD, None )
            if ret_L and ret_R:
                corner_L = cv2.cornerSubPix(GL, cor_L, (11, 11), (-1, -1), StereoCamCAL.CRITERIA)
                corner_R = cv2.cornerSubPix(GR, cor_R, (11, 11), (-1, -1), StereoCamCAL.CRITERIA)
                self.corners.append( self.P )
                self.corners_L.append( self.sort_corners( corner_L[:,0,:] ) )
                self.corners_R.append( self.sort_corners( corner_R[:,0,:] ) )
                self.cal_flag = False
                self.count = 0
                
        if self.count < StereoCamCAL.MAXCOUNT:
            self.count = self.count + 1
            if len( self.corners_L ):
                FL = cv2.drawChessboardCorners( FL, StereoCamCAL.CHESSBOARD, self.corners_L[-1], True )
            if len( self.corners_R ):
                FR = cv2.drawChessboardCorners( FR, StereoCamCAL.CHESSBOARD, self.corners_R[-1], True )

        return FL, FR

    def calibration( self ):
        if not isinstance( self.LS_Map, type( None ) ) and not isinstance( self.RS_Map, type( None ) ): return

        ret_L, ML, DL, _, _ = cv2.calibrateCamera( self.corners, self.corners_L, self.size, None, None )
        ret_R, MR, DR, _, _ = cv2.calibrateCamera( self.corners, self.corners_R, self.size, None, None )
        
        if ret_L and ret_R:
            nML, _ = cv2.getOptimalNewCameraMatrix( ML, DL, self.size, 1, self.size )
            nMR, _ = cv2.getOptimalNewCameraMatrix( MR, DR, self.size, 1, self.size )
            
            _, nML, DL, nMR, DR, R, T, E, F = cv2.stereoCalibrate( 
                self.corners, 
                self.corners_L, 
                self.corners_R, 
                nML, 
                DL, 
                nMR, 
                DR, 
                self.size, 
                StereoCamCAL.CRITERIA, 
                StereoCamCAL.FLAGS 
            )

            #p = np.array(
            #    [
            #        [ 200, 200 ],
            #        [ 400, 400 ]
            #    ],
            #    dtype=np.float32
            #)

            #self.linesL = cv2.computeCorrespondEpilines( p, 1, F )
            #self.linesR = cv2.computeCorrespondEpilines( p, 2, F )

            rL, rR, pL, pR, _, _, _ = cv2.stereoRectify(
                nML, DL, 
                nMR, DR,
                self.size,
                R,
                T, 
                1,
                (0,0)
            )

            self.LS_Map = cv2.initUndistortRectifyMap( nML, DL, rL, pL, self.size, cv2.CV_16SC2)
            self.RS_Map = cv2.initUndistortRectifyMap( nMR, DR, rR, pR, self.size, cv2.CV_16SC2)

    def rectify( self, FL, FR ):
        self.calibration()
        if not isinstance( self.LS_Map, type( None ) ) and not isinstance( self.RS_Map, type( None ) ):
            #self.drawEpilines( FL, self.linesR )
            #self.drawEpilines( FR, self.linesL )
            FL= cv2.remap( FL, self.LS_Map[0], self.LS_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            FR= cv2.remap( FR, self.RS_Map[0], self.RS_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        return FL, FR
    
    def drawEpilines( self, F : np.ndarray, lines : np.ndarray ):
        _,c,_ = F.shape
        L = lines[:,0,:]
        for r in L: 
            x0, y0 = map(int, [0, -r[2] / r[1] ]) 
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ]) 
            
            cv2.line(F, (x0, y0), (x1, y1), (0,240,0), 1)

        cv2.circle( F, (200,200), 5, (200,0,0), 1 ) 
        cv2.circle( F, (400,400), 5, (0,0,200), 1 ) 

class StereoCam( StereoCamCAL ):
    STATE_CALIBRATION = 1
    STATE_ONLINE = 2
    WINDOW_SIZE = ( 20, 20 )
    def __init__(self, portL : int = 1, portR : int = 2, RES_W = 1280, RES_H = 720) -> None:
        super().__init__( ( RES_H, RES_W ) )

        self.match_flag = False
        self.locus = 1

        self.capL = cv2.VideoCapture( portL, cv2.CAP_DSHOW )
        self.capL.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
        self.capL.set(cv2.CAP_PROP_FRAME_HEIGHT,RES_H)

        self.capR = cv2.VideoCapture( portR, cv2.CAP_DSHOW )
        self.capR.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
        self.capR.set(cv2.CAP_PROP_FRAME_HEIGHT,RES_H)

        self.state = StereoCam.STATE_CALIBRATION
        self._gen = self.gen()

    def window_crop( self, F : np.ndarray, pos : tuple[int,int] ):
        n, m = StereoCam.WINDOW_SIZE
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
        return np.sum( f*t ) / ( np.sqrt(np.sum(f**2))*np.sqrt(np.sum(t**2)) )

    def stereo_match( self, F : np.ndarray, T : np.ndarray, roi : tuple[int,int] ):
        t = self.window_crop( T, roi )
        _, m = StereoCam.WINDOW_SIZE
        _, w, _ = F.shape
        l = []
        for i in range( m, w - m ):
            f = self.window_crop( F, (roi[0], i) )
            l.append( self.NCC( f, t ) )
        l = np.array( l )
        return l.argmax() + m
        

    def _state_online( self, FL ,FR ):
        if self.match_flag:
            self.match_flag = False
            self.locus = self.stereo_match( FR, FL, (200,400) )
            self.count = 0

        if self.count < StereoCamCAL.MAXCOUNT:
            self.count = self.count + 1
            cv2.circle( FL, ( 400,200 ), 5, (0,255,0), 2 )
            cv2.circle( FR, ( self.locus,200 ), 5, (0,255,0), 2 )

        return FL, FR

    def next( self ):
        return next( self._gen )

    def gen( self ):
         while True:
              FL, FR = self.read()
              if not isinstance( FL, type(None) ) and not isinstance( FR, type(None) ):
                if self.state == StereoCam.STATE_CALIBRATION:
                    yield self._state_calibration( FL, FR )
                elif self.state == StereoCam.STATE_ONLINE:
                    yield self._state_online( *self.rectify( FL, FR ) )

    def read( self ) -> tuple:
        ret_l, FL = self.capL.read(); ret_r, FR = self.capR.read()
        if ret_l and ret_r:
            FL = cv2.rotate( FL, cv2.ROTATE_90_COUNTERCLOCKWISE )
            FR = cv2.rotate( FR, cv2.ROTATE_90_COUNTERCLOCKWISE )
            return FL, FR
        return None, None

cam = StereoCam( RES_W=520, RES_H=520 )

while True:
    FL, FR = cam.next()
    cv2.imshow( "Left", FL )
    cv2.imshow( "Right", FR )
    k = cv2.waitKey( 1 )
    if   k == ord( "q" ): break
    elif k == ord( "c" ): cam.cal_flag = True
    elif k == ord( "n" ): cam.state = StereoCam.STATE_ONLINE
    elif k == ord( "m" ): cam.match_flag = True