import numpy as np

from typing import Tuple

from src import CONFIG_FILE
from src.vot.CROI import CROI
from src.vot.FPreprocess import image_preprocess, gauss_response#, shift_matrix
from src.vot.FUtils import crop_frame#, image_show_channel, image_show

LEARNING_RATE = CONFIG_FILE.get( "LEARNING_RATE", 0.001 )
LAMBDA        = CONFIG_FILE.get( "LAMBDA", 0.0001 )
KERNEL_SIGMA  = CONFIG_FILE.get( "GAUSS_KERNEL_SIGMA", 2 )
RESPONSE_SIGMA= CONFIG_FILE.get( "GAUSS_RESPONSE_SIGMA", 2 )

class CKCF:
    def __init__( self, img : np.ndarray = None, roi : Tuple[int,int,int,int] = None ) -> None:
        self.roi = CROI()
        self.n   = LEARNING_RATE
        self.A   = None
        self.X   = None
        self.Y   = None
        if isinstance( img, np.ndarray ) and isinstance( roi, tuple ):
            self.init( img, roi )

    def init( self, img : np.ndarray, roi : Tuple[int,int,int,int] ) -> None:
        self.roi.roi = roi
        x = image_preprocess( crop_frame( img, self.roi.roi ) )
        X = fft_by_channel( x )
        self.Y = gauss_response( x, self.roi.dim, RESPONSE_SIGMA )
        K, self.X = gauss_kernel( x, X )
        self.A = train( K, self.Y )

    def update( self, img : np.ndarray ):
        if isinstance( self.X, np.ndarray ) and isinstance( self.A, np.ndarray ) and isinstance( self.Y, np.ndarray ):
            z = image_preprocess( crop_frame( img, self.roi.roi ) )
            K, Z = gauss_kernel( z, self.X, KERNEL_SIGMA )
            y, delta = detect( K, self.A )
            #image_show_channel( shift_matrix( z[:,:,0], [0,-2] ), "img100_c0_preproc" )
            self.roi.delta( delta[0], delta[1] )
            A = train( K, self.Y, LAMBDA )
            self.A = (1-self.n) * self.A + self.n * A
            self.X = (1-self.n) * self.X + self.n * Z

def gauss_kernel( x : np.ndarray, Y : np.ndarray, sigma : float = 2.0 ) -> Tuple[ np.ndarray, np.ndarray ]:
    w = Y.shape
    x_norm = np.linalg.norm( x.flatten() ) ** 2
    y_norm = np.linalg.norm( Y.flatten() ) ** 2 / (w[0] * w[1])
    try:
        h, w, c = x.shape
        C = np.zeros( ( h,w ) )
        X = np.zeros( x.shape, dtype=np.complex128 )
        for i in range( c ):
            X[:,:,i] = np.fft.fft2( x[:,:,i] )
            C = C + X[:,:,i].conj() * Y[:,:,i]
    except ValueError:
        h, w = x.shape
        X = np.fft.fft2( x )
        C = X.conj() * Y
    
    d = x_norm + y_norm - 2 * np.fft.ifft2( C )
    return np.fft.fft2( np.exp( -1 / sigma**2 * np.abs( d ) / x.size ) ), X

def fft_by_channel( x : np.ndarray ) -> np.ndarray:
    X = np.zeros( x.shape, dtype=np.complex128 )
    try:
        _, _, c = x.shape
        for i in range( c ):
            X[:,:,i] = np.fft.fft2( x[:,:,i] )
    except ValueError:
        X = np.fft.fft2( x )
    return X

def train( K : np.ndarray, Y : np.ndarray, _lambda : float = 0.0001 ) -> np.ndarray:
    return Y / ( K + _lambda )

def detect( K : np.ndarray, A : np.ndarray ) -> Tuple[ np.ndarray, tuple ]:
    y = np.real( np.fft.ifft2( K * A ) )
    delta = np.unravel_index( np.argmax( y, axis=None ), y.shape )
    return y, delta