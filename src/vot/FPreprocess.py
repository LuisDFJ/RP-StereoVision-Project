import numpy as np
from typing import Tuple

def shift_matrix( I : np.ndarray, pos : Tuple[int,int] ) -> np.ndarray:
    return np.roll( np.roll( I, -pos[0], 1 ), -pos[1], 0 )

def gauss_response( I : np.ndarray, pos : Tuple[int,int], sigma : float = 2 ) -> np.ndarray:
    try:
        h, w, _ = I.shape
    except ValueError:
        h, w = I.shape
    Y, X = np.mgrid[ 0:h, 0:w ]
    d = ( np.square(X - pos[1]) + np.square(Y - pos[0]) ) / (2 * sigma)
    g = np.exp( -d )
    return np.fft.fft2( shift_matrix( g, pos ) )

def hanning_filter( I : np.ndarray ) -> np.ndarray:
    h, w, c = I.shape
    Hx, Hy = np.meshgrid( np.hanning( w ), np.hanning( h ) )
    for i in range( c ):
        I[:,:,i] = Hx * Hy * I[:,:,i]
    return I

def image_preprocess( I : np.ndarray ) -> np.ndarray:
    L = np.log( I.astype( np.float32 ) + 1 )
    L = ( L - np.mean( L ) ) / ( np.std( L ) + 1e-6 )
    return hanning_filter( L )

