import numpy as np
from typing import Tuple

#import matplotlib.pyplot as plt
#import cv2
#def image_show_channel( I : np.ndarray, file ) -> None:
#    h, w = I.shape
#    Y, X = np.mgrid[ 0:h, 0:w ]
#    _, ax = plt.subplots()
#    ax.pcolormesh( X, Y, I, cmap=plt.colormaps["jet"])
#    plt.axis('off')
#    plt.savefig( "C:\\workspace\\projects\\MCI-stuff\\mnTracker\\{0}.png".format(file), bbox_inches="tight")
#    plt.show()
#
#def image_show( I : np.ndarray, file ) -> None:
#    cv2.namedWindow( "Debug" )
#    I = cv2.resize( I, [128,128], interpolation=cv2.INTER_AREA )
#    cv2.imwrite( "C:\\workspace\\projects\\MCI-stuff\\mnTracker\\{0}.png".format(file) , I )
#    cv2.imshow( "Debug", I )

def crop_frame( img : np.ndarray, roi : Tuple[int,int,int,int] ) -> np.ndarray:
    return img[ 
        roi[0]-roi[2] : roi[0]+roi[2],
        roi[1]-roi[3] : roi[1]+roi[3] ]