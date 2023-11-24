import cv2
from src.StereoCam import StereoCam
from src.utils import read_homography

#cam = StereoCam( RES_W=640, RES_H=520 )
HL, HR = read_homography( r".\homography", True )
cam = StereoCam( HL, HR )


cv2.namedWindow( "Left" )
cv2.setMouseCallback( "Left", cam.mouseCallback )


i = 1
while True:
    FL, FR = cam.next()
    cv2.imshow( "Left", FL )
    cv2.imshow( "Right", FR )
    k = cv2.waitKey( 1 )

    if   k == ord( "q" ): break
    elif k == ord( "c" ): cv2.imwrite( f".\media\Left_{i}.png", FL ); cv2.imwrite( f".\media\Right_{i}.png", FR ); i = i + 1
    elif k == ord( "n" ): cam.startCallback()
    

