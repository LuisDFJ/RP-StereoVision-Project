import os
import struct
import numpy as np

def read_homography( path : str, debug = False ):
    if not debug:
        with open( os.path.join( path, "HL.bin" ), 'rb') as pfile:
            file = pfile.read()
            fmt = f'@{int(len(file)/8)}d'
            HL = struct.unpack( fmt, file )
        
        with open( os.path.join( path, "HR.bin" ), 'rb') as pfile:
            file = pfile.read()
            fmt = f'@{int(len(file)/8)}d'
            HR = struct.unpack( fmt, file )

        HL = np.array( HL, dtype=np.float64 ).reshape( (3,3) ).T
        HR = np.array( HR, dtype=np.float64 ).reshape( (3,3) ).T
    else:
        HL = np.eye(3)
        HR = np.eye(3)
    return HL, HR