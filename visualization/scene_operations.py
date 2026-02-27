# http://geomalgorithms.com/a05-_intersect-1.html
# https://stackoverflow.com/questions/14607640/rotating-a-vector-in-3d-space

import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def createMhFromMV(Mrot, Vtra):
    T  = Vtra.reshape((3,1))
    RT = np.hstack((Mrot, T))
    M  = np.vstack((RT, np.array([0,0,0,1])))
    return M

def createMhFromVV(Vrot, Vtra):
    Mrot, _ = cv2.Rodrigues(Vrot)
    M = createMhFromMV(Mrot, Vtra)
    return M

def transformMhProws(Mh, Prows):
    if Prows.ndim == 1:
        Prows = Prows.reshape((1, Prows.shape[0]))
    PhRows = np.hstack((Prows, np.ones((Prows.shape[0],1))))
    PhCols = PhRows.T
    PhtranCols = Mh.dot(PhCols)
    PhtranRows = PhtranCols.T
    PtranRows = np.squeeze(PhtranRows[:, :-1])
    return PtranRows
    
def createScreenCamMs(screenO_cam, orientation=0):
    VTscreen_cam = screenO_cam * np.array([1, -1, 1]) # Shortcut en este caso de ejes específico que sirve para todo irisgo
    Mscreen_cam, Mcam_screen = createBothRotMat(np.array([0, math.pi, 0]), VTscreen_cam)

    if orientation == 1:
        VRscreen_cam = np.array([math.pi, 0, 0])# dado la vuelta
    elif orientation == 2:
        VRscreen_cam = np.array([-math.pi/2, 0, 0])# girado a la izquierda
    elif orientation == 3:
        VRscreen_cam = np.array([math.pi/2, 0, 0])# girado a la derecha
    else:
        return Mscreen_cam, Mcam_screen

    M2screen_cam, M2cam_screen = createBothRotMat(VRscreen_cam, np.array([0, 0, 0]))

    Mscreen_cam = np.dot(M2screen_cam, Mscreen_cam)
    Mcam_screen = np.dot(M2cam_screen, Mcam_screen)
    return Mscreen_cam, Mcam_screen

def update_tranformation_matrix(rotation_matrix, translation_vector):        
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        Mscreen_cam = transformation_matrix
        Mcam_screen = np.linalg.inv(transformation_matrix)
        return Mscreen_cam, Mcam_screen

def createScreenCamMsSimple(VTscreen_cam, VRscreen_cam):
    rotation_matrix = R.from_euler('xyz', VRscreen_cam).as_matrix()
    Mscreen_cam, Mcam_screen = update_tranformation_matrix(rotation_matrix, VTscreen_cam)
    
    return Mscreen_cam, Mcam_screen


def createBothRotMat(VRscreen_cam, VTscreen_cam):
    # Step 1: Create rotation matrix from Euler angles
    rotation_matrix = R.from_euler('zyx', VRscreen_cam, degrees=False).as_matrix()

    # Step 2: Construct the transformation matrix
    Mscreen_cam = np.eye(4)  # Initialize 4x4 identity matrix
    Mscreen_cam[:3, :3] = rotation_matrix  # Insert rotation matrix
    Mscreen_cam[:3, 3] = VTscreen_cam    # Insert translation vector

    Mcam_screen  = np.linalg.inv(Mscreen_cam)
    return Mscreen_cam, Mcam_screen 

def rotateVector(vector, angles):
    if angles[2] == 0:
        return vector
    vector = vector.flatten()
    R, _ = cv2.Rodrigues(angles)
    vrot = R.dot(vector)
    return vrot
    
def getNormalizedVector(point1, point2):
    gaze = point1 - point2 
    gaze = gaze / np.linalg.norm(gaze)
    return gaze 

def normalizeVector(vector):
    return vector / np.linalg.norm(vector)

def meanNormalizedVector(vector1, vector2):
    # Calculate the mean of the two vectors
    mean_vector = (vector1 + vector2) / 2.0

    # Normalize the mean vector
    mean_magnitude = np.linalg.norm(mean_vector)
    if mean_magnitude != 0:
        normalized_mean_vector = mean_vector / mean_magnitude
    else:
        # Handle the case where the mean magnitude is 0
        normalized_mean_vector = np.zeros(3)
        
    return normalized_mean_vector

class Screen:
    def __init__(self, Wcm, Hcm, Wpx, Hpx, zoom):
        self.Wcm = Wcm
        self.Hcm = Hcm
        self.Wpx = int(Wpx)
        self.Hpx = int(Hpx)

    def setSizeCm(self, Wcm, Hcm):
        self.Wcm = Wcm
        self.Hcm = Hcm        
        
    def setSizePx(self, Wpx, Hpx, zoom):
        self.Wpx = int(Wpx*zoom)
        self.Hpx = int(Hpx*zoom)
        
    def fromPxToCm(self, Ppx):
        Ppx = np.squeeze(Ppx)
        xpx = Ppx[0]
        ypx = Ppx[1]
        xcm = xpx / self.Wpx * self.Wcm
        ycm = ypx / self.Hpx * self.Hcm
        zcm = 0.
        Pcm = np.array([xcm, ycm, zcm])
        return Pcm
    
    def fromCmToPx(self, Pcm):
        xcm = Pcm[0]
        ycm = Pcm[1]
        xpx = xcm / self.Wcm * self.Wpx
        ypx = ycm / self.Hcm * self.Hpx
        Ppx = np.array([xpx, ypx])
        return Ppx
    
    def returnScreenMatrix(self):
        return np.array([[0, 0, 0], [self.Wcm, 0, 0], [self.Wcm, self.Hcm, 0], [0, self.Hcm, 0]])
    

class Plane:
    def __init__(self, V0, n):
        self.V0 = V0
        self.n = n

    def intersect(self, P0, u):
        u = u.flatten()
        s1 = self.n.dot(self.V0-P0) / self.n.dot(u)
        Ps1 = P0 + s1*u
        return Ps1

def computeCameraMatrix(focal_length, principal_point):
    mat = np.array([[focal_length[0], 0, principal_point[0]],
                    [0, focal_length[1], principal_point[1]],
                    [0, 0, 1]])
    return mat

def projectPoints(Mp, Ps):
    proj3t = Mp.dot(Ps.T)
    proj3  = proj3t.T
    z3     = np.column_stack((proj3[:,2], proj3[:,2], proj3[:,2]))
    projn  = proj3 / z3
    proj   = projn[:,[0,1]]
    return proj
    
def rotateAndCropEyesMatrix(diff, center, crop = (60, 36)):
    # Rotate and crop eyes
    angle  = np.arctan2(diff[1], diff[0])
    scale  = 40 / np.linalg.norm(diff)
    Mrot   = cv2.getRotationMatrix2D(tuple(center), math.degrees(angle), scale)
    Mrot[0, 2] += -center[0] + crop[0] * 0.5
    Mrot[1, 2] += -center[1] + crop[1] * 0.5
    return Mrot
