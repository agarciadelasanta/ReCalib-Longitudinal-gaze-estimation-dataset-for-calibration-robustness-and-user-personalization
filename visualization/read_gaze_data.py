import cv2
import json
import numpy as np
import scene_operations as scene
from head_model import load_head_model
from scene_3d_plots import trackingEnviroment3DPlotter
from scene_2d_plots import trackingScreen2DPlotter
import math 
#from ...src.modules.utils import calc_vec_and_xy_angles_error
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class Read_gaze_data:
    def __init__(self, pathImg, pathJson, crop_format=False):
        self.pathImg = pathImg
        self.pathJson = pathJson
        self.crop_format = crop_format
        
        if not(self.checkIfIsParsed()):
            return

        self.img = cv2.imread(self.pathImg)
        
        if self.img is None:
            return
        
        self.variableInitialization()

        if self.crop_format:
            self.loadJSONData_crop()
        else:
            self.loadJSONData()

        self.sceneReconstruction()

    def checkIfIsParsed(self):
        try:
            with open(self.pathJson, 'r') as j:
                self.data = json.loads(j.read())
        except Exception as e:
            print("Error loading JSON:", e)
        if self.crop_format:
            if "rotation" in self.data:
                return True
            else:
                return False
        else:
            if "hpe" in self.data:
                return True
            else:
                return False
        
    def variableInitialization(self):
        self.pogPx_screen = []
        self.eye3DCenter_cam = []
        
        self.cameraPlane_cam = scene.Plane(np.array([0,0,0]), np.array([0,0,1]))
        self.gazePrediction = None
        self.predictionHead3D_cam = None
        self.pogCmCalibratedPrediction_cam = []

        self.pogPrediction_cam = []
        self.eye3DPredictionCenter_cam = []
        self.pogPredictionPx_screen = np.array([])
        self.pogPredictionRawPx_screen = np.array([])

    def loadJSONData(self):
        self.pos = np.array([self.data["pos"]["x"], self.data["pos"]["y"]])
        self.gaze = np.array(self.data["gaze"])
        self.screen_mm = np.array([self.data["screen_mm"]["width"], self.data["screen_mm"]["height"]])
        self.screen_pixels = np.array([self.data["screen_pixels"]["height"], self.data["screen_pixels"]["width"]])
        
        self.posCam_mm = self.data["posCam_mm"]
        self.hpe = self.data["hpe"]["6d"]

        self.screen_orientation = self.data["screen_orientation"]
        self.zoom = self.data["screen_zoom"]
        self.img_res = self.data["img_res"]
        self.pogMm_cam = np.array(self.data["gaze"]["destiny"])

        if self.hpe[5] < 200:
            self.hpe[3] = self.hpe[3]
            self.hpe[4] = self.hpe[4]
            self.hpe[5] = self.hpe[5]

        self.pogMm_cam = np.array(self.data["gaze"]["destiny"])
        self.gaze =  np.array(self.data["gaze"]["vector"])
        self.eye3DCenter_cam =  np.array(self.data["gaze"]["origin"])

    def loadJSONData_crop(self):
        self.gaze = np.array(self.data["gaze"])
        #self.screen_mm = self.data["enviroment_variables"]["screen_mm"]*10

        #self.posCam_mm = [130, 10, 0]
        self.hpe = np.concatenate((self.data["rotation"], self.data["position"]))

        #self.screen_orientation = self.data["enviroment_variables"]["screen_orientation"]
        #self.img_res = self.data["enviroment_variables"]["image_resolution"]
        self.pogMm_cam = np.array(self.data["lookAtPoint"])

        if self.hpe[5] < 200:
            self.hpe[3] = self.hpe[3]*10
            self.hpe[4] = self.hpe[4]*10
            self.hpe[5] = self.hpe[5]*10

        if "rotated_crop" in self.data and self.data["rotated_crop"] != None:
            self.roll = self.data["rotated_crop"]["roll"]
            self.gaze_raw = self.data["rotated_crop"]["gaze_raw"]

        self.eye3DCenter_cam =  np.array(self.data["eye3DCenter"])
        self.leftEye3DCenter_cam =  np.array(self.data["leftEyeCenter"])
        self.rightEye3DCenter_cam =  np.array(self.data["rightEyeCenter"])
    
    def rotated_vector_correction(self):
        self.gaze_corrected = scene.rotateVector(np.asarray(self.gaze, dtype=np.float32), np.array([0, 0, -self.roll]))
        self.gaze_corrected = self.gaze_corrected / np.linalg.norm(self.gaze_corrected)
        self.gaze = self.gaze_corrected
        roll_values = np.linspace(-np.pi, np.pi, 10)

        self.gaze_corrected_range = []
        
        for roll_aux in roll_values:
            self.gaze_corrected_range.append(scene.rotateVector(np.asarray(self.gaze, dtype=np.float32), np.array([0, 0, -roll_aux])))

    def sceneReconstruction(self):
        # Head model
        self.screenParametersSetup()
        self.rotationMatrixGen()
        self.axisDefinition()
        self.screenSetup()
        self.headSetup()
        self.getPOGinTheScreen()

        if self.crop_format:
            self.rotated_vector_correction()

    def getPOGinTheScreen(self):
        #self.pos = [int(x*y*self.zoom) for x, y in zip(self.data["percentLookAtPoint"],  self.screen_pixels)]
        #self.pos = np.array([int(self.pos["x"]), int(self.pos["y"])])
        
        # Image and POG
        self.pogPx_screen = (self.pos[0], self.pos[1])

        screenOps = scene.Screen(self.screen_mm[0], 
                                 self.screen_mm[1], 
                                 self.screen_pixels[0]*self.zoom, 
                                 self.screen_pixels[1]*self.zoom,
                                 self.zoom)
        #Pog Px -> Cam
        Pogcm_screen = screenOps.fromPxToCm(self.pogPx_screen)
        self.Pogcm_cam = scene.transformMhProws(self.Mcam_screen, Pogcm_screen)

    def rotationMatrixGen(self):
        ### Rotation matrix
        self.Mscreen_cam, self.Mcam_screen = scene.createScreenCamMsSimple(self.screen_o_cam_t, self.screen_o_cam_r)
        self.VRhead_cam = np.array(self.hpe[0:3])
        self.VThead_cam = np.array(self.hpe[3:6])
        self.Mhead_cam, self.Mcam_head  = scene.createBothRotMat(self.VRhead_cam, self.VThead_cam)
        
    def axisDefinition(self):
        #head axis
        headAxis_head  = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.headAxis_cam  = scene.transformMhProws(self.Mhead_cam, headAxis_head)

        # Camera axes
        self.cameraAxis_cam = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

        # screen axis
        screenAxis_screen  = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.screenAxis_cam     = scene.transformMhProws(self.Mcam_screen, screenAxis_screen)

    def screenParametersSetup(self):
        # Screen and camera setup
        imageSize = []
        # if self.crop_format:
        #     self.screen_mm = np.array([self.data["enviroment_variables"]["screen_mm"]["height"], self.data["enviroment_variables"]["screen_mm"]["width"]]    )
        #     self.screen_pixels = np.array([self.data["enviroment_variables"]["screen_pixels"]["height"], self.data["enviroment_variables"]["screen_pixels"]["width"]])
        #     self.zoom = self.data["enviroment_variables"]["screen_zoom"]
        #     self.pos = [int(x*y*self.zoom) for x, y in zip(self.data["percentLookAtPoint"], self.screen_pixels)]
        #     self.screen_orientation = self.data["enviroment_variables"]["screen_orientation"]
        #     self.img_res = self.data["enviroment_variables"]["image_resolution"]
        #     self.screenWmm = self.screen_mm[0]
        #     self.screenHmm = self.screen_mm[1]
        #     self.screenWpx = self.screen_pixels[0]
        #     self.screenHpx = self.screen_pixels[1]
        #     self.height = self.img_res[0]
        #     self.width = self.img_res[1]
        #     self.camera_pos_x, self.camera_pos_y, self.camera_pos_z = self.data["enviroment_variables"]["posCam_mm"]
        #     self.screen_o_cam_r = np.array([0, math.pi, 0])
            
        # elif "enviroment_variables" in self.data:
        #     self.screen_mm = np.array([self.data["enviroment_variables"]["screen_mm"]["height"], self.data["enviroment_variables"]["screen_mm"]["width"]]    )
        #     self.screenWmm = self.data["enviroment_variables"]["screen_mm"]["width"]
        #     self.screenHmm = self.data["enviroment_variables"]["screen_mm"]["height"]
        #     self.zoom = self.data["enviroment_variables"]["screen_zoom"]
        #     self.screen_orientation = self.data["enviroment_variables"]["screen_orientation"]
        #     self.screenWpx = self.data["enviroment_variables"]["screen_pixels"]["width"]
        #     self.screenHpx = self.data["enviroment_variables"]["screen_pixels"]["height"]
        #     self.height = self.data["enviroment_variables"]["img_res"]["height"]
        #     self.width = self.data["enviroment_variables"]["image_resolution"]["width"]
        #     self.camera_pos_x, self.camera_pos_y, self.camera_pos_z = np.array([(self.data["enviroment_variables"]["posCam_mm"]["x"]), 
        #                                     self.data["enviroment_variables"]["posCam_mm"]["y"], 
        #                                     self.data["enviroment_variables"]["posCam_mm"]["z"]])
            
        #     if ("rotCam" not in self.data["enviroment_variables"] or 
        #             "x" not in self.data["enviroment_variables"]["rotCam"] or 
        #             "y" not in self.data["enviroment_variables"]["rotCam"] or 
        #             "z" not in self.data["enviroment_variables"]["rotCam"]):
        #         self.screen_o_cam_r = np.array([0, math.pi, 0])
        #     else:
        #         self.screen_o_cam_r = np.array([self.data["enviroment_variables"]["rotCam"]["x"], 
        #                                 self.data["enviroment_variables"]["rotCam"]["y"], 
        #                                 self.data["enviroment_variables"]["rotCam"]["z"]])
        # elif ("screen_mm" in self.data and 
                # "screen_pixels" in self.data and 
                # "screen_zoom" in self.data and 
                # "img_res" in self.data and 
                # "posCam_mm" in self.data):
        #self.screen_mm = np.array([self.data["screen_mm"]["height"], self.data["screen_mm"]["width"]])
        #self.screen_pixels = np.array([self.data["screen_pixels"]["height"], self.data["screen_pixels"]["width"]])
        self.screenWmm = self.data["screen_mm"]["width"]
        self.screenHmm = self.data["screen_mm"]["height"]
        #self.zoom = self.data["screen_zoom"]
        #self.screen_orientation = self.data["screen_orientation"]
        self.screenWpx = self.data["screen_pixels"]["width"]
        self.screenHpx = self.data["screen_pixels"]["height"]
        self.height = self.data["img_res"]["height"]
        self.width = self.data["img_res"]["width"]
        self.camera_pos_x, self.camera_pos_y, self.camera_pos_z = np.array([(self.data["posCam_mm"]["x"]), 
                                        self.data["posCam_mm"]["y"], 
                                        self.data["posCam_mm"]["z"]])

        if "rotCam" not in self.data or "x" not in self.data["rotCam"] or "y" not in self.data["rotCam"] or "z" not in self.data["rotCam"]:
            self.screen_o_cam_r = np.array([0, math.pi, 0])
        # else:
        #     self.screenWmm = 260
        #     self.screenHmm = 173
        #     self.zoom = 2
        #     self.screenWpx = 1368
        #     self.screenHpx = 912
        #     self.height = 1080
        #     self.width = 1920
        #     self.screen_orientation = 0
        #     self.camera_pos_x, self.camera_pos_y, self.camera_pos_z = np.array([self.screenWmm/2, 10, 0])
        #     self.screen_o_cam_r = np.array([0, math.pi, 0])

        if self.screen_orientation == 2 or self.screen_orientation == 3:
            self.screenHpx, self.screenWpx = self.screenWpx, self.screenHpx
            self.screenHmm, self.screenWmm = self.screenWmm, self.screenHmm

        # Set screen parameters
        # Compute camera_pos_x and camera_pos_y based on orientation
        
        if self.screen_orientation == 0:
            self.camera_pos_x =  self.camera_pos_x + self.screenWmm / 2
            self.camera_pos_y = -self.camera_pos_y
        elif self.screen_orientation == 1:
            self.camera_pos_x = self.camera_pos_x + self.screenWmm / 2
            self.camera_pos_y = self.camera_pos_y + self.screenHmm
        elif self.screen_orientation == 2:
            self.camera_pos_x = -self.camera_pos_y
            self.camera_pos_y = -self.camera_pos_x + self.screenHmm / 2
        elif self.screen_orientation == 3:
            self.camera_pos_x = self.camera_pos_y + self.screenWmm
            self.camera_pos_y = self.camera_pos_x + self.screenHmm / 2
        else:
            self.camera_pos_x =  self.camera_pos_x + self.screenWmm / 2
            self.camera_pos_y = -self.camera_pos_y
        
        #self.screen_o_cam_r = np.array([self.data["rotCam"]["x"], self.data["rotCam"]["y"], self.data["rotCam"]["z"]])

        self.screen_o_cam_t = np.array([self.camera_pos_x, self.camera_pos_y, self.camera_pos_z])  # Convert to cm
        self.screenO_px_cam = np.array([self.screen_o_cam_t[0] * self.screenWpx / self.screenWmm, self.screen_o_cam_t[1] * self.screenHpx / self.screenHmm])

    def screenSetup(self):
        screenFrame_screen = np.array([[0, 0, 0], [self.screenWmm, 0, 0], [self.screenWmm, self.screenHmm, 0], [0, self.screenHmm, 0]])
        self.screenFrame_cam    = scene.transformMhProws(self.Mcam_screen, screenFrame_screen)
        self.screen = scene.Screen(self.screenWmm, self.screenHmm, self.screenWpx, self.screenHpx, self.zoom)

    def headSetup(self):
        # Head model
        head3D_head = load_head_model()*10
        self.head3D_cam = scene.transformMhProws(self.Mhead_cam, head3D_head)

        # Eyes centers 3D
        rightOut_cam = self.head3D_cam[36]
        rightIn_cam  = self.head3D_cam[39]
        leftOut_cam  = self.head3D_cam[45]
        leftIn_cam   = self.head3D_cam[42]
        self.rightEye3DCenter_cam    = 0.5 * (rightOut_cam + rightIn_cam)
        self.leftEye3DCenter_cam     = 0.5 * (leftOut_cam + leftIn_cam)

        # Head projection
        projeMat = scene.computeCameraMatrix((self.height, self.height), (0.5*self.height, 0.5*self.width))
        head2D_cam   = scene.projectPoints(projeMat, self.head3D_cam)
        rightOut2D_cam = head2D_cam[36]
        rightIn2D_cam  = head2D_cam[39]
        leftOut2D_cam  = head2D_cam[45]
        leftIn2D_cam   = head2D_cam[42]
        self.rightVector2D_cam = rightIn2D_cam - rightOut2D_cam
        self.leftVector2D_cam  = leftOut2D_cam - leftIn2D_cam
        self.rightCenter2D_cam = 0.5 * (rightOut2D_cam + rightIn2D_cam)
        self.leftCenter2D_cam = 0.5 * (leftOut2D_cam + leftIn2D_cam)

    def addHeadPrediction(self, predictionHead3D_cam, predictionMhead_cam, predictionVRhead_cam, predictionVThead_cam):
        predHeadAxis_head  = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.predHeadAxis_cam  = scene.transformMhProws(predictionMhead_cam, predHeadAxis_head)
        self.predictionHead3D_cam = Read_gaze_data.__convert_head_format(predictionHead3D_cam)
        self.predictionHead3D_cam  = scene.transformMhProws(predictionMhead_cam, self.predictionHead3D_cam)

        vec_rot, x_rot, y_rot = Read_gaze_data.__calc_vec_and_xy_angles_error(self.hpe[0:3], predictionVRhead_cam, error_per_component=True)
        euclidean_error, [x_error, y_error, z_error] = Read_gaze_data.__calc_dist_error(self.hpe[3:6], predictionVThead_cam)
        
        print(
            "HPE_r error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, HPE_r gt: {}, HPE_t pred: {}".format(
                 vec_rot, x_rot, y_rot, self.hpe[0:3], predictionVRhead_cam
            )
        )

        print(
            "HPE_t error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, z_error: {:.2f}, HPE_t gt: {}, HPE_t pred: {}".format(
                euclidean_error, x_error, y_error, z_error, self.hpe[3:6], predictionVThead_cam, 
            )
        )

    @staticmethod
    def __convert_head_format(predictionHead3D_cam: np.ndarray):

        head_3D = np.array([np.array([-headAux.x, headAux.y, headAux.z]) for headAux in predictionHead3D_cam])

        return head_3D
        # main_key_features_indices = [
        #     33,  # Left eye outer corner
        #     133, # Left eye inner corner
        #     362, # Right eye outer corner
        #     263, # Right eye inner corner
        #     1,   # Nose tip
        #     61,  # Mouth left corner
        #     291, # Mouth right corner
        # ]

        
    def addGazePrediction(self, gazePrediction_cam: np.ndarray, eye3DPredictionCenter_cam: np.ndarray, pogPredictionRawPx_screen: np.ndarray, pogPredictionPx_screen: np.ndarray, pog_cm_cam: np.ndarray):        
        self.eye3DPredictionCenter_cam = eye3DPredictionCenter_cam
        self.gazePrediction = gazePrediction_cam
        self.pogPrediction_cam = pog_cm_cam

        #self.pogPrediction_cam = self.cameraPlane_cam.intersect(self.eye3DPredictionCenter_cam, self.gazePrediction)
        
        vec_rot_error, x_rot, y_rot = Read_gaze_data.__calc_vec_and_xy_angles_error(self.gaze, self.gazePrediction, error_per_component=True)
        print(
            "gaze_error:  {:.2f}, x_error: {:.2f}, y_error: {:.2f}, gaze gt: {}, gaze pred: {}".format(
                vec_rot_error, x_rot, y_rot, self.gaze, self.gazePrediction
            )
        )

        eye3d_error, [x_error, y_error, z_error] = Read_gaze_data.__calc_dist_error(self.eye3DCenter_cam, self.eye3DPredictionCenter_cam)

        print(
            "eye3d_error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, z_error: {:.2f}, eye3d gt: {}, eye3d pred: {}".format(
                eye3d_error, x_error, y_error, z_error, self.eye3DCenter_cam, self.eye3DPredictionCenter_cam
            )
        )

        # Pogcm_cam
        self.pogPredictionPx_screen = pogPredictionPx_screen
        self.pogPredictionRawPx_screen = pogPredictionRawPx_screen
        
        pogPX_error, [x_error, y_error] = Read_gaze_data.__calc_dist_error(self.pogPx_screen, self.pogPredictionPx_screen)

        print(
            "pogPX_error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, pogPX gt: {}, pogPX pred: {}".format(
                pogPX_error, x_error, y_error, self.pogPx_screen, self.pogPredictionPx_screen
            )
        )

        pogMM_error, [x_error, y_error, z_error] = Read_gaze_data.__calc_dist_error(pog_cm_cam, self.pogPrediction_cam)

        print(
            "pogMM_error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, z_error: {:.2f}, pogMM gt: {}, pogMM pred: {}".format(
                pogMM_error, x_error, y_error, z_error, pog_cm_cam, self.pogPrediction_cam
            )
        )

    def plot3D(self):
        #Cam coordinate system
        example = trackingEnviroment3DPlotter()
        example.appendAxis(self.cameraAxis_cam, 'CameraAx')
        example.appendAxis(self.screenAxis_cam, 'ScreenAx')
        example.appendAxis(self.headAxis_cam, 'HeadAx')
        example.appendHead(self.head3D_cam, 'Head')
        example.appendScreen(self.screenFrame_cam, 'screen')
        example.appendPoint(self.pogMm_cam, 'PoG')
        example.appendPoint(self.eye3DCenter_cam, 'eye3DCenter')
        example.appendGaze(self.pogMm_cam, self.eye3DCenter_cam, 'Gaze')
        example.appendInfGaze(self.eye3DCenter_cam, self.gaze, 'InfGaze')

        if self.predictionHead3D_cam is not None:
            example.appendAxis(self.predHeadAxis_cam, 'predHeadAx')
            example.appendHead(self.predictionHead3D_cam, 'predHead')

        if self.gazePrediction is not None:
            example.appendGaze(self.pogPrediction_cam, self.eye3DPredictionCenter_cam, 'GazePrediction')
            example.appendPoint(self.pogPrediction_cam, 'PoGPrediction')
            example.appendPoint(self.eye3DPredictionCenter_cam, 'eye3DPredictionCenter')
            radius = math.dist(np.squeeze(self.pogMm_cam), np.squeeze(self.pogPrediction_cam))
            example.appendCircle(self.pogMm_cam, radius, 'ErrorPrediction')
            example.appendInfGaze(self.eye3DPredictionCenter_cam, self.gazePrediction, 'InfGazePred')
        
        if self.pogCmCalibratedPrediction_cam != []:
            example.appendPoint(self.pogCmCalibratedPrediction_cam, 'PoGCalibratedPrediction')
            radius = math.dist(np.squeeze(self.pogMm_cam), np.squeeze(self.pogCmCalibratedPrediction_cam))
            example.appendCircle(self.pogMm_cam, radius, 'ErrorCalibrated')
        
        if self.crop_format:
            example.appendInfGaze(self.eye3DCenter_cam, self.gaze_raw, 'InfGaze_raw')
            example.appendInfGaze(self.eye3DCenter_cam, self.gaze_corrected, 'InfGaze_corrected')
            for i, gaze_aux in enumerate(self.gaze_corrected_range):
                example.appendInfGaze(self.eye3DCenter_cam, gaze_aux, f'InfGaze_corrected_range_{i}')

        example.plot()

    def plot2D(self, write = False):
        example2D = trackingScreen2DPlotter(self.screenWpx, self.screenHpx, zoom = self.zoom)
        example2D.appendCircle(point=self.pogPx_screen, radius=30, color=(255, 20, 255))

        if self.pogPredictionRawPx_screen.size > 0:
            example2D.appendPoint(self.pogPredictionPx_screen, (20, 255, 255))
            example2D.appendPoint(self.pogPredictionRawPx_screen, (255, 255, 20))
        
        self.screenO_px_cam = np.array([self.screenO_px_cam[0], self.screenO_px_cam[1]])

        # if self.crop_format:
        #     example2D.appendPoint(self.pogPredictionRawPx_screen, (255, 255, 20))
        example2D.appendCamera(self.screenO_px_cam, (255, 20, 20))

        fig = example2D.plot()

        if write:
            cv2.imwrite("img_2d.png", fig)
        else:       
            imgplot = plt.imshow(fig)
            plt.show()
    

    @staticmethod
    def __calc_vec_and_xy_angles_error(vector1, vector2, error_per_component=False):
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(vector1, vector2)
        vec_rot = np.abs(np.degrees(np.arccos(dot_product))).item()

        if error_per_component:      
            x_rot = np.degrees(np.abs(vector1[0] - vector2[0])).item()
            y_rot = np.degrees(np.abs(vector1[1] - vector2[1])).item()
            return vec_rot, x_rot, y_rot
        else:
            return vec_rot
        
    @staticmethod
    def __calc_dist_error(point1, point2, abs = False):
        # Ensure both points are 3-dimensional
        if len(point1) > 3 or len(point2) > 3:
            raise ValueError("Both points must be 3-dimensional")
        
        # Calculate dimension-wise errors
        if abs:
            dimension_errors = [abs(point1[i] - point2[i]) for i in range(len(point1))]
        else:
            dimension_errors = [(point1[i] - point2[i]) for i in range(len(point1))]
        # Calculate Euclidean distance
        euclidean_distance = math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))))

        return euclidean_distance, dimension_errors