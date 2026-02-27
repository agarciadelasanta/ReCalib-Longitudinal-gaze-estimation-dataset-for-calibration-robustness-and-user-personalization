import cv2
import json
import math
import numpy as np
from scene_operations import Plane, rotateVector, createBothRotMat, transformMhProws, computeCameraMatrix, projectPoints
from head_model import load_head_model
from scene_3d_plots import trackingEnviroment3DPlotter
from scene_2d_plots import trackingScreen2DPlotter
from math import pi
from utils import pog_converter_from_cm_2_px, pog_calc, Setup_specs
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
        
        self.cameraPlane_cam = Plane(np.array([0,0,0]), np.array([0,0,1]))
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
        self.gaze_corrected = rotateVector(np.asarray(self.gaze, dtype=np.float32), np.array([0, 0, -self.roll]))
        self.gaze_corrected = self.gaze_corrected / np.linalg.norm(self.gaze_corrected)
        self.gaze = self.gaze_corrected
        roll_values = np.linspace(-np.pi, np.pi, 10)

        self.gaze_corrected_range = []
        
        for roll_aux in roll_values:
            self.gaze_corrected_range.append(rotateVector(np.asarray(self.gaze, dtype=np.float32), np.array([0, 0, -roll_aux])))

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
        # Image and POG
        self.pogPx_screen = (self.pos[0], self.pos[1])

        # Pog Px -> Cam (using setup_specs instead of standalone Screen)
        Pogcm_screen = self.setup_specs.from_px_to_cm(self.pogPx_screen)
        self.pog_cm_cam = transformMhProws(self.setup_specs.Mcam_screen, Pogcm_screen)

    def rotationMatrixGen(self):
        ### Rotation matrix — use the transforms already computed by setup_specs
        self.VRhead_cam = np.array(self.hpe[0:3])
        self.VThead_cam = np.array(self.hpe[3:6])
        self.Mhead_cam, self.Mcam_head = createBothRotMat(self.VRhead_cam, self.VThead_cam)
        
    def axisDefinition(self):
        #head axis
        headAxis_head  = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.headAxis_cam  = transformMhProws(self.Mhead_cam, headAxis_head)

        # Camera axes
        self.cameraAxis_cam = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

        # screen axis
        screenAxis_screen  = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.screenAxis_cam     = transformMhProws(self.setup_specs.Mcam_screen, screenAxis_screen)

    def screenParametersSetup(self):
        """Build self.setup_specs from JSON data – single source of truth."""

        # --- raw values from JSON ------------------------------------------
        screen_width_mm  = self.data["screen_mm"]["width"]
        screen_height_mm = self.data["screen_mm"]["height"]
        screen_width_px  = self.data["screen_pixels"]["width"]
        screen_height_px = self.data["screen_pixels"]["height"]
        img_height       = self.data["img_res"]["height"]
        img_width        = self.data["img_res"]["width"]

        cam_x, cam_y, cam_z = (
            self.data["posCam_mm"]["x"],
            self.data["posCam_mm"]["y"],
            self.data["posCam_mm"]["z"],
        )

        if ("rotCam" in self.data
                and "x" in self.data["rotCam"]
                and "y" in self.data["rotCam"]
                and "z" in self.data["rotCam"]):
            cam_rx = self.data["rotCam"]["x"]
            cam_ry = self.data["rotCam"]["y"]
            cam_rz = self.data["rotCam"]["z"]
        else:
            cam_rx, cam_ry, cam_rz = 0.0, pi, 0.0

        # --- build Setup_specs (handles orientation + zoom internally) ------
        self.setup_specs = Setup_specs(
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            screen_width_mm=screen_width_mm,
            screen_height_mm=screen_height_mm,
            screen_orientation=self.screen_orientation,
            zoom=self.zoom,
            img_width=img_width,
            img_height=img_height,
        )

        # Apply the orientation-dependent camera position tweaks
        self.setup_specs.update_webcam_params(
            camera_pos_x=cam_x,
            camera_pos_y=cam_y,
            camera_pos_z=cam_z,
            camera_rot_x=cam_rx,
            camera_rot_y=cam_ry,
            camera_rot_z=cam_rz,
        )

    def screenSetup(self):
        self.screenFrame_cam = transformMhProws(
            self.setup_specs.Mcam_screen,
            self.setup_specs.screen_frame_screen,
        )

    def headSetup(self):
        # Head model
        head3D_head = load_head_model()*10
        self.head3D_cam = transformMhProws(self.Mhead_cam, head3D_head)

        # # Eyes centers 3D
        # rightOut_cam = self.head3D_cam[36]
        # rightIn_cam  = self.head3D_cam[39]
        # leftOut_cam  = self.head3D_cam[45]
        # leftIn_cam   = self.head3D_cam[42]
        # self.rightEye3DCenter_cam    = 0.5 * (rightOut_cam + rightIn_cam)
        # self.leftEye3DCenter_cam     = 0.5 * (leftOut_cam + leftIn_cam)

        # # Head projection
        # h = self.setup_specs.img_height
        # w = self.setup_specs.img_width
        # projeMat = computeCameraMatrix((h, h), (0.5 * h, 0.5 * w))
        # head2D_cam   = projectPoints(projeMat, self.head3D_cam)
        # rightOut2D_cam = head2D_cam[36]
        # rightIn2D_cam  = head2D_cam[39]
        # leftOut2D_cam  = head2D_cam[45]
        # leftIn2D_cam   = head2D_cam[42]
        # self.rightVector2D_cam = rightIn2D_cam - rightOut2D_cam
        # self.leftVector2D_cam  = leftOut2D_cam - leftIn2D_cam
        # self.rightCenter2D_cam = 0.5 * (rightOut2D_cam + rightIn2D_cam)
        # self.leftCenter2D_cam = 0.5 * (leftOut2D_cam + leftIn2D_cam)

    def addHeadPrediction(self, predictionHead3D_cam, predictionMhead_cam, predictionVRhead_cam, predictionVThead_cam):
        predHeadAxis_head  = 50*np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.predHeadAxis_cam  = transformMhProws(predictionMhead_cam, predHeadAxis_head)
        self.predictionHead3D_cam = Read_gaze_data.__convert_head_format(predictionHead3D_cam)
        self.predictionHead3D_cam  = transformMhProws(predictionMhead_cam, self.predictionHead3D_cam)

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

    def calc_pog_from_gaze_vec(self, eye_gaze_vector_cam, eye_center_3d_cam):
        """Compute PoG in both mm (cam frame) and px (screen frame)."""
        R_rot = self.setup_specs.Mcam_screen[:3, :3]
        eye_gaze_vector_screen = R_rot.T @ eye_gaze_vector_cam
        eye_center_3d_screen = transformMhProws(self.setup_specs.Mscreen_cam, eye_center_3d_cam)
        pog_cm_screen = pog_calc(eye_gaze_vector_screen, eye_center_3d_screen)
        pog_cm_cam = transformMhProws(self.setup_specs.Mcam_screen, pog_cm_screen)
        pog_px = pog_converter_from_cm_2_px(pog_cm_screen, self.setup_specs)
        return pog_cm_cam, pog_px

    @staticmethod
    def __convert_head_format(predictionHead3D_cam: np.ndarray):

        head_3D = np.array([np.array([-headAux.x, headAux.y, headAux.z]) for headAux in predictionHead3D_cam])

        return head_3D

    def addGazePrediction(self, gazePrediction):        
        self.eye3DPredictionCenter_cam = np.array(self.data["gaze"]["origin"])
        self.gazePrediction = gazePrediction

        self.pogPrediction_cam, self.pogPredictionPx_screen = self.calc_pog_from_gaze_vec(self.gazePrediction, self.eye3DPredictionCenter_cam)

        
        vec_rot_error, x_rot, y_rot = Read_gaze_data.__calc_vec_and_xy_angles_error(self.gaze, self.gazePrediction, error_per_component=True)
        print(
            "gaze_error:  {:.2f}, x_error: {:.2f}, y_error: {:.2f}, gaze gt: {}, gaze pred: {}".format(
                vec_rot_error, x_rot, y_rot, self.gaze, self.gazePrediction
            )
        )

        # Pogcm_cam
        
        pogPX_error, [x_error, y_error] = Read_gaze_data.__calc_dist_error(self.pogPx_screen, self.pogPredictionPx_screen)

        print(
            "pogPX_error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, pogPX gt: {}, pogPX pred: {}".format(
                pogPX_error, x_error, y_error, self.pogPx_screen, self.pogPredictionPx_screen
            )
        )

        pogMM_error, [x_error, y_error, z_error] = Read_gaze_data.__calc_dist_error(self.pog_cm_cam, self.pogPrediction_cam)

        print(
            "pogMM_error: {:.2f}, x_error: {:.2f}, y_error: {:.2f}, z_error: {:.2f}, pogMM gt: {}, pogMM pred: {}".format(
                pogMM_error, x_error, y_error, z_error, self.pog_cm_cam, self.pogPrediction_cam
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

    def plot2D(self, write=False):
        ss = self.setup_specs
        example2D = trackingScreen2DPlotter(ss.screen_width_px, ss.screen_height_px, zoom=ss.zoom)
        example2D.appendCircle(point=self.pogPx_screen, radius=30, color=(255, 20, 255))

        if self.pogPredictionRawPx_screen.size > 0:
            example2D.appendPoint(self.pogPredictionPx_screen, (20, 255, 255))
            example2D.appendPoint(self.pogPredictionRawPx_screen, (255, 255, 20))

        example2D.appendCamera(ss.camera_pos_px, (255, 20, 20))

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
    def __calc_dist_error(point1, point2, use_abs=False):
        # Ensure both points are 3-dimensional
        if len(point1) > 3 or len(point2) > 3:
            raise ValueError("Both points must be 3-dimensional")
        
        # Calculate dimension-wise errors
        if use_abs:
            dimension_errors = [abs(point1[i] - point2[i]) for i in range(len(point1))]
        else:
            dimension_errors = [(point1[i] - point2[i]) for i in range(len(point1))]
        # Calculate Euclidean distance
        euclidean_distance = math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))))

        return euclidean_distance, dimension_errors