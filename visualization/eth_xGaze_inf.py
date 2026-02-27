import os
import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms
from imutils import face_utils

from model import gaze_network


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    _, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        _, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec


def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    focal_norm = 960
    distance_norm = 600
    roiSize = (224, 224)

    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face_model.T) + ht

    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance

    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img, W, roiSize)

    landmarks_warped = cv2.perspectiveTransform(landmarks, W).reshape(landmarks.shape[0], 2)
    return img_warped, landmarks_warped


def pitchyaw_to_unit_vector(pitchyaw):
    """Convert (pitch, yaw) to 3D unit vector."""
    pitch = float(pitchyaw[0])
    yaw = float(pitchyaw[1])

    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)

    v = np.array([x, y, z], dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v


class ETHXGazeEstimator:
    """
    Usage:
        est = ETHXGazeEstimator(
            shape_predictor_path="./modules/shape_predictor_68_face_landmarks.dat",
            face_model_path="face_model.txt",
            ckpt_path="./ckpt/epoch_24_ckpt.pth.tar",
            camera_npz_path="intrinsicos_surface.npz",   # OR camera_xml_path="cam00.xml"
            device="cuda"
        )

        gaze_vec = est.predict_gaze_vector("some.png")  # returns np.array shape (3,)
    """

    def __init__(
        self,
        shape_predictor_path: str,
        face_model_path: str,
        ckpt_path: str,
        device: str = "cuda",
        camera_npz_path: str | None = None,
        camera_xml_path: str | None = None,
        use_cnn_detector: bool = False,
        cnn_detector_path: str | None = None,
    ):
        # device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # transforms
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # dlib models
        if not os.path.isfile(shape_predictor_path):
            raise FileNotFoundError(f"shape predictor not found: {shape_predictor_path}")
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        if use_cnn_detector:
            if not cnn_detector_path or not os.path.isfile(cnn_detector_path):
                raise FileNotFoundError(f"cnn detector not found: {cnn_detector_path}")
            self.face_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
            self._cnn = True
        else:
            self.face_detector = dlib.get_frontal_face_detector()
            self._cnn = False

        # camera intrinsics
        if camera_npz_path:
            if not os.path.isfile(camera_npz_path):
                raise FileNotFoundError(f"camera npz not found: {camera_npz_path}")
            data = np.load(camera_npz_path)
            self.camera_matrix = data["mtx"]
            self.camera_distortion = data["dist"]
        elif camera_xml_path:
            if not os.path.isfile(camera_xml_path):
                raise FileNotFoundError(f"camera xml not found: {camera_xml_path}")
            fs = cv2.FileStorage(camera_xml_path, cv2.FILE_STORAGE_READ)
            self.camera_matrix = fs.getNode("Camera_Matrix").mat()
            self.camera_distortion = fs.getNode("Distortion_Coefficients").mat()
            fs.release()
        else:
            raise ValueError("Provide either camera_npz_path or camera_xml_path")

        # make sure distortion has suitable shape
        self.camera_distortion = np.asarray(self.camera_distortion, dtype=np.float64).reshape(-1, 1)
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)

        # face model for solvePnP
        if not os.path.isfile(face_model_path):
            raise FileNotFoundError(f"face model not found: {face_model_path}")
        face_model_load = np.loadtxt(face_model_path)
        landmark_use = [20, 23, 26, 29, 15, 19]  # eye corners + nose corners
        self.face_model = face_model_load[landmark_use, :].astype(np.float64)  # (6,3)
        self.facePts = self.face_model.reshape(6, 1, 3)  # (6,1,3)

        # gaze model
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        self.model = gaze_network().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

    def _detect_first_face(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        dets = self.face_detector(rgb, 1)
        if len(dets) == 0:
            return None
        if self._cnn:
            # CNN detector returns mmod rectangles with confidence
            return dets[0].rect
        return dets[0]

    @torch.no_grad()
    def predict_pitchyaw(self, png_path: str) -> np.ndarray:
        """Return (pitch, yaw) as np.array shape (2,)."""
        image = cv2.imread(png_path)
        if image is None:
            raise ValueError(f"cv2.imread failed for: {png_path}")

        face_rect = self._detect_first_face(image)
        if face_rect is None:
            raise RuntimeError("No face detected")

        shape = self.predictor(image, face_rect)
        shape = face_utils.shape_to_np(shape)

        landmarks_sub = shape[[36, 39, 42, 45, 31, 35], :].astype(np.float64).reshape(6, 1, 2)

        hr, ht = estimateHeadPose(
            landmarks_sub,
            self.facePts,
            self.camera_matrix,
            self.camera_distortion,
        )

        img_norm, _ = normalizeData_face(image, self.face_model, landmarks_sub, hr, ht, self.camera_matrix)

        inp = img_norm[:, :, [2, 1, 0]]  # BGR->RGB
        inp = self.trans(inp).float().to(self.device).unsqueeze(0)

        pred = self.model(inp)[0].detach().cpu().numpy()
        return pred

    @torch.no_grad()
    def predict_gaze_vector(self, png_path: str) -> np.ndarray:
        """Return 3D unit gaze vector as np.array shape (3,)."""
        pitchyaw = self.predict_pitchyaw(png_path)
        return pitchyaw_to_unit_vector(pitchyaw)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    est = ETHXGazeEstimator(
        shape_predictor_path="./visualization/modules/shape_predictor_68_face_landmarks.dat",
        face_model_path="./visualization/face_model.txt",
        ckpt_path="./ckpt/epoch_24_ckpt.pth.tar",
        camera_npz_path="./example/input/intrinsicos_surface.npz",
        camera_xml_path=None,
        device="cuda",
    )
    auxFileName = r"C:\Projects\ReCalib-A-multi-session-gaze-dataset-for-calibration-robustness-and-user-adaptation\example\input\sample.png"
    v = est.predict_gaze_vector(auxFileName)
    print("Gaze vector:", v)