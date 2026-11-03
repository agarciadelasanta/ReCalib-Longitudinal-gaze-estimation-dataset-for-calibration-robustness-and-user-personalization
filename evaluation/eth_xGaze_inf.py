import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from model import gaze_network

# dlib / imutils are only needed for the landmark fallback and are imported lazily,
# so the JSON-landmark path runs without dlib installed.

# Single source of truth for the camera intrinsics: the released calibration of the acquisition
# webcam (1920x1080). OpenCV distortion order: k1, k2, p1, p2, k3.
DEFAULT_CAMERA_NPZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "docs", "camera_intrinsics.npz")


def load_camera_npz(npz_path=None):
    """Return (camera_matrix (3,3), distortion (5,1)) as float64; defaults to the released npz."""
    npz_path = npz_path or DEFAULT_CAMERA_NPZ
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"camera npz not found: {npz_path}")
    data = np.load(npz_path)
    return (np.asarray(data["mtx"], dtype=np.float64),
            np.asarray(data["dist"], dtype=np.float64).reshape(-1, 1))

# MediaPipe indices matching the ETH 6-point face model / dlib (36, 39, 42, 45, 31, 35):
# image-left eye outer, inner, image-right eye inner, outer, image-left nose, image-right nose.
MEDIAPIPE_ETH_LANDMARKS = (33, 133, 362, 263, 129, 358)
DLIB_ETH_LANDMARKS = [36, 39, 42, 45, 31, 35]


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    _, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        _, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec


def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    """Return the warped face patch, warped landmarks, the normalization rotation R and
    the ETH face centre in camera coordinates (mm, shape (3,))."""
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
    return img_warped, landmarks_warped, R, face_center.reshape(3)


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


def load_face_model(face_model_path: str):
    """Return the (6,3) ETH face model used for solvePnP."""
    if not os.path.isfile(face_model_path):
        raise FileNotFoundError(f"face model not found: {face_model_path}")
    face_model_load = np.loadtxt(face_model_path)
    landmark_use = [20, 23, 26, 29, 15, 19]  # eye corners + nose corners
    return face_model_load[landmark_use, :].astype(np.float64)


def landmarks_from_json(data) -> np.ndarray | None:
    """(6,1,2) landmarks in ETH order from a ReCalib JSON dict, or None if a key is missing."""
    mesh = data.get("head_pose", {}).get("mediapipe_face_mesh_2d")
    if mesh is None:
        return None
    try:
        pts = [mesh[str(i)] for i in MEDIAPIPE_ETH_LANDMARKS]
    except KeyError:
        return None
    return np.asarray(pts, dtype=np.float64).reshape(6, 1, 2)


def dlib_eth_landmarks(detector, predictor, image_bgr, cnn=False) -> np.ndarray | None:
    """(6,1,2) landmarks in ETH order from a dlib detector/predictor, or None if no face is found."""
    from imutils import face_utils
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 1)
    if len(dets) == 0:
        return None
    # CNN detector returns mmod rectangles with confidence
    face_rect = dets[0].rect if cnn else dets[0]
    shape = face_utils.shape_to_np(predictor(image_bgr, face_rect))
    return shape[DLIB_ETH_LANDMARKS, :].astype(np.float64).reshape(6, 1, 2)


def prepare_frame(image, landmarks6, face_model, camera, distortion):
    """Head pose + normalization for one frame.

    Returns (img_norm BGR 224x224, R, face_center_cam). R maps camera -> normalized
    coordinates, so a normalized gaze vector g_norm returns to the camera frame as R.T @ g_norm.
    """
    facePts = face_model.reshape(6, 1, 3)
    hr, ht = estimateHeadPose(landmarks6, facePts, camera, distortion)
    img_norm, _, R, face_center = normalizeData_face(image, face_model, landmarks6, hr, ht, camera)
    return img_norm, R, face_center


def denormalize_gaze(pitchyaw, R):
    """Normalized-space (pitch, yaw) -> unit gaze vector in the camera frame."""
    return (R.T @ pitchyaw_to_unit_vector(pitchyaw).astype(np.float64)).astype(np.float32)


class ETHXGazeEstimator:
    """
    Usage:
        est = ETHXGazeEstimator(
            face_model_path="face_model.txt",
            ckpt_path="./ckpt/epoch_24_ckpt.pth.tar",
            device="cuda",
            shape_predictor_path="./modules/shape_predictor_68_face_landmarks.dat",  # dlib fallback only
        )

        gaze_vec = est.predict_gaze_vector("some.png")  # camera-frame unit vector, shape (3,)

    Camera intrinsics default to docs/camera_intrinsics.npz. Pass camera_npz_path /
    camera_xml_path to override them. Inputs are the image and the intrinsics only: landmarks come
    from dlib on the image (landmark_source="dlib", needs shape_predictor_path). "json" uses the
    dataset's MediaPipe landmarks (corrected order) and "dlib_then_json" falls back to them; both
    use model estimates stored in the JSON, so they are not image-only.
    """

    def __init__(
        self,
        face_model_path: str,
        ckpt_path: str,
        device: str = "cpu",
        shape_predictor_path: str | None = None,
        camera_npz_path: str | None = None,
        camera_xml_path: str | None = None,
        use_cnn_detector: bool = False,
        cnn_detector_path: str | None = None,
        landmark_source: str = "dlib",
    ):
        if landmark_source not in ("dlib", "json", "dlib_then_json"):
            raise ValueError(f"unknown landmark_source: {landmark_source}")
        if landmark_source != "json" and shape_predictor_path is None:
            raise ValueError("landmark_source 'dlib' needs shape_predictor_path")
        self.landmark_source = landmark_source
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

        # dlib models (optional fallback)
        self.predictor = None
        self.face_detector = None
        self._cnn = False
        if shape_predictor_path is not None:
            import dlib
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

        # camera intrinsics
        if camera_xml_path and not camera_npz_path:
            if not os.path.isfile(camera_xml_path):
                raise FileNotFoundError(f"camera xml not found: {camera_xml_path}")
            fs = cv2.FileStorage(camera_xml_path, cv2.FILE_STORAGE_READ)
            self.camera_matrix = fs.getNode("Camera_Matrix").mat()
            self.camera_distortion = fs.getNode("Distortion_Coefficients").mat()
            fs.release()
        else:
            self.camera_matrix, self.camera_distortion = load_camera_npz(camera_npz_path)

        # make sure distortion has suitable shape
        self.camera_distortion = np.asarray(self.camera_distortion, dtype=np.float64).reshape(-1, 1)
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)

        # face model for solvePnP
        self.face_model = load_face_model(face_model_path)

        # gaze model
        if not os.path.isfile(ckpt_path):
            print(f"Pretrained model could not be found, please make sure you followed the instructions in evaluation/ckpt/README.txt")
            exit()
        self.model = gaze_network(pretrained=False).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

    def detect_landmarks_dlib(self, image_bgr) -> np.ndarray | None:
        """(6,1,2) landmarks in ETH order from dlib, or None if no face is found."""
        if self.predictor is None:
            raise RuntimeError("dlib fallback requested but shape_predictor_path was not given")
        return dlib_eth_landmarks(self.face_detector, self.predictor, image_bgr, self._cnn)

    def _landmarks_json(self, png_path):
        json_path = os.path.splitext(png_path)[0] + ".json"
        if not os.path.isfile(json_path):
            return None
        import json
        with open(json_path, "r") as f:
            return landmarks_from_json(json.load(f))

    def _landmarks(self, image, png_path):
        if self.landmark_source == "json":
            lm = self._landmarks_json(png_path)
        else:
            lm = self.detect_landmarks_dlib(image)
            if lm is None and self.landmark_source == "dlib_then_json":
                lm = self._landmarks_json(png_path)
        if lm is None:
            raise RuntimeError("No face detected / no landmarks available")
        return lm

    @torch.no_grad()
    def predict_normalized(self, png_path: str):
        """Return (pitch_yaw in the NORMALIZED frame, R, face_center_cam)."""
        image = cv2.imread(png_path)
        if image is None:
            raise ValueError(f"cv2.imread failed for: {png_path}")

        landmarks = self._landmarks(image, png_path)
        img_norm, R, face_center = prepare_frame(
            image, landmarks, self.face_model, self.camera_matrix, self.camera_distortion
        )

        inp = img_norm[:, :, [2, 1, 0]]  # BGR->RGB
        inp = self.trans(inp).float().to(self.device).unsqueeze(0)

        pred = self.model(inp)[0].detach().cpu().numpy()
        return pred, R, face_center

    @torch.no_grad()
    def predict_pitchyaw(self, png_path: str) -> np.ndarray:
        """Return (pitch, yaw) in the NORMALIZED frame (not comparable with camera-frame labels)."""
        return self.predict_normalized(png_path)[0]

    @torch.no_grad()
    def predict_gaze_vector(self, png_path: str) -> np.ndarray:
        """Return the 3D unit gaze vector in the CAMERA frame (R^T applied), shape (3,)."""
        pitchyaw, R, _ = self.predict_normalized(png_path)
        return denormalize_gaze(pitchyaw, R)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    est = ETHXGazeEstimator(
        face_model_path=os.path.join(REPO_ROOT, "evaluation", "face_model.txt"),
        ckpt_path=os.path.join(REPO_ROOT, "evaluation", "ckpt", "epoch_24_ckpt.pth.tar"),
        shape_predictor_path=os.path.join(REPO_ROOT, "evaluation", "modules", "shape_predictor_68_face_landmarks.dat"),
        device="cpu",
    )
    auxFileName = os.path.join(REPO_ROOT, "example", "07_00_02_img-0040.png")
    v = est.predict_gaze_vector(auxFileName)
    print("Gaze vector (camera frame):", v)
