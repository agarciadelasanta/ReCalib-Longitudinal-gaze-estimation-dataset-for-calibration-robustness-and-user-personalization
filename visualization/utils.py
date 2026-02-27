from pathlib import Path
import numpy as np
from math import pi
from typing import Tuple

from os.path import exists, splitext, islink, join, isdir
from numpy.typing import NDArray

NDArrayF = NDArray[np.float64]


class Setup_specs:
    # ── screen -------------------------------------------------------------
    screen_width_px : int
    screen_height_px: int
    screen_width_mm : float
    screen_height_mm: float
    screen_orientation: int = 0        # 0-3 like before
    zoom: float = 1.0                  # legacy ‘zoom’

    # ── camera -------------------------------------------------------------
    camera_pos_x : float = 0.0         # mm (screen frame)
    camera_pos_y : float = 0.0
    camera_pos_z : float = 500.0
    camera_rot_x : float = 0.0         # rad
    camera_rot_y : float = pi
    camera_rot_z : float = 0.0

    # ── derived (filled in __post_init__) ----------------------------------
    width_ratio : float = 0.0          # [px / mm]
    height_ratio: float = 0.0
    Mscreen_cam : NDArrayF | None = None
    Mcam_screen: NDArrayF | None = None

    # ───────────────────────────────────────────────────────────────────────
    # Initialise / update helpers
    # ───────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        # apply zoom ONCE at construction
        self.screen_width_px  = int(self.screen_width_px  * self.zoom)
        self.screen_height_px = int(self.screen_height_px * self.zoom)
        self._recompute()

    def update_screen_features(
        self,
        screen_width_px : int,
        screen_height_px: int,
        screen_width_mm : float,
        screen_height_mm: float,
        screen_orientation: int,
        zoom: float = 1.0,
    ):
        """Legacy API – called by Mamu when screen data change."""
        if screen_orientation in (2, 3):  # portrait
            screen_width_px, screen_height_px = screen_height_px, screen_width_px
            screen_width_mm, screen_height_mm = screen_height_mm, screen_width_mm

        self.screen_width_px  = int(screen_width_px * zoom)
        self.screen_height_px = int(screen_height_px * zoom)
        self.screen_width_mm  = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.screen_orientation = screen_orientation
        self.zoom = zoom
        self._recompute()

    def update_webcam_params(
        self,
        camera_pos_x : float,
        camera_pos_y : float,
        camera_pos_z : float,
        camera_rot_x : float = 0.0,
        camera_rot_y : float = pi,
        camera_rot_z : float = 0.0,
    ):
        """Legacy API – keeps the same coordinate tweaks you had."""
        # position depends on orientation exactly like before
        match self.screen_orientation:
            case 0:  # landscape
                self.camera_pos_x = camera_pos_x + self.screen_width_mm / 2
                self.camera_pos_y = -camera_pos_y
            case 1:  # upside-down
                self.camera_pos_x = camera_pos_x + self.screen_width_mm / 2
                self.camera_pos_y = camera_pos_y + self.screen_height_mm
            case 2:  # 90° CCW
                self.camera_pos_x = -camera_pos_y
                self.camera_pos_y = -camera_pos_x + self.screen_height_mm / 2
            case 3:  # 90° CW
                self.camera_pos_x = camera_pos_y + self.screen_width_mm
                self.camera_pos_y = camera_pos_x + self.screen_height_mm / 2

        self.camera_pos_z, self.camera_rot_x, self.camera_rot_y, self.camera_rot_z = (
            camera_pos_z,
            camera_rot_x,
            camera_rot_y,
            camera_rot_z,
        )
        self._recompute()

    # ───────────────────────────────────────────────────────────────────────
    # Maths helpers
    # ───────────────────────────────────────────────────────────────────────
    def _recompute(self):
        self.width_ratio  = self.screen_width_px  / self.screen_width_mm
        self.height_ratio = self.screen_height_px / self.screen_height_mm

        rot = R.from_euler(
            "xyz",
            (self.camera_rot_x, self.camera_rot_y, self.camera_rot_z),
        ).as_matrix()
        tra = np.array([self.camera_pos_x, self.camera_pos_y, self.camera_pos_z])

        self.Mscreen_cam, self.Mcam_screen = self._build_transforms(rot, tra)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_transforms(Rmat: NDArrayF, t: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        M = np.eye(4)
        M[:3, :3] = Rmat
        M[:3, 3] = t
        return M, np.linalg.inv(M)

def checkIfisAValidPNGPair(pngFile):
    if not (pngFile.endswith('.png') or pngFile.endswith('.jpg')) :
        return False, "", ""

    auxFileJsonName = splitext(pngFile)[0]+'.json'

    if not (exists(auxFileJsonName)):
        return False, "", ""
    
    return True, pngFile, auxFileJsonName

@staticmethod
def pog_converter_from_cm_2_px(
    pog_cm: NDArrayF | tuple[float, float], setup_specs: Setup_specs
) -> NDArrayF:
    pog_cm = np.asarray(pog_cm, dtype=float)
    return np.array(
        [
            pog_cm[0] * setup_specs.width_ratio,
            pog_cm[1] * setup_specs.height_ratio,
        ]
    )

@staticmethod
def pog_calc(
    eye_gaze_vector_screen: NDArrayF, eye_center_screen: NDArrayF
) -> NDArrayF:
    """Intersect the gaze ray with the *physical* screen plane (z=0)."""
    n = np.array([0.0, 0.0, 1.0])
    u = eye_gaze_vector_screen / np.linalg.norm(eye_gaze_vector_screen)
    s = - (n @ eye_center_screen) / (n @ u)
    return eye_center_screen + s * u  # in mm