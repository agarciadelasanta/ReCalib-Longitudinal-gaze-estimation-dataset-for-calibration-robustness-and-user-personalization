import numpy as np
from math import pi
from typing import Tuple
from dataclasses import dataclass, field

from os.path import exists, splitext, islink, join, isdir
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

NDArrayF = NDArray[np.float64]


@dataclass
class Setup_specs:
    """Centralised screen + camera setup.

    Construct with raw JSON values, then (optionally) call
    ``update_webcam_params`` to apply orientation-dependent camera tweaks.
    """

    # ── screen -------------------------------------------------------------
    screen_width_px : int  = 0
    screen_height_px: int  = 0
    screen_width_mm : float = 0.0
    screen_height_mm: float = 0.0
    screen_orientation: int = 0        # 0-3
    zoom: float = 1.0

    # ── camera -------------------------------------------------------------
    camera_pos_x : float = 0.0         # mm (screen frame)
    camera_pos_y : float = 0.0
    camera_pos_z : float = 500.0
    camera_rot_x : float = 0.0         # rad
    camera_rot_y : float = pi
    camera_rot_z : float = 0.0

    # ── image resolution (kept here for convenience) -----------------------
    img_width : int = 0
    img_height: int = 0

    # ── derived (filled in __post_init__) ----------------------------------
    width_ratio : float = field(init=False, default=0.0)
    height_ratio: float = field(init=False, default=0.0)
    Mscreen_cam : NDArrayF = field(init=False, default=None)
    Mcam_screen : NDArrayF = field(init=False, default=None)

    # ───────────────────────────────────────────────────────────────────────
    # Initialise / update helpers
    # ───────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        # handle portrait orientations
        if self.screen_orientation in (2, 3):
            self.screen_width_px, self.screen_height_px = self.screen_height_px, self.screen_width_px
            self.screen_width_mm, self.screen_height_mm = self.screen_height_mm, self.screen_width_mm

        # apply zoom
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
        """Apply orientation-dependent camera position tweaks."""
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
            case _:  # fallback – same as landscape
                self.camera_pos_x = camera_pos_x + self.screen_width_mm / 2
                self.camera_pos_y = -camera_pos_y

        self.camera_pos_z  = camera_pos_z
        self.camera_rot_x  = camera_rot_x
        self.camera_rot_y  = camera_rot_y
        self.camera_rot_z  = camera_rot_z
        self._recompute()

    # ───────────────────────────────────────────────────────────────────────
    # Coordinate conversions (replaces the old Screen helper)
    # ───────────────────────────────────────────────────────────────────────
    def from_px_to_cm(self, Ppx) -> NDArrayF:
        """Convert a 2-D pixel coordinate to mm in the screen frame (z=0)."""
        Ppx = np.squeeze(Ppx)
        xcm = Ppx[0] / self.screen_width_px  * self.screen_width_mm
        ycm = Ppx[1] / self.screen_height_px * self.screen_height_mm
        return np.array([xcm, ycm, 0.0])

    def from_cm_to_px(self, Pcm) -> NDArrayF:
        """Convert a mm coordinate in the screen frame to pixels."""
        xpx = Pcm[0] / self.screen_width_mm  * self.screen_width_px
        ypx = Pcm[1] / self.screen_height_mm * self.screen_height_px
        return np.array([xpx, ypx])

    @property
    def screen_frame_screen(self) -> NDArrayF:
        """Four corners of the screen rectangle in screen-mm coords."""
        w, h = self.screen_width_mm, self.screen_height_mm
        return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]])

    @property
    def camera_pos_screen(self) -> NDArrayF:
        """Camera position vector in screen-mm coords."""
        return np.array([self.camera_pos_x, self.camera_pos_y, self.camera_pos_z])

    @property
    def camera_pos_px(self) -> NDArrayF:
        """Camera origin projected to screen pixels (2-D)."""
        return np.array([
            self.camera_pos_x * self.width_ratio - self.screen_width_px/(2*self.zoom),
            self.camera_pos_y * self.height_ratio // self.zoom,
        ])

    # ───────────────────────────────────────────────────────────────────────
    # Maths helpers
    # ───────────────────────────────────────────────────────────────────────
    def _recompute(self):
        if self.screen_width_mm > 0 and self.screen_height_mm > 0:
            self.width_ratio  = self.screen_width_px  / self.screen_width_mm
            self.height_ratio = self.screen_height_px / self.screen_height_mm
        else:
            self.width_ratio  = 0.0
            self.height_ratio = 0.0

        rot = R.from_euler(
            "xyz",
            (self.camera_rot_x, self.camera_rot_y, self.camera_rot_z),
        ).as_matrix()
        tra = np.array([self.camera_pos_x, self.camera_pos_y, self.camera_pos_z])

        self.Mscreen_cam, self.Mcam_screen = self._build_transforms(rot, tra)

    @staticmethod
    def _build_transforms(Rmat: NDArrayF, t: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        M = np.eye(4)
        M[:3, :3] = Rmat
        M[:3, 3] = t
        return M, np.linalg.inv(M)


# ───────────────────────────────────────────────────────────────────────────
# Standalone utility functions
# ───────────────────────────────────────────────────────────────────────────

def checkIfisAValidPNGPair(pngFile):
    if not (pngFile.endswith('.png') or pngFile.endswith('.jpg')):
        return False, "", ""

    auxFileJsonName = splitext(pngFile)[0] + '.json'

    if not exists(auxFileJsonName):
        return False, "", ""

    return True, pngFile, auxFileJsonName


def pog_converter_from_cm_2_px(
    pog_cm: NDArrayF | tuple[float, float], setup_specs: Setup_specs
) -> NDArrayF:
    pog_cm = np.asarray(pog_cm, dtype=float)
    return np.array([
        pog_cm[0] * setup_specs.width_ratio,
        pog_cm[1] * setup_specs.height_ratio,
    ])


def pog_calc(
    eye_gaze_vector_screen: NDArrayF, eye_center_screen: NDArrayF
) -> NDArrayF:
    """Intersect the gaze ray with the *physical* screen plane (z=0)."""
    n = np.array([0.0, 0.0, 1.0])
    u = eye_gaze_vector_screen / np.linalg.norm(eye_gaze_vector_screen)
    s = -(n @ eye_center_screen) / (n @ u)
    return eye_center_screen + s * u  # in mm