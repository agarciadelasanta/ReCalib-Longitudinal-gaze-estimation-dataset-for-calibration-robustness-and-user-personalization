"""Stand-alone pseudo-labelling of ReCalib images (image + setup + screen target -> JSON)."""
from .labeller import Labeller, load_setup


def label_image(image, pog_px, setup, image_name=None, ids=None, model_path=None, depth_scale=1.0) -> dict:
    """Label a single image (convenience wrapper; use Labeller for many images)."""
    with Labeller(setup, model_path=model_path) as labeller:
        return labeller.label(image, pog_px, image_name=image_name, ids=ids, depth_scale=depth_scale)


__all__ = ["Labeller", "label_image", "load_setup"]
