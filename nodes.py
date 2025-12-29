import hashlib
import os

import numpy as np
import torch

from PIL import Image, ImageOps, ImageSequence


HEIC_EXTS = {".heic", ".heif"}


def _register_preview_route_if_possible() -> None:
    """Register /heic_preview once the PromptServer is available."""
    try:
        from aiohttp import web
        from io import BytesIO
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        if instance is None or not hasattr(instance, "routes"):
            return
        if getattr(instance, "_heic_preview_route_registered", False):
            return
        instance._heic_preview_route_registered = True

        @instance.routes.get("/heic_preview")
        async def heic_preview(request: web.Request):
            filename = request.rel_url.query.get("filename")
            if not filename:
                return web.Response(status=400, text="filename is required")

            folder_paths, _ = _get_comfy_modules()
            if not folder_paths.exists_annotated_filepath(filename):
                return web.Response(status=404, text="file not found")

            image_path = folder_paths.get_annotated_filepath(filename)
            _try_register_heif_opener()
            try:
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGBA")
            except Exception as e:
                return web.Response(status=500, text=f"failed to decode image: {e}")

            bio = BytesIO()
            img.save(bio, format="PNG")
            return web.Response(body=bio.getvalue(), content_type="image/png")
    except Exception:
        return


def _get_comfy_modules():
    try:
        import folder_paths  # provided by ComfyUI
        import node_helpers  # provided by ComfyUI
    except Exception as e:
        raise RuntimeError(
            "This node must run inside a ComfyUI environment (missing folder_paths/node_helpers)."
        ) from e

    return folder_paths, node_helpers


def _try_register_heif_opener() -> bool:
    """Register HEIF/HEIC opener for Pillow. Returns True if available."""
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        return True
    except Exception:
        return False


def _is_heic_path(name: str) -> bool:
    _, ext = os.path.splitext(str(name))
    return ext.lower() in HEIC_EXTS


def _list_heic_files_in_input_dir() -> list[str]:
    folder_paths, _ = _get_comfy_modules()
    input_dir = folder_paths.get_input_directory()
    try:
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
    except Exception:
        return []

    allowed_ext = {".heic", ".heif"}
    out = []
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in allowed_ext:
            out.append(f)

    return sorted(out)


def _list_image_files_in_input_dir() -> list[str]:
    """Return all supported image files (PNG, JPG, WEBP, HEIC/HEIF)."""
    folder_paths, _ = _get_comfy_modules()
    input_dir = folder_paths.get_input_directory()
    try:
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
    except Exception:
        return []

    # Support all common image formats + HEIC
    allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".gif"}
    out = []
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in allowed_ext:
            out.append(f)

    return sorted(out)


class LoadImagePlusHEIC:
    @classmethod
    def INPUT_TYPES(cls):
        _register_preview_route_if_possible()
        files = _list_image_files_in_input_dir()
        return {
            "required": {
                "image": (files, {"image_upload": True}),
            },
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        if _is_heic_path(image) and not _try_register_heif_opener():
            raise RuntimeError(
                "HEIC/HEIF support is not available. Install dependency: pip install pillow-heif"
            )
        else:
            _try_register_heif_opener()

        folder_paths, node_helpers = _get_comfy_modules()

        if not folder_paths.exists_annotated_filepath(image):
            raise FileNotFoundError(f"Image not found in input path: {image}")

        image_path = folder_paths.get_annotated_filepath(image)

        try:
            img = node_helpers.pillow(Image.open, image_path)
        except Exception as e:
            if _is_heic_path(image):
                raise RuntimeError(
                    "Failed to open HEIC/HEIF image. Ensure pillow-heif is installed: pip install pillow-heif\n"
                    f"File: {image}\nError: {e}"
                )
            raise RuntimeError(f"Failed to open image: {image}\nError: {e}")

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_t = torch.from_numpy(image_np)[None,]

            if "A" in i.getbands():
                mask_np = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask_t = 1.0 - torch.from_numpy(mask_np)
            elif i.mode == "P" and "transparency" in i.info:
                mask_np = (
                    np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                )
                mask_t = 1.0 - torch.from_numpy(mask_np)
            else:
                mask_t = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image_t)
            output_masks.append(mask_t.unsqueeze(0))

        if not output_images:
            raise RuntimeError(f"No frames could be decoded from image: {image}")

        if len(output_images) > 1 and getattr(img, "format", None) not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(cls, image):
        folder_paths, _ = _get_comfy_modules()
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                m.update(chunk)
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        folder_paths, _ = _get_comfy_modules()
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODE_CLASS_MAPPINGS = {
    "LoadImagePlusHEIC": LoadImagePlusHEIC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagePlusHEIC": "Load Image (HEIC)",
}
