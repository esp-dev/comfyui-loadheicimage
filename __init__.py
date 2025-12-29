from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Frontend extension (drag&drop HEIC/HEIF upload handling)
WEB_DIRECTORY = "web"


def _register_preview_route() -> None:
	"""Serve HEIC/HEIF previews as PNG for browsers that can't render HEIC."""
	try:
		from aiohttp import web
		from io import BytesIO
		from PIL import Image, ImageOps

		from server import PromptServer
		import folder_paths

		from .nodes import _try_register_heif_opener

		instance = getattr(PromptServer, "instance", None)
		if instance is None or not hasattr(instance, "routes"):
			return

		# Avoid duplicate route registration if module reloads.
		if getattr(instance, "_heic_preview_route_registered", False):
			return
		instance._heic_preview_route_registered = True

		@instance.routes.get("/heic_preview")
		async def heic_preview(request: web.Request):
			filename = request.rel_url.query.get("filename")
			if not filename:
				return web.Response(status=400, text="filename is required")

			if not folder_paths.exists_annotated_filepath(filename):
				return web.Response(status=404, text="file not found")

			path = folder_paths.get_annotated_filepath(filename)

			_try_register_heif_opener()

			try:
				img = Image.open(path)
				img = ImageOps.exif_transpose(img)
				img = img.convert("RGBA")
			except Exception as e:
				return web.Response(status=500, text=f"failed to decode image: {e}")

			bio = BytesIO()
			img.save(bio, format="PNG")
			return web.Response(body=bio.getvalue(), content_type="image/png")

	except Exception:
		# Never fail import because preview route couldn't register.
		return


_register_preview_route()
