from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import settings
from .routers import characters, crops, pages, volumes

app = FastAPI(title="NomnaOCR Annotation Tool")

app.include_router(volumes.router)
app.include_router(pages.router)
app.include_router(crops.router)
app.include_router(characters.router)

if settings.IMAGE_ROOT.exists():
    app.mount("/static/images", StaticFiles(directory=settings.IMAGE_ROOT), name="images")

if settings.FRONTEND_DIST.exists():
    # Vite's hashed JS/CSS/font chunks live under dist/assets/.
    app.mount(
        "/assets",
        StaticFiles(directory=settings.FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    # React Router paths (e.g. /page/<volume>/<page>, /gallery) aren't real
    # files, so a plain StaticFiles mount 404s on direct navigation/refresh.
    # Serve the matching static file if one exists (favicon, etc.), else
    # fall back to index.html and let the client-side router take over.
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = settings.FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
