"""WSGI/ASGI entrypoint expected by some deployment platforms.

This module exposes a variable named `app` (FastAPI instance) so platforms
like Vercel can discover the application. It imports the FastAPI app
defined in `src/predict_api.py`.

Run locally with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
try:
    # import the FastAPI app from src.predict_api
    from src.predict_api import app  # type: ignore
except Exception:
    # fallback: try importing from predict_api when running without package context
    from predict_api import app  # type: ignore


if __name__ == "__main__":
    # Allow running this file directly for local testing
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
