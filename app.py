import os
import sys
from pathlib import Path

# Add src/ directory to sys.path so hybrid_rag imports seamlessly
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_rag.settings import settings  # noqa: E402

# Ensure alembic migrations run if database_url is configured
if settings.database_url and "localhost" not in settings.database_url:
    try:
        from alembic.config import Config

        from alembic import command

        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        print("QueryFusion: Database migrations up to date.")
    except Exception as e:
        print(f"QueryFusion: Notice during migration check: {e}")

from hybrid_rag.app import create_app  # noqa: E402

fastapi_app = create_app()

try:
    import gradio as gr

    with gr.Blocks(title="QueryFusion RAG") as demo:
        gr.Markdown(
            "# QueryFusion RAG Platform\n\n"
            "The platform is running! Access the full interactive web workbench at **[Launch App](/)**."
        )

    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
except ImportError:
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
