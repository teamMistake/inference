from datetime import datetime
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from .routers import JamoRouter
from .settings import get_settings


def create_app() -> FastAPI:
    app = FastAPI()

    init_cors(app)
    init_settings(app)
    init_middleware(app)
    init_router(app)
    return app


def init_settings(app: FastAPI):
    @app.on_event("startup")
    def startup_event():
        pass

    @app.get("/")
    async def index():
        """ELB check"""
        current_time = datetime.utcnow().strftime("%Y.%m.%d %H:%M:%S")
        msg = f"Notification API (UTC: {current_time})"
        return Response(msg)


def init_cors(app: FastAPI):
    setting = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=setting.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def init_middleware(app: FastAPI):
    pass


def init_router(app: FastAPI):
    app.include_router(JamoRouter)


app = create_app()
