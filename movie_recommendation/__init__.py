import os

from flask import Flask
from flask_cors import CORS

from .config import Config
from .exceptions import register_error_handlers
from .frontend import frontend_bp
from .logging_config import configure_logging
from .resources import load_resources
from .routes import api_bp
from .storage import init_app as init_storage


def create_app(config_object=None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(config_object or Config)

    configure_logging(app)
    register_error_handlers(app)
    CORS(app, origins=app.config.get("CORS_ORIGINS", ["*"]))

    init_storage(app)
    app.register_blueprint(frontend_bp)
    app.register_blueprint(api_bp)
    load_resources()

    app.logger.info("Flask application initialized")
    return app
