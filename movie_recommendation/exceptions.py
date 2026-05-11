from typing import Any, Dict, Optional

from flask import jsonify, request


class AppError(Exception):
    status_code = 500

    def __init__(self, message: str, status_code: int = None, payload: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code or self.status_code
        self.payload = payload or {}

    def to_dict(self) -> Dict[str, Any]:
        data = {"message": self.message}
        data.update(self.payload)
        return data


class InvalidUsage(AppError):
    status_code = 400


class NotFoundError(AppError):
    status_code = 404


class ExternalAPIError(AppError):
    status_code = 502


def register_error_handlers(app):
    @app.errorhandler(AppError)
    def handle_app_error(error: AppError):
        app.logger.warning("AppError: %s", error.message)
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(404)
    def handle_not_found(error):
        app.logger.warning("Resource not found: %s", request.path)
        response = jsonify({"message": "Not found"})
        response.status_code = 404
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error):
        app.logger.exception("Unhandled exception:")
        response = jsonify({"message": "Internal server error"})
        response.status_code = 500
        return response
