import os
from flask import Flask
from app.routes import api as api_blueprint
from app.config.settings import Config

#Inicializar a aplicação Flask

def create_app(config_class=Config):
    """
    Factory pattern para criar a aplicação Flask
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Registrar blueprints
    app.register_blueprint(api_blueprint.api_bp)
    
    # Certificar que as pastas necessárias existem
    os.makedirs('logs', exist_ok=True)
    
    return app