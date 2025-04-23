# app/routes/api.py
import os
import logging
from flask import Blueprint, jsonify, request, current_app
from app.services.model_service import load_model, predict_disease
from app.services.s3_service import download_image_from_url, load_image_from_path
import boto3
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# Configuração de logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

# AWS S3 Configuração de conexão
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "**")

# PostgreSQL Configuração de Conexão
DATABASE_URL = os.getenv("DATABASE_URL", "**")
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Result(Base):
    __tablename__ = "**"
    id = Column(Integer, primary_key=True)
    image_url = Column(String)
    disease = Column(String)
    treatment = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Criar blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/health', methods=['GET'])
def health_check():
    try:
        model_status = load_model()
        if model_status:
            return jsonify({
                'status': 'ok',
                'message': 'API funcionando corretamente'
            }), 200
        else:
            logger.warning("Modelo não carregado")
            return jsonify({
                'status': 'error',
                'message': 'Modelo não carregado'
            }), 500
    except Exception as e:
        logger.error(f"Erro na verificação de saúde: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500

@api_bp.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()
        if not data or ('image_url' not in data and 'image_path' not in data):
            logger.warning("URL ou caminho da imagem não fornecido")
            return jsonify({
                'status': 'error',
                'message': 'URL ou caminho da imagem não fornecido'
            }), 400

        image = None
        temp_path = None
        if 'image_url' in data:
            logger.info(f"Processando imagem da URL: {data['image_url']}")
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            s3_client.download_file(BUCKET_NAME, data['image_url'].split('/')[-1], temp_path)
            image = load_image_from_path(temp_path)
        else:
            logger.info(f"Processando imagem local: {data['image_path']}")
            image = load_image_from_path(data['image_path'])

        if image is None:
            logger.error("Não foi possível carregar a imagem")
            return jsonify({
                'status': 'error',
                'message': 'Não foi possível carregar a imagem'
            }), 400

        result = predict_disease(image)
        
        # Salvar no PostgreSQL
        session = Session()
        new_result = Result(
            image_url=data.get('image_url', data.get('image_path')),
            disease=result.get('disease', 'unknown'),
            treatment=result.get('treatment', 'Consult an expert.')
        )
        session.add(new_result)
        session.commit()
        session.close()

        # Limpeza de arquivos
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'status': 'success',
            'data': result
        }), 200

    except Exception as e:
        logger.error(f"Erro ao processar previsão: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500

@api_bp.route('/upload-predict', methods=['POST'])
def upload_predict():
    
    try:
        if 'file' not in request.files:
            logger.warning("Nenhum arquivo enviado")
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo enviado'
            }), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("Nenhum arquivo selecionado")
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo selecionado'
            }), 400

        # Upload do S3
        filename = secure_filename(file.filename)
        s3_filename = f"plants/{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
        s3_client.upload_fileobj(file, BUCKET_NAME, s3_filename)
        image_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_filename}"

        # Download da imagem temporariamente para o processamento
        temp_path = f"temp_{s3_filename.split('/')[-1]}"
        s3_client.download_file(BUCKET_NAME, s3_filename, temp_path)
        image = load_image_from_path(temp_path)

        if image is None:
            logger.error("Não foi possível processar a imagem")
            return jsonify({
                'status': 'error',
                'message': 'Não foi possível processar a imagem'
            }), 400

        result = predict_disease(image)

        
        session = Session()
        new_result = Result(
            image_url=image_url,
            disease=result.get('disease', 'unknown'),
            treatment=result.get('treatment', 'Consult an expert.')
        )
        session.add(new_result)
        session.commit()
        session.close()

        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'status': 'success',
            'data': result,
            'image_url': image_url
        }), 200

    except Exception as e:
        logger.error(f"Erro ao processar upload e previsão: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500