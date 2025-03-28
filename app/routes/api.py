#Rotas da api
import os
from flask import Blueprint, jsonify, request, current_app
from app.services.model_service import load_model, predict_disease
from app.services.s3_service import download_image_from_url, load_image_from_path
import logging
from werkzeug.utils import secure_filename

# Configuração de logging
logger = logging.getLogger(__name__)

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
            return jsonify({
                'status': 'error',
                'message': 'Modelo não carregado'
            }), 500
    
    except Exception as e:
        logger.error(f"Erro na verificação: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    #Endpoint para fazer previsões de doenças em plantas
    {
        "image_path": "data/sample_images/planta.jpg" # EXEMPLO DO CAMINHO DA FOTO
    }
    try:
        # Obter dados da requisição
        data = request.get_json()
        
        # Processar imagem a partir do caminho
        image = None
        
        if data and 'image_url' in data:
            logger.info(f"Processando imagem da URL: {data['image_url']}")
            image = download_image_from_url(data['image_url'])
        elif data and 'image_path' in data:
            logger.info(f"Processando imagem local: {data['image_path']}")
            image = load_image_from_path(data['image_path'])
        else:
            return jsonify({
                'status': 'error', 
                'message': 'URL ou caminho da imagem não fornecido'
            }), 400
        
        if image is None:
            return jsonify({
                'status': 'error',
                'message': 'Não foi possível carregar a imagem'
            }), 400    
    
        result = predict_disease(image)        
        
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
    #Endpoint para fazer upload de uma imagem e obter previsão
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo enviado'
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo selecionado'
            }), 400
            
        # Salvar o arquivo temporariamente
        uploads_dir = os.path.join(current_app.root_path, '../data/uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(uploads_dir, filename)
        file.save(filepath)
        
        image = load_image_from_path(filepath)
        
        if image is None:
            return jsonify({
                'status': 'error',
                'message': 'Não foi possível processar a imagem'
            }), 400
            
        result = predict_disease(image)
        
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao processar upload e previsão: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500