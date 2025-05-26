from flask import Blueprint, jsonify, request, current_app
from PIL import Image
import logging
import traceback               
from app.services.model_service import load_model, predict_disease

api_bp = Blueprint('api', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)

@api_bp.route('/predict', methods=['POST'])
def predict():
    # Carrega o modelo (se ainda não tiver carregado)
    if not load_model():
        return jsonify({"status": "error", "message": "Falha ao carregar o modelo"}), 500

    # Valida envio do arquivo
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Nenhum arquivo enviado"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Nome de arquivo inválido"}), 400

    # Tenta abrir a imagem
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"status":"error","message":"Formato de imagem inválido"}), 400

    # Executa a predição
    try:
        result = predict_disease(img)
        return jsonify({"status": "success", "data": result}), 200

    except Exception as e:
        # imprime stack completo no log
        tb = traceback.format_exc()
        current_app.logger.error("Erro ao predizer:\n%s", tb)
        # devolve mensagem e parte do traceback (só em dev)
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": tb.splitlines()[-5:]
        }), 500
