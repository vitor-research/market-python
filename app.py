import requests
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import warnings
from flask_cors import CORS # 1. Importe a biblioteca

warnings.filterwarnings("ignore")


app = Flask(__name__)

# 2. Habilite o CORS para todas as rotas e todas as origens
CORS(app)

COINS = [
    "BTC", "ETH", "SOL", "ARB", "OP", "WIF", "SUI", "APT", "RENDER", "NEAR", "AVAX", "LINK", "DOGE", "ONDO", "PYTH", "TIA", "SEI", "IMX"
]

# ==============================================================================
# CACHE GLOBAL EM MEMÓRIA RAM
# ==============================================================================
MODELS_CACHE = {}

def load_models_into_memory():
    global MODELS_CACHE
    MODELS_CACHE.clear()
    
    for coin in COINS:
        model_path = f"./models/{coin}_model.onnx"
        # Cria uma sessão de inferência para cada moeda
        MODELS_CACHE[coin] = rt.InferenceSession(model_path)
        
    print(f"[Sistema] {len(MODELS_CACHE)} modelos ONNX carregados na RAM!")

# Carrega os modelos assim que o script é executado
load_models_into_memory()

def get_data(coin, days=3):
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": int((datetime.now()-timedelta(days=days)).timestamp()*1000)}}
    try:
        response = requests.post(url, json=payload).json()
        df = pd.DataFrame(response)
        df[['o', 'h', 'l', 'c', 'v']] = df[['o', 'h', 'l', 'c', 'v']].astype(float)
        df['mid'] = (df['h'] + df['l']) / 2
        return df
    except: return None

def scan_opportunities(threshold_param):
    opportunities = []
    
    for coin in COINS:
        df = get_data(coin, days=3)
        print(coin)
        try:
            # Recalcular features
            tr = pd.concat([df['h']-df['l'], abs(df['h']-df['c'].shift()), abs(df['l']-df['c'].shift())], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
            df['vol_rel'] = df['v'] / df['v'].rolling(20).mean()
            df['mom'] = df['mid'].pct_change(5)
            
            curr_mid = df['mid'].iloc[-1]
            curr_atr = df['atr'].iloc[-1]
            atr_pct = curr_atr/curr_mid * 5
            features = df[['mom', 'atr', 'vol_rel']].iloc[[-1]].values.astype(np.float32)
            
            # Obter a sessão ONNX da moeda
            sess = MODELS_CACHE[coin]
            input_name = sess.get_inputs()[0].name
            
            # Executar inferência: 
            # O resultado do ONNX para classificação geralmente retorna 
            # [0] = label final, [1] = probabilidades
            label, result = sess.run(None, {input_name: features})
            
            # Ajuste: probs costuma ser uma lista de dicionários ou array dependendo da versão
            # Vamos pegar a probabilidade real (geralmente indexada como [0])
            probs = result[0]
            probs = [probs.get(-1,0), 0, probs.get(1, 0)]
            
            
            # Probs: [Short, Neutro, Long]
            if probs[2] > threshold_param:
                opportunities.append({
                    "coin": coin,
                    "is_buy": True,
                    "prob": probs[2],
                    "tp": round(1 + (atr_pct * 3.0), 6),
                    "sl": round(1 - (atr_pct * 2.0), 6),
                    "leverage": 4
                })
            elif probs[0] > threshold_param:
                opportunities.append({
                    "coin": coin,
                    "is_buy": False,
                    "prob": probs[0],
                    "tp": round(1 - (atr_pct * 3.0), 6),
                    "sl": round(1 + (atr_pct * 2.0), 6),
                    "leverage": 4
                })
        except Exception as e:
            print(e)
    
    opportunities.sort(key=lambda x: x['prob'], reverse=True)
    
    return opportunities

# ==============================================================================
# ENDPOINTS DA API
# ==============================================================================
@app.route('/scan', methods=['GET'])
def scan_market():
    """Escaneia o mercado usando os modelos já carregados na RAM."""
    try:
        threshold_param = float(request.args.get('threshold', 0.60))
    except ValueError:
        return jsonify({"error": "O parâmetro threshold deve ser um número."}), 400

    if not MODELS_CACHE:
        return jsonify({"error": "Modelos não carregados na RAM. Rode o treinamento e faça /reload."}), 404

    signals_found = scan_opportunities(threshold_param)

    return jsonify({
        "status": "sucesso",
        "total_signals": len(signals_found),
        "signals": signals_found
    }), 200


@app.route('/reload', methods=['POST'])
def reload_models():
    """
    Endpoint administrativo. 
    Use após rodar o treinador.py para atualizar os cérebros sem reiniciar a API.
    """
    load_models_into_memory()
    return jsonify({
        "status": "sucesso", 
        "mensagem": f"{len(MODELS_CACHE)} modelos atualizados na memória RAM."
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 
    # debug=False é recomendado em produção para evitar recarregamentos duplos