import os
import glob
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from flask import Flask, request, jsonify
import warnings
from flask_cors import CORS # 1. Importe a biblioteca

warnings.filterwarnings("ignore")


app = Flask(__name__)

# 2. Habilite o CORS para todas as rotas e todas as origens
CORS(app)

# Configurações Base
INTERVAL = "15m"
TAKE_PROFIT_PCT = 0.010
STOP_LOSS_PCT = 0.005
RISK_ACCOUNT = 0.02
BASE_URL = "https://api.hyperliquid.xyz/info"

# ==============================================================================
# CACHE GLOBAL EM MEMÓRIA RAM
# ==============================================================================
MODELS_CACHE = {}

def load_models_into_memory():
    """Carrega todos os arquivos .joblib do disco para a memória RAM."""
    global MODELS_CACHE
    MODELS_CACHE.clear()
    
    modelos_salvos = glob.glob("modelos/modelo_*.joblib")
    if not modelos_salvos:
        print("[Aviso] Nenhum modelo encontrado na pasta /modelos/ durante a inicialização.")
        return
        
    for caminho in modelos_salvos:
        symbol = os.path.basename(caminho).replace("modelo_", "").replace(".joblib", "")
        # Carrega no dicionário global
        MODELS_CACHE[symbol] = joblib.load(caminho)
        
    print(f"[Sistema] {len(MODELS_CACHE)} modelos carregados na RAM com sucesso!")

# Carrega os modelos assim que o script é executado
load_models_into_memory()

# ==============================================================================
# FUNÇÕES DE DADOS
# ==============================================================================
def get_recent_data(symbol):
    """Baixa os últimos 3 dias de dados para montar a EMA 200 e features do momento."""
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (3 * 24 * 60 * 60 * 1000) 
    
    payload = {
        "type": "candleSnapshot", 
        "req": {"coin": symbol, "interval": INTERVAL, "startTime": start_time, "endTime": end_time}
    }
    resp = requests.post(BASE_URL, json=payload).json()
    
    if not resp: return pd.DataFrame()
    
    df = pd.DataFrame(resp)
    for col in ["o", "h", "l", "c", "v"]: df[col] = df[col].astype(float)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).drop_duplicates("t")
    
    df['ret_1'] = df['close'].pct_change()
    df['ret_3'] = df['close'].pct_change(3)
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['macro_trend'] = np.where(df['close'] > df['ema_200'], 1, -1)
    df['dist_ema_20'] = (df['close'] / df['close'].ewm(span=20).mean()) - 1
    
    return df.dropna().reset_index(drop=True)

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

    features = ['ret_1', 'ret_3', 'vol_ratio', 'dist_ema_20', 'macro_trend']
    signals_found = []
    
    # Varre as moedas que estão na memória RAM
    for symbol, model in MODELS_CACHE.items():
        df = get_recent_data(symbol)
        if df.empty: continue
        
        live_candle = df.iloc[-1:]
        
        # PREDIÇÃO EM MILISSEGUNDOS (Lendo da RAM)
        probs = model.predict_proba(live_candle[features])[0]
        macro = live_candle['macro_trend'].values[0]
        current_price = live_candle['close'].values[0]
        
        signal = None
        certeza = 0
        
        if probs[1] > threshold_param and macro == 1:
            signal = "LONG"
            certeza = probs[1]
        elif probs[2] > threshold_param and macro == -1:
            signal = "SHORT"
            certeza = probs[2]
            
        if signal:
            tp_price = current_price * (1 + TAKE_PROFIT_PCT) if signal == "LONG" else current_price * (1 - TAKE_PROFIT_PCT)
            sl_price = current_price * (1 - STOP_LOSS_PCT) if signal == "LONG" else current_price * (1 + STOP_LOSS_PCT)
            alavancagem_ideal = RISK_ACCOUNT / STOP_LOSS_PCT
            
            signals_found.append({
                "coin": symbol,
                "is_buy": signal == "LONG",
                "thresold": round(certeza * 100, 2),
                "current_price": round(current_price, 4),
                "tp": round(tp_price, 4),
                "sl": round(sl_price, 4),
                "leverage": int(alavancagem_ideal)
            })

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