"""
================================================================================
 EXECUTOR STAT-ARB (LIVE TRADING) COM ONNX - VERSÃO ENXUTA
 - O TypeScript cuida da margem, saldo e netting.
 - Este script faz apenas inferência ML e validação de contexto.
 - Retorna a direção (is_buy) e o peso (beta_weight) para alocação.
================================================================================
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import onnxruntime as rt
import ccxt
from urllib.parse import unquote

class Config:
    exchange_id = "hyperliquid"
    timeframe = "1h"
    lookback_bars = 150              
    
    # Filtros de Gatilho e Contexto (Idênticos ao Backtest)
    zscore_window = 60
    entry_z = 1.5                    
    
    ml_proba_high = 0.65             
    ml_proba_med = 0.55              
    ml_proba_min = 0.50              
    extreme_z = 2.5                  

    model_dir = "./models"

CFG = Config()

def fetch_live_data(symbols: list, cfg: Config) -> pd.DataFrame:
    exchange = getattr(ccxt, cfg.exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    data = {}
    for sym in symbols:
        try:
            batch = exchange.fetch_ohlcv(sym, timeframe=cfg.timeframe, limit=cfg.lookback_bars)
            if batch:
                df = pd.DataFrame(batch, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                data[sym] = df.drop_duplicates(subset="timestamp").set_index("timestamp")["close"]
        except Exception:
            continue
            
    if not data: return pd.DataFrame()
    return pd.DataFrame(data).ffill().dropna()

def check_pairs_to_close(current_coins_in_positions):
    """
    Verifica o Z-Score atual de todos os pares ativos.
    Se |Z-Score| > 4.5, encerra as posições dos ativos envolvidos.
    """
    # print(">> Verificando integridade estatística dos pares (Stop Loss Estrutural)...")

    portfolio_path = os.path.join(CFG.model_dir, "portfolio.json")
    if not os.path.exists(portfolio_path): return []
            
    with open(portfolio_path, "r") as f:
        portfolio = json.load(f)
        
    if not portfolio: return []


    pairs_to_close = []
    
    # 1. Identificar quais pares estão com posições abertas
    active_pairs = {}
    for coin in current_coins_in_positions:
        for pair_id, meta in portfolio.items():
            if coin == meta['asset_y'].split("/")[0] or coin == meta['asset_x'].split("/")[0]:
                active_pairs[pair_id] = meta

    # 2. Calcular Z-Score em tempo real para cada par ativo
    for pair_id, meta in active_pairs.items():
        # Busca dados atualizados para calcular o Z-Score agora
        prices = fetch_live_data([meta['asset_y'], meta['asset_x']], CFG)
        
        if prices.empty: continue
        
        # Calcula Z-Score (usando a mesma lógica do backtest)
        spread = prices[meta['asset_y']] - meta['beta'] * prices[meta['asset_x']]
        mean = spread.rolling(CFG.zscore_window).mean().iloc[-1]
        std = spread.rolling(CFG.zscore_window).std().iloc[-1]
        
        current_z = (spread.iloc[-1] - mean) / (std + 1e-9)
        
        # print(f"🔍 Monitoramento [{pair_id}]: Z-Score = {current_z:.2f}")

        # 3. Limite de Segurança de 4.5
        if abs(current_z) > 4.5 or abs(current_z) < 1.5:
            # print(f"🚨 STOP LOSS ESTRUTURAL [ {pair_id} ]! Z-Score: {current_z:.2f}")
            
            # Fecha as duas pernas do par
            # Aqui assumimos que você tem uma função para fechar (market order oposta ou zerar)
            # Exemplo genérico de fechamento:
            for asset in [meta['asset_y'].split("/")[0], meta['asset_x'].split("/")[0]]:
                # Comando para zerar a posição (Hyperliquid geralmente aceita volume 0 para zerar)
                # Substitua pela sua função real de envio de ordem
                pairs_to_close.append(asset)
            
            # print(f"✅ Par {pair_id} encerrado com sucesso.")
    
    return pairs_to_close

def build_features(price_y: pd.Series, price_x: pd.Series, beta: float, cfg: Config) -> pd.DataFrame:
    # A ordem exata das features é crucial para o ONNX
    spread = price_y - beta * price_x
    zscore = (spread - spread.rolling(cfg.zscore_window).mean()) / spread.rolling(cfg.zscore_window).std()
    
    feat = pd.DataFrame(index=spread.index)
    feat["zscore"] = zscore
    feat["zscore_abs"] = zscore.abs()
    feat["spread_vol_short"] = spread.rolling(10).std()
    feat["spread_vol_long"] = spread.rolling(cfg.zscore_window).std()
    feat["vol_ratio"] = feat["spread_vol_short"] / (feat["spread_vol_long"] + 1e-9)
    feat["momentum_y"] = price_y.pct_change(10)
    feat["momentum_x"] = price_x.pct_change(10)
    
    return feat.replace([np.inf, -np.inf], np.nan).dropna()

def get_live_signals() -> list:
    portfolio_path = os.path.join(CFG.model_dir, "portfolio.json")
    if not os.path.exists(portfolio_path): return []
        
    with open(portfolio_path, "r") as f:
        portfolio = json.load(f)
        
    if not portfolio: return []

    unique_symbols = set()
    for meta in portfolio.values():
        unique_symbols.add(meta["asset_y"])
        unique_symbols.add(meta["asset_x"])
        
    prices = fetch_live_data(list(unique_symbols), CFG)
    if prices.empty: return []

    active_signals = []
    
    for pair_id, meta in portfolio.items():
        y_sym, x_sym = meta["asset_y"], meta["asset_x"]
        if y_sym not in prices.columns or x_sym not in prices.columns:
            continue
            
        feat = build_features(prices[y_sym], prices[x_sym], meta["beta"], CFG)
        if feat.empty: continue
            
        current_features = feat.iloc[-1:].values 
        z = feat["zscore"].iloc[-1]
        vol_ratio = feat["vol_ratio"].iloc[-1]
        
        # 1. Padronização salva do treino
        scaler_mean = np.array(meta["scaler_mean"])
        scaler_scale = np.array(meta["scaler_scale"])
        X_scaled = (current_features - scaler_mean) / scaler_scale
        
        # 2. Inferência ONNX
        sess = rt.InferenceSession(meta["onnx_model"], providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name 
        pred_onx = sess.run([label_name], {input_name: X_scaled.astype(np.float32)})[0]
        
        p = float(pred_onx[0].get(1, 0.0)) 
        
        # 3. Lógica de Contexto (IDÊNTICA AO BACKTEST)
        high_conviction = p >= CFG.ml_proba_high
        med_conviction_calm = (p >= CFG.ml_proba_med) and (vol_ratio < 1.0)
        extreme_stretch = (abs(z) >= CFG.extreme_z) and (p >= CFG.ml_proba_min)

        action = None
        if high_conviction or med_conviction_calm or extreme_stretch:
            if z > CFG.entry_z:
                action = "SHORT_SPREAD"
            elif z < -CFG.entry_z:
                action = "LONG_SPREAD"
                
        if action:
            # Limpa o formato para o TypeScript (ex: XYZ-BTC/USDC:USDC vira BTC)
            coin_y = y_sym.split("/")[0].replace("XYZ-", "")
            coin_x = x_sym.split("/")[0].replace("XYZ-", "")
            
            active_signals.append({
                "pair_id": pair_id,
                "ml_probability": round(p, 4),
                "zscore": round(z, 2),
                "execution": [
                    {
                        "coin": coin_y,
                        "is_buy": action == "LONG_SPREAD",
                        "weight": 1.0 # O TypeScript usa isso para alocar 1 parte do capital
                    },
                    {
                        "coin": coin_x,
                        "is_buy": action == "SHORT_SPREAD",
                        "weight": round(abs(meta["beta"]), 4) # O TypeScript multiplica o capital por isso (Hedge)
                    }
                ]
            })
            
    return active_signals

if __name__ == "__main__":
    try:
        # Não precisa mais receber argumento de capital via linha de comando
        ordens = get_live_signals()
        print(json.dumps(ordens))
    except Exception as e:
        print(json.dumps([]))