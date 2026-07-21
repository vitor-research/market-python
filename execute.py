import os
import gc  # Garbage Collector
import json
import time
import datetime
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURAÇÕES DA ESTRATÉGIA
# ==========================================
BOTTLENECK_DIM = 7
LR = 0.0026
ENTRY_Z = 3.0049
EXIT_Z = 0.1976

LEVERAGE = 2
ALLOCATION_PER_LEG = 0.95 
HEDGE_ASSET = "BTC"       

UNIVERSE_SIZE = 45       
TIMEFRAME = "1h"         
DAYS_HISTORY = 45        
EPOCHS = 60

# Caminhos para salvar a memória do robô
MODEL_PATH = "stat_arb_model.pth"
STATS_PATH = "market_stats.json"

base_url = "https://api.hyperliquid.xyz"

class StatArbAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(StatArbAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# MOTOR DE DADOS OTIMIZADO (LOW-MEMORY)
# ==========================================
def fetch_candle_data(coin):
    end_time = int(time.time() * 1000)
    start_time = end_time - (DAYS_HISTORY * 24 * 60 * 60 * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": TIMEFRAME, "startTime": start_time, "endTime": end_time}}
    try:
        response = requests.post(f"{base_url}/info", json=payload).json()
        if not response or not isinstance(response, list): 
            return None
        
        # OTIMIZAÇÃO 1: Mantém apenas a coluna de preço de fechamento ('c')
        prices = [float(candle['c']) for candle in response]
        
        # OTIMIZAÇÃO 2: Converte para float32 para economizar 50% de RAM
        return pd.Series(prices, dtype=np.float32)
    except: 
        return None

def get_market_matrix():
    universe = requests.post(f"{base_url}/info", json={"type": "meta"}).json().get("universe", [])
    
    # Filtro de qualidade rápido (removendo memecoins básicas)
    blacklist = ["PEPE", "WIF", "DOGE", "SHIB", "BONK", "FLOKI", "kPEPE"]
    coins = [c["name"] for c in universe if c["name"] not in blacklist][:UNIVERSE_SIZE]
    
    data = {}
    for coin in coins:
        series = fetch_candle_data(coin)
        if series is not None and len(series) > (DAYS_HISTORY * 24 * 0.8): # Pelo menos 80% dos dados
            data[coin] = series
        time.sleep(0.05)
        
    df = pd.DataFrame(data).ffill().bfill()
    
    # Libera o dicionário temporário da memória
    del data
    gc.collect()
    
    return df

# ==========================================
# CÉREBRO DO ROBÔ OTIMIZADO (LOW-MEMORY)
# ==========================================
def run_trading_cycle(positions, verify_new_pair=False, train_mode=False, is_retry = False):
    agora = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Logs amigáveis para você saber exatamente o que o bot está fazendo
    # acao = "PROCURAR ENTRADAS" if verify_new_pair else "WATCHDOG (SAÍDA)"
    # estado = "TREINANDO IA" if train_mode else "USANDO MEMÓRIA"
    # print(f"\n[{agora}] Ciclo: {acao} | Cérebro: {estado}")

    # =========================================================
    # 1. OBTENÇÃO DE DADOS (Low-Memory)
    # =========================================================
    try:
        prices_df = get_market_matrix()
    except RuntimeError as e:
        if is_retry:
            return {"type": "error", "msg": "Falha ao baixar dados"}

        else:
            return run_trading_cycle(positions, train_mode, verify_new_pair, is_retry = True)
    
    if prices_df.empty:
        return {"type": "error", "msg": "Falha ao baixar dados"}

    coins_list = list(prices_df.columns)
    returns_df = prices_df.pct_change().dropna().astype(np.float32)
    
    del prices_df 
    gc.collect()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns_df).astype(np.float32)
    X_tensor = torch.tensor(X_scaled)
    last_real_returns = returns_df.iloc[-1].values 
    
    del returns_df
    gc.collect()

    model = StatArbAutoencoder(X_scaled.shape[1], BOTTLENECK_DIM)
    criterion = nn.MSELoss()

    # =========================================================
    # 2. GESTÃO DO MODELO (Treinar vs. Carregar)
    # =========================================================
    if train_mode:
        # TREINAMENTO PESADO (Rode 1x ao dia)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(X_tensor), X_tensor)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PATH)
        del optimizer, loss
        gc.collect()

        # Calcula a normalidade histórica
        model.eval()
        with torch.no_grad():
            reconstructed_all = model(X_tensor).numpy()
            errors_all = np.abs(X_scaled - reconstructed_all)

        error_mean = errors_all.mean(axis=0)  
        error_std = errors_all.std(axis=0)    

        stats = {
            "means": error_mean.tolist(),
            "stds": error_std.tolist(),
            "coins": coins_list
        }
        with open(STATS_PATH, 'w') as f:
            json.dump(stats, f)
            
        del reconstructed_all, errors_all
        gc.collect()

    else:
        # CARREGAMENTO LEVE (Rode o resto do dia)
        if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
            return {"type": "error", "msg": "Modelo ausente. Rode com train_mode=True 1x."}

        
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except RuntimeError as e:
          if is_retry:
            return {"type": "error", "msg": "Falha ao carregar o modelo treinado. Aguarde o ciclo de Treino."}

          else:
            return run_trading_cycle(positions, train_mode, verify_new_pair, is_retry = True)
            
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
            
        error_mean = np.array(stats["means"], dtype=np.float32)
        error_std = np.array(stats["stds"], dtype=np.float32)

    # =========================================================
    # 3. INFERÊNCIA RÁPIDA DO MOMENTO ATUAL
    # =========================================================
    # Independente de ter treinado ou não, fazemos a predição só da ÚLTIMA VELA
    model.eval()
    with torch.no_grad():
        last_tensor = X_tensor[-1].unsqueeze(0) # Pega só a vela de agora
        reconstructed_last = model(last_tensor).numpy()[0]
        last_errors = np.abs(X_scaled[-1] - reconstructed_last)

    # =========================================================
    # 4. DECISÃO (Verificar Novo Par vs. Watchdog)
    # =========================================================
    if verify_new_pair:
        # LÓGICA DE ENTRADA
        if len(positions) == 0:
            zscores = (last_errors - error_mean) / (error_std + 1e-8)
            
            max_idx = np.argmax(np.abs(zscores))
            max_z_coin = coins_list[max_idx]
            max_z_val = abs(zscores[max_idx])

            if max_z_val > ENTRY_Z and max_z_coin != HEDGE_ASSET:
                real_val = last_real_returns[max_idx]
                pred_val = reconstructed_last[max_idx]
                is_buy_anomaly = bool(real_val < pred_val)

                return {
                    "type": "open",
                    "pair": [
                        {"coin": max_z_coin, "is_buy": is_buy_anomaly},
                        {"coin": HEDGE_ASSET, "is_buy": not is_buy_anomaly}
                    ],
                    # "stats": stats
                }
            return {"type": "wait", "msg": f"Mercado calmo. Maior Z-Score: {max_z_coin} ({max_z_val:.2f})"}
        return {"type": "wait", "msg": "Posições ativas. Aguardando saída."}

    else:
        # LÓGICA DE SAÍDA (WATCHDOG)
        if len(positions) > 0:
            main_coin = [coin for coin in positions if coin != HEDGE_ASSET][0]
            
            if main_coin not in coins_list:
                return {"type": "error", "msg": "Moeda aberta sumiu dos dados."}

            idx = coins_list.index(main_coin)
            
            z_atual = (last_errors[idx] - error_mean[idx]) / (error_std[idx] + 1e-8)
            z_atual_abs = abs(float(z_atual))

            if z_atual_abs <= EXIT_Z:
                return {"type": "close", "coin": main_coin, "z_score": z_atual_abs}
            return {"type": "hold", "coin": main_coin, "z_score": z_atual_abs}
            
        return {"type": "wait", "msg": "Nenhuma posição aberta."}

# ==========================================
# MOCK DE EXECUÇÃO LOCAL (PARA TESTE)
# ==========================================
if __name__ == "__main__":
    print("🚀 SISTEMA STAT-ARB CARREGADO")
    
    # Simula o TypeScript pedindo para treinar e buscar entradas (Roda a cada 1 hora)
    posicoes_mock = []
    print("\n--- SIMULANDO CHAMADA DO TYPESCRIPT: HORA EM HORA ---")
    resultado_treino = run_trading_cycle(posicoes_mock, True, False)
    print("RESPOSTA JSON PARA O TYPESCRIPT:", resultado_treino)
    
    # Se o robô sugerisse uma entrada, o TS adicionaria ela em 'posicoes_mock'. 
    # Vamos simular que estamos comprados em ARB para testar o Watchdog:
    posicoes_mock = ["ARB", "BTC"]
    
    print("\n--- SIMULANDO CHAMADA DO TYPESCRIPT: 5 EM 5 MINUTOS ---")
    # resultado_watchdog = run_trading_cycle(posicoes_mock, False)
    # print("RESPOSTA JSON PARA O TYPESCRIPT:", resultado_watchdog)