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
BOTTLENECK_DIM = 6
LR = 0.0012
ENTRY_Z = 4.0412
EXIT_Z = 0.2856

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
def run_trading_cycle(positions, train_mode=True, is_retry=False):
    agora = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{agora}] Iniciando ciclo {'TREINO' if train_mode else 'WATCHDOG'} (Modo Low-Memory)...")

    # 1. Obtenção e Tratamento de Dados (Foco em descartar o que não usa)
    prices_df = get_market_matrix()
    if prices_df.empty:
        return {"type": "error", "msg": "Falha ao baixar dados da Hyperliquid"}

    coins_list = list(prices_df.columns)
    
    # Calcula os retornos e descarta os preços originais para economizar RAM
    returns_df = prices_df.pct_change().dropna().astype(np.float32)
    del prices_df 
    gc.collect()

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns_df).astype(np.float32)
    X_tensor = torch.tensor(X_scaled) # Mais eficiente que FloatTensor
    
    # Pegamos a última linha real (numpy) para uso posterior
    last_real_returns = returns_df.iloc[-1].values 
    
    # Limpeza pesada
    del returns_df
    gc.collect()

    # 2. Definição do Modelo
    model = StatArbAutoencoder(X_scaled.shape[1], bottleneck_dim=BOTTLENECK_DIM)
    criterion = nn.MSELoss()

    if train_mode:
        # TREINAMENTO
        optimizer = optim.Adam(model.parameters(), lr=LR)
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(X_tensor), X_tensor)
            loss.backward()
            optimizer.step()

        # Salva o modelo e limpa variáveis de treino
        torch.save(model.state_dict(), MODEL_PATH)
        del optimizer, loss
        gc.collect()

        # Avaliação global para gerar o stats.json
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor).numpy()
            errors = np.abs(X_scaled - reconstructed)

        error_mean = errors.mean(axis=0)  
        error_std = errors.std(axis=0)    

        stats = {
            "means": error_mean.tolist(),
            "stds": error_std.tolist(),
            "coins": coins_list
        }
        with open(STATS_PATH, 'w') as f:
            json.dump(stats, f)

        # Procura oportunidades se não houver posições
        if len(positions) == 0:
            # Z-Score do último candle
            last_errors = errors[-1]
            zscores = (last_errors - error_mean) / (error_std + 1e-8)
            
            max_idx = np.argmax(np.abs(zscores))
            max_z_coin = coins_list[max_idx]
            max_z_val = abs(zscores[max_idx])

            if max_z_val > ENTRY_Z and max_z_coin != HEDGE_ASSET:
                real_val = last_real_returns[max_idx]
                pred_val = reconstructed[-1][max_idx]
                is_buy_anomaly = bool(real_val < pred_val)

                return {
                    "type": "open",
                    "pair": [
                        {"coin": max_z_coin, "is_buy": is_buy_anomaly},
                        {"coin": HEDGE_ASSET, "is_buy": not is_buy_anomaly}
                    ],
                    "stats": stats
                }
            return {"type": "wait"}
        return {"type": "wait", "msg": "Posições ativas."}

    else:
        # WATCHDOG (5 em 5 minutos) - Inferência super leve
        if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
            return {"type": "error", "msg": "Modelo ausente. Aguarde o Treino."}

        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except RuntimeError as e:
          if is_retry:
            return {"type": "error", "msg": "Falha ao carregar o modelo treinado. Aguarde o ciclo de Treino."}

          else:
            return run_trading_cycle(positions, train_mode=False, is_retry = True)
        model.eval()

        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)

        if len(positions) > 0:
            main_coin = [coin for coin in positions if coin != HEDGE_ASSET][0]
            
            if main_coin not in stats["coins"]:
                return {"type": "error", "msg": "Moeda não mapeada."}

            idx = stats["coins"].index(main_coin)

            with torch.no_grad():
                # Fazemos a inferência APENAS do último candle (X_tensor[-1]) para economizar cálculo
                last_tensor = X_tensor[-1].unsqueeze(0)
                reconstructed = model(last_tensor).numpy()[0]
                current_error = np.abs(X_scaled[-1][idx] - reconstructed[idx])

            mean = stats["means"][idx]
            std = stats["stds"][idx]
            z_atual = (current_error - mean) / (std + 1e-8)
            z_atual_abs = abs(float(z_atual))

            if z_atual_abs <= EXIT_Z:
                return {"type": "close", "coin": main_coin, "z_score": z_atual_abs}
            return {"type": "hold", "coin": main_coin, "z_score": z_atual_abs}
            
        return {"type": "wait"}

# ==========================================
# MOCK DE EXECUÇÃO LOCAL (PARA TESTE)
# ==========================================
if __name__ == "__main__":
    print("🚀 SISTEMA STAT-ARB CARREGADO")
    
    # Simula o TypeScript pedindo para treinar e buscar entradas (Roda a cada 1 hora)
    posicoes_mock = []
    print("\n--- SIMULANDO CHAMADA DO TYPESCRIPT: HORA EM HORA ---")
    resultado_treino = run_trading_cycle(posicoes_mock, train_mode=True)
    # print("RESPOSTA JSON PARA O TYPESCRIPT:", resultado_treino)
    
    # Se o robô sugerisse uma entrada, o TS adicionaria ela em 'posicoes_mock'. 
    # Vamos simular que estamos comprados em ARB para testar o Watchdog:
    posicoes_mock = ["ARB", "BTC"]
    
    print("\n--- SIMULANDO CHAMADA DO TYPESCRIPT: 5 EM 5 MINUTOS ---")
    resultado_watchdog = run_trading_cycle(posicoes_mock, train_mode=False)
    print("RESPOSTA JSON PARA O TYPESCRIPT:", resultado_watchdog)