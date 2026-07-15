import os
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

# ==========================================
# 2. AUTOENCODER
# ==========================================
class StatArbAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(StatArbAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, BOTTLENECK_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 3. MOTOR DE DADOS
# ==========================================
def fetch_candle_data(coin):
    end_time = int(time.time() * 1000)
    start_time = end_time - (DAYS_HISTORY * 24 * 60 * 60 * 1000)
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": TIMEFRAME, "startTime": start_time, "endTime": end_time}}
    try:
        response = requests.post(f"{base_url}/info", json=payload).json()
        if not response or not isinstance(response, list): return None
        df = pd.DataFrame(response)
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('datetime', inplace=True)
        return pd.to_numeric(df['c'])
    except: return None

def get_market_matrix():
    universe = requests.post(f"{base_url}/info", json={"type": "meta"}).json().get("universe", [])
    blacklist = ["PEPE", "WIF", "DOGE", "SHIB", "BONK", "FLOKI", "kPEPE"] 
    
    filtered_coins = []
    for c in universe:
        name = c["name"]
        # Filtro: Exclui moedas da blacklist e garante que o nome não contenha palavras comuns de memes
        if name not in blacklist and len(name) < 10: 
            filtered_coins.append(name)
            
    # Pegamos as moedas que restaram, limitando ao seu UNIVERSE_SIZE
    coins = filtered_coins[:UNIVERSE_SIZE + 10]
    
    data = {}
    for coin in coins:
        series = fetch_candle_data(coin)
        if series is not None and not series.empty: data[coin] = series
        time.sleep(0.05)
        
    df = pd.DataFrame(data).dropna(axis=1, thresh=int(DAYS_HISTORY*24*0.9)).ffill().bfill()
    return df[df.columns[:UNIVERSE_SIZE]]


# ==========================================
# 4. O CÉREBRO DO ROBÔ (API PARA O TYPESCRIPT)
# ==========================================
def run_trading_cycle(positions, train_mode=True, is_retry = False):
    agora = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{agora}] Iniciando ciclo de {'TREINO (1h)' if train_mode else 'WATCHDOG (5min)'}...")
    # print(f"💰 Patrimônio Base: U$ {equity:.2f} | Posições Abertas: {len(positions)}")
    print(f"Posições Abertas: {len(positions)}")

    # Baixa dados recentes
    prices_df = get_market_matrix()
    if prices_df.empty:
        return {"type": "error", "msg": "Falha ao baixar dados da Hyperliquid"}

    returns_df = prices_df.pct_change().dropna()
    coins_list = list(returns_df.columns)

    # Padroniza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns_df)
    X_tensor = torch.FloatTensor(X_scaled)

    model = StatArbAutoencoder(X_scaled.shape[1])
    criterion = nn.MSELoss()

    if train_mode:
        # ---------------------------------------------------------
        # MODO TREINO: APRENDE O MERCADO E PROCURA ENTRADAS
        # ---------------------------------------------------------
        optimizer = optim.Adam(model.parameters(), lr=LR)
        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(X_tensor), X_tensor)
            loss.backward()
            optimizer.step()

        # Salva o "Cérebro" para o Watchdog poder usar depois
        torch.save(model.state_dict(), MODEL_PATH)

        # Calcula a média e o desvio padrão dos erros
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            errors = torch.abs(X_tensor - reconstructed).numpy()

        error_df = pd.DataFrame(errors, index=returns_df.index, columns=coins_list)
        error_mean = error_df.mean(axis=0)  
        error_std = error_df.std(axis=0)    

        # Salva as estatísticas para o Watchdog
        stats = {
            "means": error_mean.tolist(),
            "stds": error_std.tolist(),
            "coins": coins_list
        }
        with open(STATS_PATH, 'w') as f:
            json.dump(stats, f)

        # Procura oportunidades APENAS se não houver posições
        if len(positions) == 0:
            zscore_df = (error_df - error_mean) / (error_std + 1e-8)
            current_zscores = zscore_df.iloc[-1]
            predicted_df = pd.DataFrame(reconstructed.numpy(), index=returns_df.index, columns=coins_list)

            max_z_coin = current_zscores.abs().idxmax()
            max_z_val = abs(current_zscores[max_z_coin])

            # Verifica se atingiu o gatilho e garante que não operamos BTC contra BTC
            if max_z_val > ENTRY_Z and max_z_coin != HEDGE_ASSET:
                print(f"🚨 ANOMALIA DETECTADA: {max_z_coin} (Z-Score: {max_z_val:.2f})")
                
                real_val = returns_df[max_z_coin].iloc[-1]
                pred_val = predicted_df[max_z_coin].iloc[-1]
                is_buy_anomaly = bool(real_val < pred_val)

                return {
                    "type": "open",
                    "pair": [
                        {"coin": max_z_coin, "is_buy": is_buy_anomaly},
                        {"coin": HEDGE_ASSET, "is_buy": not is_buy_anomaly}
                    ],
                    "stats": stats
                }
            else:
                print(f"Zzz... Mercado eficiente. Maior anomalia é {max_z_coin} (Z={max_z_val:.2f}).")
                return {"type": "wait"}
        else:
            return {"type": "wait", "msg": "Posições ativas. Aguardando saída via Watchdog."}

    else:
        # ---------------------------------------------------------
        # MODO WATCHDOG: CARREGA A MEMÓRIA E VERIFICA SAÍDAS
        # ---------------------------------------------------------
        if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
            return {"type": "error", "msg": "Arquivos do modelo não encontrados. Aguarde o ciclo de Treino."}

        # Carrega o modelo treinado na última hora
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except RuntimeError as e:
          if is_retry:
            return {"type": "error", "msg": "Falha ao carregar o modelo treinado. Aguarde o ciclo de Treino."}

          else:
            return run_trading_cycle(positions, train_mode=True, is_retry = True)

        model.eval()

        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)

        if len(positions) > 0:
            main_coin = [coin for coin in positions if coin != HEDGE_ASSET][0]
            
            if main_coin not in stats["coins"]:
                return {"type": "error", "msg": "A moeda aberta não está na matriz de dados."}

            idx = stats["coins"].index(main_coin)

            # Roda inferência apenas para capturar o erro exato de agora
            with torch.no_grad():
                reconstructed = model(X_tensor)
                current_errors = torch.abs(X_tensor[-1] - reconstructed[-1]).numpy()

            # Z-Score Residual instantâneo
            mean = stats["means"][idx]
            std = stats["stds"][idx]
            z_atual = (current_errors[idx] - mean) / (std + 1e-8)
            z_atual_abs = abs(float(z_atual))

            print(f"📊 Monitorando {main_coin} | Z-Score Residual: {z_atual_abs:.2f} (Alvo: <= {EXIT_Z})")

            if z_atual_abs <= EXIT_Z:
                print("✅ A anomalia sumiu! Fechando operação para garantir lucro.")
                return {
                    "type": "close", 
                    "coin": main_coin, 
                    "z_score_atual": z_atual_abs
                }
            else:
                return {
                    "type": "hold", 
                    "coin": main_coin, 
                    "z_score_atual": z_atual_abs
                }
        else:
            return {"type": "wait", "msg": "Nenhuma posição aberta no momento."}

# ==========================================
# MOCK DE EXECUÇÃO LOCAL (PARA TESTE)
# ==========================================
if __name__ == "__main__":
    print("🚀 SISTEMA STAT-ARB CARREGADO")
    
    # Simula o TypeScript pedindo para treinar e buscar entradas (Roda a cada 1 hora)
    posicoes_mock = []
    print("\n--- SIMULANDO CHAMADA DO TYPESCRIPT: HORA EM HORA ---")
    resultado_treino = run_trading_cycle(posicoes_mock, train_mode=True)
    print("RESPOSTA JSON PARA O TYPESCRIPT:", resultado_treino)
    
    # Se o robô sugerisse uma entrada, o TS adicionaria ela em 'posicoes_mock'. 
    # Vamos simular que estamos comprados em ARB para testar o Watchdog:
    posicoes_mock = ["ARB", "BTC"]
    
    print("\n--- SIMULANDO CHAMADA DO TYPESCRIPT: 5 EM 5 MINUTOS ---")
    resultado_watchdog = run_trading_cycle(posicoes_mock, train_mode=False)
    print("RESPOSTA JSON PARA O TYPESCRIPT:", resultado_watchdog)