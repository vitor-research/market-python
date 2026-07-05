import os
import math
import time
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app import load_models_into_memory, fetch_signals

# Carrega os dados do arquivo .env
load_dotenv()

# load_models_into_memory()

# ==========================================
# CONFIGURAÇÕES E ESTADO GLOBAL
# ==========================================

class Env:
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY")

ACCOUNT_DETAILS = {
    "positions": [],
    "balance": 0.0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Lógica de STARTUP ---
    scheduler = AsyncIOScheduler()
    await scan_task()
    scheduler.add_job(scan_task, 'interval', minutes=1)
    scheduler.start()
    print("Scheduler de 1 minuto iniciado via Lifespan!")
    
    # O yield indica que a aplicação está rodando
    yield
    
    # --- Lógica de SHUTDOWN (Opcional, mas recomendado) ---
    scheduler.shutdown()
    print("Scheduler encerrado de forma segura.")

# ==========================================
# INICIALIZAÇÃO DO FASTAPI (Atualizado)
# ==========================================
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
# Inicialização do cliente Hyperliquid (Mainnet)
account = Account.from_key(Env.PRIVATE_KEY)
info_client = Info(constants.MAINNET_API_URL, skip_ws=True)
exchange_client = Exchange(account, constants.MAINNET_API_URL, vault_address=account.address)

# ==========================================
# FUNÇÕES UTILITÁRIAS
# ==========================================

def format_sz(size: float, decimals: int) -> float:
    factor = 10 ** decimals
    # Arredonda para baixo para evitar erro de margem insuficiente
    truncated_size = math.floor(size * factor) / factor
    return truncated_size

def format_px(price: float) -> str:
    # 1. Isola os 5 algarismos significativos permitidos pela HL
    sig_fig_price = float(f"{price:.5g}")
    str_price = str(sig_fig_price)
    
    # 2. Proteção contra Notação Científica
    if 'e' in str_price.lower():
        str_price = f"{sig_fig_price:.10f}".rstrip('0').rstrip('.')
        
    return str_price

def get_coin_index(coin: str) -> int:
    meta_and_asset_ctxs = info_client.meta_and_asset_ctxs()
    universe = meta_and_asset_ctxs[0]["universe"]
    for i, asset in enumerate(universe):
        if asset["name"] == coin:
            return i
    raise ValueError(f"Moeda {coin} não encontrada no universo.")

# ==========================================
# LÓGICA DE NEGOCIAÇÃO (HYPERLIQUID)
# ==========================================

def cancel_position(position: dict):
    mids = info_client.all_mids()
    price = float(mids[position["coin"]])
    adjusted_price = price * 1.001 if position["is_buy"] else price * 0.998
    final_price = float(format_px(adjusted_price))
    
    # Construção manual da ordem de cancelamento similar ao TS
    order_req = {
        "coin": position["coin"],
        "is_buy": not position["is_buy"],
        "sz": position["size"] * 2,
        "reduce_only": True,
        "limit_px": final_price,
        "order_type": {
            "trigger": {
                "isMarket": True,
                "triggerPx": final_price,
                "tpsl": "sl"
            }
        }
    }
    
    # Executa a ordem bruta via o método interno do exchange_client
    exchange_client.bulk_orders([order_req])

def make_order(coin: str, usd_size: float, leverage: int, tp: float, sl: float, is_buy: bool):
    meta_and_asset_ctxs = info_client.meta_and_asset_ctxs()
    universe = meta_and_asset_ctxs[0]["universe"]
    asset_ctxs = meta_and_asset_ctxs[1]

    coin_index = get_coin_index(coin)
    
    
    max_leverage = universe[coin_index]["maxLeverage"]
    safe_leverage = min(max(1, round(leverage)), max_leverage)
    sz_decimals = universe[coin_index]["szDecimals"]
    
    price = float(asset_ctxs[coin_index]["markPx"])
    adjusted_price = price * 1.001 if is_buy else price * 0.998
    final_price = float(format_px(adjusted_price))
    
    size = format_sz((usd_size * safe_leverage) / final_price, sz_decimals)
    
    # Atualiza Alavancagem
    exchange_client.update_leverage(safe_leverage, coin, is_cross=True)
    
    tp = float(format_px(tp))
    sl = float(format_px(sl))
    
    # Estrutura de múltiplas ordens (Entry + TP + SL)
    orders_req = [
        {
            "coin": coin,
            "is_buy": is_buy,
            "sz": size,
            "reduce_only": False,
            "limit_px": final_price,
            "order_type": {"limit": {"tif": "Gtc"}}
        },
        {
            "coin": coin,
            "is_buy": not is_buy,
            "sz": size * 2,
            "reduce_only": True,
            "limit_px": tp,
            "order_type": {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": tp,
                    "tpsl": "tp"
                }
            }
        },
        {
            "coin": coin,
            "is_buy": not is_buy,
            "sz": size * 2,
            "reduce_only": True,
            "limit_px": sl,
            "order_type": {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": sl,
                    "tpsl": "sl"
                }
            }
        }
    ]
    
    response = exchange_client.bulk_orders(orders_req)
    
    # Extrai o OID da ordem limit (resting)
    try:
        resting_oid = None
        print(response)
        for status in response["data"]["statuses"]:
            if isinstance(status, dict) and "resting" in status:
                resting_oid = status["resting"]["oid"]
                break
        
        if resting_oid:
            ACCOUNT_DETAILS["positions"].append({
                "oid": resting_oid,
                "coin": coin,
                "size": float(size),
                "notional": size * final_price,
                "entry_price": final_price,
                "unrealized_pnl": 0.0,
                "is_buy": is_buy,
                "time_open": int(time.time() * 1000)
            })
    except Exception as e:
        print(f"Erro ao processar resposta da ordem: {e}")

def get_position_details(wallet_address: str, coin: str):
    fills = info_client.user_fills(wallet_address)
    coin_fills = [f for f in fills if f.get("coin") == coin]
    coin_fills.sort(key=lambda x: x.get("time", 0), reverse=True)
    
    if not coin_fills:
        return {"error": "Nenhuma transação encontrada para esta moeda."}
        
    last_fill = coin_fills[0]
    return {
        "oid": last_fill.get("oid"),
        "time_open": last_fill.get("time"),
        "side": last_fill.get("side"),
        "price": last_fill.get("px")
    }

def update_account_info(wallet_address: str):
# 1. Substituído clearinghouse_state por user_state
    clearinghouse_state = info_client.user_state(wallet_address)
    
    total_balance = 0.0
    asset_positions = clearinghouse_state.get("assetPositions", [])
    
    if len(asset_positions) > 0:
        total_balance = float(clearinghouse_state["marginSummary"]["accountValue"])
    else:
        # 2. Substituído spot_clearinghouse_state por spot_user_state
        spot_state = info_client.spot_user_state(wallet_address)
        total_balance = float(spot_state["balances"][0]["total"])
    new_positions = []
    
    for p in asset_positions:
        pos_data = p["position"]
        coin = pos_data["coin"]
        
        # Procura se já existe no cache local
        existing_pos = next((x for x in ACCOUNT_DETAILS["positions"] if x["coin"] == coin), None)
        
        if existing_pos:
            existing_pos.update({
                "notional": float(pos_data["positionValue"]),
                "entry_price": float(pos_data["entryPx"]),
                "unrealized_pnl": float(pos_data["unrealizedPnl"])
            })
            new_positions.append(existing_pos)
        else:
            details = get_position_details(wallet_address, coin)
            
            new_positions.append({
                "oid": details.get("oid"),
                "coin": coin,
                "size": abs(float(pos_data["szi"])),
                "notional": float(pos_data["positionValue"]),
                "entry_price": float(pos_data["entryPx"]),
                "unrealized_pnl": float(pos_data["unrealizedPnl"]),
                "is_buy": float(pos_data["szi"]) > 0,
                "time_open": details.get("time_open", int(time.time() * 1000)),
            })

    ACCOUNT_DETAILS["balance"] = total_balance
    ACCOUNT_DETAILS["positions"] = new_positions

async def scan_task():
    # try:
        wallet_address = account.address
        n_positions = len(ACCOUNT_DETAILS["positions"])
        
        # Atualiza infos bloqueando (SDK é síncrono, se quiser liberar o event loop use asyncio.to_thread)
        update_account_info(wallet_address)
        
        current_time = int(time.time() * 1000)
        positions = ACCOUNT_DETAILS["positions"]
        
        # Verifica se posições foram fechadas ou se passaram 15 minutos
        condition_met = (
            len(positions) == 0 or 
            n_positions > len(positions) or 
            (len(positions) > 0 and (current_time - positions[0]["time_open"] > 15 * 60 * 1000))
        )
        
        if condition_met:
            result1 = await fetch_signals(0.10)
            await asyncio.sleep(1.0) # sleep assíncrono
            result2 = await fetch_signals(0.10)
            
            if len(result1) > 0 and len(result2) > 0 and result1[0]["coin"] == result2[0]["coin"]:
                signal = result2[0]
                coin = signal["coin"]
                leverage = signal.get("leverage", 1)
                tp = signal.get("tp")
                sl = signal.get("sl")
                is_buy = signal.get("is_buy")
                
                if len(ACCOUNT_DETAILS["positions"]) > 0:
                    if ACCOUNT_DETAILS["positions"][0]["coin"] != coin:
                        cancel_position(ACCOUNT_DETAILS["positions"][0])
                        update_account_info(wallet_address)
                        make_order(coin, ACCOUNT_DETAILS["balance"], leverage, tp, sl, is_buy)
                else:
                    make_order(coin, ACCOUNT_DETAILS["balance"], leverage, tp, sl, is_buy)
                    
    # except Exception as e:
    #     print(f"Erro no scan: {e}")

# ==========================================
# ROTAS FASTAPI (Substitui o bloco fetch do Worker)
# ==========================================

class OpenOrderPayload(BaseModel):
    coin: str
    is_buy: bool
    usd_size: float
    leverage: float
    tp: float
    sl: float

@app.get("/setup")
async def setup_route():
    update_account_info(account.address)
    return {"status": "ok", "message": "Account info updated"}

@app.post("/open")
async def open_route(payload: OpenOrderPayload):
    make_order(
        payload.coin, 
        payload.usd_size, 
        payload.leverage, 
        payload.tp, 
        payload.sl, 
        payload.is_buy
    )
    return {"status": "ok", "message": "Order placed"}

@app.get("/scan")
async def scan_route():
    await scan_task()
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"status": "ok"}


# Para rodar o servidor, use o comando no terminal:
# uvicorn main:app --host 0.0.0.0 --port 8000