"""
model/backtest.py

Production-grade backtester with:
 - Realistic Indian market transaction costs (STT, brokerage, slippage)
 - Multi-stock portfolio support (runs across all stocks in the universe)
 - Configurable via config/config.yaml
"""
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import xgboost as xgb
import glob
import yaml
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sqlalchemy import text
from config.database import engine
from config.logger import get_logger
from feature_engineering.feature_store import calculate_technicals

logger = get_logger(__name__)

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

BT_CONFIG = config.get("backtest", {})
INITIAL_CAPITAL   = BT_CONFIG.get("initial_capital", 100000)
TRADE_AMOUNT      = BT_CONFIG.get("trade_amount", 20000)
STOP_LOSS         = BT_CONFIG.get("stop_loss", -0.05)
TAKE_PROFIT       = BT_CONFIG.get("take_profit", 0.15)

# Transaction costs (Indian equity delivery)
BROKERAGE_PCT     = BT_CONFIG.get("brokerage_pct", 0.0003)   # 0.03% per side
STT_BUY_PCT       = BT_CONFIG.get("stt_buy_pct", 0.001)      # 0.1% on buy
STT_SELL_PCT      = BT_CONFIG.get("stt_sell_pct", 0.001)     # 0.1% on sell
SLIPPAGE_PCT      = BT_CONFIG.get("slippage_pct", 0.001)     # 0.1% execution slippage


def load_latest_model() -> xgb.XGBClassifier:
    files = glob.glob("model/artifacts/model_cls_*.json")
    if not files:
        raise FileNotFoundError("No trained model found in model/artifacts/")
    latest = max(files, key=os.path.getctime)
    logger.info(f"Loading model: {latest}")
    model = xgb.XGBClassifier()
    model.load_model(latest)
    return model


def calculate_trade_cost(price: float, shares: int, side: str) -> float:
    """
    Returns the total transaction cost for a buy or sell.
    Costs: brokerage (both sides) + STT (both sides) + slippage (both sides).
    """
    value = price * shares
    brokerage = value * BROKERAGE_PCT
    stt       = value * (STT_BUY_PCT if side == "BUY" else STT_SELL_PCT)
    slippage  = value * SLIPPAGE_PCT
    total_cost = brokerage + stt + slippage
    logger.debug(f"  [{side}] Value=₹{value:.0f} | Brokerage=₹{brokerage:.2f} | STT=₹{stt:.2f} | Slippage=₹{slippage:.2f}")
    return total_cost


def run_backtest_for_symbol(symbol: str, model: xgb.XGBClassifier, threshold: float = 0.60) -> dict:
    """
    Runs a backtest for a single stock symbol.
    Returns a dict of summary metrics.
    """
    logger.info(f"--- Backtesting {symbol} ---")

    query = text("""
        SELECT f.*, p.close as price_close, p.date as trade_date
        FROM features_daily f
        JOIN stocks s ON f.stock_id = s.stock_id
        JOIN prices p ON f.stock_id = p.stock_id AND f.date = p.date
        WHERE s.symbol = :symbol
        ORDER BY f.date ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol})

    if df.empty:
        logger.warning(f"No feature data found for {symbol}. Skipping.")
        return {}

    booster = model.get_booster()
    model_features = booster.feature_names

    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[model_features]

    # Filter to 2024+ for recent relevant results
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["trade_date"] >= pd.Timestamp("2024-01-01")].copy()
    X = X.loc[df.index]

    if df.empty:
        logger.warning(f"Not enough recent data for {symbol}. Skipping.")
        return {}

    # Generate predictions
    probs = model.predict_proba(X)[:, 1]
    df["probability"] = probs
    df["prediction"] = (probs > threshold).astype(int)

    # Simulate trades with cost deductions
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = []

    for _, row in df.iterrows():
        price = row["price_close"]
        date  = row["trade_date"]

        signal = "HOLD"

        if position == 0:
            if row["probability"] > threshold and row.get("rsi_14", 50) < 70:
                signal = "BUY"
        elif position > 0:
            current_profit_pct = (price - entry_price) / entry_price
            if (row["rsi_14"] > 70
                    or row["probability"] < 0.4
                    or current_profit_pct >= TAKE_PROFIT
                    or current_profit_pct <= STOP_LOSS):
                signal = "SELL"

        if signal == "BUY" and position == 0:
            shares = int(TRADE_AMOUNT / price)
            if shares == 0:
                continue
            cost = calculate_trade_cost(price, shares, "BUY")
            capital -= (shares * price) + cost
            position    = shares
            entry_price = price
            trades.append({"date": date, "type": "BUY", "price": price, "shares": shares, "cost": cost})

        elif signal == "SELL" and position > 0:
            cost    = calculate_trade_cost(price, position, "SELL")
            revenue = (position * price) - cost
            gross_profit = (position * price) - (position * entry_price)
            net_profit   = gross_profit - cost - trades[-1]["cost"]  # subtract buy cost too
            capital += revenue
            trades.append({"date": date, "type": "SELL", "price": price, "shares": position, "profit": net_profit, "cost": cost})
            position = 0

        equity_curve.append({"date": date, "equity": capital + (position * price)})

    if not equity_curve:
        return {}

    final_equity   = equity_curve[-1]["equity"]
    total_return   = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    sell_trades    = [t for t in trades if t["type"] == "SELL"]
    wins           = [t for t in sell_trades if t.get("profit", 0) > 0]
    win_rate       = len(wins) / len(sell_trades) * 100 if sell_trades else 0
    total_costs    = sum(t.get("cost", 0) for t in trades)

    logger.info(f"  {symbol}: Return={total_return:.2f}% | Win Rate={win_rate:.1f}% | Trade Costs=₹{total_costs:.2f}")

    return {
        "symbol":          symbol,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity":    round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "num_round_trips": len(sell_trades),
        "win_rate_pct":    round(win_rate, 1),
        "total_costs":     round(total_costs, 2),
        "equity_curve":    equity_curve,
    }


def run_portfolio_backtest(threshold: float = 0.60):
    """
    Runs the backtest across ALL active stocks in the portfolio universe
    and outputs an aggregate summary.
    """
    logger.info("=" * 60)
    logger.info("STARTING MULTI-STOCK PORTFOLIO BACKTEST")
    logger.info(f"Transaction Costs Active: Brokerage={BROKERAGE_PCT*100:.3f}% | "
                f"STT Buy={STT_BUY_PCT*100:.2f}% | STT Sell={STT_SELL_PCT*100:.2f}% | "
                f"Slippage={SLIPPAGE_PCT*100:.2f}%")
    logger.info("=" * 60)

    model = load_latest_model()

    # Fetch all active stocks from DB
    with engine.connect() as conn:
        stocks = pd.read_sql(
            text("SELECT symbol FROM stocks WHERE is_active = true"),
            conn
        )

    if stocks.empty:
        logger.error("No active stocks found in DB.")
        return

    symbols = stocks["symbol"].tolist()
    logger.info(f"Running backtest for {len(symbols)} stocks: {symbols}")

    all_results = []
    combined_equity = {}

    for symbol in symbols:
        result = run_backtest_for_symbol(symbol, model, threshold)
        if result:
            all_results.append(result)
            # Aggregate equity curves
            for point in result["equity_curve"]:
                d = str(point["date"].date())
                combined_equity[d] = combined_equity.get(d, 0) + point["equity"]

    if not all_results:
        logger.error("No backtest results generated.")
        return

    # ------ Summary Table ------
    summary_df = pd.DataFrame([{
        k: v for k, v in r.items() if k != "equity_curve"
    } for r in all_results])

    logger.info("\n" + "=" * 60)
    logger.info("PORTFOLIO BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info(f"\nAvg Return:   {summary_df['total_return_pct'].mean():.2f}%")
    logger.info(f"Avg Win Rate: {summary_df['win_rate_pct'].mean():.1f}%")
    logger.info(f"Total Costs:  ₹{summary_df['total_costs'].sum():.2f}")

    # Save CSV
    os.makedirs("model/experiments", exist_ok=True)
    summary_df.drop(columns=["equity_curve"], errors="ignore").to_csv(
        "model/experiments/portfolio_backtest_results.csv", index=False
    )
    logger.info("Results saved to model/experiments/portfolio_backtest_results.csv")

    # ------ Combined Equity Curve Plot ------
    if combined_equity:
        edf = pd.DataFrame(list(combined_equity.items()), columns=["date", "total_equity"])
        edf["date"] = pd.to_datetime(edf["date"])
        edf = edf.sort_values("date")

        plt.figure(figsize=(14, 6))
        plt.plot(edf["date"], edf["total_equity"], label="Combined Portfolio Equity", color="royalblue", linewidth=2)
        plt.axhline(y=INITIAL_CAPITAL * len(all_results), color="gray", linestyle="--", label="Starting Capital")
        plt.title("Multi-Stock Portfolio Backtest Equity Curve (After Transaction Costs)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (₹)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest_portfolio.png", dpi=150)
        logger.info("Portfolio equity curve saved to backtest_portfolio.png")

    return summary_df


if __name__ == "__main__":
    run_portfolio_backtest()
