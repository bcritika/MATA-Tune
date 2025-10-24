import pandas as pd
import numpy as np
import json
import requests
import os
import warnings
import yfinance as yf
from dotenv import load_dotenv
from typing import TypedDict, Dict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

warnings.filterwarnings("ignore", message="YF.download() has changed argument auto_adjust")

# ---------- Environment ---------- #
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_KEY"))

# ---------- Helpers ---------- #
def parse_json_safely(text: str):
    """Remove code fences and parse JSON if possible."""
    if not text:
        return None
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None

def normalize_companies_to_tickers(obj) -> List[str]:
    """Normalize various possible LLM outputs into a flat list of uppercase tickers."""
    if isinstance(obj, dict) and "companies" in obj:
        obj = obj["companies"]

    tickers = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                tickers.append(item.upper())
            elif isinstance(item, dict):
                t = item.get("ticker") or item.get("symbol") or item.get("name") or ""
                if t:
                    tickers.append(str(t).upper())
    elif isinstance(obj, dict):
        for _, arr in obj.items():
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, str):
                        tickers.append(item.upper())
                    elif isinstance(item, dict):
                        t = item.get("ticker") or item.get("symbol") or item.get("name") or ""
                        if t:
                            tickers.append(str(t).upper())
    return sorted(set(tickers))

# ---------- State ---------- #
class PortfolioState(TypedDict, total=False):
    user: dict
    indicators: dict
    industries: list
    asset_classes: list
    companies: list
    sentiment: dict
    fundamental: dict
    technical: dict
    portfolio: dict
    recommendation: dict

# ---------- User ---------- #
class UserProfile:
    def __init__(self, name, risk_profile, return_requirements, preferred_assets, capital):
        self.name = name
        self.risk_profile = risk_profile
        self.return_requirements = return_requirements
        self.preferred_assets = preferred_assets
        self.capital = capital

    def to_dict(self):
        return {
            "name": self.name,
            "risk_profile": self.risk_profile,
            "return_requirements": self.return_requirements,
            "preferred_assets": self.preferred_assets,
            "capital": self.capital
        }

class UserNode:
    def __init__(self, user: UserProfile):
        self.user = user

    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[UserNode] Starting")
        user_dict = self.user.to_dict()
        print("User:", user_dict)
        print("[UserNode] Completed\n")
        return {"user": user_dict}

# ---------- Economic Indicators ---------- #
class EconomicIndicators:
    def __init__(self):
        self.api_key = os.getenv("FRED_KEY")

    def _fetch_from_fred(self, series_id):
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.api_key}&file_type=json"
            resp = requests.get(url, timeout=10).json()
            if resp.get("observations"):
                return float(resp["observations"][-1]["value"])
        except Exception:
            return None
        return None

    def get_all(self):
        try:
            usd = yf.Ticker("DX-Y.NYB").history(period="1mo")["Close"].iloc[-1]
        except Exception:
            usd = None
        return {
            "m2": self._fetch_from_fred("M2SL"),
            "unemployment": self._fetch_from_fred("UNRATE"),
            "gdp": self._fetch_from_fred("A191RL1Q225SBEA"),
            "consumer_spending": self._fetch_from_fred("PCE"),
            "usd_strength": usd,
        }

class EconomicIndicatorsNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[EconomicIndicatorsNode] Starting")
        econ = EconomicIndicators()
        indicators = econ.get_all()
        state["indicators"] = indicators
        print("Indicators:", indicators)
        print("[EconomicIndicatorsNode] Completed\n")
        return state

# ---------- Industries ---------- #
class IndustriesNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[IndustriesNode] Starting")
        indicators = state.get("indicators", {})
        prompt = f"You are an investment strategist. Indicators: {indicators}. Return valid JSON with 'industries' and 'reasoning'."
        response = llm.invoke(prompt)
        data = parse_json_safely(response.content)
        if not isinstance(data, dict):
            state["industries"] = []
        else:
            state["industries"] = data.get("industries", [])
        print("Industries:", state.get("industries"))
        print("[IndustriesNode] Completed\n")
        return state

# ---------- Asset Classes ---------- #
class AssetClassesNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[AssetClassesNode] Starting")
        industries = state.get("industries", [])
        indicators = state.get("indicators", {})
        user = state.get("user", {})
        prompt = f"Industries: {industries}, Indicators: {indicators}, User: {user}. Return JSON with 'asset_classes'."
        response = llm.invoke(prompt)
        data = parse_json_safely(response.content)
        if not isinstance(data, dict):
            state["asset_classes"] = []
        else:
            state["asset_classes"] = data.get("asset_classes", [])
        print("Asset classes:", state.get("asset_classes"))
        print("[AssetClassesNode] Completed\n")
        return state

# ---------- Companies ---------- #
class CompaniesNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[CompaniesNode] Starting")
        industries = state.get("industries", [])
        asset_classes = state.get("asset_classes", [])
        indicators = state.get("indicators", {})
        user = state.get("user", {})
        prompt = f"Industries: {industries}, Asset classes: {asset_classes}, Indicators: {indicators}, User: {user}. Return JSON with a 'companies' key containing stock tickers."
        response = llm.invoke(prompt)
        data = parse_json_safely(response.content)
        tickers = normalize_companies_to_tickers(data)
        state["companies"] = tickers
        print("Companies (tickers):", tickers)
        print("[CompaniesNode] Completed\n")
        return state

# ---------- Sentiment ---------- #
class SentimentNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[SentimentNode] Starting")
        tickers = state.get("companies", []) or []
        if not tickers:
            print("No companies found, skipping sentiment.")
            state["sentiment"] = {}
            return state
        prompt = f"Evaluate market sentiment for these tickers: {tickers}. Return JSON with 'sentiment'."
        response = llm.invoke(prompt)
        data = parse_json_safely(response.content)
        if not isinstance(data, dict) or "sentiment" not in data:
            state["sentiment"] = {t: "unknown" for t in tickers}
        else:
            s = data["sentiment"]
            state["sentiment"] = {k.upper(): v for k, v in s.items()}
        print("Sentiment:", state["sentiment"])
        print("[SentimentNode] Completed\n")
        return state

# ---------- Technical Indicators ---------- #
class TechnicalIndicatorsNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[TechnicalIndicatorsNode] Starting")
        tickers = state.get("companies", []) or []
        technicals = {}
        for t in tickers:
            try:
                data = yf.download(t, period="6mo", interval="1d", progress=False)
                if data.empty:
                    raise ValueError("No price data")
                close = data["Close"]
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = -delta.clip(upper=0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                technicals[t] = {
                    "RSI": round(float(rsi.iloc[-1].item()), 2),
                    "MACD": round(float(macd.iloc[-1].item()), 2),
                    "Signal": round(float(signal.iloc[-1].item()), 2),
                    "Trend": "bullish" if macd.iloc[-1].item() > signal.iloc[-1].item() else "bearish",
                }
            except Exception as e:
                technicals[t] = {"error": str(e)}
        state["technical"] = technicals
        print("Collected technicals for", len(technicals), "tickers.")
        print("[TechnicalIndicatorsNode] Completed\n")
        return state

# ---------- Fundamentals ---------- #
class FundamentalIndicatorsNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[FundamentalIndicatorsNode] Starting")
        tickers = state.get("companies", []) or []
        fundamentals = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                fundamentals[t] = {
                    "PE": info.get("forwardPE") or info.get("trailingPE"),
                    "ROE": info.get("returnOnEquity"),
                    "EPS": info.get("trailingEps"),
                    "P/B": info.get("priceToBook"),
                    "Debt/Equity": info.get("debtToEquity"),
                    "MarketCap": info.get("marketCap"),
                    "DividendYield": info.get("dividendYield"),
                }
            except Exception as e:
                fundamentals[t] = {"error": str(e)}
        state["fundamental"] = fundamentals
        print("Collected fundamentals for", len(fundamentals), "companies.")
        print("[FundamentalIndicatorsNode] Completed\n")
        return state

# ---------- Portfolio Optimization ---------- #
class PortfolioOptimizationNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[PortfolioOptimizationNode] Starting")
        sentiment = state.get("sentiment", {})
        technical = state.get("technical", {})
        fundamental = state.get("fundamental", {})
        scores = {}
        for c in sentiment.keys():
            score = 0
            if sentiment.get(c) == "positive": score += 2
            elif sentiment.get(c) == "neutral": score += 1
            else: score += 1
            if technical.get(c, {}).get("Trend") == "bullish": score += 1
            pe = fundamental.get(c, {}).get("PE")
            if pe and pe < 20: score += 1
            scores[c] = score
        total = sum(np.exp(v) for v in scores.values()) if scores else 1
        weights = {c: round(np.exp(v) / total, 3) for c, v in scores.items()}
        state["portfolio"] = weights
        print("Portfolio weights:", weights)
        print("[PortfolioOptimizationNode] Completed\n")
        return state

# ---------- Recommendation ---------- #
class RecommendationNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("[RecommendationNode] Starting")
        portfolio = state.get("portfolio", {})
        if not portfolio:
            state["recommendation"] = {"portfolio": {}, "note": "No portfolio generated."}
            return state
        user = state.get("user", {})
        industries = state.get("industries", [])
        note = f"Portfolio for {user.get('name')} with {len(portfolio)} stocks. Industries: {industries}."
        state["recommendation"] = {"portfolio": portfolio, "note": note}
        print("Final recommendation:", state["recommendation"])
        print("[RecommendationNode] Completed\n")
        return state

# ---------- Graph Setup ---------- #
workflow = StateGraph(PortfolioState)
workflow.set_entry_point("user")

user = UserProfile(
    name="Ritika",
    risk_profile="high",
    return_requirements=0.15,
    preferred_assets=[],
    capital=3500
)

workflow.add_node("user", UserNode(user))
workflow.add_node("econ", EconomicIndicatorsNode())
workflow.add_node("industries", IndustriesNode())
workflow.add_node("asset_classes", AssetClassesNode())
workflow.add_node("companies", CompaniesNode())
workflow.add_node("sentiment", SentimentNode())
workflow.add_node("fundamental", FundamentalIndicatorsNode())
workflow.add_node("technical", TechnicalIndicatorsNode())
workflow.add_node("portfolio_opt", PortfolioOptimizationNode())
workflow.add_node("recommendation", RecommendationNode())

workflow.add_edge("user", "econ")
workflow.add_edge("econ", "industries")
workflow.add_edge("industries", "asset_classes")
workflow.add_edge("asset_classes", "companies")
workflow.add_edge("companies", "sentiment")
workflow.add_edge("sentiment", "fundamental")
workflow.add_edge("fundamental", "technical")
workflow.add_edge("technical", "portfolio_opt")
workflow.add_edge("portfolio_opt", "recommendation")

workflow.set_finish_point("recommendation")

# ---------- Run ---------- #
app = workflow.compile()
initial_state = PortfolioState()
final_state = app.invoke(initial_state)

print("\n==============================")
print("FINAL OUTPUT")
print("==============================")
portfolio_df = pd.DataFrame(
    list(final_state['portfolio'].items()),
    columns=['Ticker', 'Weight']
)
portfolio_df['Weight (%)'] = (portfolio_df['Weight'] * 100).round(2)
portfolio_df = portfolio_df[['Ticker', 'Weight (%)']]

# Print as a nicely formatted table
print(portfolio_df.to_string(index=False))
