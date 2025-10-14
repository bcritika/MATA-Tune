import pandas as pd
import numpy as np
import langgraph
import json
import langchain
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List
import requests
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import yfinance as yf
from langgraph.graph import add_messages
from typing_extensions import Annotated
import warnings

warnings.filterwarnings("ignore", message="YF.download() has changed argument auto_adjust")

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPEN_AI_KEY"))

# ---------------- STATE ---------------- #
class PortfolioState(TypedDict, total=False):
    user: dict
    indicators: dict
    industries: list
    assets: list
    companies: list
    sentiment: dict
    fundamental: dict
    technical: dict
    recommendation: dict


# ---------------- USER ---------------- #
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
        print("\n🟦 [UserNode] Starting...")
        user_dict = self.user.to_dict()
        print("User:", user_dict)
        print("🟩 [UserNode] Completed.\n")
        return {"user": user_dict}


# ---------------- ECONOMIC INDICATORS ---------------- #
class EconomicIndicators:
    def __init__(self):
        self.api_key = os.getenv("FRED_KEY")
        self.m2 = None
        self.unemployment = None
        self.gdp = None
        self.consumer_spending = None
        self.usd_strength = None

    def _fetch_from_fred(self, series_id):
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.api_key}&file_type=json"
        resp = requests.get(url).json()
        if resp.get("observations"):
            return float(resp["observations"][-1]["value"])
        return None

    def get_all(self):
        return {
            "m2": self._fetch_from_fred("M2SL"),
            "unemployment": self._fetch_from_fred("UNRATE"),
            "gdp": self._fetch_from_fred("A191RL1Q225SBEA"),
            "consumer_spending": self._fetch_from_fred("PCE"),
            "usd_strength": yf.Ticker("DX-Y.NYB").history(period="1mo")["Close"].iloc[-1],
        }

class EconomicIndicatorsNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [EconomicIndicatorsNode] Starting...")
        econ = EconomicIndicators()
        indicators = econ.get_all()
        state["indicators"] = indicators
        print("Indicators:", indicators)
        print("🟩 [EconomicIndicatorsNode] Completed.\n")
        return state


# ---------------- INDUSTRIES ---------------- #
class IndustriesNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [IndustriesNode] Starting...")
        indicators = state.get("indicators", {})
        prompt = f"You are an investment strategist. Indicators: {indicators}. Return valid JSON with 'industries' and 'reasoning'."
        response = llm.invoke(prompt)
        response_text = response.content.strip().replace("```json", "").replace("```", "")
        try:
            industries_data = json.loads(response_text)
            state["industries"] = industries_data.get("industries", [])
            state["industries_reasoning"] = industries_data.get("reasoning", {})
        except Exception as e:
            print("⚠️ JSON parse failed in IndustriesNode:", e)
            print("Raw output:", response_text)
            state["industries"] = [response_text]
        print("Industries:", state.get("industries"))
        print("🟩 [IndustriesNode] Completed.\n")
        return state


# ---------------- ASSET CLASSES ---------------- #
class AssetClassesNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [AssetClassesNode] Starting...")
        industries = state.get("industries", [])
        indicators = state.get("indicators", {})
        user = state.get("user", {})
        prompt = f"Industries: {industries}, Indicators: {indicators}, User: {user}. Return valid JSON with 'asset_classes'."
        response = llm.invoke(prompt)
        response_text = response.content.strip().replace("```json", "").replace("```", "")
        try:
            data = json.loads(response_text)
            state["asset_classes"] = data.get("asset_classes", [])
            state["asset_classes_reasoning"] = data.get("reasoning", {})
        except Exception as e:
            print("⚠️ JSON parse failed in AssetClassesNode:", e)
            print("Raw output:", response_text)
            state["asset_classes"] = [response_text]
        print("Asset classes:", state.get("asset_classes"))
        print("🟩 [AssetClassesNode] Completed.\n")
        return state


# ---------------- COMPANIES ---------------- #
class CompaniesNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [CompaniesNode] Starting...")
        industries = state.get("industries", [])
        asset_classes = state.get("asset_classes", [])
        indicators = state.get("indicators", {})
        user = state.get("user", {})
        prompt = f"Industries: {industries}, Asset classes: {asset_classes}, Indicators: {indicators}, User: {user}. Return JSON with companies."
        response = llm.invoke(prompt)
        response_text = response.content.strip().replace("```json", "").replace("```", "")
        try:
            data = json.loads(response_text)
            state["companies"] = data.get("companies", [])
            state["companies_reasoning"] = data.get("reasoning", {})
        except Exception as e:
            print("⚠️ JSON parse failed in CompaniesNode:", e)
            print("Raw output:", response_text)
            state["companies"] = [response_text]
        print("Companies:", state.get("companies"))
        print("🟩 [CompaniesNode] Completed.\n")
        return state


# ---------------- SENTIMENT ---------------- #
class SentimentNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [SentimentNode] Starting...")
        companies = state.get("companies", [])
        if not companies:
            print("⚠️ No companies found — skipping sentiment analysis.")
            state["sentiment"] = {}
            return state
        prompt = f"Evaluate sentiment for companies {companies}. Return JSON with 'sentiment' and 'reasoning'."
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            state["sentiment"] = data.get("sentiment", {})
        except Exception as e:
            print("⚠️ JSON parse failed in SentimentNode:", e)
            state["sentiment"] = {c: "unknown" for c in companies}
        print("Sentiment:", state.get("sentiment"))
        print("🟩 [SentimentNode] Completed.\n")
        return state


# ---------------- TECHNICAL ---------------- #
class TechnicalIndicatorsNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [TechnicalIndicatorsNode] Starting...")

        companies_state = state.get("companies", [])
        technicals = {}

        # ✅ Handle both shapes: nested dict or flat list
        if isinstance(companies_state, dict):
            tickers = [
                c["ticker"].upper()
                for sector in companies_state.values()
                for c in sector
                if "ticker" in c
            ]
        elif isinstance(companies_state, list):
            tickers = [c.upper() for c in companies_state]
        else:
            tickers = []

        print(f"✅ Extracted {len(tickers)} tickers for technicals: {tickers}")

        for t in tickers:
            try:
                # Download last 6 months of daily close data
                data = yf.download(t, period="6mo", interval="1d", progress=False)
                if data.empty:
                    raise ValueError("No price data available")

                close = data["Close"]

                # --- RSI (14-day) ---
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = -delta.clip(upper=0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                # --- MACD (12-26) ---
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()

                technicals[t] = {
                    "RSI": round(rsi.iloc[-1], 2),
                    "MACD": round(macd.iloc[-1], 2),
                    "Signal": round(signal.iloc[-1], 2),
                    "Trend": "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish"
                }

            except Exception as e:
                technicals[t] = {"error": str(e)}

        state["technical"] = technicals
        print(f"📈 Collected technicals for {len(technicals)} tickers.")
        print("🟩 [TechnicalIndicatorsNode] Completed.\n")
        return state


# ---------------- FUNDAMENTALS ---------------- #
class FundamentalIndicatorsNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [FundamentalIndicatorsNode] Starting...")

        companies_state = state.get("companies", [])
        fundamentals = {}

        tickers = []

        for sector in companies_state:
            rec_stocks = companies_state[sector]["recommended_stocks"]
            temp = [s.split("(")[-1].strip(")") for s in rec_stocks]
            tickers.extend(temp) 

        # ✅ Handle both: flat list of tickers OR dict of industries
        if isinstance(companies_state, dict):
            # Flatten to list of tickers from nested dict
            tickers = [
                c["ticker"].upper()
                for industry in companies_state.values()
                for c in industry
                if "ticker" in c
            ]
        elif isinstance(companies_state, list):
            # Already flat list
            tickers = [c.upper() for c in companies_state]
        else:
            tickers = []

        print(f"✅ Extracted {len(tickers)} tickers for fundamentals: {tickers}")

        for t in tickers:
            try:
                ticker = yf.Ticker(t)
                info = ticker.info

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
        print(f"📊 Collected fundamentals for {len(fundamentals)} companies.")
        print("🟩 [FundamentalIndicatorsNode] Completed.\n")
        return state



# ---------------- PORTFOLIO OPTIMIZATION ---------------- #
class PortfolioOptimizationNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [PortfolioOptimizationNode] Starting...")
        sentiment = state.get("sentiment", {})
        technical = state.get("technical", {})
        fundamental = state.get("fundamental", {})
        user = state.get("user", {})
        scores = {}
        for c in sentiment.keys():
            score = 0
            if sentiment.get(c) == "positive": score += 2
            elif sentiment.get(c) == "neutral": score += 1
            else: score += 1
            if technical.get(c, {}).get("Trend") == "bullish": score += 1
            if fundamental.get(c, {}).get("PE") and fundamental[c]["PE"] < 20: score += 1
            scores[c] = score
        total = sum(np.exp(v) for v in scores.values()) if scores else 1
        weights = {c: round(np.exp(v) / total, 3) for c, v in scores.items()}
        state["portfolio"] = weights
        print("Portfolio weights:", weights)
        print("🟩 [PortfolioOptimizationNode] Completed.\n")
        return state


# ---------------- RECOMMENDATION ---------------- #
class RecommendationNode:
    def __call__(self, state: PortfolioState) -> PortfolioState:
        print("\n🟦 [RecommendationNode] Starting...")
        portfolio = state.get("portfolio", {})
        if not portfolio:
            print("⚠️ No portfolio generated.")
            state["recommendation"] = {
                "portfolio": {},
                "note": "No portfolio could be generated."
            }
            return state
        user = state.get("user", {})
        industries = state.get("industries", [])
        note = f"Portfolio for {user.get('name')} with {len(portfolio)} stocks. Industries: {industries}."
        state["recommendation"] = {"portfolio": portfolio, "note": note}
        print("Final Recommendation:", state["recommendation"])
        print("🟩 [RecommendationNode] Completed.\n")
        return state


# ---------------- GRAPH SETUP ---------------- #
workflow = StateGraph(PortfolioState)
workflow.set_entry_point("user")
user = UserProfile(
    name="Alice",
    risk_profile="moderate",
    return_requirements=0.08,
    preferred_assets=["Tech Stocks", "Dividend Stocks"],
    capital=100000
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

# ---------------- RUN ---------------- #
app = workflow.compile()
initial_state = PortfolioState()
final_state = app.invoke(initial_state)

print("\n==============================")
print("🏁 FINAL OUTPUT")
print("==============================")
print(final_state["recommendation"])
