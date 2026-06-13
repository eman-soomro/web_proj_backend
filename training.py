import os
import time
from pytrends.request import TrendReq  

class TrainingAbstraction:

    def __init__(self, domain, keywords, prediction_horizon=30, sleep_time=5):
        self.domain = domain
        self.keywords = keywords
        self.prediction_horizon = prediction_horizon
        self.sleep_time = sleep_time

        self.data = None
        self.results = {}

        # Save graphs inside Flask's static folder
        self.graph_dir = os.path.join(os.path.dirname(__file__), "static", "graphs")
        os.makedirs(self.graph_dir, exist_ok=True)

    def _ingest_data(self):
        pytrends = TrendReq()
        pytrends.build_payload(self.keywords, timeframe="today 3-m")
        time.sleep(self.sleep_time)

        self.data = pytrends.interest_over_time()
        if self.data.empty:
            raise Exception("No Google Trends data found for the provided keywords.")

    def _plot_forecast(self, keyword, historical_df, forecast_df):
        import matplotlib
        matplotlib.use("Agg")  
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.scatter(historical_df["ds"], historical_df["y"], color="blue", marker="o", s=30, label="Historical")
        plt.scatter(forecast_df["ds"], forecast_df["yhat"], color="red", marker="x", s=50, label="Forecast")

        plt.title(f"Google Trends Forecast - {keyword}")
        plt.xlabel("Date")
        plt.ylabel("Interest Score")
        plt.legend()
        plt.grid(True)

        filename = f"{keyword.replace(' ', '_')}_forecast.png"
        filepath = os.path.join(self.graph_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        # Return a public URL path
        return f"/static/graphs/{filename}"

    def _forecast_keyword(self, keyword):
        import pandas as pd
        from prophet import Prophet

        df = self.data.reset_index()
        df.rename(columns={df.columns[0]: "ds"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])

        prophet_df = df[["ds", keyword]].rename(columns={keyword: "y"})
        prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 10:
            raise Exception(f"Insufficient data available for keyword: {keyword}")

        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                        changepoint_prior_scale=0.5)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=self.prediction_horizon)
        forecast = model.predict(future)
        forecast_future = forecast[["ds", "yhat"]].tail(self.prediction_horizon)

        current_interest = float(prophet_df["y"].iloc[-1])
        forecast_interest = float(forecast_future["yhat"].mean())
        change_percent = ((forecast_interest - current_interest) / max(current_interest, 1)) * 100

        graph_url = self._plot_forecast(keyword, prophet_df, forecast_future)

        return {
            "trend": "up" if change_percent > 0 else "down" if change_percent < 0 else "flat",
            "current_interest": round(current_interest, 2),
            "forecast_interest": round(forecast_interest, 2),
            "change_percent": round(change_percent, 2),
            "graph": graph_url,  # <-- now a public URL
            "forecast": [
                {"date": str(row["ds"].date()), "predicted_interest": round(float(row["yhat"]), 2)}
                for _, row in forecast_future.iterrows()
            ]
        }

    def run_training_pipeline(self):
        self._ingest_data()
        for keyword in self.keywords:
            try:
                self.results[keyword] = self._forecast_keyword(keyword)
            except Exception as e:
                self.results[keyword] = {"error": str(e)}

        return {
            "data_ingested": True,
            "data_preprocessed": True,
            "model_trained": True,
            "domain": self.domain,
            "prediction_horizon": self.prediction_horizon,
            "keywords": self.results
        }
