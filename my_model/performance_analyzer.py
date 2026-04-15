"""Model performance analyzer with model-name parameter.

Usage examples:
    python my_model/performance_analyzer.py --model enhanced2
    python my_model/performance_analyzer.py --model enhanced2 --compare enhanced
    python my_model/performance_analyzer.py --model enhanced --top-loss 8
"""

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.prelude_template import load_data


MODEL_REGISTRY = {
    "enhanced": {
        "module": "my_model.model_development_enhanced",
        "output_dir": "my_model/output_enhanced",
        "label": "Enhanced",
    },
    "enhanced1": {
        "module": "my_model.model_development_enhanced1",
        "output_dir": "my_model/output_enhanced1",
        "label": "Enhanced1",
    },
    "enhanced2": {
        "module": "my_model.model_development_enhanced2",
        "output_dir": "my_model/output_enhanced2",
        "label": "Enhanced2",
    },
    "enhanced3": {
        "module": "my_model.model_development_enhanced3",
        "output_dir": "my_model/output_enhanced3",
        "label": "Enhanced3",
    },
    "enhanced4": {
        "module": "my_model.model_development_enhanced4",
        "output_dir": "my_model/output_enhanced4",
        "label": "Enhanced4",
    },
}


@dataclass
class ModelBundle:
    name: str
    label: str
    output_dir: Path
    module: Any
    metrics: dict
    results_df: pd.DataFrame
    features_df: pd.DataFrame


class PerformanceAnalyzer:
    def __init__(self, model_name: str, compare_name: str | None = None, top_loss: int = 10):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Choices: {list(MODEL_REGISTRY)}")
        if compare_name and compare_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown compare model '{compare_name}'. Choices: {list(MODEL_REGISTRY)}")

        self.model_name = model_name
        self.compare_name = compare_name
        self.top_loss = top_loss

        self.btc_df = load_data()
        self.primary = self._load_bundle(model_name)
        self.compare = self._load_bundle(compare_name) if compare_name else None

    def _load_bundle(self, model_name: str) -> ModelBundle:
        cfg = MODEL_REGISTRY[model_name]
        output_dir = Path(cfg["output_dir"])
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"{metrics_path} not found. Please run backtest first for model '{model_name}'."
            )

        module = importlib.import_module(cfg["module"])
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        rows = []
        for w in payload["window_level_data"]:
            rows.append(
                {
                    "window": w["window"],
                    "start_date": pd.to_datetime(w["start_date"]),
                    "dynamic_pct": w["dynamic_percentile"],
                    "uniform_pct": w["uniform_percentile"],
                    "excess": w["excess_percentile"],
                    "dynamic_spd": w["dynamic_sats_per_dollar"],
                    "uniform_spd": w["uniform_sats_per_dollar"],
                }
            )

        results_df = pd.DataFrame(rows)
        results_df["year"] = results_df["start_date"].dt.year
        results_df["month"] = results_df["start_date"].dt.month
        results_df["is_win"] = results_df["excess"] > 0

        features_df = module.precompute_features(self.btc_df)

        return ModelBundle(
            name=model_name,
            label=cfg["label"],
            output_dir=output_dir,
            module=module,
            metrics=payload["summary_metrics"],
            results_df=results_df,
            features_df=features_df,
        )

    @staticmethod
    def _window_end(start_date: pd.Timestamp) -> pd.Timestamp:
        return start_date + pd.Timedelta(days=365)

    def _compute_window_detail(self, bundle: ModelBundle, start_date: pd.Timestamp) -> dict:
        end_date = self._window_end(start_date)
        prices = self.btc_df.loc[start_date:end_date, "PriceUSD_coinmetrics"].copy()
        if prices.empty:
            return {}

        weights = bundle.module.compute_window_weights(
            features_df=bundle.features_df,
            start_date=start_date,
            end_date=end_date,
            current_date=end_date,
        ).reindex(prices.index).fillna(0.0)

        n = len(prices)
        if n == 0:
            return {}

        uniform_w = np.full(n, 1.0 / n)
        inv_price = 1e8 / prices.values
        diff_w = weights.values - uniform_w
        contribution = diff_w * inv_price

        q25 = prices.quantile(0.25)
        q75 = prices.quantile(0.75)
        cheap_mask = prices <= q25
        expensive_mask = prices >= q75

        cheap_w = float(weights[cheap_mask].sum())
        expensive_w = float(weights[expensive_mask].sum())

        weighted_buy_price = float((weights.values * prices.values).sum())
        uniform_buy_price = float(prices.mean())

        early_half_w = float(weights.iloc[: n // 2].sum())
        late_half_w = float(weights.iloc[n // 2 :].sum())

        return {
            "cheap_weight": cheap_w,
            "expensive_weight": expensive_w,
            "cheap_expensive_ratio": cheap_w / max(expensive_w, 1e-12),
            "weighted_buy_price": weighted_buy_price,
            "uniform_buy_price": uniform_buy_price,
            "buy_price_edge_pct": (uniform_buy_price / weighted_buy_price - 1.0) * 100,
            "early_half_weight": early_half_w,
            "late_half_weight": late_half_w,
            "spd_alpha_decomp": float(contribution.sum()),
            "top_positive_days": pd.DataFrame(
                {
                    "date": prices.index,
                    "price": prices.values,
                    "weight": weights.values,
                    "uniform_weight": uniform_w,
                    "weight_diff": diff_w,
                    "contribution": contribution,
                }
            )
            .nlargest(5, "contribution")
            .to_dict("records"),
            "top_negative_days": pd.DataFrame(
                {
                    "date": prices.index,
                    "price": prices.values,
                    "weight": weights.values,
                    "uniform_weight": uniform_w,
                    "weight_diff": diff_w,
                    "contribution": contribution,
                }
            )
            .nsmallest(5, "contribution")
            .to_dict("records"),
        }

    def print_summary(self):
        m = self.primary.metrics
        print("\n" + "=" * 78)
        print(f"MODEL SUMMARY: {self.primary.name} ({self.primary.label})")
        print("=" * 78)
        print(f"score:                 {m['score']:.4f}")
        print(f"win_rate:              {m['win_rate']:.4f}%")
        print(f"exp_decay_percentile:  {m['exp_decay_percentile']:.4f}%")
        print(f"mean_excess:           {m['mean_excess']:.4f}%")
        print(f"median_excess:         {m['median_excess']:.4f}%")
        print(f"mean_ratio:            {m['mean_ratio']:.4f}")
        print(f"median_ratio:          {m['median_ratio']:.4f}")
        print(f"windows:               {m['wins']}/{m['total_windows']} wins")

    def print_compare_summary(self):
        if self.compare is None:
            return
        a = self.primary.metrics
        b = self.compare.metrics

        print("\n" + "=" * 78)
        print(f"COMPARE: {self.primary.name} vs {self.compare.name}")
        print("=" * 78)

        def line(k: str):
            da = a[k]
            db = b[k]
            diff = da - db
            print(f"{k:22s} {da:10.4f} | {db:10.4f} | diff={diff:+.4f}")

        for key in [
            "score",
            "win_rate",
            "exp_decay_percentile",
            "mean_excess",
            "median_excess",
            "mean_ratio",
            "median_ratio",
        ]:
            line(key)

    def print_yearly_breakdown(self):
        df = self.primary.results_df
        yearly = (
            df.groupby("year")
            .agg(
                excess_mean=("excess", "mean"),
                excess_median=("excess", "median"),
                win_rate=("is_win", "mean"),
                count=("window", "count"),
            )
            .round(4)
        )
        yearly["win_rate"] = yearly["win_rate"] * 100

        print("\n" + "=" * 78)
        print("YEARLY BREAKDOWN")
        print("=" * 78)
        print(yearly)

    def print_yearly_shape_diagnostics(self):
        print("\n" + "=" * 78)
        print("YEARLY WEIGHT SHAPE DIAGNOSTICS")
        print("=" * 78)

        rows = []
        for year, grp in self.primary.results_df.groupby("year"):
            sample = grp.sort_values("start_date").iloc[:: max(len(grp) // 12, 1)]
            details = []
            for _, r in sample.iterrows():
                det = self._compute_window_detail(self.primary, r["start_date"])
                if det:
                    details.append(det)

            if not details:
                continue

            rows.append(
                {
                    "year": int(year),
                    "avg_cheap_w": np.mean([d["cheap_weight"] for d in details]),
                    "avg_expensive_w": np.mean([d["expensive_weight"] for d in details]),
                    "cheap_minus_exp": np.mean([d["cheap_weight"] - d["expensive_weight"] for d in details]),
                    "avg_buy_price_edge%": np.mean([d["buy_price_edge_pct"] for d in details]),
                    "avg_early_half_w": np.mean([d["early_half_weight"] for d in details]),
                    "avg_late_half_w": np.mean([d["late_half_weight"] for d in details]),
                }
            )

        out = pd.DataFrame(rows).round(4)
        print(out)

    def print_weight_shape_diagnostics(self):
        print("\n" + "=" * 78)
        print("WEIGHT SHAPE DIAGNOSTICS (sampled windows)")
        print("=" * 78)

        df = self.primary.results_df.sort_values("start_date")
        sample = df.iloc[:: max(len(df) // 8, 1)][["start_date", "excess"]]

        rows = []
        for _, r in sample.iterrows():
            det = self._compute_window_detail(self.primary, r["start_date"])
            if not det:
                continue
            rows.append(
                {
                    "start_date": r["start_date"].date(),
                    "excess": round(float(r["excess"]), 3),
                    "cheap_w": round(det["cheap_weight"], 4),
                    "expensive_w": round(det["expensive_weight"], 4),
                    "cheap/exp": round(det["cheap_expensive_ratio"], 3),
                    "buy_price_edge%": round(det["buy_price_edge_pct"], 3),
                    "early_half_w": round(det["early_half_weight"], 4),
                    "late_half_w": round(det["late_half_weight"], 4),
                }
            )

        print(pd.DataFrame(rows))

    def print_worst_windows_detail(self):
        print("\n" + "=" * 78)
        print(f"WORST {self.top_loss} WINDOWS: WHY THEY LOST")
        print("=" * 78)

        worst = self.primary.results_df.nsmallest(self.top_loss, "excess")
        for _, row in worst.iterrows():
            start = row["start_date"]
            end = self._window_end(start)
            det = self._compute_window_detail(self.primary, start)
            if not det:
                continue

            print(
                f"\n[{start.date()} -> {end.date()}] excess={row['excess']:.3f}% "
                f"(dyn={row['dynamic_pct']:.2f}, uni={row['uniform_pct']:.2f})"
            )
            print(
                f"  cheap_w={det['cheap_weight']:.4f}, expensive_w={det['expensive_weight']:.4f}, "
                f"cheap/exp={det['cheap_expensive_ratio']:.3f}"
            )
            print(
                f"  weighted_buy_price={det['weighted_buy_price']:.2f}, "
                f"uniform_buy_price={det['uniform_buy_price']:.2f}, "
                f"buy_price_edge={det['buy_price_edge_pct']:.3f}%"
            )
            print(
                f"  early_half_w={det['early_half_weight']:.4f}, "
                f"late_half_w={det['late_half_weight']:.4f}, "
                f"spd_alpha_decomp={det['spd_alpha_decomp']:.2f}"
            )

            print("  top negative contribution days:")
            for d in det["top_negative_days"]:
                dt = pd.to_datetime(d["date"]).date()
                print(
                    f"    {dt} | price={d['price']:.2f} | w={d['weight']:.5f} "
                    f"vs uni={d['uniform_weight']:.5f} | Δw={d['weight_diff']:+.5f} "
                    f"| contrib={d['contribution']:.2f}"
                )

    def run(self):
        self.print_summary()
        self.print_compare_summary()
        self.print_yearly_breakdown()
        self.print_yearly_shape_diagnostics()
        self.print_weight_shape_diagnostics()
        self.print_worst_windows_detail()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze backtest metrics by model name")
    parser.add_argument(
        "--model",
        type=str,
        default="enhanced3",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to analyze (maps to model file + output folder)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Optional model to compare against",
    )
    parser.add_argument(
        "--top-loss",
        type=int,
        default=10,
        help="How many worst windows to print in detail",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyzer = PerformanceAnalyzer(
        model_name=args.model,
        compare_name=args.compare,
        top_loss=args.top_loss,
    )
    analyzer.run()
