#!/usr/bin/env python3
"""
Calculate feature importance for RL models.

Usage:
    python feature_importance.py --model-dir user_data/models/MtfScalperRL_v1/ --pair BTC/USDT:USDT

Features:
    - Permutation importance analysis
    - Correlation with actions
    - Feature usage frequency
    - Recommendations for feature pruning
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Error: stable-baselines3 required")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)


class FeatureImportanceAnalyzer:
    """Analyzes feature importance for RL models."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.importance_scores = {}

    def load_model(self):
        """Load trained PPO model."""
        print(f"Loading model from {self.model_path}...")

        model_file = self.model_path / "trained_model.zip"
        if not model_file.exists():
            print(f"Error: Model file not found: {model_file}")
            sys.exit(1)

        self.model = PPO.load(str(model_file))
        print("Model loaded successfully")

    def calculate_permutation_importance(
        self,
        dataframe: pd.DataFrame,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate feature importance using permutation method.

        For each feature:
        1. Get baseline predictions
        2. Shuffle the feature
        3. Get new predictions
        4. Measure prediction difference
        """
        if self.model is None:
            self.load_model()

        print(f"\nCalculating permutation importance on {n_samples} samples...")

        # Get feature columns
        feature_cols = [col for col in dataframe.columns if col.startswith("%-")]

        if len(feature_cols) == 0:
            print("Error: No features found (columns starting with %-)")
            return pd.DataFrame()

        print(f"Found {len(feature_cols)} features")

        # Sample data
        if len(dataframe) > n_samples:
            sample_df = dataframe.sample(n=n_samples, random_state=42)
        else:
            sample_df = dataframe

        # Get baseline predictions
        X = sample_df[feature_cols].values
        baseline_actions, _ = self.model.predict(X, deterministic=True)

        # Calculate importance for each feature
        importance_scores = {}

        for i, feature in enumerate(feature_cols):
            # Create copy with shuffled feature
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])

            # Get predictions with shuffled feature
            shuffled_actions, _ = self.model.predict(X_shuffled, deterministic=True)

            # Measure change in predictions
            action_change_rate = (baseline_actions != shuffled_actions).mean()
            importance_scores[feature] = action_change_rate

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(feature_cols)} features...")

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        }).sort_values('importance', ascending=False)

        self.importance_scores['permutation'] = importance_df
        return importance_df

    def calculate_action_correlation(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlation between features and model actions.
        """
        if self.model is None:
            self.load_model()

        print("\nCalculating feature-action correlations...")

        feature_cols = [col for col in dataframe.columns if col.startswith("%-")]

        # Get predictions
        X = dataframe[feature_cols].values
        actions, _ = self.model.predict(X, deterministic=True)

        # Calculate correlations
        correlations = {}
        for feature in feature_cols:
            feature_values = dataframe[feature].values
            # Use absolute correlation
            corr = abs(np.corrcoef(feature_values, actions)[0, 1])
            correlations[feature] = corr

        # Create DataFrame
        corr_df = pd.DataFrame({
            'feature': list(correlations.keys()),
            'correlation': list(correlations.values())
        }).sort_values('correlation', ascending=False)

        self.importance_scores['correlation'] = corr_df
        return corr_df

    def analyze_feature_statistics(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze feature statistics (variance, range, etc.).
        """
        print("\nAnalyzing feature statistics...")

        feature_cols = [col for col in dataframe.columns if col.startswith("%-")]

        stats = []
        for feature in feature_cols:
            values = dataframe[feature].values

            stat_dict = {
                'feature': feature,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'cv': np.std(values) / (abs(np.mean(values)) + 1e-10),  # Coefficient of variation
                'nan_pct': np.isnan(values).mean() * 100,
                'zero_pct': (values == 0).mean() * 100
            }
            stats.append(stat_dict)

        stats_df = pd.DataFrame(stats)
        self.importance_scores['statistics'] = stats_df
        return stats_df

    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on importance analysis.
        """
        recommendations = []

        if 'permutation' in self.importance_scores:
            perm_df = self.importance_scores['permutation']

            # Low importance features
            low_importance = perm_df[perm_df['importance'] < 0.01]
            if len(low_importance) > 0:
                recommendations.append(
                    f"Consider removing {len(low_importance)} low-importance features: "
                    f"{', '.join(low_importance['feature'].head(5).tolist())}"
                )

            # High importance features
            high_importance = perm_df[perm_df['importance'] > 0.2]
            if len(high_importance) > 0:
                recommendations.append(
                    f"Critical features (keep these!): "
                    f"{', '.join(high_importance['feature'].tolist())}"
                )

        if 'statistics' in self.importance_scores:
            stats_df = self.importance_scores['statistics']

            # Constant or near-constant features
            constant = stats_df[stats_df['std'] < 0.001]
            if len(constant) > 0:
                recommendations.append(
                    f"Remove constant features: {', '.join(constant['feature'].tolist())}"
                )

            # High zero percentage
            mostly_zero = stats_df[stats_df['zero_pct'] > 90]
            if len(mostly_zero) > 0:
                recommendations.append(
                    f"Consider removing sparse features (>90% zeros): "
                    f"{', '.join(mostly_zero['feature'].tolist())}"
                )

        return recommendations

    def generate_plots(self, output_dir: str = "./analysis/"):
        """Generate feature importance plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating plots in {output_path}...")

        # Plot 1: Top features by permutation importance
        if 'permutation' in self.importance_scores:
            perm_df = self.importance_scores['permutation']

            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = perm_df.head(20)

            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance (Action Change Rate)')
            ax.set_title('Top 20 Features by Permutation Importance')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plot_path = output_path / 'feature_importance.png'
            plt.savefig(plot_path, dpi=100)
            print(f"Saved: {plot_path}")
            plt.close()

        # Plot 2: Feature correlations
        if 'correlation' in self.importance_scores:
            corr_df = self.importance_scores['correlation']

            fig, ax = plt.subplots(figsize=(10, 8))
            top_corr = corr_df.head(20)

            ax.barh(range(len(top_corr)), top_corr['correlation'])
            ax.set_yticks(range(len(top_corr)))
            ax.set_yticklabels(top_corr['feature'])
            ax.set_xlabel('Absolute Correlation with Actions')
            ax.set_title('Top 20 Features by Action Correlation')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plot_path = output_path / 'feature_correlation.png'
            plt.savefig(plot_path, dpi=100)
            print(f"Saved: {plot_path}")
            plt.close()

        # Plot 3: Feature statistics
        if 'statistics' in self.importance_scores:
            stats_df = self.importance_scores['statistics']

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Coefficient of variation
            top_cv = stats_df.nlargest(20, 'cv')
            axes[0].barh(range(len(top_cv)), top_cv['cv'])
            axes[0].set_yticks(range(len(top_cv)))
            axes[0].set_yticklabels(top_cv['feature'])
            axes[0].set_xlabel('Coefficient of Variation')
            axes[0].set_title('Most Variable Features')
            axes[0].grid(axis='x', alpha=0.3)

            # Zero percentage
            top_zero = stats_df.nlargest(20, 'zero_pct')
            axes[1].barh(range(len(top_zero)), top_zero['zero_pct'])
            axes[1].set_yticks(range(len(top_zero)))
            axes[1].set_yticklabels(top_zero['feature'])
            axes[1].set_xlabel('Zero Percentage (%)')
            axes[1].set_title('Sparsest Features')
            axes[1].grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plot_path = output_path / 'feature_statistics.png'
            plt.savefig(plot_path, dpi=100)
            print(f"Saved: {plot_path}")
            plt.close()

    def generate_report(self, output_dir: str = "./analysis/"):
        """Generate feature importance report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\nGenerating report...")

        report = "# Feature Importance Analysis\n\n"

        # Permutation importance
        if 'permutation' in self.importance_scores:
            perm_df = self.importance_scores['permutation']
            report += "## Top 10 Features by Permutation Importance\n\n"
            report += perm_df.head(10).to_markdown(index=False)
            report += "\n\n"

            report += "## Bottom 10 Features (Least Important)\n\n"
            report += perm_df.tail(10).to_markdown(index=False)
            report += "\n\n"

        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            report += "## Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"

        # Statistics summary
        if 'statistics' in self.importance_scores:
            stats_df = self.importance_scores['statistics']
            report += "## Feature Statistics Summary\n\n"
            report += f"- Total features: {len(stats_df)}\n"
            report += f"- Features with >50% zeros: {len(stats_df[stats_df['zero_pct'] > 50])}\n"
            report += f"- Near-constant features (std < 0.01): {len(stats_df[stats_df['std'] < 0.01])}\n"
            report += "\n"

        # Save report
        report_path = output_path / 'feature_importance_report.md'
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Saved: {report_path}")

    def run_full_analysis(
        self,
        dataframe: pd.DataFrame,
        output_dir: str = "./analysis/"
    ):
        """Run complete feature importance analysis."""
        print("=" * 60)
        print("Feature Importance Analysis")
        print("=" * 60)

        # Load model
        self.load_model()

        # Run analyses
        self.calculate_permutation_importance(dataframe)
        self.calculate_action_correlation(dataframe)
        self.analyze_feature_statistics(dataframe)

        # Generate outputs
        self.generate_plots(output_dir)
        self.generate_report(output_dir)

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

        # Print summary
        if 'permutation' in self.importance_scores:
            perm_df = self.importance_scores['permutation']
            print(f"\nTop 5 most important features:")
            for i, row in perm_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate feature importance for RL models"
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to parquet/feather file with features (optional, for testing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis/',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples for permutation importance'
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = FeatureImportanceAnalyzer(args.model_dir)

    # Load data if provided
    if args.data_file:
        print(f"Loading data from {args.data_file}...")
        if args.data_file.endswith('.parquet'):
            dataframe = pd.read_parquet(args.data_file)
        elif args.data_file.endswith('.feather'):
            dataframe = pd.read_feather(args.data_file)
        elif args.data_file.endswith('.csv'):
            dataframe = pd.read_csv(args.data_file)
        else:
            print("Error: Unsupported file format. Use .parquet, .feather, or .csv")
            sys.exit(1)

        print(f"Loaded {len(dataframe)} rows")

        # Run analysis
        analyzer.run_full_analysis(dataframe, args.output_dir)
    else:
        print("Note: No data file provided. Load model only for inspection.")
        analyzer.load_model()
        print("\nModel architecture:")
        print(analyzer.model.policy)


if __name__ == '__main__':
    main()
