#!/usr/bin/env python3

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))

from src.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ragas_evaluation():
    logger.info("Running RAGAS evaluation...")
    try:
        from test_ragas import run_ragas_evaluation as ragas_eval
        ragas_eval()
        return True
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return False


def run_deepeval_evaluation():
    logger.info("Running DeepEval evaluation...")
    try:
        from test_deepeval import run_deepeval_evaluation as deepeval_eval
        deepeval_eval()
        return True
    except Exception as e:
        logger.error(f"DeepEval evaluation failed: {e}")
        return False


def compare_results():
    logger.info("Comparing evaluation results...")

    results_dir = "evaluation_results"
    if not os.path.exists(results_dir):
        logger.warning("No evaluation results found. Run evaluations first.")
        return

    import json

    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frameworks": {}
    }

    ragas_summary_path = os.path.join(results_dir, "ragas_summary.json")
    deepeval_summary_path = os.path.join(results_dir, "deepeval_summary.json")

    if os.path.exists(ragas_summary_path):
        with open(ragas_summary_path, 'r') as f:
            comparison["frameworks"]["ragas"] = json.load(f)
    else:
        logger.warning("RAGAS summary not found")

    if os.path.exists(deepeval_summary_path):
        with open(deepeval_summary_path, 'r') as f:
            comparison["frameworks"]["deepeval"] = json.load(f)
    else:
        logger.warning("DeepEval summary not found")

    comparison_path = os.path.join(results_dir, "framework_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Framework comparison saved to: {comparison_path}")

    print("\n" + "="*60)
    print("FRAMEWORK COMPARISON SUMMARY")
    print("="*60)

    for framework_name, framework_data in comparison["frameworks"].items():
        print(f"\n{framework_name.upper()}:")
        print(f"  Total Questions: {framework_data.get('total_questions', 'N/A')}")

        if framework_name == "ragas":
            for metric, score in framework_data.items():
                if isinstance(score, (int, float)) and metric != 'total_questions':
                    print(f"  {metric.replace('_', ' ').title()}: {score:.4f}")

        elif framework_name == "deepeval":
            agg_scores = framework_data.get('aggregated_scores', {})
            for metric_name, stats in agg_scores.items():
                if isinstance(stats, dict):
                    avg_score = stats.get('average_score', 0)
                    success_rate = stats.get('success_rate', 0)
                    print(f"  {metric_name}:")
                    print(f"    Average Score: {avg_score:.4f}")
                    print(f"    Success Rate: {success_rate:.4f}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Run RAG evaluation tests')
    parser.add_argument('--framework', choices=['ragas', 'deepeval', 'both'],
                       default='both', help='Which evaluation framework to run')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results from both frameworks')

    args = parser.parse_args()

    logger.info("Checking prerequisites...")

    required_vars = ['GEMINI_API_KEY']
    missing_vars = [var for var in required_vars if not getattr(settings, var.lower(), None)]

    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.info("Please copy .env.example to .env and fill in the required values")
        return False

    results = {}

    if args.framework in ['ragas', 'both']:
        logger.info("Starting RAGAS evaluation...")
        results['ragas'] = run_ragas_evaluation()

    if args.framework in ['deepeval', 'both']:
        logger.info("Starting DeepEval evaluation...")
        results['deepeval'] = run_deepeval_evaluation()

    if args.compare or args.framework == 'both':
        compare_results()

    success_count = sum(results.values())
    total_count = len(results)

    logger.info(f"\nEvaluation Summary: {success_count}/{total_count} frameworks completed successfully")

    if success_count == total_count:
        logger.info("All evaluations completed successfully!")
        logger.info("\nCheck the 'evaluation_results' directory for detailed reports")
        return True
    else:
        logger.warning("Some evaluations failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)