import json
import pandas as pd
import sys
import os
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.rag_system import RAGSystem
from src.vector_store import MongoVectorStore
from src.config import settings

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric
)
from deepeval.test_case import LLMTestCase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepEvalEvaluator:
    def __init__(self, rag_system: RAGSystem, test_data_path: str):
        self.rag_system = rag_system
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()

    def _load_test_data(self):
        with open(self.test_data_path, 'r') as f:
            return json.load(f)

    def prepare_test_cases(self) -> List[LLMTestCase]:
        test_cases = []

        logger.info(f"Preparing {len(self.test_data)} test cases for DeepEval...")

        for i, item in enumerate(self.test_data):
            question = item['question']
            expected_answer = item['expected_answer']

            logger.info(f"Processing test case {i+1}: {question[:50]}...")

            try:
                result = self.rag_system.query(question)
                answer = result['answer']
                retrieved_docs = result['retrieved_documents']

                retrieval_context = [doc['content'] for doc in retrieved_docs] if retrieved_docs else [""]

                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    expected_output=expected_answer,
                    retrieval_context=retrieval_context
                )

                test_cases.append(test_case)

            except Exception as e:
                logger.error(f"Error processing test case {i+1} '{question}': {e}")

                test_case = LLMTestCase(
                    input=question,
                    actual_output="Error generating answer",
                    expected_output=expected_answer,
                    retrieval_context=[""]
                )
                test_cases.append(test_case)

        logger.info(f"Successfully prepared {len(test_cases)} test cases")
        return test_cases

    def get_evaluation_metrics(self):
        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.7),
            HallucinationMetric(threshold=0.3),
            ToxicityMetric(threshold=0.5),
            BiasMetric(threshold=0.5)
        ]
        return metrics

    def evaluate_with_deepeval(self, test_cases: List[LLMTestCase] = None):
        if test_cases is None:
            test_cases = self.prepare_test_cases()

        metrics = self.get_evaluation_metrics()

        logger.info("Starting DeepEval evaluation...")

        results = {}

        for metric in metrics:
            metric_name = metric.__class__.__name__
            logger.info(f"Evaluating with {metric_name}...")

            try:
                results[metric_name] = []

                for i, test_case in enumerate(test_cases):
                    logger.info(f"  Processing test case {i+1}/{len(test_cases)}")

                    try:
                        metric.measure(test_case)
                        score = metric.score
                        success = metric.success

                        results[metric_name].append({
                            'test_case_index': i,
                            'question': test_case.input,
                            'score': score,
                            'success': success,
                            'reason': getattr(metric, 'reason', 'No reason provided')
                        })

                    except Exception as e:
                        logger.error(f"Error evaluating test case {i+1} with {metric_name}: {e}")
                        results[metric_name].append({
                            'test_case_index': i,
                            'question': test_case.input,
                            'score': 0.0,
                            'success': False,
                            'reason': f"Evaluation error: {str(e)}"
                        })

                logger.info(f"Completed {metric_name} evaluation")

            except Exception as e:
                logger.error(f"Error during {metric_name} evaluation: {e}")
                results[metric_name] = []

        logger.info("DeepEval evaluation completed!")
        return results

    def calculate_aggregate_scores(self, results: Dict[str, List[Dict]]):
        aggregated = {}

        for metric_name, metric_results in results.items():
            if metric_results:
                scores = [r['score'] for r in metric_results if r['score'] is not None]
                success_rate = sum(1 for r in metric_results if r['success']) / len(metric_results)

                aggregated[metric_name] = {
                    'average_score': sum(scores) / len(scores) if scores else 0.0,
                    'success_rate': success_rate,
                    'total_evaluations': len(metric_results),
                    'successful_evaluations': sum(1 for r in metric_results if r['success']),
                    'min_score': min(scores) if scores else 0.0,
                    'max_score': max(scores) if scores else 0.0
                }
            else:
                aggregated[metric_name] = {
                    'average_score': 0.0,
                    'success_rate': 0.0,
                    'total_evaluations': 0,
                    'successful_evaluations': 0,
                    'min_score': 0.0,
                    'max_score': 0.0
                }

        return aggregated

    def generate_detailed_report(self, results: Dict[str, List[Dict]], aggregated: Dict):
        report = []
        report.append("="*60)
        report.append("DEEPEVAL EVALUATION REPORT")
        report.append("="*60)
        report.append()

        report.append("OVERALL METRICS:")
        report.append("-" * 20)
        for metric_name, stats in aggregated.items():
            report.append(f"\n{metric_name}:")
            report.append(f"  Average Score: {stats['average_score']:.4f}")
            report.append(f"  Success Rate:  {stats['success_rate']:.4f}")
            report.append(f"  Min Score:     {stats['min_score']:.4f}")
            report.append(f"  Max Score:     {stats['max_score']:.4f}")
            report.append(f"  Evaluations:   {stats['successful_evaluations']}/{stats['total_evaluations']}")

        report.append()
        report.append("PERFORMANCE BY QUESTION CATEGORY:")
        report.append("-" * 35)

        categories = {}
        for i, item in enumerate(self.test_data):
            category = item.get('category', 'unknown')
            if category not in categories:
                categories[category] = []

            category_scores = {}
            for metric_name, metric_results in results.items():
                if i < len(metric_results) and metric_results[i]['score'] is not None:
                    category_scores[metric_name] = metric_results[i]['score']

            if category_scores:
                categories[category].append(category_scores)

        for category, scores_list in categories.items():
            if scores_list:
                report.append(f"\n{category.replace('_', ' ').title()}:")
                metric_names = set()
                for scores in scores_list:
                    metric_names.update(scores.keys())

                for metric_name in sorted(metric_names):
                    metric_scores = [s[metric_name] for s in scores_list if metric_name in s]
                    if metric_scores:
                        avg_score = sum(metric_scores) / len(metric_scores)
                        report.append(f"  {metric_name}: {avg_score:.4f}")

        report.append()
        report.append("FAILED EVALUATIONS:")
        report.append("-" * 20)
        for metric_name, metric_results in results.items():
            failed_cases = [r for r in metric_results if not r['success']]
            if failed_cases:
                report.append(f"\n{metric_name} - {len(failed_cases)} failed cases:")
                for case in failed_cases[:3]:
                    report.append(f"  Question: {case['question'][:60]}...")
                    report.append(f"  Reason: {case['reason']}")
                if len(failed_cases) > 3:
                    report.append(f"  ... and {len(failed_cases) - 3} more")

        report.append()
        report.append("="*60)

        return "\n".join(report)

    def save_results(self, results: Dict[str, List[Dict]], output_dir: str = "evaluation_results"):
        os.makedirs(output_dir, exist_ok=True)

        aggregated = self.calculate_aggregate_scores(results)

        detailed_results = []
        for i, item in enumerate(self.test_data):
            row = {
                'question': item['question'],
                'expected_answer': item['expected_answer'],
                'category': item.get('category', 'unknown')
            }

            for metric_name, metric_results in results.items():
                if i < len(metric_results):
                    row[f"{metric_name}_score"] = metric_results[i]['score']
                    row[f"{metric_name}_success"] = metric_results[i]['success']
                else:
                    row[f"{metric_name}_score"] = None
                    row[f"{metric_name}_success"] = False

            detailed_results.append(row)

        df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(output_dir, "deepeval_detailed_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to: {csv_path}")

        report = self.generate_detailed_report(results, aggregated)
        report_path = os.path.join(output_dir, "deepeval_evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to: {report_path}")

        summary = {
            "framework": "DeepEval",
            "total_questions": len(self.test_data),
            "metrics_evaluated": list(results.keys()),
            "aggregated_scores": aggregated
        }

        summary_path = os.path.join(output_dir, "deepeval_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")


def run_deepeval_evaluation():
    logger.info("Initializing RAG system for DeepEval evaluation...")

    try:
        vector_store = MongoVectorStore(
            mongodb_uri=settings.mongodb_uri,
            database_name=settings.mongodb_database,
            collection_name=settings.mongodb_collection,
            embedding_model=settings.embedding_model
        )

        if not vector_store.health_check():
            logger.error("MongoDB connection failed. Please ensure MongoDB is running.")
            return

        rag_system = RAGSystem(vector_store)

        test_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_datasets.json')

        evaluator = DeepEvalEvaluator(rag_system, test_data_path)

        logger.info("Preparing test cases...")
        test_cases = evaluator.prepare_test_cases()

        logger.info("Running DeepEval evaluation...")
        results = evaluator.evaluate_with_deepeval(test_cases)

        aggregated = evaluator.calculate_aggregate_scores(results)
        print(evaluator.generate_detailed_report(results, aggregated))
        evaluator.save_results(results)

    except Exception as e:
        logger.error(f"Error during DeepEval evaluation setup: {e}")


if __name__ == "__main__":
    run_deepeval_evaluation()