import json
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.rag_system import RAGSystem
from src.vector_store import MongoVectorStore
from src.document_processor import DocumentProcessor
from src.config import settings

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy
)
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagasEvaluator:
    def __init__(self, rag_system: RAGSystem, test_data_path: str):
        self.rag_system = rag_system
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()

    def _load_test_data(self):
        with open(self.test_data_path, 'r') as f:
            return json.load(f)

    def prepare_evaluation_dataset(self):
        questions = []
        ground_truths = []
        answers = []
        contexts = []

        logger.info(f"Processing {len(self.test_data)} test questions...")

        for item in self.test_data:
            question = item['question']
            expected_answer = item['expected_answer']

            logger.info(f"Processing question: {question[:50]}...")

            try:
                result = self.rag_system.query(question)
                answer = result['answer']
                retrieved_docs = result['retrieved_documents']

                context = [doc['content'] for doc in retrieved_docs] if retrieved_docs else [""]

                questions.append(question)
                ground_truths.append(expected_answer)
                answers.append(answer)
                contexts.append(context)

            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                questions.append(question)
                ground_truths.append(expected_answer)
                answers.append("Error generating answer")
                contexts.append([""])

        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        })

        return dataset

    def evaluate_with_ragas(self, dataset: Dataset = None):
        if dataset is None:
            dataset = self.prepare_evaluation_dataset()

        logger.info("Starting RAGAS evaluation...")

        metrics = [
            answer_relevancy,
            answer_similarity,
            answer_correctness,
            faithfulness,
            context_precision,
            context_recall,
            context_relevancy
        ]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics
            )

            logger.info("RAGAS evaluation completed successfully!")
            return result

        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            return None

    def generate_detailed_report(self, result):
        if result is None:
            return "Evaluation failed - no results to report"

        report = []
        report.append("="*60)
        report.append("RAGAS EVALUATION REPORT")
        report.append("="*60)
        report.append()

        report.append("OVERALL METRICS:")
        report.append("-" * 20)
        for metric, score in result.items():
            if isinstance(score, (int, float)):
                report.append(f"{metric.replace('_', ' ').title()}: {score:.4f}")
        report.append()

        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()

            report.append("DETAILED ANALYSIS:")
            report.append("-" * 20)

            for metric in ['answer_relevancy', 'answer_similarity', 'answer_correctness',
                          'faithfulness', 'context_precision', 'context_recall', 'context_relevancy']:
                if metric in df.columns:
                    scores = df[metric].dropna()
                    if len(scores) > 0:
                        report.append(f"\n{metric.replace('_', ' ').title()}:")
                        report.append(f"  Mean: {scores.mean():.4f}")
                        report.append(f"  Std:  {scores.std():.4f}")
                        report.append(f"  Min:  {scores.min():.4f}")
                        report.append(f"  Max:  {scores.max():.4f}")

            report.append()
            report.append("PERFORMANCE BY QUESTION CATEGORY:")
            report.append("-" * 35)

            categories = {}
            for i, item in enumerate(self.test_data):
                if i < len(df):
                    category = item.get('category', 'unknown')
                    if category not in categories:
                        categories[category] = []

                    row_data = {}
                    for metric in ['answer_relevancy', 'answer_similarity', 'answer_correctness']:
                        if metric in df.columns and i < len(df[metric]):
                            score = df[metric].iloc[i]
                            if pd.notna(score):
                                row_data[metric] = score

                    if row_data:
                        categories[category].append(row_data)

            for category, scores_list in categories.items():
                if scores_list:
                    report.append(f"\n{category.replace('_', ' ').title()}:")
                    for metric in ['answer_relevancy', 'answer_similarity', 'answer_correctness']:
                        metric_scores = [s[metric] for s in scores_list if metric in s]
                        if metric_scores:
                            avg_score = sum(metric_scores) / len(metric_scores)
                            report.append(f"  {metric.replace('_', ' ').title()}: {avg_score:.4f}")

        report.append()
        report.append("="*60)

        return "\n".join(report)

    def save_results(self, result, output_dir: str = "evaluation_results"):
        os.makedirs(output_dir, exist_ok=True)

        if result is not None and hasattr(result, 'to_pandas'):
            df = result.to_pandas()

            csv_path = os.path.join(output_dir, "ragas_detailed_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Detailed results saved to: {csv_path}")

        report = self.generate_detailed_report(result)
        report_path = os.path.join(output_dir, "ragas_evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to: {report_path}")

        summary = {
            "framework": "RAGAS",
            "total_questions": len(self.test_data),
            "metrics_evaluated": [
                "answer_relevancy", "answer_similarity", "answer_correctness",
                "faithfulness", "context_precision", "context_recall", "context_relevancy"
            ]
        }

        if result is not None:
            for metric, score in result.items():
                if isinstance(score, (int, float)):
                    summary[metric] = float(score)

        summary_path = os.path.join(output_dir, "ragas_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")


def run_ragas_evaluation():
    logger.info("Initializing RAG system for RAGAS evaluation...")

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

        evaluator = RagasEvaluator(rag_system, test_data_path)

        logger.info("Preparing evaluation dataset...")
        dataset = evaluator.prepare_evaluation_dataset()

        logger.info("Running RAGAS evaluation...")
        result = evaluator.evaluate_with_ragas(dataset)

        if result is not None:
            print(evaluator.generate_detailed_report(result))
            evaluator.save_results(result)
        else:
            logger.error("RAGAS evaluation failed")

    except Exception as e:
        logger.error(f"Error during RAGAS evaluation setup: {e}")


if __name__ == "__main__":
    run_ragas_evaluation()