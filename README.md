# RAG Testing Project: Comprehensive Guide with Ragas and DeepEval

This project demonstrates how to build and evaluate a Retrieval-Augmented Generation (RAG) application using both **Ragas** and **DeepEval** evaluation frameworks. The system uses MongoDB for vector storage, Gemini AI for language generation, and comprehensive testing methodologies for RAG evaluation.

## 🏗️ Project Architecture

```
rag-testing-project/
├── src/                     # Core application code
│   ├── config.py           # Configuration management
│   ├── document_processor.py # Document processing and embedding
│   ├── vector_store.py     # MongoDB vector storage
│   └── rag_system.py       # Main RAG implementation
├── tests/                  # Evaluation scripts
│   ├── test_ragas.py       # Ragas evaluation implementation
│   └── test_deepeval.py    # DeepEval evaluation implementation
├── data/                   # Sample data and test datasets
│   ├── documents/          # Sample markdown documents
│   └── test_datasets.json  # Evaluation test cases
├── scripts/                # Utility scripts
│   ├── setup_data.py       # Database setup script
│   └── run_evaluations.py  # Evaluation runner
├── config/                 # Configuration files
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **MongoDB** running locally or remotely
3. **Gemini AI API key** from Google AI Studio

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd rag-testing-project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

5. **Set up the database:**
   ```bash
   python scripts/setup_data.py
   ```

6. **Run evaluations:**
   ```bash
   python scripts/run_evaluations.py --framework both
   ```

## 📋 Detailed Setup Guide

### 1. Environment Configuration

Create a `.env` file with the following variables:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (with defaults)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=rag_testing
MONGODB_COLLECTION=documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
GEMINI_MODEL=gemini-1.5-pro
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7
```

### 2. MongoDB Setup

**Option 1: Local MongoDB**
```bash
# Install MongoDB Community Edition
# Start MongoDB service
mongod --dbpath /your/data/path
```

**Option 2: MongoDB Atlas (Cloud)**
```bash
# Update MONGODB_URI in .env to your Atlas connection string
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

**Option 3: Docker**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 3. Gemini AI API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🔧 Core Components

### RAG System Architecture

The system consists of several key components:

**Document Processor (`src/document_processor.py`)**
- Loads and processes markdown documents
- Splits documents into chunks
- Generates embeddings using SentenceTransformers

**Vector Store (`src/vector_store.py`)**
- MongoDB-based vector storage
- Cosine similarity search
- Document indexing and retrieval

**RAG System (`src/rag_system.py`)**
- Integrates retrieval and generation
- Uses Gemini AI for response generation
- Handles query processing pipeline

### Sample Documents

The project includes comprehensive markdown documents covering:
- **Machine Learning Basics** - Fundamental ML concepts
- **Deep Learning Guide** - Neural networks and architectures
- **Data Science Workflow** - End-to-end data science process
- **Python Programming** - Python language fundamentals

## 📊 Evaluation Frameworks

### Ragas Evaluation

**Metrics Evaluated:**
- **Answer Relevancy** - How relevant the answer is to the question
- **Answer Similarity** - Semantic similarity to expected answer
- **Answer Correctness** - Factual accuracy of the response
- **Faithfulness** - How well the answer is grounded in retrieved context
- **Context Precision** - Precision of retrieved context
- **Context Recall** - Recall of relevant context
- **Context Relevancy** - Relevance of retrieved context

**Running Ragas:**
```bash
python tests/test_ragas.py
```

### DeepEval Evaluation

**Metrics Evaluated:**
- **Answer Relevancy** - Relevance of generated answers
- **Faithfulness** - Factual consistency with context
- **Contextual Precision** - Precision of context retrieval
- **Contextual Recall** - Completeness of context retrieval
- **Contextual Relevancy** - Quality of retrieved context
- **Hallucination** - Detection of fabricated information
- **Toxicity** - Content safety evaluation
- **Bias** - Bias detection in responses

**Running DeepEval:**
```bash
python tests/test_deepeval.py
```

## 🎯 Test Dataset

The evaluation uses a comprehensive test dataset (`data/test_datasets.json`) with 15 questions covering:

**Question Categories:**
- **Definition** - Basic concept explanations
- **Classification** - Categorization questions
- **Comparison** - Comparative analysis
- **Technical Details** - Specific technical information
- **Process** - Step-by-step procedures
- **Evaluation** - Assessment methodologies
- **Architecture Comparison** - System architecture differences
- **Data Preprocessing** - Data preparation techniques
- **Model Problems** - Common ML issues
- **Programming Concepts** - Programming fundamentals
- **Advanced Techniques** - Complex methodologies
- **Evaluation Metrics** - Assessment measures
- **Data Analysis** - Analysis techniques
- **Programming Syntax** - Code structure
- **Training Challenges** - ML training difficulties

## 📈 Understanding Results

### Ragas Output

The Ragas evaluation produces:
- **Overall metrics** with scores from 0-1 (higher is better)
- **Performance by category** showing domain-specific performance
- **Detailed CSV** with per-question results
- **Summary report** with key insights

**Key Metrics to Monitor:**
- `answer_relevancy > 0.7` - Good relevance
- `faithfulness > 0.8` - High factual accuracy
- `context_precision > 0.6` - Good retrieval precision

### DeepEval Output

The DeepEval evaluation provides:
- **Success rates** for each metric
- **Average scores** across all questions
- **Failed evaluations** with detailed reasons
- **Category-wise performance** analysis

**Key Metrics to Monitor:**
- `AnswerRelevancyMetric` success rate > 80%
- `FaithfulnessMetric` score > 0.7
- `HallucinationMetric` score < 0.3 (lower is better)

## 🛠️ Customization

### Adding New Documents

1. Place markdown files in `data/documents/`
2. Run setup script to reprocess:
   ```bash
   python scripts/setup_data.py
   ```

### Adding New Test Cases

Edit `data/test_datasets.json`:
```json
{
  "question": "Your question here",
  "expected_answer": "Expected response",
  "contexts": ["relevant_document.md"],
  "category": "your_category"
}
```

### Modifying Evaluation Metrics

**Ragas Customization:**
```python
# In tests/test_ragas.py
metrics = [
    answer_relevancy,
    faithfulness,
    # Add or remove metrics as needed
]
```

**DeepEval Customization:**
```python
# In tests/test_deepeval.py
metrics = [
    AnswerRelevancyMetric(threshold=0.8),  # Adjust thresholds
    FaithfulnessMetric(threshold=0.7),
    # Add custom metrics
]
```

### Configuration Options

Modify `src/config.py` or environment variables:
- `CHUNK_SIZE` - Document chunk size
- `CHUNK_OVERLAP` - Overlap between chunks
- `TOP_K_RETRIEVAL` - Number of documents to retrieve
- `SIMILARITY_THRESHOLD` - Minimum similarity for retrieval

## 🔍 Troubleshooting

### Common Issues

**1. MongoDB Connection Failed**
```bash
# Check if MongoDB is running
mongo --eval "db.adminCommand('ismaster')"

# Or check the service status
sudo systemctl status mongod
```

**2. Gemini API Key Issues**
- Verify API key is correctly set in `.env`
- Check API key permissions in Google AI Studio
- Ensure you have sufficient quota

**3. Memory Issues with Large Documents**
```python
# Reduce chunk size in .env
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

**4. Slow Evaluation Performance**
```python
# Reduce test dataset size for testing
# Edit data/test_datasets.json and remove some entries
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Optimization

### Database Optimization

1. **Create proper indexes:**
   ```python
   # Indexes are automatically created in MongoVectorStore
   collection.create_index([("doc_id", 1)])
   collection.create_index([("chunk_id", 1)], unique=True)
   ```

2. **Use appropriate chunk sizes:**
   - Smaller chunks (500-800 tokens): Better precision
   - Larger chunks (1000-1500 tokens): Better context

### Embedding Optimization

1. **Choose appropriate embedding models:**
   ```python
   # Fast but less accurate
   EMBEDDING_MODEL=all-MiniLM-L6-v2

   # Slower but more accurate
   EMBEDDING_MODEL=all-mpnet-base-v2
   ```

2. **Batch processing for large datasets:**
   ```python
   # Process documents in batches
   batch_size = 100
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       process_batch(batch)
   ```

## 🧪 Advanced Testing

### Custom Evaluation Metrics

Create custom evaluation functions:

```python
def custom_coherence_metric(question, answer, context):
    # Implement your custom metric
    score = calculate_coherence(answer)
    return score

# Add to evaluation pipeline
custom_results = []
for test_case in test_cases:
    score = custom_coherence_metric(
        test_case.question,
        test_case.answer,
        test_case.context
    )
    custom_results.append(score)
```

### A/B Testing Different Configurations

```python
configs = [
    {"chunk_size": 500, "top_k": 3},
    {"chunk_size": 1000, "top_k": 5},
    {"chunk_size": 1500, "top_k": 7}
]

for config in configs:
    # Run evaluation with each config
    results = run_evaluation(config)
    save_results(f"results_{config['chunk_size']}.json", results)
```

## 📚 Additional Resources

### Documentation
- [Ragas Documentation](https://docs.ragas.io/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Gemini AI Documentation](https://ai.google.dev/docs)

### Research Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Include tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with:
   - Error message
   - Configuration details
   - Steps to reproduce

---

**Happy RAG Testing! 🚀**