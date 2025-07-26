# Language Learning Curriculum API

A sophisticated language learning curriculum generator with REST API that uses LlamaIndex for semantic knowledge retrieval and Google Gemini 2.0 Flash for intelligent content generation.

## Features

- **REST API**: Full REST API with FastAPI for easy integration
- **Semantic Knowledge Retrieval**: Uses LlamaIndex for intelligent search through language learning content
- **Multi-language Support**: French and Spanish with extensible architecture
- **Real-world Scenarios**: Cafe Order, Hotel Check-in, Shopping
- **AI Content Generation**: Uses Google Gemini 2.0 Flash for intelligent curriculum creation
- **Structured Output**: JSON-formatted curriculum with scenarios, questions, and corrections
- **Optimized Performance**: Pre-computed knowledge contexts for fast response times

## Architecture

The curriculum agent combines:
- **LlamaIndex**: Semantic search and retrieval of contextual knowledge
- **Knowledge Base**: Pre-defined vocabulary, grammar, and interaction guidelines
- **Real-world Data**: Authentic menu items, phrases, and cultural notes
- **AI Generation**: Google Gemini 2.0 Flash for intelligent content creation
- **Fallback System**: Graceful degradation when API unavailable

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv curriculum_env

# Activate environment
source curriculum_env/bin/activate  # On macOS/Linux
# or
curriculum_env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the API

```bash
# Start the API server
python api.py
```

### 4. Access API Documentation

Visit: http://localhost:8000/docs

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Available Scenarios
```bash
curl http://localhost:8000/available-scenarios
```

### Generate Curriculum
```bash
curl -X POST http://localhost:8000/generate-curriculum \
  -H "Content-Type: application/json" \
  -d '{"target_language": "French", "scenario": "Cafe Order"}'
```

### Cache Statistics
```bash
curl http://localhost:8000/cache-stats
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and cache stats |
| `/available-scenarios` | GET | Available scenarios and languages |
| `/generate-curriculum` | POST | Generate curriculum content |
| `/cache-stats` | GET | Cache statistics |
| `/docs` | GET | Interactive API documentation |

## Output Structure

```json
{
  "scenario_scene": "You are in a charming French café...",
  "curriculum_questions": [
    {
      "question": "How would you greet someone?",
      "expected_response": "Bonjour, monsieur/madame"
    }
  ],
  "correction_examples": [
    {
      "incorrect_phrase": "I want coffee",
      "correct_phrase": "Je voudrais un café",
      "explanation": "Use polite forms and proper articles"
    }
  ]
}
```

## Supported Features

- **Languages**: French, Spanish
- **Scenarios**: Cafe Order, Hotel Check-in, Shopping
- **Performance**: Pre-computed knowledge contexts for fast response times

## LlamaIndex Benefits

- **Semantic Search**: Intelligent retrieval of relevant language learning content
- **Context Awareness**: Understands relationships between vocabulary, grammar, and cultural context
- **Scalable Knowledge**: Easy to add new scenarios, languages, and proficiency levels
- **Accurate Retrieval**: Finds the most relevant information for each specific query

## Files

- `curriculum_agent.py` - Main curriculum agent with LlamaIndex implementation
- `api.py` - REST API server using FastAPI
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Knowledge Base Structure

The LlamaIndex implementation includes:
- **Vocabulary Focus**: Level-appropriate vocabulary for each scenario
- **Grammar Focus**: Progressive grammar structures from A1 to B1
- **Interaction Patterns**: Real-world conversation patterns
- **Cultural Context**: Authentic cultural notes and etiquette
- **Teaching Guidelines**: Specific instructions for each scenario-level combination

## License

This project is open source and available under the MIT License. 