# Kriya Bot

## Overview
Event details query assistant powered by RAG (Retrieval-Augmented Generation)

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Environment
- Create `.env` file
- Add `GEMINI_API_KEY=your_api_key`

## Usage

### Workflow
1. Initialize database
```python
# First, call store route
/store  # Loads PDF and creates vector store

# Then query events
/query  # Submit event-related questions
```

### Query Example
```json
{
    "query": "When is the next event?"
}
```

