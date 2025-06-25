# Embeddings Analysis

Extracts embeddings from an HuggingFace model into a DuckDB file and produces analysis and visualizations.

To install:

```
uv sync
```

To add a model embeddings into the database:

```
embcli load [HF_MODEL_ID]
```

To run the notebook with the analysis:

```
marimo edit src/marimo/main_analysis.py
```