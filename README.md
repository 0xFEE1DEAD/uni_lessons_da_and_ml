# Learning about data analysis and machine learning

This project requires uv package manager. For install it please visit this page https://docs.astral.sh/uv/getting-started/installation/

For create virtual environment and install dependencies use command
```
# for install cuda torch
uv sync --extra cu128
# for install cpu torch
uv sync --extra cpu
```
For run scripts use command
```
uv run <file>
```

you may need to install additional dependency spacy
```
uv run -- spacy download ru_core_news_sm
```