# Easy Webpage Summarizer

A Python script designed to summarize webpages from specified URLs using the LangChain framework and the ChatOllama model. It leverages advanced language models to generate detailed summaries, making it an invaluable tool for quickly understanding the content of web-based documents.

## Requirements

[ollama](https://ollama.com/) must be installed and served

```bash
ollama run llama3:instruct
```

```bash
pip install -r requirements.txt
```

## Features

- Summarization of webpages and youtube videos directly from URLs.
- Translates to Turkish language (other languages will be added soon!)
- Integration with LangChain and ChatOllama for state-of-the-art summarization.
- Command-line interface for easy use and integration into workflows.
- Structured output with summary and tags

## Usage

To use the webpage summarizer, run the script from the command line, providing the URL of the document you wish to summarize:

```bash
python app/summarizer.py -u "http://example.com/document"
```

Replace `http://example.com/document` with the actual URL of the document you want to summarize.

The output will include:
- Title of the document
- A structured summary of 120-220 Chinese characters
- Tags representing key topics in the document

### Web UI

To use the webpage summarizer in you web browser, you can also try gradio app.

```bash
python app/webui.py
```

![gradio](assets/gradio.png)

## Docker

```bash
docker build -t web-summarizer:local .
docker compose up -d
```

## API Server

The project also includes a FastAPI server for serving summaries via HTTP:

```bash
python app/web_summarizer_api.py
```

To use the API:

```bash
curl -X POST "http://127.0.0.1:8001/summarize" -H "Content-Type: application/json" -d '{"url": "http://example.com/document"}'
```

## Development

To contribute to the development of this script, clone the repository, make your changes, and submit a pull request. We welcome contributions that improve the script's functionality or extend its capabilities.

- [x] Summarize youtube videos
- [x] Dockerize project
- [ ] Translate to different languages
- [ ] Streaming text output on gradio
- [ ] Serve on web

## License

This script is released under the MIT License. See the [LICENSE](./LICENSE) file in the repository for full details.