Reading the project's search provider file to include accurate provider details (name, usage, and configuration) in the README. I'll open the backend provider file mentioned.

Read 
Completed with input: {
  "path": "d:\\ngxp_internship\\multi_source_researcher\\backend\\src\\search_providers\\tavily_provider.py"
}

# multi-source-researcher

A research assistant that collects and aggregates search results from multiple providers to support multi-source research workflows (web, knowledge bases, and vector stores). This repository contains the backend search provider implementations and shared types used by the agent.

This README gives an overview of the project, setup and development instructions, configuration notes (including the Tavily provider implementation), and contribution guidance so it can be pasted into GitHub.

---

## Table of contents

- Project overview
- Key features
- Architecture & important files
- Requirements
- Quick start (development)
- Configuration
- Example: using the Tavily provider
- Running tests
- Contributing
- Troubleshooting
- License

---

## Project overview

multi-source-researcher is an extensible backend for gathering research results from multiple search sources (web search APIs, internal knowledge bases, vector search, etc.), normalizing them into a common SearchResult model, and returning them for further processing (deduplication, credibility scoring, citation generation, UI consumption).

The code is organized so that additional search providers can be added by implementing a common provider interface.

---

## Key features

- Provider abstraction layer for pluggable search providers
- Normalized SearchResult model for consistent downstream handling
- Example provider that integrates with the Tavily SDK (see below)
- Async-friendly search methods
- Clear error wrapping and provider availability checks

---

## Architecture & important files

- backend/src/search_providers/ — search provider implementations
  - `tavily_provider.py` — Tavily provider implementation (uses the official tavily-python SDK)
- backend/src/search_providers/base.py — base provider interface and exceptions (provider must implement `search`)
- backend/src/state.py — shared SearchResult model
- backend/src/config.py — centralized configuration object used by providers (expects API keys and other settings)

(Adjust paths above if your workspace layout differs; file links point to the local development tree.)

---

## Requirements

- Python 3.8+ (recommend 3.10+)
- Install project dependencies (see quick start below)
- An API key for any external search providers you enable (for example Tavily)

---

## Quick start (development)

1. Clone the repository (if not already done):
   git clone <your-repo-url>

2. Create and activate a virtual environment:
   python -m venv .venv
   - Windows (PowerShell):
     .venv\Scripts\Activate.ps1
   - Windows (cmd):
     .venv\Scripts\activate.bat
   - macOS / Linux:
     source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

   If the repository does not provide a requirements file, install the SDKs you need. For the Tavily provider:
   pip install tavily

4. Configure credentials (see Configuration section).

5. Start the backend or run the part of the system you need for development (project may supply e.g., an `app.py` or `run` script). If there is a run script, use:
   python backend/app.py
   (Replace above with the actual entrypoint in this repository.)

---

## Configuration

Providers rely on a centralized configuration object imported as `config` (example: `from src.config import config`). For the Tavily provider the code expects `config.tavily_api_key` to be set.

Two common ways to provide configuration:

- Environment variables (recommended for secrets). Example:
  - On Windows (PowerShell):
    $env:TAVILY_API_KEY = "sk-..."
  - On Linux/macOS:
    export TAVILY_API_KEY="sk-..."

- `.env` file loaded at startup (if the project uses python-dotenv). Example `.env`:
  TAVILY_API_KEY=sk-...

Make sure the configuration code maps the environment variable into `config.tavily_api_key`. If there is a `backend/src/config.py`, adapt it to read env vars or document how to set `config` fields.

---

## Example: the Tavily provider

The repository includes an async provider implementation for Tavily in:

`tavily_provider.py`

Highlights from the implementation:

- Class name: `TavilySearchProvider`
- Expects `config.tavily_api_key` to be set. If not set it raises `ProviderUnavailableError`.
- Uses the official Tavily SDK (`TavilyClient`) to perform searches.
- Provides an async `search(query: str, max_results: Optional[int]) -> List[SearchResult]` method that:
  - Calls the synchronous client via `asyncio.to_thread(...)`
  - Skips results that do not include a URL (to avoid broken links in the frontend)
  - Wraps unexpected errors in a `SearchError` with details

Example usage pattern (conceptual — adapt to your application wiring):

- Ensure `config.tavily_api_key` is set (from environment or config file)
- Instantiate the provider:
  from src.search_providers.tavily_provider import TavilySearchProvider
  provider = TavilySearchProvider()
- Call provider.search(...) from an async context:
  results = await provider.search("example query", max_results=5)

If you'd like, a small helper CLI or script can be written to demo the provider by loading the config, creating the provider, and printing results.

---

## Running tests

If the repository contains tests, run them with the project's test runner. Example with pytest:
pip install -r dev-requirements.txt  # if there's a dev requirements file
pytest -q

If no test suite is present yet, consider adding unit tests that mock external providers (e.g., mock TavilyClient) to validate `search()` behavior and error handling.

---

## Contributing

Contributions are welcome. Suggested guidelines:

- Open an issue for new features or bug reports
- Create small, focused pull requests
- Add tests for new features and bug fixes
- Keep provider implementations isolated behind the `SearchProvider` interface
- Document any new configuration variables in README or a dedicated docs file

When adding a new search provider:
1. Implement the provider class in `backend/src/search_providers/`
2. Raise `ProviderUnavailableError` if required credentials are missing
3. Return `List[SearchResult]` objects with `url`, `title`, `snippet`, and `source_type`

---

## Troubleshooting

- "Provider not configured" errors:
  - Confirm the relevant API key is present in `config` (for Tavily: `config.tavily_api_key`)
  - Confirm you set the correct environment variable and that config reads it

- Missing/blank URLs in results:
  - The Tavily provider intentionally skips results with no URL to avoid broken frontend links. If you want to handle such cases differently, modify `tavily_provider.py` accordingly.

- SDK or dependency errors:
  - Ensure the correct SDK (e.g., `tavily`) is installed in the active virtual environment.

---

## Next steps / ideas

- Add end-to-end example (script or small HTTP API) that accepts a query and returns normalized results from multiple providers for easier demoing and integration tests.
- Add caching and rate-limit handling for providers.
- Add credibility scoring and citation generation modules downstream of collected SearchResult items.

---

## License

Add a license file appropriate to your project. If unsure, consider MIT for permissive open-source usage. Example: include a `LICENSE` file with MIT text and update this README.

---

## Contact

If you need a tailored README update (e.g., to include exact install commands, project entrypoint, or examples of other providers in this repo), tell me which runner/entrypoint and any additional files to reference and I will update this README accordingly.
