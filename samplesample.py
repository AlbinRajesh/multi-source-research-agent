import os
import requests

owner = "assafelovic"
repo = "gpt-researcher"
branch = "main"
target_folder = "gpt_researcher_reference" # New isolated folder

# Exact relative file paths in assafelovic/gpt-researcher
files_to_download = [
    "gpt_researcher/utils/costs.py",
    "gpt_researcher/utils/llm.py",
    "gpt_researcher/retrievers/searx/searx.py",
    "gpt_researcher/context/compression.py"
]

for file_path in files_to_download:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    print(f"Downloading from GPT-Researcher: {file_path}...")
    
    response = requests.get(raw_url)
    if response.status_code == 200:
        # Prepend the target folder so it stays completely separate
        local_path = os.path.join(target_folder, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f" -> Saved safely to {local_path}")
    else:
        print(f" -> Failed to download (Status: {response.status_code})")

print("\nAll GPT-Researcher files downloaded into the new folder!")