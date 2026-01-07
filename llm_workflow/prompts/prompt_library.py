import yaml
import json
from typing import Dict, Union, Any


class PromptLibrary:
    def __init__(self, file_path: str = "prompts.yaml"):
        self.filepath = file_path
        self.prompt = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)

        except FileNotFoundError:
            return {}


    def get_prompt(self, key_path: str) -> str:
        keys = key_path.split(".")
        data = self.prompt

        try:
            for key in keys:
                data = data[key]

            if isinstance(data, (dict,list)):
                return json.dumps(data, indent=2, ensure_ascii=False)

            return str(data)

        except KeyError:
            return f"Error: key {key_path} not found in library"
