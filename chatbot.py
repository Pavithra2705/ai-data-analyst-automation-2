import os
import requests
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Automatically load API keys from .env file
load_dotenv()

class LLaMAChat:
    """
    Groq-based data analyst chatbot.
    Automatically loads GROQ_API_KEY from .env file or environment variables.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", groq_api_key: Optional[str] = None):
        self.model_name = model_name
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found. Please create a .env file in your project folder and add:\n"
                "GROQ_API_KEY=your_actual_key_here"
            )

    def _build_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = ""
        if context:
            shape = context.get("shape")
            columns = context.get("columns")
            missing = context.get("missing_values")
            summary = context.get("summary")
            sample = context.get("sample_data")

            ctx_lines = []
            if shape:
                ctx_lines.append(f"- Shape: {shape}")
            if columns:
                ctx_lines.append(f"- Columns: {columns}")
            if missing:
                ctx_lines.append(f"- Missing values: {missing}")
            if summary:
                ctx_lines.append(f"- Summary: {list(summary)[:5]}")
            if sample:
                ctx_lines.append(f"- Sample rows: {sample}")

            ctx = "\n".join(ctx_lines)

        prompt = (
            "You are an expert data analyst. "
            "Provide clear insights, explain reasoning, and use examples where possible.\n\n"
        )
        if ctx:
            prompt += f"Dataset context:\n{ctx}\n\n"
        prompt += f"User question:\n{query}\n\nAnswer:"
        return prompt

    def _call_groq(self, prompt: str, timeout: int = 120) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an AI assistant that helps data analysts."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        try:
            response = requests.post(self.groq_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return "⚠️ No response from Groq API."
        except requests.exceptions.RequestException as e:
            return f"❌ API Request Failed: {e}"

    def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        prompt = self._build_prompt(query, context)
        return self._call_groq(prompt)
