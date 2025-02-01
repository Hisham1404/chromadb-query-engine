import requests
import json
from config import GROQ_API_KEY, TOGETHER_API_KEY, NVIDIA_API_KEY, MODELS

class ModelAPI:
    @staticmethod
    def query_local(prompt: str, model: str = "llama3.2:1b") -> str:
        try:
            print(f"Sending request to Ollama with model: {model}")
            response = requests.post(
                MODELS["local"]["endpoint"],
                json={"model": model, "prompt": prompt, "stream": False}
            )
            
            if response.status_code != 200:
                print(f"Ollama API returned status code: {response.status_code}")
                return f"Error: Ollama API returned status {response.status_code}"
                
            json_response = response.json()
            if not json_response.get("response"):
                print("Ollama API returned no response field")
                return "Error: No response from Ollama"
                
            return json_response.get("response", "")
            
        except Exception as e:
            print(f"Local API error details: {str(e)}")
            return f"Local API error: {str(e)}"

    @staticmethod
    def query_groq(prompt: str) -> str:
        try:
            response = requests.post(
                MODELS["groq"]["endpoint"],
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": MODELS["groq"]["name"],
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Groq API error: {str(e)}"

    @staticmethod
    def query_together(prompt: str) -> str:
        try:
            response = requests.post(
                MODELS["together"]["endpoint"],
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
                json={
                    "model": MODELS["together"]["name"],
                    "prompt": prompt,
                    "max_tokens": 512
                }
            )
            return response.json()["choices"][0]["text"]
        except Exception as e:
            return f"Together API error: {str(e)}"

    @staticmethod
    def query_nvidia(prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
                "model": MODELS["nvidia"]["name"]
            }
            
            response = requests.post(
                f"{MODELS['nvidia']['endpoint']}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"NVIDIA API Error Status: {response.status_code}")
                print(f"Response: {response.text}")
                return f"NVIDIA API Error: {response.text}"
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"NVIDIA API error details: {str(e)}")
            print(f"Full response: {response.text if 'response' in locals() else 'No response'}")
            return f"NVIDIA API error: {str(e)}"

    @staticmethod
    def query_github(prompt: str) -> str:
        try:
            response = requests.post(
                MODELS["github"]["endpoint"],
                headers=MODELS["github"]["headers"],
                json={
                    "model": MODELS["github"]["name"],
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"GitHub API error: {str(e)}"

def get_model_response(prompt: str, provider: str = "local") -> str:
    model_apis = {
        "local": ModelAPI.query_local,
        "groq": ModelAPI.query_groq,
        "together": ModelAPI.query_together,
        "nvidia": ModelAPI.query_nvidia,
        "github": ModelAPI.query_github
    }
    
    if provider not in model_apis:
        return f"Error: Unknown provider '{provider}'"
    
    return model_apis[provider](prompt)
