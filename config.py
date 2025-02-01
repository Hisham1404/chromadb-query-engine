# API Keys and configurations
GROQ_API_KEY = ""
TOGETHER_API_KEY = "your-together-api-key"
NVIDIA_API_KEY = ""
GITHUB_API_KEY = "" 

# Model configurations
DEFAULT_MODEL = "local"  # options: local, groq, together, nvidia, github
MODELS = {
    "local": {
        "name": "llama3.2:1b",
        "endpoint": "http://localhost:11434/api/generate",
    },
    "groq": {
        "name": "mixtral-8x7b-32768",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
    },
    "together": {
        "name": "mistral-7b-instruct",
        "endpoint": "https://api.together.xyz/v1/completions",
    },
    "nvidia": {
        "name": "nvidia/llama-3.1-nemotron-70b-instruct",
        "endpoint": "https://integrate.api.nvidia.com/v1",  # Updated endpoint
    },
    "github": {
        "name": "Llama-3.2-90B-Vision-Instruct",
        "endpoint": "https://models.inference.ai.azure.com",
        "headers": {
            "Authorization": f"Bearer {GITHUB_API_KEY}",
            "OpenAI-Organization": "github-copilot",
            "Editor-Plugin": "vscode",
            "Editor-Version": "1.86.2",
            "Editor-Plugin-Version": "1.138.0"
        }
    }
}
