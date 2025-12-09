import os
import logging
from enum import Enum
from typing import Optional, Literal

import instructor
from openai import OpenAI
from groq import Groq

logger = logging.getLogger(__name__)

class Provider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"

def get_available_providers() -> list[Provider]:
    """Check environment variables to see which providers are available."""
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append(Provider.OPENAI)
    if os.getenv("GROQ_API_KEY"):
        providers.append(Provider.GROQ)
    return providers

def create_client(provider: Provider):
    """Create an instructor-patched client for the specified provider."""
    
    if provider == Provider.OPENAI:
        logger.info("Initializing OpenAI client")
        return instructor.from_openai(OpenAI())
        
    elif provider == Provider.GROQ:
        logger.info("Initializing Groq client")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file or environment.")
        
        client = Groq(api_key=groq_api_key)
        return instructor.patch(client, mode=instructor.Mode.JSON)
    
    raise ValueError(f"Unsupported provider: {provider}")

def get_model_for_provider(provider: Provider) -> str:
    """Return the recommended model for each provider."""
    if provider == Provider.OPENAI:
        return "gpt-4o"
    elif provider == Provider.GROQ:
        return "openai/gpt-oss-120b"
    return "gpt-3.5-turbo"
