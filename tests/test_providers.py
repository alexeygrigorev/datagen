import os
from unittest.mock import patch
from datagen.providers import Provider, get_available_providers, get_model_for_provider

def test_get_available_providers_none():
    with patch.dict(os.environ, {}, clear=True):
        assert get_available_providers() == []

def test_get_available_providers_openai():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        assert get_available_providers() == [Provider.OPENAI]

def test_get_available_providers_groq():
    with patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test"}, clear=True):
        assert get_available_providers() == [Provider.GROQ]

def test_get_available_providers_multiple():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "GROQ_API_KEY": "gsk-test"}, clear=True):
        providers = get_available_providers()
        assert Provider.OPENAI in providers
        assert Provider.GROQ in providers

def test_get_model_for_provider():
    assert get_model_for_provider(Provider.OPENAI) == "gpt-4o"
    assert get_model_for_provider(Provider.GROQ) == "llama-3.3-70b-versatile"
