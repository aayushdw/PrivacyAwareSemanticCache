"""
LLM Provider for Gemini API integration.

This module provides a simple interface to interact with Google's Gemini API
for generating text responses. The API key is loaded from the .env file.
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai


class GeminiProvider:
    """
    A provider class for interacting with Google's Gemini API.

    Attributes:
        api_key (str): The Gemini API key loaded from environment variables.
        model_name (str): The name of the Gemini model to use.
        model: The configured Gemini model instance.
        mock_mode (bool): If True, returns mock responses instead of calling the API.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the Gemini provider.

        Args:
            model_name (str): The Gemini model to use. Defaults to "gemini-2.5-flash".
            api_key (str, optional): API key for Gemini. If not provided, will load
                                    from GEMINI_API_KEY environment variable.
            mock_mode (bool): If True, returns "LLM called" instead of making API requests.
                            Useful for testing without consuming API quota. Defaults to False.

        Raises:
            ValueError: If API key is not found in environment variables or provided
                       (only when mock_mode is False).
        """
        # Load environment variables from .env file
        load_dotenv()

        # Store mock mode setting
        self.mock_mode = mock_mode

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        # Only require API key if not in mock mode
        if not mock_mode:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found. Please set it in your .env file "
                    "or pass it as a parameter."
                )

            # Configure the Gemini API
            genai.configure(api_key=self.api_key)

            # Store model name and initialize model
            self.model_name = model_name
            self.model = genai.GenerativeModel(model_name)

            print(f"✓ GeminiProvider initialized with model: {model_name}")
        else:
            self.model_name = model_name
            self.model = None
            print(f"✓ GeminiProvider initialized in MOCK MODE (model: {model_name})")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        top_k: int = 40,
        **kwargs
    ) -> str:
        """
        Generate text response from Gemini model.

        Args:
            prompt (str): The input prompt/query to send to the model.
            temperature (float): Controls randomness (0.0-1.0). Higher = more random.
                               Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate.
            top_p (float): Nucleus sampling threshold. Defaults to 0.95.
            top_k (int): Top-k sampling parameter. Defaults to 40.
            **kwargs: Additional parameters to pass to the generation config.

        Returns:
            str: The generated text response from Gemini (or "LLM called" if in mock mode).

        Raises:
            Exception: If the API call fails.
        """
        # Return mock response if in mock mode
        if self.mock_mode:
            return "LLM called"

        try:
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                **kwargs
            }

            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            raise Exception(f"Error generating response from Gemini: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        return {
            "model_name": self.model_name,
            "api_configured": bool(self.api_key),
            "mock_mode": self.mock_mode
        }


# Example usage
if __name__ == "__main__":
    # Example 1: Mock Mode (for testing without consuming API quota)
    print("\n=== Example 1: Mock Mode ===")
    mock_provider = GeminiProvider(mock_mode=True)
    prompt = "What is semantic caching in the context of LLMs?"
    response = mock_provider.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Model info: {mock_provider.get_model_info()}\n")

    "=== Example 2: Real API Mode (commented out to save quota) ==="
    "# Uncomment the following lines to make real API calls:"
    "# provider = GeminiProvider()"
    "# response = provider.generate('What is semantic caching?')"
    "# print(f'Response: {response}')\n"

    # Example 3: Multiple mock calls
    print("=== Example 3: Multiple Mock Calls ===")
    queries = [
        "How does federated learning work?",
        "What is differential privacy?",
        "Explain semantic similarity."
    ]
    for query in queries:
        response = mock_provider.generate(query)
        print(f"Query: {query}")
        print(f"Response: {response}\n")
