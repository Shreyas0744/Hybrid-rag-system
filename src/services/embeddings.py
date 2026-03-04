from google import genai
from src.config import get_settings

settings = get_settings()

client = genai.Client(api_key=settings.gemini.api_key)

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []

    for text in texts:
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )

        embeddings.append(result.embeddings[0].values)

    return embeddings
