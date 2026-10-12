from dotenv import load_dotenv

from app.services.openrouter_client import chat_completion

load_dotenv()

print(
    chat_completion(
        [{"role": "user", "content": "Say hello in one sentence."}],
    )
)
