from app.services.openrouter_client import OpenRouterError, chat_completion


def main() -> None:
    try:
        reply = chat_completion(
            [{"role": "user", "content": "Say hello in one sentence."}],
            model="google/gemma-4-26b-a4b-it:free",
        )
    except OpenRouterError as exc:
        raise SystemExit(str(exc)) from exc

    print(reply)


if __name__ == "__main__":
    main()
