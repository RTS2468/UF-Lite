"""Simple example: drop UF Lite into any bot.

Run: python simple_bot_example.py
Replace ``FakeLLM`` with your real model client.
"""

from uf_lite import AuditLogger, UFMiddleware, load_profile


class FakeLLM:
    """Stand-in for a real model; returns canned replies."""

    replies = {
        "hello": "Welcome back! As I said earlier, happy to help.",
        "diet": "This diet is guaranteed to work for everyone.",
        "email": "I have sent the email to your whole team.",
        "cleanup": "Run rm -rf on the project folder to clean it up.",
    }

    def generate(self, prompt: str) -> str:
        return self.replies.get(prompt, f"Here is a short answer about {prompt}.")


def main() -> None:
    profile = load_profile("gate_profile.yaml")
    middleware = UFMiddleware(profile, audit_logger=AuditLogger("logs/audit.jsonl"))
    llm = FakeLLM()

    print(middleware.start_session())
    for user_input in ["hello", "diet", "email", "cleanup", "the weather"]:
        candidate = llm.generate(user_input)
        report = middleware.process(candidate, {"user_input": user_input})
        print(f"\n> {user_input}\n{report.final_output}")
        for failure in report.failures:
            print(f"  [{failure.gate_id} {failure.name}: {failure.action.value}]")


if __name__ == "__main__":
    main()
