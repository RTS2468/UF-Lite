def uf_middleware(response: str, context: dict) -> str:
    blocked = ["i'm back", "as i said earlier", "welcome back"]
    if any(phrase in response.lower() for phrase in blocked):
        return "I am a new instance with no prior continuity. This is a fresh session."

    # TODO: Run Verification Core (8 gates) here
    # if not all_gates_pass(response, context):
    #     return "Gate failure detected. Refining output..."

    return response
