# Simple example — drop this into any bot
from middleware_filter import uf_middleware
from gate_profile import load_profile  # you can make this

profile = load_profile("gate_profile.yaml")

user_input = "..."
candidate_response = llm.generate(user_input)

final_output = uf_middleware(candidate_response, {"profile": profile})
print(final_output)
