import sys
import time
from google import genai

client = genai.Client()  # reads GOOGLE_API_KEY from the environment

if len(sys.argv) == 1:
    print("Gemini models available to your key:")
    names = sorted(m.name.replace("models/", "") for m in client.models.list())
    for n in names:
        if "gemini" in n:
            print(" ", n)
    print(f"({len(names)} models total)")
else:
    model = sys.argv[1]
    start = time.time()
    resp = client.models.generate_content(model=model, contents="Reply with the words: hello from Gemini")
    print(f"{model} -> {resp.text.strip()}  ({time.time() - start:.1f}s)")
