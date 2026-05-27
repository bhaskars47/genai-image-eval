#!/usr/bin/env python3
# list_models.py
# Prints all models available for your Gemini API key.
# Run: python3 list_models.py --api-key YOUR_KEY
#
# Look for models that support "generateContent" AND have image output capability.

import argparse
from google import genai

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)
args = parser.parse_args()

client = genai.Client(api_key=args.api_key)

print("\nModels available for your API key:\n")
print(f"{'Model name':<55} {'Supported methods'}")
print("-" * 90)

for model in client.models.list():
    methods = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", [])
    print(f"{model.name:<55} {methods}")

print()
