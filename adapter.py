"""
HTTP adapter — exposes your env via the BenchAnything four-endpoint protocol.

Local dev:
    python adapter.py
    python adapter.py --port 9000
"""

import argparse

from bench_common.env_sdk import serve
from env import JengaBenchEnv

# Image observations require the Mesocosm env_sdk interface from:
# https://github.com/swecc-uw/swecc-core/commit/d4b81907456b17f50a878d40980b5e6aa9b74c9b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"JengaBenchEnv adapter -> http://{args.host}:{args.port}")
    serve(JengaBenchEnv, host=args.host, port=args.port)
