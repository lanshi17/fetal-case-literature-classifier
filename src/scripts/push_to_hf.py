#!/usr/bin/env python3
"""Upload the case-only DoRA bundle to the Hugging Face Hub.

Prerequisites
-------------
    hf auth login            # or: export HF_TOKEN=hf_...
    # uploads go to https://huggingface.co (not the hf-mirror download endpoint)

Usage
-----
    .venv/bin/python src/scripts/push_to_hf.py --repo-id <user-or-org>/prenatal-case-dora --public
    .venv/bin/python src/scripts/push_to_hf.py --repo-id <user-or-org>/prenatal-case-dora --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING = ROOT / "artifacts" / "hf_upload" / "case_dora"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, help="e.g. username/prenatal-case-dora")
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--endpoint", default="https://huggingface.co")
    ap.add_argument("--token", default=None, help="defaults to HF_TOKEN / cached login")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--public", dest="private", action="store_false")
    ap.set_defaults(private=False)
    ap.add_argument("--message", default="Upload case-only DoRA model")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    token = args.token
    api = HfApi(endpoint=args.endpoint, token=token)

    try:
        user = api.whoami()
        print(f"authenticated as: {user.get('name')}")
    except HfHubHTTPError as exc:
        print(
            "Hugging Face authentication failed.\n"
            f"  endpoint: {args.endpoint}\n"
            f"  error: {exc}\n"
            "Run `hf auth login` or set a valid HF_TOKEN, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    staging = args.staging.resolve()
    if not (staging / "model.safetensors").exists():
        raise SystemExit(f"missing {staging/'model.safetensors'}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"repo: https://huggingface.co/{args.repo_id} ({'private' if args.private else 'public'})")

    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
        ignore_patterns=[".*", "__pycache__/*"],
    )
    print(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
