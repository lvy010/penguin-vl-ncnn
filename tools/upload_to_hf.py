#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Upload an exported ncnn model directory (the output of extract_tokenizer.py +
# export_penguinvl.py + rename_decoder_blobs.py) to a Hugging Face model repo.
#
# Security: this script NEVER takes a token on the command line and NEVER writes
# one to disk. Authenticate first with `hf auth login` (recommended), or set the
# HF_TOKEN environment variable in your shell. That keeps the token out of your
# shell history and out of the repo.
#
# Usage:
#   hf auth login                      # paste your token once, interactively
#   python tools/upload_to_hf.py \
#       --repo <your-username>/penguin-vl-2b-ncnn \
#       --dir  ./penguin-vl-2b-ncnn \
#       --create --private

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Upload an exported penguin-vl ncnn model dir to Hugging Face."
    )
    ap.add_argument("--repo", required=True,
                    help="target repo id, e.g. 'username/penguin-vl-2b-ncnn'")
    ap.add_argument("--dir", required=True,
                    help="local exported model directory to upload")
    ap.add_argument("--create", action="store_true",
                    help="create the repo if it does not exist")
    ap.add_argument("--private", action="store_true",
                    help="make the repo private (only used with --create)")
    ap.add_argument("--commit-message", default="upload penguin-vl-ncnn export")
    ap.add_argument("--allow-weights", action="store_true",
                    help="also upload *.bin weight files (large). Off by default "
                         "so you don't accidentally push several GB.")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"[error] --dir not found: {args.dir}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("[error] pip install huggingface_hub first")

    # Token comes from `hf auth login` (stored credential) or HF_TOKEN; we do NOT
    # accept it as an argument on purpose.
    token = os.environ.get("HF_TOKEN")  # falls back to the stored login if None
    api = HfApi(token=token)

    try:
        who = api.whoami()
        print(f"[ok] authenticated as: {who.get('name', '<unknown>')}")
    except Exception as e:
        sys.exit(f"[error] not authenticated. Run `hf auth login` first. ({e})")

    if args.create:
        api.create_repo(repo_id=args.repo, repo_type="model",
                        private=args.private, exist_ok=True)
        print(f"[ok] repo ready: {args.repo} (private={args.private})")

    # By default skip the big weight blobs; params/vocab/merges/model.json are the
    # contract-critical, reviewable files. Pass --allow-weights to include *.bin.
    ignore = None if args.allow_weights else ["*.bin"]
    if ignore:
        print("[info] skipping *.bin weight files (pass --allow-weights to include)")

    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.dir,
        ignore_patterns=ignore,
        commit_message=args.commit_message,
    )
    print(f"[ok] uploaded {args.dir} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
