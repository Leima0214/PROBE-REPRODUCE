from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print("Load PromptEnhancedViT + LightweightDetectionHead for PROBE inference.")
    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
