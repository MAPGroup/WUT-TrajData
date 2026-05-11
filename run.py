import argparse


MODEL_REGISTRY = {
    "simple": {
        "description": "Physics-based and classical machine learning baselines.",
        "models": {
            "cv": "benchmark/simple_baselines/",
            "ca": "benchmark/simple_baselines/",
            "kf": "benchmark/simple_baselines/",
            "ekf": "benchmark/simple_baselines/",
            "knn": "benchmark/simple_baselines/",
            "svr": "benchmark/simple_baselines/",
            "hmm": "benchmark/simple_baselines/",
        },
    },
    "basic_dl": {
        "description": "Basic deep learning models implemented under a unified framework.",
        "models": {
            "lstm": "benchmark/basic_dl/",
            "gru": "benchmark/basic_dl/",
            "bilstm": "benchmark/basic_dl/",
            "transformer": "benchmark/basic_dl/",
        },
    },
    "external": {
        "description": "Adapted open-source trajectory prediction models.",
        "models": {
            "led": "benchmark/external_models/LED/",
            "tutr": "benchmark/external_models/TUTR/",
            "social_lstm": "benchmark/external_models/Social_LSTM/",
            "social_gan": "benchmark/external_models/Social_GAN/",
            "pecnet": "benchmark/external_models/PECNet/",
            "st_mgt": "benchmark/external_models/ST_MGT/",
            "msrl": "benchmark/external_models/MSRL/",
            "traisformer": "benchmark/external_models/TrAISformer/",
        },
    },
}


def list_models():
    print("\nAvailable benchmark groups:\n")

    for group_name, group_info in MODEL_REGISTRY.items():
        print(f"[{group_name}]")
        print(f"  {group_info['description']}")
        print("  Models:")

        for model_name, model_path in group_info["models"].items():
            print(f"    - {model_name}: {model_path}")

        print()


def show_model_info(group: str, model: str):
    if group not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported group: {group}")

    models = MODEL_REGISTRY[group]["models"]

    if model not in models:
        raise ValueError(f"Unsupported model: {model} in group: {group}")

    model_path = models[model]

    print("\nModel information")
    print("-----------------")
    print(f"Group : {group}")
    print(f"Model : {model}")
    print(f"Path  : {model_path}")

    if group == "simple":
        print("\nSuggested usage:")
        print("  Please refer to benchmark/simple_baselines/README.md")

    elif group == "basic_dl":
        print("\nSuggested usage:")
        print("  Please refer to benchmark/basic_dl/README.md")

    elif group == "external":
        print("\nSuggested usage:")
        print(f"  Please refer to {model_path}/README_WUT.md")
        print("  The original project structure is preserved as much as possible.")


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight entry point for WUT-TrajData benchmark."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark models.",
    )

    parser.add_argument(
        "--group",
        type=str,
        choices=["simple", "basic_dl", "external"],
        help="Benchmark group.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name.",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.group and args.model:
        show_model_info(args.group, args.model)
        return

    parser.print_help()


if __name__ == "__main__":
    main()