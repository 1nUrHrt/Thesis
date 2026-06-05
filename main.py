import argparse

from Evaluate import run, run_train, run_test


def main():
    parser = argparse.ArgumentParser(description="Experiment CLI")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command"
    )

    parser_train = subparsers.add_parser("train", help="Run training only")
    parser_train.add_argument(
        "--configs",
        default="./configs",
        help="Path to a JSON configuration file or a directory containing JSON config files (default: ./configs)",
    )

    parser_test = subparsers.add_parser("test", help="Run test only")
    parser_test.add_argument(
        "--configs",
        default="./configs",
        help="Path to a JSON configuration file or a directory containing JSON config files (default: ./configs)",
    )

    parser_run = subparsers.add_parser("run", help="Run training and test")
    parser_run.add_argument(
        "--configs",
        default="./configs",
        help="Path to a JSON configuration file or a directory containing JSON config files (default: ./configs)",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_train(args.configs)
    elif args.command == "test":
        run_test(args.configs)
    elif args.command == "run":
        run(args.configs)


if __name__ == "__main__":
    main()
