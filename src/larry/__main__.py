import argparse
import asyncio

from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="larry")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("test-jaw")
    enroll = sub.add_parser("enroll")
    enroll.add_argument("name")
    args = parser.parse_args()

    if args.cmd == "enroll":
        from larry.cli.enroll import main as enroll_main
        enroll_main(args.name)
    elif args.cmd == "test-jaw":
        from larry.cli.test_jaw import main as test_jaw_main
        test_jaw_main()
    else:
        from larry.pipeline import run
        asyncio.run(run())


if __name__ == "__main__":
    main()
