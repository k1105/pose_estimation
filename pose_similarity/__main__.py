import argparse
from pathlib import Path
from .cli import main as search_main
from .create_comparison import main as comparison_main


def main():
    parser = argparse.ArgumentParser(
        prog="python -m pose_similarity",
        description="ポーズ類似度検索と比較画像作成のツール"
    )
    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # 類似度検索用のサブコマンド
    search_parser = subparsers.add_parser("search", help="類似度検索を実行")
    search_parser.add_argument("json_path", type=Path, help="対象 JSON ファイルへの相対/絶対パス")
    search_parser.add_argument("--id", type=int, required=True, help="検索対象の id")
    search_parser.add_argument("--top", type=int, default=5, help="表示する上位件数 (デフォルト: 5)")

    # 比較画像作成用のサブコマンド
    comparison_parser = subparsers.add_parser("compare", help="比較画像を作成")
    comparison_parser.add_argument("json_path", type=Path, help="対象 JSON ファイルへの相対/絶対パス")
    comparison_parser.add_argument("--id", type=int, help="検索対象の id")
    comparison_parser.add_argument("--random", action="store_true", help="ランダムにIDを選択")
    comparison_parser.add_argument("--count", type=int, default=10, help="ランダムに選択するIDの数 (デフォルト: 10)")
    comparison_parser.add_argument("--output", type=Path, default=Path("comparison.jpg"), 
                                  help="出力画像のパス (デフォルト: comparison.jpg)")

    args = parser.parse_args()

    if args.command == "search":
        search_main(args)
    elif args.command == "compare":
        comparison_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 