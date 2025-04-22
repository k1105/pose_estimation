import argparse
from pathlib import Path
from .cli import main as search_main
from .create_comparison import main as comparison_main
from .chain_comparison import main as chain_main
from .visualize_network import main as network_main


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

    # 連鎖的な類似画像作成用のサブコマンド
    chain_parser = subparsers.add_parser("chain", help="連鎖的な類似画像を作成")
    chain_parser.add_argument("json_path", type=Path, help="対象 JSON ファイルへの相対/絶対パス")
    chain_parser.add_argument("--id", type=int, required=True, help="開始ID")
    chain_parser.add_argument("--count", type=int, required=True, help="生成する画像の枚数")
    chain_parser.add_argument("--output", type=Path, help="出力ディレクトリのパス (デフォルト: chain_images_id{id})")

    # ネットワーク可視化用のサブコマンド
    network_parser = subparsers.add_parser(
        "network",
        help="Create network visualization of pose similarities"
    )
    network_parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the JSON file"
    )
    network_parser.add_argument(
        "--output",
        type=Path,
        default=Path("network.html"),
        help="Output HTML file path (default: network.html)"
    )
    network_parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Distance threshold for edges (default: 50.0)"
    )
    network_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN (default: 5)"
    )
    network_parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples for HDBSCAN core points (default: 3)"
    )
    network_parser.add_argument(
        "--embed",
        type=str,
        choices=["mds", "umap"],
        default="mds",
        help="Embedding method to use (default: mds)"
    )
    network_parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (default: 15)"
    )
    network_parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP (default: 0.1)"
    )
    network_parser.set_defaults(
        handler=lambda args: create_network_visualization(
            args.json_path,
            args.output,
            args.threshold,
            args.min_cluster_size,
            args.min_samples,
            args.embed,
            args.n_neighbors,
            args.min_dist
        )
    )

    args = parser.parse_args()

    # chainコマンドのデフォルト出力ディレクトリを設定
    if args.command == "chain" and not args.output:
        args.output = Path(f"chain_images_id{args.id}")

    if args.command == "search":
        search_main(args)
    elif args.command == "compare":
        comparison_main(args)
    elif args.command == "chain":
        chain_main(args)
    elif args.command == "network":
        network_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 