"""Command-line interface: ``gang-mof example | fit | sample``."""
from __future__ import annotations

import argparse
import sys

from ._version import __version__


def _cmd_example(args):
    from .data import load_qmof_example
    load_qmof_example(as_frame=True).to_csv(args.out, index=False)
    print(f"wrote 1,000 example QMOF MOFs to {args.out}")


def _cmd_fit(args):
    from .data import NodeDataset
    from .model import GANg, GANgConfig
    data = NodeDataset.from_csv(args.data, node_id=args.node_id, target=args.target, schema=args.schema,
                                features=args.features.split(",") if args.features else None)
    train, val, test = data.split(1 - args.validation - args.test, args.validation, args.test, args.seed)
    cfg = GANgConfig.paper() if args.paper else GANgConfig(iterations=args.iterations)
    model = GANg(config=cfg, conditional=not args.unconditional, random_state=args.seed, device=args.device)
    model.fit(train, validation_data=val)
    model.save(args.out)
    if len(test):
        test.to_dataframe().to_csv(f"{args.out}/test_nodes.csv", index=False)
    print(f"saved model to {args.out}")


def _cmd_sample(args):
    from .model import GANg
    model = GANg.load(args.model, device=args.device)
    result = model.sample(args.n, condition=args.condition, random_state=args.seed)
    model.attach(result, k=args.k)
    model.evaluate(result, surrogate=not args.no_surrogate)
    result.export(args.out)
    print(result.summary())
    print(f"results written to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gang-mof", description="GANg: auditable graph-node synthesis for MOF similarity networks")
    p.add_argument("--version", action="version", version=f"gang-mof {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("example", help="write the bundled 1,000-MOF example table to CSV")
    e.add_argument("--out", default="qmof_example_1000.csv")
    e.set_defaults(func=_cmd_example)

    f = sub.add_parser("fit", help="train GANg on a CSV of real nodes")
    f.add_argument("--data", required=True)
    f.add_argument("--node-id", default=None)
    f.add_argument("--target", default=None, help="conditioning target column (omit for unconditional)")
    f.add_argument("--schema", default="auto", choices=["auto", "qmof106", "generic"])
    f.add_argument("--features", default=None, help="comma-separated feature columns (generic schema)")
    f.add_argument("--validation", type=float, default=0.15)
    f.add_argument("--test", type=float, default=0.15)
    f.add_argument("--iterations", type=int, default=3000)
    f.add_argument("--paper", action="store_true", help="use the paper configuration (6000 updates)")
    f.add_argument("--unconditional", action="store_true")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cpu")
    f.add_argument("--out", default="gang_model")
    f.set_defaults(func=_cmd_fit)

    s = sub.add_parser("sample", help="generate, attach and audit nodes with a saved model")
    s.add_argument("--model", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--condition", type=float, default=None, help="requested target value")
    s.add_argument("--k", type=int, default=15)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--no-surrogate", action="store_true")
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", default="gang_results")
    s.set_defaults(func=_cmd_sample)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
