"""Train grouped-CV voltage models and evaluate the frozen selection."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step5-run', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--families', nargs='+', choices=['svr', 'krr', 'dnn'], default=['svr', 'krr', 'dnn'])
    parser.add_argument('--populations', nargs='+', choices=['mixed_ions', 'li_only'], default=['mixed_ions', 'li_only'])
    args = parser.parse_args()
    try:
        run(ROOT, args.step5_run, epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
            families=args.families, populations=args.populations)
        return 0
    except KeyboardInterrupt:
        print('\nTraining interrupted. Rerun the same command to reuse completed folds.')
        return 130
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'[ERROR] {exc}')
        print('Completed folds are retained. Share the error before changing the configuration.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
