import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Podaj liczbę całkowitą większą od zera.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Porównanie modeli na CIFAR-10.")
    parser.add_argument("--train-per-class", type=positive_int, default=4500,
                        help="Liczba obrazów treningowych na klasę (domyślnie %(default)s).")
    parser.add_argument("--val-per-class", type=positive_int, default=500,
                        help="Liczba obrazów walidacyjnych na klasę (domyślnie %(default)s).")
    parser.add_argument("--test-per-class", type=positive_int, default=1000,
                        help="Liczba obrazów testowych na klasę (domyślnie %(default)s).")
    parser.add_argument("--epochs", type=positive_int, default=1000,
                        help="Limit epok CNN i głowicy (domyślnie %(default)s).")
    parser.add_argument("--fine-tune-epochs", type=positive_int, default=1000,
                        help="Limit dodatkowych epok dostrajania (domyślnie %(default)s).")
    parser.add_argument("--batch-size", type=positive_int, default=128,
                        help="Wielkość partii (domyślnie %(default)s).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Ziarno losowania (domyślnie 2026).")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results",
                        help="Katalog nadrzędny na wyniki eksperymentów.")
    args = parser.parse_args()
    if args.train_per_class + args.val_per_class > 5000 or args.test_per_class > 1000:
        parser.error("Na klasę dostępnych jest 5000 obrazów do treningu i walidacji oraz 1000 do testu.")
    if not 0 <= args.seed < 2**32:
        parser.error("Ziarno musi należeć do zakresu od 0 do 2³² − 1.")

    from experiment import ImageClassificationExperiment

    ImageClassificationExperiment(
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output=args.output,
    ).run()


if __name__ == "__main__":
    main()
