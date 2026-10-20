import itertools


def expand_grid(specs):
    for spec in specs:
        keys = list(spec)
        for values in itertools.product(*(spec[key] for key in keys)):
            yield dict(zip(keys, values))


def selection_key(metrics):
    return metrics["macro_f1"], metrics["balanced_accuracy"], metrics["accuracy"]
