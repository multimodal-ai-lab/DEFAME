from .averitec.benchmark import AVeriTeC
from .benchmark import Benchmark
from .fever.benchmark import FEVER
from .verite.benchmark import VERITE
from .newsclippings.benchmark import NewsCLIPpings
from .dgm4.benchmark import DGM4
from .mocheg.benchmark import MOCHEG
from .claimreview.benchmark import ClaimReview2024

BENCHMARK_REGISTRY = {
    AVeriTeC,
    FEVER,
    VERITE,
    NewsCLIPpings,
    DGM4,
    MOCHEG,
    ClaimReview2024,
}


def load_benchmark(name: str, **kwargs) -> Benchmark:
    print("Loading benchmark...", end="")
    for benchmark in BENCHMARK_REGISTRY:
        if name == benchmark.shorthand:
            b = benchmark(**kwargs)
            print(" done.")
            return b
    raise ValueError(f"No benchmark named '{name}'.")
