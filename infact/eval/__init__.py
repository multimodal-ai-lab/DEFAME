from .averitec.benchmark import AVeriTeC
from .benchmark import Benchmark
from .fever.benchmark import FEVER
from .verite.benchmark import VERITE
from .newsclippings.benchmark import NewsCLIPpings
from .dgm4.benchmark import DGM4
from .MOCHEG.benchmark import MOCHEG

BENCHMARK_REGISTRY = {
    AVeriTeC,
    FEVER,
    VERITE,
    NewsCLIPpings,
    DGM4,
    MOCHEG,
}


def load_benchmark(name: str, n_samples: int = None, **kwargs) -> Benchmark:
    print("Loading benchmark...", end="")
    for benchmark in BENCHMARK_REGISTRY:
        if name == benchmark.shorthand:
            b = benchmark(n_samples=n_samples, **kwargs)
            print(" done.")
            return b
    raise ValueError(f"No benchmark named '{name}'.")
