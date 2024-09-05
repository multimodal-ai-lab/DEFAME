from .averitec.benchmark import AVeriTeC
from .benchmark import Benchmark
from .fever.benchmark import FEVER
from .verite.benchmark import VERITE

BENCHMARK_REGISTRY = {
    AVeriTeC,
    FEVER,
    VERITE
}


def load_benchmark(name: str, **kwargs) -> Benchmark:
    for benchmark in BENCHMARK_REGISTRY:
        if name == benchmark.shorthand:
            return benchmark(**kwargs)
    raise ValueError(f"No benchmark named '{name}'.")
