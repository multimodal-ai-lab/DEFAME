from .averitec.benchmark import AVeriTeC
from .benchmark import Benchmark
from .fever.benchmark import FEVER
from .verite.benchmark import VERITE
from .newsclippings.benchmark import NewsCLIPpings

BENCHMARK_REGISTRY = {
    AVeriTeC,
    FEVER,
    VERITE,
    NewsCLIPpings,
}


def load_benchmark(name: str, **kwargs) -> Benchmark:
    for benchmark in BENCHMARK_REGISTRY:
        if name == benchmark.shorthand:
            return benchmark(**kwargs)
    raise ValueError(f"No benchmark named '{name}'.")
