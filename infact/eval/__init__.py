from infact.eval.averitec.benchmark import AVeriTeC
from infact.eval.benchmark import Benchmark
from infact.eval.fever.benchmark import FEVER
from infact.eval.verite.benchmark import VERITE

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
