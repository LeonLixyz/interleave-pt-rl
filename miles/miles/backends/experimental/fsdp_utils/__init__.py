__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]


def __getattr__(name: str):
    """Load GPU/runtime-heavy FSDP modules only when their symbols are used."""
    if name == "FSDPTrainRayActor":
        from .actor import FSDPTrainRayActor

        return FSDPTrainRayActor
    if name == "load_fsdp_args":
        from .arguments import load_fsdp_args

        return load_fsdp_args
    raise AttributeError(name)
