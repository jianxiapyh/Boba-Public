__all__ = ["SpringMassSystemWarp", "InvPhyTrainerWarp"]


def __getattr__(name):
    if name == "SpringMassSystemWarp":
        from .model import SpringMassSystemWarp

        return SpringMassSystemWarp
    if name == "InvPhyTrainerWarp":
        from .engine import InvPhyTrainerWarp

        return InvPhyTrainerWarp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
