from abc import ABC
import inspect


class Action(ABC):
    """Executed by the actor. Performing an Action yields Evidence.
    Each action should validate its arguments. Docstrings of the implementing
    subclasses are automatically used as descriptions for the LLM. Make sure
    to provide a docstring on class-level describing the purpose of the action and
    provide a docstring in the __init__() method explaining the usage of the parameters.
    See existing subclasses for examples."""
    name: str
    requires_image: bool = False
    additional_info: str = None
    _init_parameters: dict = None

    def _save_parameters(self, local_variables: dict):
        """Memorizes the parameters provided to the __init__() method of the action.
        Used for the string representation of the action, needed in the Report.
        Always call this first in the __init__() method of an action."""
        init_signature = inspect.signature(self.__init__)

        # Only save non-None parameters
        init_parameters = dict()
        for param in init_signature.parameters:
            value = local_variables[param]
            if value is not None:
                init_parameters[param] = value
        self._init_parameters = init_parameters

    def __str__(self):
        if self._init_parameters:
            param_str = ", ".join([f"{k}={v.__repr__()}" for k, v in self._init_parameters.items()])
            return f"{self.name}({param_str})"
        else:
            return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


def get_action_documentation(action_cls: type[Action]) -> str:
    """Returns an LLM-friendly representation of the given action class,
    explaining the action's purpose and usage (with signature)."""
    description = inspect.getdoc(action_cls).strip()
    usage = inspect.getdoc(action_cls.__init__).strip(" \t\n")

    # Get signature and remove 'self' from it
    parameters = dict(inspect.signature(action_cls.__init__).parameters)
    parameters.pop("self")
    signature = "(" + ", ".join([str(p) for p in parameters.values()]) + ")"

    # Put all together into a nice and readable string
    documentation = f"""
`{action_cls.name}`: {description}
Signature:
```python
{action_cls.name}{signature}
```
Usage:
{usage}
""".strip()
    if action_cls.additional_info:
        documentation += f"\n{action_cls.additional_info}"
    return documentation
