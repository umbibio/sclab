from typing import Callable, Type


# full class definition is in .dataset/processor/step/_processor_step_base.py
class ProcessorStepBase:
    name: str
    description: str


methods_registry: dict[str, list[ProcessorStepBase]] = {}


def register_sclab_method(
    category: str,
    name: str | None = None,
    description: str | None = None,
    order: int | None = None,
) -> Callable:
    """
    Decorator to register a class as a sclab method.

    Args:
        category: The category to register the method under (e.g., "Processing")
        name: Optional display name for the method. If None, uses the class name.
        description: Optional description of the method.
        order: Optional ordering within the category. Lower numbers appear first.

    Returns:
        Decorated class
    """

    def decorator(cls: Type[ProcessorStepBase]) -> Type[ProcessorStepBase]:
        if name:
            cls.name = name

        if description:
            cls.description = description

        if order is not None:
            cls.order = order

        # Initialize the category in the registry if it doesn't exist
        if category not in methods_registry:
            methods_registry[category] = []

        methods_list = methods_registry[category]

        # Add the class to the registry
        methods_list.append(cls)

        # Sort the methods by order
        methods_registry[category] = sorted(methods_list, key=lambda x: x.order)

        return cls

    return decorator


def get_sclab_methods():
    methods = {}

    for category, methods_list in methods_registry.items():
        methods[category] = sorted(methods_list, key=lambda x: x.order)

    return methods
