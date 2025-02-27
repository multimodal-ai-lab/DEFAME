_platform_handlers = {}  # Maps platform name to handler instance

def register_platform_handler(platform_names: str, handler_instance):
    """Register a handler instance for the specified platforms."""
    for platform in platform_names:
        _platform_handlers[platform.lower()] = handler_instance


def get_handler_for_platform(platform: str):
    """Get the appropriate handler for a platform."""
    return _platform_handlers.get(platform.lower())


def get_supported_platforms():
    """Get a list of supported platforms."""
    return list(_platform_handlers.keys())

    