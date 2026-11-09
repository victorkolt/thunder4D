"""matplotlib is optional: the plots of the design helper are a convenience, not a dependency."""


def pyplot():
    """``matplotlib.pyplot``, or an ImportError that says how to install it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError('plotting needs matplotlib: install it with  uv add "thunder4d[examples]"  '
                          "(or  pip install matplotlib)") from err
    return plt
