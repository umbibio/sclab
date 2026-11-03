from ipywidgets import Dropdown


def set_options_keep_value(dropdown: Dropdown, options) -> None:
    """Replace ``dropdown.options``, keeping the current value if still offered.

    ipywidgets resets the selection to the first option whenever the options
    change. The restore happens inside ``hold_sync`` so the frontend receives
    one atomic update (options and selected index) instead of the transient
    reset, which it can render as an empty dropdown.
    """
    old_value = dropdown.value
    available = options.values() if isinstance(options, dict) else options
    with dropdown.hold_sync():
        dropdown.options = options
        if old_value in available:
            dropdown.value = old_value
