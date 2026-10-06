from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict


def parse_definition(definition: str) -> list[str]:
    """Split an enumerator definition into its entries.

    Args:
        definition: Comma-separated entries. Whitespace around each entry is
            trimmed and empty entries are dropped.

    Returns:
        The entries in definition order.
    """
    return [s.strip() for s in definition.split(",") if s.strip()]


class Enumerator(ComfyNodeABC):
    """Map a selected entry of a user-defined list to an integer.

    The ``selection`` combo is populated on the frontend from ``definition``
    by ``web/enumerator.js``.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """Declare the node inputs.

        Returns:
            The ``definition`` string, the ``selection`` combo, and the
            ``first_value`` int.
        """
        return {
            "required": {
                "definition": (IO.STRING, {"default": "first, second, third"}),
                "selection": (IO.COMBO, {"options": []}),
                "first_value": (IO.INT, {"default": 0}),
            }
        }

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("value",)
    FUNCTION = "enumerate"
    CATEGORY = "utils"

    @classmethod
    def VALIDATE_INPUTS(cls, selection: str) -> bool:
        """Accept any selection so ComfyUI skips its static combo check.

        The options come from ``definition`` at runtime, so :meth:`enumerate`
        validates the selection instead.

        Args:
            selection: The selected entry.

        Returns:
            Always ``True``.
        """
        return True

    def enumerate(self, definition: str, selection: str, first_value: int) -> tuple[int]:
        """Compute the enumerated value of the selected entry.

        Args:
            definition: Comma-separated entries, parsed by
                :func:`parse_definition`.
            selection: The selected entry.
            first_value: The value of the first entry.

        Returns:
            ``first_value`` plus the index of ``selection`` in the definition.

        Raises:
            ValueError: If ``selection`` is not an entry of ``definition``.
        """
        options = parse_definition(definition)
        if selection not in options:
            raise ValueError(f"Enumerator: '{selection}' is not in definition {options}")
        return (first_value + options.index(selection),)


NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {
    "Enumerator": Enumerator,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "Enumerator": "Enumerator",
}
