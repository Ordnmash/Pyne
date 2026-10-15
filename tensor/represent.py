def __repr__(self) -> str:
  def fmt(x):
    if isinstance(x, float):
      return f"{x:.4f}"
    return str(x)

  width = max((len(fmt(x)) for x in self.storage), default=1)

  def format_tensor(obj, level=0):

    if not isinstance(obj, list):
      return fmt(obj).rjust(width)

    if len(obj) == 0:
      return "[]"

    if not isinstance(obj[0], list):
      values = [fmt(x).rjust(width) for x in obj]
      return "[" + ", ".join(values) + "]"

    # Higher-dimensional tensor
    lines = []

    for i, item in enumerate(obj):
      formatted = format_tensor(item, level + 1)

      if i > 0:
        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
          lines.append("\n" * (level + 2))
        else:
          lines.append("\n")
        lines.append(" " * (len("pyne.tensor(") + level))
      lines.append(formatted)

      if i != len(obj) - 1:
        lines.append(",")

    return "[" + "".join(lines) + "]"

  return f"pyne.tensor({format_tensor(self.data)})"