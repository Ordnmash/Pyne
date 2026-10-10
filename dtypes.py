"""data types for tensor object!"""

class Float:
  type = 'float'
  def __init__(self):
    self.type = 'float'

  def __repr__(self):
    return f"pyne.float"

  def __eq__(self):
    # ==
    return self.type

class Long:
  type = 'long'
  def __init__(self):
    self.type = 'long'

  def __repr__(self):
    return f"pyne.long"

  def __eq__(self):
    return self.type

class Bool:
  type = 'bool'
  def __init__(self):
    self.type = 'bool'

  def __repr__(self):
    return f"pyne.bool"

  def __eq__(self):
    return self.type
