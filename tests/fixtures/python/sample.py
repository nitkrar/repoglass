"""Fixture for extraction tests. Every construct here is deliberate.

Hand-authored so expected counts can be verified by reading the file.
"""
import os                          # module-scope reference

CONSTANT = 1                       # module-scope assignment


def top_level():                   # def 1
    """Reference inside a definition: enclosing is top_level."""
    return os.getcwd()


def outer():                       # def 2
    def inner():                   # def 3  -- nested; enclosing of helper() is inner
        return helper()
    return inner


def helper():                      # def 4
    return CONSTANT


class Thing:                       # def 5
    def method(self):              # def 6
        return helper()

    def tiny(self):
        pass


top_level()                        # module-scope reference, no enclosing def
