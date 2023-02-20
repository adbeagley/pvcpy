"""Module to hold class and method decorators"""


def abstract_class(cls_):
    """Decorate a class, overriding __new__.
    Preventing a class from being instantiated similar to abc.ABCMeta
    but does not require an abstract method.
    """

    def __new__(cls, *args, **kwargs):
        if cls is cls_:
            raise TypeError(
                (f"{cls.__name__} is an abstract class"
                 "and may not be instantiated.")
            )
        return object.__new__(cls)

    cls_.__new__ = __new__
    return cls_
