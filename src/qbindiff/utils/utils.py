import logging
from functools import cached_property, lru_cache, partial, update_wrapper
from typing import Callable, TypeVar, TypeAlias

T = TypeVar("T")
UserFuncType: TypeAlias = Callable[..., T]


def instance_lru_cache(
    method: UserFuncType | None = None,
    *,
    maxsize: int | None = 128,
    typed: bool = False
) -> UserFuncType | Callable[[UserFuncType], UserFuncType]:
    """
    Least-recently-used cache decorator for instance methods.

    The cache follows the lifetime of an object, it is stored on the object, not on the
    class. Wrapper around functools.lru_cache so all the parameters other than `self`
    must be hashables.

    The parameters *maxsize* and *typed* are forwarded to functools.lru_cache decorator.

    View the cache statistics named tuple (hits, misses, maxsize, currsize) with
    f.cache_info(). Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.
    """

    def decorator(wrapped: UserFuncType) -> UserFuncType:
        @cached_property
        def wrapper(self: object) -> UserFuncType:
            return lru_cache(maxsize=maxsize, typed=typed)(
                update_wrapper(partial(wrapped, self), wrapped)
            )

        return wrapper

    return decorator if method is None else decorator(method)


def instance_cache(
    method: UserFuncType | None = None,
) -> UserFuncType | Callable[[UserFuncType], UserFuncType]:
    """Wrapper around instance_lru_cache(maxsize=None)"""
    return instance_lru_cache(method, maxsize=None)


def is_debug() -> bool:
    """Returns True if the current logging level is set to debug"""
    return logging.root.level <= logging.DEBUG
