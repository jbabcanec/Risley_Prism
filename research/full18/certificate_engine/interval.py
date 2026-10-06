"""Exact-rational, outward-rounded intervals for the bounded certificate engine.

All endpoints are fractions on a dyadic grid. Every arithmetic operation rounds
its lower endpoint down and its upper endpoint up, using integer arithmetic only.
The default grid is 2**-80. Binary operations between intervals use the coarser
of their explicit precisions. Fractions, integers and rational/decimal strings
are accepted; binary floating-point inputs are deliberately rejected.

These are enclosing computations, not a claim that any optical box is feasible.
Domain failures (notably zero-crossing division and uncertain square-root sign)
must be handled by a caller as unresolved, or settled by a separate certificate.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial, isqrt
from typing import Iterator


_PRECISION: ContextVar[int] = ContextVar("certificate_interval_bits", default=80)


class IntervalDomainError(ArithmeticError):
    """An operation lacks a required sign/domain certificate on its input box."""


def _check_bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 8 <= bits <= 4096:
        raise ValueError("interval precision must be an integer from 8 to 4096")
    return bits


@contextmanager
def precision(bits: int) -> Iterator[None]:
    """Temporarily set the constructor precision; existing I objects are immutable."""
    token = _PRECISION.set(_check_bits(bits))
    try:
        yield
    finally:
        _PRECISION.reset(token)


def _rational(value: object) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("boolean is not an exact-rational interval endpoint")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("use an int, Fraction or rational string; floats are forbidden")


def _floor_grid(value: Fraction, bits: int) -> Fraction:
    scale = 1 << bits
    return Fraction((value.numerator * scale) // value.denominator, scale)


def _ceil_grid(value: Fraction, bits: int) -> Fraction:
    scale = 1 << bits
    return Fraction(-((-value.numerator * scale) // value.denominator), scale)


@dataclass(frozen=True, init=False, slots=True)
class I:
    """Closed interval [lo, hi] with explicit outward dyadic precision in bits."""

    lo: Fraction
    hi: Fraction
    bits: int

    def __init__(self, lo: object, hi: object = None, *, bits: int | None = None):
        if isinstance(lo, I):
            if hi is not None:
                raise TypeError("an interval copy cannot also specify an upper endpoint")
            low, high = lo.lo, lo.hi
            actual_bits = lo.bits if bits is None else _check_bits(bits)
        else:
            low = _rational(lo)
            high = low if hi is None else _rational(hi)
            actual_bits = _PRECISION.get() if bits is None else _check_bits(bits)
        if low > high:
            raise ValueError("lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lo", _floor_grid(low, actual_bits))
        object.__setattr__(self, "hi", _ceil_grid(high, actual_bits))
        object.__setattr__(self, "bits", actual_bits)

    def _pair(self, other: object) -> tuple[I, int]:
        if isinstance(other, I):
            return other, min(self.bits, other.bits)
        return I(other, bits=self.bits), self.bits

    def __add__(self, other: object) -> I:
        other, bits = self._pair(other)
        return I(self.lo + other.lo, self.hi + other.hi, bits=bits)

    __radd__ = __add__

    def __neg__(self) -> I:
        return I(-self.hi, -self.lo, bits=self.bits)

    def __sub__(self, other: object) -> I:
        other, bits = self._pair(other)
        return I(self.lo - other.hi, self.hi - other.lo, bits=bits)

    def __rsub__(self, other: object) -> I:
        other, _ = self._pair(other)
        return other - self

    def __mul__(self, other: object) -> I:
        other, bits = self._pair(other)
        values = (self.lo * other.lo, self.lo * other.hi,
                  self.hi * other.lo, self.hi * other.hi)
        return I(min(values), max(values), bits=bits)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> I:
        other, bits = self._pair(other)
        if other.lo <= 0 <= other.hi:
            raise IntervalDomainError("divisor interval contains zero")
        values = (self.lo / other.lo, self.lo / other.hi,
                  self.hi / other.lo, self.hi / other.hi)
        return I(min(values), max(values), bits=bits)

    def __rtruediv__(self, other: object) -> I:
        other, _ = self._pair(other)
        return other / self

    def square(self) -> I:
        """Enclose x*x with the repeated occurrence treated as the same scalar."""
        high = max(self.lo * self.lo, self.hi * self.hi)
        low = Fraction(0) if self.lo <= 0 <= self.hi else min(
            self.lo * self.lo, self.hi * self.hi)
        return I(low, high, bits=self.bits)

    def __pow__(self, exponent: int) -> I:
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("interval exponent must be an integer")
        if exponent < 0:
            return 1 / (self ** (-exponent))
        result = I(1, bits=self.bits)
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result * factor
            remaining >>= 1
            if remaining:
                factor = factor.square()
        return result

    def sqrt(self) -> I:
        """Enclose nonnegative square roots by integer square-root comparisons.

        A negative lower endpoint raises even if the upper endpoint is positive:
        this evaluator does not silently erase an unresolved physical sign test.
        Exact roots on the output dyadic grid are retained without an extra ulp.
        """
        if self.lo < 0:
            raise IntervalDomainError("square-root radicand is not certified nonnegative")
        scale = 1 << self.bits
        square_scale = scale * scale
        low_n = self.lo.numerator * square_scale
        high_n = self.hi.numerator * square_scale
        low_root = isqrt(low_n // self.lo.denominator)
        high_root = isqrt(high_n // self.hi.denominator)
        if high_root * high_root * self.hi.denominator != high_n:
            high_root += 1
        return I(Fraction(low_root, scale), Fraction(high_root, scale), bits=self.bits)

    def __abs__(self) -> I:
        if self.lo >= 0:
            return self
        if self.hi <= 0:
            return -self
        return I(0, max(-self.lo, self.hi), bits=self.bits)

    @property
    def width(self) -> Fraction:
        """Exact width of the stored enclosure, with no floating-point conversion."""
        return self.hi - self.lo

    @property
    def midpoint(self) -> Fraction:
        return (self.lo + self.hi) / 2

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    def contains(self, value: object) -> bool:
        if isinstance(value, I):
            return self.lo <= value.lo and value.hi <= self.hi
        value = _rational(value)
        return self.lo <= value <= self.hi

    def hull(self, *others: object) -> I:
        low, high, bits = self.lo, self.hi, self.bits
        for value in others:
            other = value if isinstance(value, I) else I(value, bits=bits)
            low, high = min(low, other.lo), max(high, other.hi)
            bits = min(bits, other.bits)
        return I(low, high, bits=bits)

    def json(self) -> list[str]:
        """Return exact fraction-string endpoints; callers record bits separately."""
        return [str(self.lo), str(self.hi)]

    to_json = json

    @classmethod
    def from_json(cls, value: object, *, bits: int | None = None) -> I:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("interval JSON must have exactly two rational endpoints")
        return cls(value[0], value[1], bits=bits)

    fromjson = from_json


def _atan_reciprocal_bounds(denominator: int, bits: int) -> tuple[Fraction, Fraction]:
    """Alternating-series bounds for atan(1/denominator), denominator > 1."""
    x = Fraction(1, denominator)
    power = x
    total = Fraction(0)
    threshold = Fraction(1, 1 << (bits + 8))
    k = 0
    while True:
        term = power / (2 * k + 1)
        total += term if k % 2 == 0 else -term
        next_power = power * x * x
        next_term = next_power / (2 * k + 3)
        if next_term <= threshold:
            following = total + (-next_term if k % 2 == 0 else next_term)
            return min(total, following), max(total, following)
        power = next_power
        k += 1


@lru_cache(maxsize=16)
def _pi_at_bits(bits: int) -> I:
    # Machin's identity: pi = 16 atan(1/5) - 4 atan(1/239).
    low5, high5 = _atan_reciprocal_bounds(5, bits)
    low239, high239 = _atan_reciprocal_bounds(239, bits)
    return I(16 * low5 - 4 * high239, 16 * high5 - 4 * low239, bits=bits)


def pi_interval(*, bits: int | None = None) -> I:
    """Certified pi enclosure from exact rational alternating-series bounds."""
    return _pi_at_bits(_PRECISION.get() if bits is None else _check_bits(bits))


def sincos_small(x: I | Fraction | int | str, *, terms: int = 32) -> tuple[I, I]:
    """Return (sin(x), cos(x)) enclosures for an interval contained in [-1, 1].

    Horner evaluation uses outward interval arithmetic. The sine polynomial has
    degree 2*terms-1 and the cosine polynomial degree 2*terms-2. Appending their
    next zero Taylor coefficients gives Lagrange errors bounded respectively by
    M**(2*terms+1)/(2*terms+1)! and M**(2*terms)/(2*terms)!, M=max(abs endpoints).
    The derivative bound is one on the real line. No floating-point trig is used.
    """
    x = x if isinstance(x, I) else I(x)
    if isinstance(terms, bool) or not isinstance(terms, int) or not 1 <= terms <= 256:
        raise ValueError("Taylor terms must be an integer from 1 to 256")
    magnitude = max(abs(x.lo), abs(x.hi))
    if magnitude > 1:
        raise IntervalDomainError("sincos_small requires an interval in [-1, 1]")
    x2 = x.square()
    sine_poly = I(0, bits=x.bits)
    cosine_poly = I(0, bits=x.bits)
    for k in range(terms - 1, -1, -1):
        sign = 1 if k % 2 == 0 else -1
        sine_poly = sine_poly * x2 + Fraction(sign, factorial(2 * k + 1))
        cosine_poly = cosine_poly * x2 + Fraction(sign, factorial(2 * k))
    sine_poly = sine_poly * x
    sine_error = magnitude ** (2 * terms + 1) / factorial(2 * terms + 1)
    cosine_error = magnitude ** (2 * terms) / factorial(2 * terms)
    sine = sine_poly + I(-sine_error, sine_error, bits=x.bits)
    cosine = cosine_poly + I(-cosine_error, cosine_error, bits=x.bits)
    return sine, cosine


def tan_small(x: I | Fraction | int | str, *, terms: int = 32) -> I:
    sine, cosine = sincos_small(x, terms=terms)
    return sine / cosine


def tan_pi_fraction(numerator: int, denominator: int, *, bits: int | None = None,
                    terms: int = 32) -> I:
    """Enclose tan(pi*numerator/denominator), requiring the angle box in [-1,1]."""
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError("angle numerator must be an integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator == 0:
        raise ValueError("angle denominator must be a nonzero integer")
    return tan_small(pi_interval(bits=bits) * Fraction(numerator, denominator), terms=terms)
