"""Exact composition checks for the integer-stoichiometry formulas in this CSV.

Supported: element symbols, positive integer counts, nested parentheses/brackets.
Unsupported notation (fractional occupancies, hydrates, isotope/charge suffixes,
variables) is rejected for review; this is not a general chemistry parser or a
crystal-structure validator. No oxidation state or physical stability is inferred.
"""

from collections import Counter
from functools import lru_cache, reduce
from math import gcd
import re


ELEMENTS = frozenset((
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
    'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce '
    'Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn '
    'Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
).split())
TOKEN = re.compile(r'[A-Z][a-z]?|[0-9]+|[()\[\]]')


@lru_cache(maxsize=40000)
def composition(formula: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(formula, str) or not formula or len(formula) > 1024:
        raise ValueError('empty or oversized formula')
    tokens = TOKEN.findall(formula)
    if ''.join(tokens) != formula:
        raise ValueError('unsupported formula notation')
    position = 0

    def multiplier():
        nonlocal position
        if position < len(tokens) and tokens[position].isdigit():
            token = tokens[position]
            position += 1
            if len(token) > 7 or token.startswith('0') or not 1 <= int(token) <= 1000000:
                raise ValueError('invalid stoichiometric count')
            return int(token)
        return 1

    def group(closing=None, depth=0):
        nonlocal position
        if depth > 12:
            raise ValueError('formula nesting limit exceeded')
        counts = Counter()
        while position < len(tokens):
            token = tokens[position]
            if token in (')', ']'):
                if token != closing or not counts:
                    raise ValueError('unbalanced or empty formula group')
                position += 1
                return counts
            position += 1
            if token in ('(', '['):
                nested = group(')' if token == '(' else ']', depth + 1)
                factor = multiplier()
                for element, count in nested.items():
                    counts[element] += count * factor
            elif token in ELEMENTS:
                counts[token] += multiplier()
            else:
                raise ValueError('unknown element or misplaced number')
            if sum(counts.values()) > 1000000000:
                raise ValueError('formula atom-count limit exceeded')
        if closing is not None or not counts:
            raise ValueError('unbalanced or empty formula')
        return counts

    return tuple(sorted(group().items()))


def reduced(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    if not counts:
        return ()
    divisor = reduce(gcd, counts.values())
    return tuple(sorted((element, count // divisor) for element, count in counts.items()))


def host_and_loading(formula: str, ion: str):
    counts = dict(composition(formula))
    working_count = counts.pop(ion, 0)
    if not counts:
        raise ValueError('no non-working-ion host remains')
    divisor = reduce(gcd, counts.values())
    atom_fraction = working_count / (working_count + sum(counts.values()))
    return reduced(counts), working_count / divisor, atom_fraction


def host_key(host) -> str:
    return '|'.join(f'{element}:{count}' for element, count in host)
