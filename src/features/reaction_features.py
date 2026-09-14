"""Fixed 740-column composition/symmetry adaptation, not the paper's 237 features.

This module reads an explicit predictor whitelist. It does not fit encoders,
inspect targets, or use electrode IDs, source warnings or split information.
The same feature_vector function is used for CSV rows and prediction requests.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from src.data.chemistry import ELEMENTS, composition, host_and_loading
from src.data.preparation import crystal_system


SCHEMA_VERSION = 'reaction-composition-symmetry-v1'
ELEMENT_ORDER = tuple((
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
    'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce '
    'Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn '
    'Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
).split())
if len(ELEMENT_ORDER) != 118 or set(ELEMENT_ORDER) != ELEMENTS:
    raise RuntimeError('Element table and formula parser disagree')
ATOMIC_NUMBER = {symbol: i + 1 for i, symbol in enumerate(ELEMENT_ORDER)}
IONS = ('Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Al', 'Y')
ION_PERIOD = {'Li': 2, 'Na': 3, 'K': 4, 'Mg': 3, 'Ca': 4, 'Zn': 4, 'Al': 3, 'Y': 5}
SYSTEMS = ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic')
REQUIRED_INPUTS = (
    'working_ion', 'formula_charge', 'formula_discharge',
    'crystal_system_charge', 'spacegroup_number_charge',
    'crystal_system_discharge', 'spacegroup_number_discharge',
)
OPTIONAL_INPUTS = ('fracA_charge', 'fracA_discharge')
FRACTION_TOLERANCE = 1e-6


def feature_schema():
    columns = []
    def add(name, block, description):
        columns.append({'index': len(columns), 'name': name, 'block': block, 'description': description})
    for side in ('charge', 'discharge'):
        for element in ELEMENT_ORDER:
            add(f'{side}_atom_fraction_{element}', 'endpoint_element_fractions',
                f'{element} atoms / total atoms in the {side} endpoint')
    for side in ('charge', 'discharge'):
        for name, desc in [
            ('atomic_number_mean', 'atom-fraction-weighted mean atomic number'),
            ('atomic_number_std', 'atom-fraction-weighted population standard deviation of atomic number'),
            ('atomic_number_min', 'minimum atomic number among present elements'),
            ('atomic_number_max', 'maximum atomic number among present elements'),
            ('n_elements', 'number of distinct elements'),
            ('fraction_entropy', 'negative sum of atom fraction times its natural logarithm'),
            ('fraction_l2_norm', 'square root of sum of squared atom fractions'),
        ]:
            add(f'{side}_{name}', 'endpoint_composition_summaries', f'{side}: {desc}')
    for ion in IONS:
        add('working_ion_' + ion, 'working_ion', f'1 if working ion is {ion}, otherwise 0')
    add('working_ion_atomic_number', 'working_ion', 'Atomic number of the working element')
    add('working_ion_period', 'working_ion', 'Periodic-table period of the working element')
    for name, desc in [
        ('fracA_charge', 'Working-ion atom fraction at the charge endpoint, recomputed from formula'),
        ('fracA_discharge', 'Working-ion atom fraction at the discharge endpoint, recomputed from formula'),
        ('delta_fracA', 'Discharge atom fraction minus charge atom fraction'),
        ('ion_per_host_charge', 'Charge working-ion count per reduced non-working-ion host'),
        ('ion_per_host_discharge', 'Discharge working-ion count per reduced non-working-ion host'),
        ('delta_ion_per_host', 'Increase in working-ion count per reduced host'),
    ]:
        add(name, 'concentration_interval', desc)
    for side in ('charge', 'discharge'):
        for system in SYSTEMS:
            add(f'{side}_crystal_system_{system}', 'crystal_system', f'1 if {side} crystal system is {system}')
    for side in ('charge', 'discharge'):
        for number in range(1, 231):
            add(f'{side}_spacegroup_{number:03d}', 'spacegroup', f'1 if {side} space-group number is {number}')
    return {
        'version': SCHEMA_VERSION, 'feature_count': len(columns), 'columns': columns,
        'required_inputs': list(REQUIRED_INPUTS), 'optional_inputs': list(OPTIONAL_INPUTS),
        'element_order': list(ELEMENT_ORDER), 'ion_period': ION_PERIOD,
        'fraction_tolerance': FRACTION_TOLERANCE,
        'representation_status': 'Documented adaptation; not the paper\'s original 237-feature specification',
        'source_notes': [
            'Atomic numbers use the 118-element periodic-table order. No empirical property table is imputed.',
            'Space groups are categorical one-hot indicators, not ordinal numeric magnitudes.',
            'Crystal systems and space groups are redundant in part; that redundancy is explicit.',
        ],
    }


SCHEMA = feature_schema()
FEATURE_NAMES = tuple(c['name'] for c in SCHEMA['columns'])
SCHEMA_HASH = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class ReactionInput:
    working_ion: str
    formula_charge: str
    formula_discharge: str
    crystal_system_charge: str
    spacegroup_number_charge: int
    crystal_system_discharge: str
    spacegroup_number_discharge: int
    fracA_charge: float | None = None
    fracA_discharge: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping):
        missing = [key for key in REQUIRED_INPUTS if key not in values]
        if missing:
            raise ValueError('Missing prediction inputs: ' + ', '.join(missing))
        data = {key: values[key] for key in REQUIRED_INPUTS}
        for side in ('charge', 'discharge'):
            key = 'spacegroup_number_' + side
            val = data[key]
            try:
                if isinstance(val, bool) or not math.isfinite(float(val)) or int(float(val)) != float(val):
                    raise ValueError
                data[key] = int(float(val))
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f'{key} must be an integer from 1 to 230') from None
            for key in ('crystal_system_' + side, 'formula_' + side):
                if not isinstance(data[key], str) or not data[key]:
                    raise ValueError(f'{key} must be a nonempty string')
            data['crystal_system_' + side] = data['crystal_system_' + side].strip().lower()
        for key in OPTIONAL_INPUTS:
            value = values.get(key)
            if value is None or value == '':
                data[key] = None
            else:
                try:
                    if isinstance(value, bool):
                        raise ValueError
                    data[key] = float(value)
                    if not math.isfinite(data[key]) or not 0 <= data[key] < 1:
                        raise ValueError
                except (ValueError, TypeError):
                    raise ValueError(f'{key} must be a finite atom fraction') from None
        return cls(**data)


def feature_vector(values: Mapping | ReactionInput) -> np.ndarray:
    # Round-trip dataclass inputs through the same normalizer as CSV/API mappings.
    if isinstance(values, ReactionInput):
        values = vars(values)
    reaction = ReactionInput.from_mapping(values)
    ion = reaction.working_ion
    if ion not in IONS:
        raise ValueError('Unsupported working ion')
    hosts, loadings, fractions, compositions = [], [], [], []
    for side in ('charge', 'discharge'):
        formula = getattr(reaction, 'formula_' + side)
        host, loading, fraction = host_and_loading(formula, ion)
        hosts.append(host)
        loadings.append(loading)
        fractions.append(fraction)
        compositions.append(dict(composition(formula)))
        supplied = getattr(reaction, 'fracA_' + side)
        if supplied is not None and abs(supplied - fraction) > FRACTION_TOLERANCE:
            raise ValueError(f'{side} atom fraction disagrees with its formula')
        sg = getattr(reaction, 'spacegroup_number_' + side)
        system = getattr(reaction, 'crystal_system_' + side)
        if not 1 <= sg <= 230 or crystal_system(sg) != system:
            raise ValueError(f'{side} symmetry inputs are inconsistent')
    if hosts[0] != hosts[1]:
        raise ValueError('The two endpoints must have the same non-working-ion host composition')
    if loadings[1] <= loadings[0]:
        raise ValueError('Working-ion loading must increase from charge to discharge')
    result = []
    for counts in compositions:
        total = sum(counts.values())
        result.extend(counts.get(symbol, 0) / total for symbol in ELEMENT_ORDER)
    for counts in compositions:
        total = sum(counts.values())
        weights = [(ATOMIC_NUMBER[element], count / total) for element, count in counts.items()]
        mean = sum(z * f for z, f in weights)
        result.extend([
            mean, math.sqrt(sum(f * (z - mean) ** 2 for z, f in weights)),
            min(z for z, _ in weights), max(z for z, _ in weights), len(counts),
            -sum(f * math.log(f) for _, f in weights), math.sqrt(sum(f * f for _, f in weights)),
        ])
    result.extend(float(ion == symbol) for symbol in IONS)
    result.extend([ATOMIC_NUMBER[ion], ION_PERIOD[ion], fractions[0], fractions[1],
                   fractions[1] - fractions[0], loadings[0], loadings[1], loadings[1] - loadings[0]])
    for side in ('charge', 'discharge'):
        system = getattr(reaction, 'crystal_system_' + side)
        result.extend(float(system == name) for name in SYSTEMS)
    for side in ('charge', 'discharge'):
        sg = getattr(reaction, 'spacegroup_number_' + side)
        result.extend(float(sg == number) for number in range(1, 231))
    array = np.asarray(result, dtype=np.float64)
    if array.shape != (len(FEATURE_NAMES),) or not np.isfinite(array).all():
        raise ValueError('Feature construction produced an invalid vector')
    return array


def feature_matrix(rows) -> np.ndarray:
    rows = list(rows)
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    vectors = []
    for i, row in enumerate(rows):
        try:
            vectors.append(feature_vector(row))
        except ValueError as exc:
            raise ValueError(f'Feature input row {i + 1}: {exc}') from None
    return np.stack(vectors)
