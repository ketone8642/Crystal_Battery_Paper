"""Validated request/response contracts for voltage inference."""

from typing import Annotated, Literal
import math

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from src.data.chemistry import host_and_loading
from src.features.reaction_features import feature_vector


Ion = Literal['Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Al', 'Y']
Family = Literal['auto', 'dnn', 'svr', 'krr']
Population = Literal['auto', 'li_only', 'mixed_ions']
CrystalSystem = Literal['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
Formula = Annotated[str, Field(min_length=1, max_length=256)]
SpaceGroup = Annotated[StrictInt, Field(ge=1, le=230)]
EXAMPLE = {
    'working_ion': 'Li', 'formula_charge': 'FePO4', 'formula_discharge': 'LiFePO4',
    'crystal_system_charge': 'orthorhombic', 'spacegroup_number_charge': 62,
    'crystal_system_discharge': 'orthorhombic', 'spacegroup_number_discharge': 62,
    'model_family': 'auto', 'training_population': 'auto',
}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False,
                              json_schema_extra={'examples': [EXAMPLE]})
    working_ion: Ion
    formula_charge: Formula
    formula_discharge: Formula
    crystal_system_charge: CrystalSystem
    spacegroup_number_charge: SpaceGroup
    crystal_system_discharge: CrystalSystem
    spacegroup_number_discharge: SpaceGroup
    fracA_charge: Annotated[float | None, Field(ge=0, lt=1)] = None
    fracA_discharge: Annotated[float | None, Field(ge=0, lt=1)] = None
    model_family: Family = 'auto'
    training_population: Population = 'auto'

    @field_validator('formula_charge', 'formula_discharge', 'crystal_system_charge', 'crystal_system_discharge', mode='before')
    @classmethod
    def trim_strings(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator('fracA_charge', 'fracA_discharge', mode='before')
    @classmethod
    def fraction_number(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))):
            raise ValueError('Atom fraction must be a JSON number, not text or a boolean')
        return value

    @model_validator(mode='after')
    def valid_reaction(self):
        feature_vector(self.model_dump())
        if self.training_population == 'li_only' and self.working_ion not in ('Li', 'Na', 'K'):
            raise ValueError('Li-only routing is supported for Li, Na and K; use mixed_ions for this ion')
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    items: Annotated[list[PredictionRequest], Field(min_length=1, max_length=128)]


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    intervals: Annotated[list[PredictionRequest], Field(min_length=2, max_length=50)]

    @model_validator(mode='after')
    def contiguous_profile(self):
        first = self.intervals[0]
        host, _, _ = host_and_loading(first.formula_charge, first.working_ion)
        previous = None
        for index, interval in enumerate(self.intervals):
            if interval.working_ion != first.working_ion:
                raise ValueError('All profile intervals must use the same working ion')
            if (interval.model_family, interval.training_population) != (first.model_family, first.training_population):
                raise ValueError('Use the same model choices for every profile interval')
            current_host, start, _ = host_and_loading(interval.formula_charge, interval.working_ion)
            if current_host != host:
                raise ValueError('All profile intervals must have the same reduced host')
            if previous is not None:
                _, end, _ = host_and_loading(previous.formula_discharge, previous.working_ion)
                if not math.isclose(start, end, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f'Interval {index + 1} must start where the previous interval ends; gaps and overlaps are not allowed')
                if (previous.spacegroup_number_discharge, previous.crystal_system_discharge) != (interval.spacegroup_number_charge, interval.crystal_system_charge):
                    raise ValueError('Shared profile endpoints must have matching symmetry metadata')
            previous = interval
        return self


class ModelReference(BaseModel):
    family: str
    training_population: str
    candidate_id: str
    representation: str
    training_rows: int
    training_ions: list[str]
    cv_mean_mae_V: float


class IntervalReference(BaseModel):
    formula_charge: str
    formula_discharge: str
    fracA_charge: float
    fracA_discharge: float
    ion_per_host_charge: float
    ion_per_host_discharge: float
    loading_convention: str = 'working-ion atoms per reduced non-working-ion host'


class PredictionResult(BaseModel):
    run_id: str
    working_ion: str
    predicted_voltage_V: float
    interval: IntervalReference
    model: ModelReference
    evaluation_reference: dict
    extrapolates_working_ion: bool
    changed_training_constant_features: int
    warnings: list[str]


class BatchResult(BaseModel):
    run_id: str
    count: int
    predictions: list[PredictionResult]


class ProfileResult(BatchResult):
    profile_type: str = 'average voltage over each contiguous insertion interval'
    x_axis: str = 'working-ion atoms per reduced host'
