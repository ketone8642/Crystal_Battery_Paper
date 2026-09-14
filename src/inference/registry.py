"""Load one completed local experiment; preserve its CV choices and provenance."""

import math
from pathlib import Path
import re
from threading import RLock

import numpy as np

from src.data.chemistry import host_and_loading
from src.features.reaction_features import SCHEMA_HASH, feature_matrix
from src.inference.bundle import VoltageBundle
from src.training.io_utils import read_json, sha, verify_files, write_json


POPULATIONS = ('mixed_ions', 'li_only')
FAMILIES = ('svr', 'krr', 'dnn')
CONFIG_RELATIVE = Path('config') / 'active_model.json'


def valid_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'step6_[0-9a-f]{12}', run_id):
        raise ValueError('Use a completed Step 6 run name such as step6_b8e6065bfc4e')


class ModelRegistry:
    def __init__(self, project_root, run_id, *, allow_test_artifacts=False):
        valid_run_id(run_id)
        self.root, self.run_id = Path(project_root).resolve(), run_id
        self.lock = RLock()
        self.bundles = {}
        report_dir = self.root / 'reports' / run_id
        manifest_path = report_dir / 'step6_manifest.json'
        manifest = read_json(manifest_path)
        if manifest.get('status') != 'complete':
            raise ValueError('Step 6 is incomplete; model configuration requires a completed run')
        purpose = manifest['configuration'].get('purpose')
        if purpose != 'research' and not (allow_test_artifacts and purpose == 'synthetic_smoke'):
            raise ValueError('Synthetic verification artifacts cannot serve as research models')
        self.synthetic = purpose == 'synthetic_smoke'
        self.manifest_sha256 = sha(manifest_path)
        mapping = manifest['output_sha256']

        def tracked(path):
            relative = path.relative_to(self.root).as_posix()
            if relative not in mapping:
                raise ValueError(f'Required serving file is absent from the completed manifest: {relative}')
            verify_files(self.root, {relative: mapping[relative]})

        tracked(report_dir / 'step6_report.json')
        tracked(report_dir / 'selection.json')
        self.report = read_json(report_dir / 'step6_report.json')
        selection = read_json(report_dir / 'selection.json')
        if self.report['run_id'] != run_id or self.report['configuration'] != manifest['configuration']:
            raise ValueError('The report and completed run configuration do not match')
        if selection['selection_uses_holdout_or_transfer'] is not False:
            raise ValueError('Expected model selection based only on development CV')
        if self.report['selection_sha256_before_evaluation'] != sha(report_dir / 'selection.json'):
            raise ValueError('Frozen selection does not match the evaluated selection')
        self.defaults = selection['cv_selected_family']
        for population in POPULATIONS:
            info = self.report['populations'].get(population)
            if not info or set(info['bundles']) != set(FAMILIES):
                raise ValueError('Step 7 requires all three final models for both populations')
            if self.defaults.get(population) not in FAMILIES or info['cv_selected_family'] != self.defaults[population]:
                raise ValueError('Inconsistent CV-selected model family')
            for family in FAMILIES:
                directory = self.root / 'models' / population / run_id / family
                if info['bundles'][family] != directory.relative_to(self.root).as_posix():
                    raise ValueError('Unexpected model bundle location in the report')
                tracked(directory / 'bundle.json')
                metadata = read_json(directory / 'bundle.json')
                for relative in metadata['files_sha256']:
                    path = (directory / relative).resolve()
                    if not path.is_relative_to(directory.resolve()):
                        raise ValueError('A model bundle file points outside its directory')
                    tracked(path)
                candidate = info['selected'][family]['candidate']
                if metadata['candidate'] != candidate or selection['selected'][population][family] != candidate:
                    raise ValueError('The bundle is not the candidate frozen during selection')
                if metadata['population'] != population or metadata['schema_hash'] != SCHEMA_HASH:
                    raise ValueError('Bundle population or feature schema does not match')
                if candidate['family'] != family:
                    raise ValueError('The reported candidate family does not match its bundle')
                bundle = VoltageBundle(directory)
                if bundle.preprocessor.metadata['model'] != population or bundle.regressor.family != family:
                    raise ValueError('Loaded model population/family does not match the report')
                self.bundles[(population, family)] = bundle

    @classmethod
    def from_config(cls, project_root, *, allow_test_artifacts=False):
        root = Path(project_root)
        config = read_json(root / CONFIG_RELATIVE)
        valid_run_id(config['step6_run'])
        manifest = root / 'reports' / config['step6_run'] / 'step6_manifest.json'
        if sha(manifest) != config['step6_manifest_sha256']:
            raise ValueError('The configured training run changed after API configuration')
        return cls(root, config['step6_run'], allow_test_artifacts=allow_test_artifacts)

    def model_reference(self, population, family):
        bundle = self.bundles[(population, family)]
        return {'family': family, 'training_population': population,
                'candidate_id': bundle.candidate['id'], 'representation': bundle.candidate['representation'],
                'training_rows': bundle.metadata['training_rows'], 'training_ions': bundle.metadata['training_ions'],
                'cv_mean_mae_V': self.report['populations'][population]['selected'][family]['mean_fold_mae_V']}

    def catalog(self):
        cards = []
        for (population, family), bundle in self.bundles.items():
            evaluation = self.report['populations'][population]['evaluation'][family]
            cards.append({**self.model_reference(population, family),
                          'selected_by_cv': self.defaults[population] == family,
                          'holdout': evaluation['holdout'], 'held_out_Na': evaluation['held_out_Na'],
                          'held_out_K': evaluation['held_out_K']})
        return {'run_id': self.run_id, 'models': cards, 'default_family_by_population': self.defaults,
                'median_baseline_by_population': {p: self.report['populations'][p]['evaluation'].get('median_baseline', {}) for p in POPULATIONS},
                'default_population_by_ion': {ion: ('li_only' if ion in ('Li', 'Na', 'K') else 'mixed_ions')
                    for ion in ('Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Al', 'Y')},
                'routing_basis': 'Population routing was specified before testing; family choice uses development CV.',
                'evaluation_note': 'Dataset-level errors are not confidence intervals for individual predictions.',
                'data_source': 'synthetic verification only' if self.synthetic else 'Materials Project computed insertion voltages'}

    def route(self, request):
        population = request.training_population
        if population == 'auto':
            population = 'li_only' if request.working_ion in ('Li', 'Na', 'K') else 'mixed_ions'
        if population == 'li_only' and request.working_ion not in ('Li', 'Na', 'K'):
            raise ValueError('Li-only routing is supported only for Li, Na and K')
        family = self.defaults[population] if request.model_family == 'auto' else request.model_family
        return population, family

    def predict(self, requests):
        if not requests or len(requests) > 128:
            raise ValueError('Provide between 1 and 128 reactions')
        raw = [request.model_dump() for request in requests]
        X = feature_matrix(raw)
        grouped = {}
        for index, request in enumerate(requests):
            grouped.setdefault(self.route(request), []).append(index)
        output = [None] * len(requests)
        # Shared TF/sklearn models execute sequentially in this local CPU app.
        with self.lock:
            for (population, family), indices in grouped.items():
                bundle = self.bundles[(population, family)]
                voltages = bundle.predict_matrix(X[indices])
                for index, voltage in zip(indices, voltages):
                    if not math.isfinite(float(voltage)):
                        raise ValueError('The trained model returned a nonfinite voltage')
                    request = requests[index]
                    _, low, frac_low = host_and_loading(request.formula_charge, request.working_ion)
                    _, high, frac_high = host_and_loading(request.formula_discharge, request.working_ion)
                    changed = np.abs(X[index, ~bundle.preprocessor.mask] - bundle.preprocessor.raw_mean[~bundle.preprocessor.mask]) > 1e-9
                    extrapolates = request.working_ion not in bundle.metadata['training_ions']
                    warnings = ['Prediction is based on computed reference voltages; it does not establish electrode stability or experimental performance.',
                                'A calibrated uncertainty interval is not available.']
                    if extrapolates:
                        warnings.append(f'{request.working_ion} was absent from this model\'s training ions. This is a transfer prediction without fine-tuning.')
                    if np.any(changed):
                        warnings.append('Some input descriptors differ from values that were constant during training; the model could not learn their effects.')
                    if voltage < 0:
                        warnings.append('The predicted voltage is negative and is reported without clipping.')
                    if voltage < -3 or voltage > 10:
                        warnings.append('The predicted voltage crosses the project\'s review bounds; check the inputs and reference calculations.')
                    if self.synthetic:
                        warnings.append('SYNTHETIC SOFTWARE TEST ONLY: this model is not a research result.')
                    evaluation = self.report['populations'][population]['evaluation'][family]
                    if request.working_ion in ('Na', 'K'):
                        reference = evaluation['held_out_' + request.working_ion]
                        reference = {'scope': f'held_out_{request.working_ion}_same_database', 'all': reference['all'],
                                     'known_development_group': reference.get('known_development_group'),
                                     'new_development_group': reference.get('new_development_group')}
                    else:
                        reference = {'scope': population + '_grouped_holdout', 'all': evaluation['holdout']['all'],
                                     'requested_ion': evaluation['holdout']['per_ion'].get(request.working_ion)}
                    output[index] = {'run_id': self.run_id, 'working_ion': request.working_ion,
                        'predicted_voltage_V': float(voltage),
                        'interval': {'formula_charge': request.formula_charge, 'formula_discharge': request.formula_discharge,
                                     'fracA_charge': frac_low, 'fracA_discharge': frac_high,
                                     'ion_per_host_charge': low, 'ion_per_host_discharge': high},
                        'model': self.model_reference(population, family), 'evaluation_reference': reference,
                        'extrapolates_working_ion': extrapolates,
                        'changed_training_constant_features': int(changed.sum()), 'warnings': warnings}
        return output


def configure_run(project_root, run_id, *, allow_test_artifacts=False):
    root = Path(project_root).resolve()
    valid_run_id(run_id)
    manifest_path = root / 'reports' / run_id / 'step6_manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('status') != 'complete':
        raise ValueError('Complete Step 6 before configuring the prediction API')
    verify_files(root, manifest['output_sha256'])
    registry = ModelRegistry(root, run_id, allow_test_artifacts=allow_test_artifacts)
    from app.schemas import EXAMPLE, PredictionRequest
    # Exercise the real shared feature generator and every deserialized model.
    probes = [PredictionRequest(**{**EXAMPLE, 'training_population': p, 'model_family': f})
              for p in POPULATIONS for f in FAMILIES]
    registry.predict(probes)
    config_path = root / CONFIG_RELATIVE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, {'step6_run': run_id, 'step6_manifest_sha256': sha(manifest_path)})
    return registry
