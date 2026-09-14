"""Portable numeric transforms fitted on development training rows only.

Each CV transform has its own training partition. A separate transform is fitted
on all development rows for final development-trained models. Saved NPZ states
use allow_pickle=False and contain no Python/sklearn objects.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from uuid import uuid4

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.features.reaction_features import FEATURE_NAMES, SCHEMA_HASH, feature_matrix


MODEL_IONS = {'mixed_ions': ('Li', 'Mg', 'Ca', 'Zn', 'Al', 'Y'), 'li_only': ('Li',)}
CONSTANT_TOLERANCE = 1e-12


def save_numeric_archive(path, **arrays):
    """Write atomically, reload every member, then publish the completed file."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid4().hex[:8] + '.tmp')
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError('Saved numeric archive has missing members')
            for key, expected in arrays.items():
                if not np.array_equal(archive[key], np.asarray(expected)):
                    raise ValueError(f'Saved numeric archive verification failed: {key}')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def digest_strings(values):
    return hashlib.sha256(json.dumps(list(map(str, values)), separators=(',', ':')).encode()).hexdigest()


@dataclass
class FeatureData:
    X: np.ndarray
    y: np.ndarray
    row_uid: np.ndarray
    group_id: np.ndarray
    ion: np.ndarray
    fold: np.ndarray
    role: str
    model: str

    def validate(self):
        n = len(self.X)
        if self.X.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(self.X).all():
            raise ValueError('Invalid feature matrix dimensions or nonfinite values')
        if any(array.shape != (n,) for array in (self.y, self.row_uid, self.group_id, self.ion, self.fold)):
            raise ValueError('Feature data arrays have inconsistent row counts')
        if not np.isfinite(self.y).all() or not np.issubdtype(self.fold.dtype, np.integer):
            raise ValueError('Invalid targets or validation folds')
        if len(set(self.row_uid)) != n or any(not str(x) for x in self.row_uid) or any(not str(x) for x in self.group_id):
            raise ValueError('Missing or repeated feature row identities/groups')
        if self.model not in MODEL_IONS or self.role not in ('development', 'holdout', 'held_out_Na', 'held_out_K'):
            raise ValueError('Unknown feature dataset role/model')
        allowed = MODEL_IONS[self.model] if self.role in ('development', 'holdout') else (self.role.rsplit('_', 1)[1],)
        if any(ion not in allowed for ion in self.ion):
            raise ValueError('Unexpected working ion in dataset partition')
        if self.role == 'development':
            if np.any(self.fold < 0):
                raise ValueError('Development rows need CV fold assignments')
            group_folds = {}
            for gid, fold in zip(self.group_id, self.fold):
                if gid in group_folds and group_folds[gid] != fold:
                    raise ValueError('A development group spans multiple validation folds')
                group_folds[gid] = fold
        elif np.any(self.fold != -1):
            raise ValueError('Holdout/transfer rows must not have validation-fold assignments')

    @classmethod
    def from_rows(cls, rows, role, model):
        if model not in MODEL_IONS:
            raise ValueError('Unknown model dataset')
        rows = list(rows)
        if any(row['partition'] != role for row in rows):
            raise ValueError('CSV partition does not match the requested feature role')
        if any(str(row['eligible_for_baseline']) != 'True' for row in rows):
            raise ValueError('Incomplete or quarantined rows cannot enter feature datasets')
        fold_col = 'mixed_cv_fold' if model == 'mixed_ions' else 'li_cv_fold'
        item = cls(
            feature_matrix(rows), np.asarray([float(r['average_voltage_V']) for r in rows], dtype=np.float64),
            np.asarray([r['row_uid'] for r in rows], dtype=str),
            np.asarray([r['group_id'] for r in rows], dtype=str),
            np.asarray([r['working_ion'] for r in rows], dtype=str),
            np.asarray([int(r[fold_col]) if role == 'development' else -1 for r in rows], dtype=np.int64),
            role, model,
        )
        item.validate()
        return item

    def save(self, path):
        self.validate()
        save_numeric_archive(path, X=self.X, y=self.y, row_uid=self.row_uid, group_id=self.group_id,
                            ion=self.ion, fold=self.fold, role=self.role, model=self.model,
                            schema_hash=SCHEMA_HASH, feature_names=np.asarray(FEATURE_NAMES))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive['schema_hash']) != SCHEMA_HASH or tuple(archive['feature_names']) != FEATURE_NAMES:
                raise ValueError('Feature schema or column order does not match this code version')
            result = cls(*(archive[name] for name in ('X', 'y', 'row_uid', 'group_id', 'ion', 'fold')),
                         str(archive['role']), str(archive['model']))
        result.validate()
        return result


@dataclass
class Preprocessor:
    mask: np.ndarray
    raw_mean: np.ndarray
    scale_mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    metadata: dict

    def validate(self):
        active = int(self.mask.sum())
        components = self.metadata.get('n_components')
        if self.metadata.get('schema_hash') != SCHEMA_HASH:
            raise ValueError('Preprocessor feature schema does not match')
        if self.mask.dtype != np.bool_ or self.mask.shape != (len(FEATURE_NAMES),):
            raise ValueError('Invalid preprocessor feature mask')
        if self.raw_mean.shape != self.mask.shape or self.components.shape != (components, active):
            raise ValueError('Invalid preprocessor dimensions')
        if any(a.shape != (active,) for a in (self.scale_mean, self.scale, self.pca_mean)):
            raise ValueError('Invalid scaler dimensions')
        if self.explained_variance_ratio.shape != (components,) or np.any(self.scale <= 0):
            raise ValueError('Invalid PCA variance or scaler scale')
        if not all(np.isfinite(a).all() for a in (self.raw_mean, self.scale_mean, self.scale,
                                                 self.pca_mean, self.components, self.explained_variance_ratio)):
            raise ValueError('Preprocessor state contains nonfinite values')

    def transform_scaled(self, X):
        self.validate()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or not np.isfinite(X).all():
            raise ValueError('Expected a finite matrix in the fixed feature order')
        return (X[:, self.mask] - self.scale_mean) / self.scale

    def transform(self, X):
        transformed = (self.transform_scaled(X) - self.pca_mean) @ self.components.T
        if not np.isfinite(transformed).all():
            raise ValueError('Preprocessing produced nonfinite output')
        return transformed

    def support_diagnostics(self, X):
        X = np.asarray(X, dtype=np.float64)
        # A changed training-constant feature is outside the observed support;
        # these diagnostics are not calibrated prediction uncertainty.
        changed = np.abs(X[:, ~self.mask] - self.raw_mean[~self.mask]) > 1e-9
        names = np.asarray(FEATURE_NAMES)[~self.mask]
        return {'rows_with_changed_training_constant_features': int(np.any(changed, axis=1).sum()),
                'changed_training_constant_feature_names': list(names[np.any(changed, axis=0)])}

    def save(self, path):
        self.validate()
        save_numeric_archive(path, mask=self.mask, raw_mean=self.raw_mean, scale_mean=self.scale_mean,
                            scale=self.scale, pca_mean=self.pca_mean, components=self.components,
                            explained_variance_ratio=self.explained_variance_ratio,
                            metadata_json=json.dumps(self.metadata, sort_keys=True, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            item = cls(*(archive[name] for name in ('mask', 'raw_mean', 'scale_mean', 'scale', 'pca_mean',
                                                    'components', 'explained_variance_ratio')),
                       json.loads(str(archive['metadata_json'])))
        item.validate()
        return item


def fit_preprocessor(development: FeatureData, *, fold=None, n_components=80):
    development.validate()
    if development.role != 'development':
        raise ValueError('Preprocessing may only be fitted using development data')
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError('n_components must be a positive integer')
    if fold is not None and fold not in set(development.fold):
        raise ValueError('Requested validation fold is absent')
    selected = np.ones(len(development.X), dtype=bool) if fold is None else development.fold != fold
    validation = ~selected
    training_groups = set(development.group_id[selected])
    if training_groups & set(development.group_id[validation]):
        raise ValueError('Training and validation share a group')
    X = development.X[selected]
    mask = np.ptp(X, axis=0) > CONSTANT_TOLERANCE
    if min(len(X) - 1, int(mask.sum())) < n_components:
        raise ValueError('Not enough training rows or varying columns for the requested PCA size')
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, svd_solver='full', whiten=False)
    # Limit numerical threads so repeated small CPU fits remain responsive.
    with threadpool_limits(limits=2):
        scaled = scaler.fit_transform(X[:, mask])
        pca.fit(scaled)
    metadata = {
        'schema_hash': SCHEMA_HASH, 'model': development.model,
        'fit_scope': 'all_development' if fold is None else 'cv_training',
        'validation_fold': int(fold) if fold is not None else None,
        'training_rows': int(selected.sum()), 'validation_rows': int(validation.sum()),
        'training_groups': len(training_groups), 'training_validation_group_overlap': 0,
        'training_row_uid_sha256': digest_strings(development.row_uid[selected]),
        'training_features_sha256': hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        'training_group_ids_sha256': digest_strings(sorted(training_groups)),
        'active_columns': int(mask.sum()), 'removed_training_constant_columns': int((~mask).sum()),
        'constant_tolerance': CONSTANT_TOLERANCE, 'n_components': n_components,
        'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()),
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'sklearn_version': sklearn.__version__, 'pca_solver': 'full', 'whiten': False,
        'targets_used_for_fitting': False,
    }
    item = Preprocessor(mask, X.mean(axis=0), scaler.mean_, scaler.scale_, pca.mean_,
                        pca.components_, pca.explained_variance_ratio_, metadata)
    reference = pca.transform(scaled)
    actual = item.transform(X)
    if not np.allclose(actual, reference, atol=1e-9, rtol=1e-9):
        raise ValueError('Portable transform differs from the sklearn reference')
    return item


def prepare_fold(development, fold, saved_preprocessor=None, n_components=80):
    development.validate()
    if development.role != 'development' or fold not in set(development.fold):
        raise ValueError('A development dataset and existing validation fold are required')
    train = development.fold != fold
    valid = ~train
    pp = saved_preprocessor or fit_preprocessor(development, fold=fold, n_components=n_components)
    if (pp.metadata['fit_scope'] != 'cv_training' or pp.metadata['model'] != development.model
            or pp.metadata['validation_fold'] != fold
            or pp.metadata['training_row_uid_sha256'] != digest_strings(development.row_uid[train])
            or pp.metadata['training_features_sha256'] != hashlib.sha256(np.ascontiguousarray(development.X[train]).tobytes()).hexdigest()):
        raise ValueError('Preprocessor was not fitted on this CV training partition')
    return pp.transform(development.X[train]), development.y[train], pp.transform(development.X[valid]), development.y[valid]
