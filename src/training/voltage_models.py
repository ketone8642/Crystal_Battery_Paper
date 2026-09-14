"""DNN, SVR and KRR regressors, with training-only target transformations."""

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits


def runtime_versions(include_tensorflow=True):
    result = {'python': platform.python_version()}
    for name in ['numpy', 'scipy', 'scikit-learn', 'joblib'] + (['tensorflow'] if include_tensorflow else []):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name != 'tensorflow':
                raise
            result[name] = importlib.metadata.version('tensorflow-cpu')
    return result


def tensorflow():
    # Step 2 selected native Windows CPU execution. Set before importing TF.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    import tensorflow as tf
    return tf


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Metrics require aligned finite one-dimensional arrays')
    if not len(y):
        return {'n': 0, 'mae_V': None, 'rmse_V': None, 'r2': None, 'bias_V': None}
    error = prediction - y
    sst = float(np.sum((y - y.mean()) ** 2))
    return {'n': len(y), 'mae_V': float(np.mean(np.abs(error))),
            'rmse_V': float(np.sqrt(np.mean(error ** 2))),
            'r2': float(1 - np.sum(error ** 2) / sst) if len(y) > 1 and sst > 0 else None,
            'bias_V': float(error.mean())}


def candidates():
    """A small, predeclared search; its quality must be assessed, not assumed."""
    output = []
    for representation in ('pca', 'scaled'):
        for family, settings in (
            ('svr', [{'C': 10.0, 'gamma': 0.1, 'epsilon': 0.1},
                     {'C': 10.0, 'gamma': 'scale', 'epsilon': 0.1},
                     {'C': 100.0, 'gamma': 'scale', 'epsilon': 0.1}]),
            ('krr', [{'alpha': 0.01, 'gamma': 0.1},
                     {'alpha': 0.01, 'gamma': 'scale'},
                     {'alpha': 1.0, 'gamma': 'scale'}]),
            ('dnn', [{'learning_rate': 0.001, 'l2': 0.0001}]),
        ):
            for index, parameters in enumerate(settings):
                output.append({'id': f'{family}_{representation}_{index}', 'family': family,
                               'representation': representation, 'parameters': parameters})
    return output


def represent(preprocessor, X, representation):
    if representation == 'pca':
        return preprocessor.transform(X)
    if representation == 'scaled':
        return preprocessor.transform_scaled(X)
    raise ValueError('Unknown feature representation')


@dataclass
class Regressor:
    family: str
    estimator: object
    offset: float
    target_scale: float
    info: dict

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.info['input_dimensions'] or not np.isfinite(X).all():
            raise ValueError('Invalid regressor input')
        if not len(X):
            return np.empty(0, dtype=float)
        with threadpool_limits(limits=2):
            if self.family == 'dnn':
                parts = [np.asarray(self.estimator(X[i:i+512].astype('float32'), training=False)).reshape(-1)
                         for i in range(0, len(X), 512)]
                predicted = np.concatenate(parts)
            else:
                predicted = self.estimator.predict(X)
        predicted = np.asarray(predicted, dtype=float).reshape(-1) * self.target_scale + self.offset
        if not np.isfinite(predicted).all():
            raise ValueError('Model returned nonfinite predictions')
        return predicted

    def save(self, directory):
        directory = Path(directory)
        if self.family == 'dnn':
            self.estimator.save(directory / 'model.keras')
        else:
            joblib.dump(self.estimator, directory / 'model.joblib', compress=3)
        (directory / 'regressor.json').write_text(json.dumps({
            'family': self.family, 'offset': self.offset, 'target_scale': self.target_scale,
            'info': self.info}, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        info = json.loads((directory / 'regressor.json').read_text(encoding='utf-8'))
        family = info['family']
        if family == 'dnn':
            estimator = tensorflow().keras.models.load_model(directory / 'model.keras', compile=False, safe_mode=True)
        elif family in ('svr', 'krr'):
            # Call only for a trusted local bundle after its file hashes are checked.
            estimator = joblib.load(directory / 'model.joblib')
        else:
            raise ValueError('Unknown regressor family')
        return cls(family, estimator, info['offset'], info['target_scale'], info['info'])


def fit_regressor(candidate, X, y, *, seed=2026, epochs=100, batch_size=64, progress=None):
    """Only training arrays are accepted. No validation or holdout argument exists."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or len(X) < 2 or X.shape[1] < 1:
        raise ValueError('Invalid training matrix or target shape')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Training inputs must be finite')
    family, parameters = candidate['family'], dict(candidate['parameters'])
    offset, target_scale = 0.0, 1.0
    info = {'input_dimensions': X.shape[1], 'training_rows': len(X), 'seed': seed,
            'parameters': parameters, 'target_transform': 'none'}
    if family in ('svr', 'krr'):
        gamma = parameters.pop('gamma')
        if gamma == 'scale':
            variance = float(X.var())
            gamma = 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        if not isinstance(gamma, (float, int)) or gamma <= 0:
            raise ValueError('Invalid RBF gamma')
        info['resolved_gamma'] = gamma
        if family == 'svr':
            estimator = SVR(kernel='rbf', gamma=gamma, cache_size=256, **parameters)
        else:
            offset = float(y.mean())
            info['target_transform'] = 'subtract_training_mean'
            estimator = KernelRidge(kernel='rbf', gamma=gamma, **parameters)
        with threadpool_limits(limits=2):
            estimator.fit(X, y - offset)
    elif family == 'dnn':
        if epochs < 1 or batch_size < 1:
            raise ValueError('Epochs and batch size must be positive')
        tf = tensorflow()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        offset = float(y.mean())
        target_scale = float(y.std())
        if target_scale < 1e-12:
            target_scale = 1.0
        regularizer = tf.keras.regularizers.L2(parameters['l2'])
        estimator = tf.keras.Sequential([
            tf.keras.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(60, activation='relu', kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(1),
        ])
        estimator.compile(optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=parameters['learning_rate'], rho=0.9, momentum=0.0, epsilon=1e-7, centered=False),
            loss='mean_squared_error', jit_compile=False)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 2
        options.experimental_deterministic = True
        dataset = tf.data.Dataset.from_tensor_slices((X.astype('float32'),
            ((y - offset) / target_scale).astype('float32')[:, None]))
        dataset = dataset.shuffle(len(y), seed=seed, reshuffle_each_iteration=True).batch(batch_size).with_options(options)

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                loss = float(logs['loss'])
                if not np.isfinite(loss):
                    raise ValueError('DNN training loss became nonfinite')
                if progress and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs):
                    progress(f'epoch {epoch + 1}/{epochs}; training objective={loss:.6f}')

        history = estimator.fit(dataset, epochs=epochs, verbose=0, shuffle=False, callbacks=[Progress()])
        info.update({'target_transform': 'standardize_using_training_mean_and_std',
                     'epochs': epochs, 'batch_size': batch_size,
                     'training_objective': 'standardized_target_MSE_plus_L2',
                     'loss_history': list(map(float, history.history['loss'])),
                     'early_stopping': False})
    else:
        raise ValueError('Unknown model family')
    return Regressor(family, estimator, offset, target_scale, info)
