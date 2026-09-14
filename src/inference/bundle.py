"""Load a trusted local Step 6 bundle and predict with the shared features."""

from pathlib import Path

from src.features.preprocessing import Preprocessor
from src.features.reaction_features import SCHEMA_HASH, feature_matrix
from src.training.io_utils import read_json, verify_files
from src.training.voltage_models import Regressor, represent, runtime_versions


class VoltageBundle:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory / 'bundle.json')
        if self.metadata.get('status') != 'complete' or self.metadata.get('schema_hash') != SCHEMA_HASH:
            raise ValueError('Incomplete bundle or incompatible feature schema')
        verify_files(self.directory, self.metadata['files_sha256'])
        self.preprocessor = Preprocessor.load(self.directory / 'preprocessor.npz')
        self.candidate = self.metadata['candidate']
        if self.preprocessor.metadata['fit_scope'] != 'all_development':
            raise ValueError('Final bundle needs a development-only preprocessor')
        if self.candidate['family'] in ('svr', 'krr'):
            current = runtime_versions(False)
            if current['scikit-learn'] != self.metadata['runtime']['scikit-learn']:
                raise ValueError('Use the scikit-learn version recorded in bundle.json to load this model')
        self.regressor = Regressor.load(self.directory)

    def predict_matrix(self, X):
        return self.regressor.predict(represent(self.preprocessor, X, self.candidate['representation']))

    def predict(self, reactions):
        """Return voltages in the input order. No confidence interval is implied."""
        return self.predict_matrix(feature_matrix(reactions))
