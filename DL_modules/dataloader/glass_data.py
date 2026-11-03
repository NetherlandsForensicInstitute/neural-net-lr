from pathlib import Path

import numpy as np
from lrbenchmark.data.models import DataSet


class GlassData(DataSet):
    """
    LA-ICP-MS measurements of elemental concentration from floatglass.

    Data are read from disk. This class expects to find two files:
    - aata.csv contains semicolon-separated measurement values
    - labels.csv contains the labels

    TODO: refer to README or add more info here?
    """

    def __init__(self, resource_dir: str):
        self.resource_dir = Path(resource_dir)

    def _get_source_description(self, rownum: int) -> str:
        return f"{self.resource_dir}/data.csv:{rownum}"

    def get_instances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return glass measurement data as a tuple of features, labels and meta-data.

        The features values are elemental concentrations on a log_10 basis, and normalized to Si.

        The labels are unique identifiers of a glass particle. Each particle is from a different reference window. An
        instance is a replicate measurement on a glass particle.

        The meta values of an instance are a concatenation of the filename and a row number, e.g. "training.csv:22".
        """
        features = np.loadtxt(self.resource_dir / "data.csv", delimiter=";")
        labels = np.loadtxt(self.resource_dir / "labels.csv").astype(int)
        meta = np.vectorize(self._get_source_description)(np.arange(len(labels)))
        return features, labels, meta
