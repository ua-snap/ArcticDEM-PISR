"""Config loading and data-directory layout."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


class Paths:
    """Standard locations under the data root."""

    def __init__(self, cfg):
        self.root = Path(cfg["data_root"])
        res = cfg["arcticdem"]["resolution_m"]
        ver = cfg["arcticdem"]["version"]
        self.ancillary = self.root / "ancillary"
        self.raw = self.root / "raw" / f"arcticdem_mosaic_v{ver}_{res}m"
        self.dem = self.root / "dem"
        self.pisr = self.root / "pisr"
        self.output = self.root / "output"
        self.qc = self.root / "qc"
        self.figures = self.root / "figures"
        self.logs = self.root / "logs"

        self.aoi_3338 = self.ancillary / "aoi_3338_bbox.gpkg"
        self.aoi_3413 = self.ancillary / "aoi_3413_download.gpkg"
        self.alaska_boundary = self.ancillary / "alaska_boundary_census.gpkg"
        self.target_grid = self.ancillary / "target_grid.json"
        self.tile_list = self.ancillary / f"tile_list_{res}m.csv"
        self.manifest = self.raw / "manifest.csv"

    def mkdirs(self):
        for d in (self.ancillary, self.raw, self.dem, self.pisr, self.output, self.qc,
                  self.figures, self.logs):
            d.mkdir(parents=True, exist_ok=True)
