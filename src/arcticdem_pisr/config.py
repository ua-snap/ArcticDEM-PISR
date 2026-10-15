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
        self.dem_3338 = self.dem / f"arcticdem_v{ver}_{res}m_dem_3338.tif"            # warped, voids as NaN
        self.dem_3338_filled = self.dem / f"arcticdem_v{ver}_{res}m_dem_3338_filled.tif"  # small voids filled
        self.dem_fill_mask = self.dem / f"arcticdem_v{ver}_{res}m_dem_3338_fillmask.tif"   # 1 = interpolated
        self.datamask_3338 = self.dem / f"arcticdem_v{ver}_{res}m_datamask_3338.tif"     # ArcticDEM datamask, nearest
        self.water_mask = self.dem / f"arcticdem_v{ver}_{res}m_water_3338.tif"           # 1 = water (PISR NoData)
        self.pisr_3338 = self.pisr / f"pisr_equinox_{res}m_3338.tif"                     # final, water masked
        self.pisr_3338_unmasked = self.pisr / f"pisr_equinox_{res}m_3338_unmasked.tif"   # before water masking
        self.tii_3338 = self.pisr / f"terrain_insolation_index_{res}m_3338.tif"         # TII, water masked

    def mkdirs(self):
        for d in (self.ancillary, self.raw, self.dem, self.pisr, self.output, self.qc,
                  self.figures, self.logs):
            d.mkdir(parents=True, exist_ok=True)
