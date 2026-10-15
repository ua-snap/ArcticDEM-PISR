# ArcticDEM-PISR

Potential incoming solar radiation (PISR) for Alaska, computed from the ArcticDEM mosaic and used as a
combined slope/aspect/latitude covariate in further modeling.

**Product:** daily clear-sky total insolation (direct + diffuse, kWh m⁻² day⁻¹) for an idealized equinox day,
on a 32 m EPSG:3338 (Alaska Albers) grid covering Alaska's bounding box, delivered as a CF-1.8 netCDF
(`pisr`, `terrain_insolation_index`, `elevation`, float32) for a Rasdaman coverage:
`/beegfs/CMIP6/jdpaul3/ArcticDEM-PISR_data/output/arcticdem_v4.1_pisr_equinox_32m_3338.nc`

**Terrain insolation index (TII)** = PISR computed as if every cell were at 64°N (true slope and aspect kept)
minus PISR of flat ground at 64°N. 0 = like flat ground, > 0 = warm (sun-facing) slope, < 0 = cold slope. The
same slope and aspect give the same value anywhere, so TII is a latitude- and elevation-free slope/aspect variable
for composite indices that already include temperature or permafrost layers.

Team-facing summary with figures and QC: [docs/REPORT.md](docs/REPORT.md).

## Method summary

| Item | Choice |
|---|---|
| Source DEM | ArcticDEM mosaic v4.1, 32 m (PGC; AWS Open Data COGs), EPSG:3413 |
| Grid | DEM warped (cubic) to 32 m EPSG:3338, pixel edges on multiples of 32 m, bounds = Alaska (Census) bbox |
| DEM voids | Enclosed voids ≤ 1000 cells (~1 km²) filled by inverse-distance interpolation; larger voids stay NoData |
| Water | PISR and TII set to NoData over flattened water (ArcticDEM `datamask = 0` and slope < 0.05°) |
| TII | Same model with latitude fixed at 64°N (`config.yaml: tii.ref_lat_deg`), minus flat ground at 64°N |
| Day | Idealized equinox: solar declination 0, mean Earth–Sun distance (March and September identical) |
| Shading | Local slope/aspect self-shading only. No cast shadows from surrounding terrain |
| Atmosphere | Lumped clear-sky transmittance τ = 0.70, Kasten & Young (1989) air mass, no elevation correction |
| Direct | I₀ τᵐ cos i, I₀ = 1361 W m⁻² |
| Diffuse | I₀ sin h (0.271 − 0.294 τᵐ) (1 + cos β)/2 (Liu & Jordan 1960, isotropic sky) |
| Integration | 5-minute steps from sunrise to sunset |
| Terrain | Horn (1981) gradients, converted to **true** east/north gradients with the local projection Jacobian |

**Why the Jacobian matters.** Grid north is not true north. Over Alaska the gap is 85–140° in EPSG:3413
and up to ~30° in EPSG:3338. Albers is also not conformal (up to ~3° angular distortion on the North Slope)
and has ±2.5% scale distortion. The model converts every grid gradient to a true-frame gradient, so aspect
effects are correct anywhere in the grid. Tests confirm this against an independent pvlib calculation
(max relative difference 3×10⁻⁵ across 84 slope/aspect/site/projection cases).

## Repository layout

```
config.yaml                  paths and parameters
environment.yml              micromamba env "arcticdem-pisr" (environment.lock.yml = exact versions)
src/arcticdem_pisr/          package: config.py, solar.py (numba PISR kernel)
scripts/01_download/         make_aoi.py, select_tiles.py, download_tiles.py, validate_tiles.py, coverage_map.py
scripts/02_pisr/             warp_dem.py, fill_voids.py, run_pisr.py, mask_water.py, compare_grids.py
scripts/03_post/             qc.py, figures.py, to_netcdf.py
slurm/                       Slurm wrappers for each stage (Chinook)
tests/                       pytest: analytic checks + pvlib cross-checks
```

## Running

Always run through the env (activation sets `PROJ_DATA`; without it GDAL can't find `proj.db`):

```bash
micromamba create -f environment.yml
pip install -e . --no-deps                      # inside the env
R="micromamba run -n arcticdem-pisr python -I"

# Phase 1: AOI, tile list, download (internet needed: login/transfer node)
$R scripts/01_download/make_aoi.py
$R scripts/01_download/select_tiles.py
sbatch slurm/01_download.slurm                 # download + validate + VRTs

# Phase 2: warp DEM to 3338, compute PISR
sbatch slurm/02a_warp_dem.slurm
sbatch slurm/02b_fill_voids.slurm
sbatch slurm/02c_run_pisr.slurm                # restartable
sbatch slurm/02d_mask_water.slurm
sbatch slurm/02c_run_pisr.slurm --product tii   # terrain insolation index (uses the water mask)

# Phase 3: QC, figures, netCDF
sbatch slurm/03_post.slurm

# Tests
micromamba run -n arcticdem-pisr python -I -m pytest tests/ -o pythonpath=tests
```

Data live under `data_root` in `config.yaml` (`ancillary/`, `raw/`, `dem/`, `pisr/`, `output/`, `qc/`,
`figures/`, `logs/`).

## Notes and limitations

- Neighbouring ArcticDEM tiles overlap by 96 m, and each tile's outermost pixel ring is an edge artifact
  (differences up to ~5 m). Tiles are cropped to their nominal 100 km footprints before mosaicking, which
  removes seams.
- The extent is the full Alaska bounding box, so it includes parts of Yukon, British Columbia, and Chukotka
  where ArcticDEM has data. Areas without ArcticDEM coverage (e.g. BC south of 60°N, open ocean) are NoData.
- Cells next to DEM NoData are NoData (the 3×3 slope window must be complete).
- Water is NoData in `pisr`. ArcticDEM v4.1 mosaics flatten lakes and sea and flag them `datamask = 0`, but
  `datamask = 0` also covers DEM voids and some PGC-filled land. So water = `datamask = 0` AND valid in the unfilled
  DEM AND slope < 0.05° (`config.yaml: water_mask`). Mask: `dem/arcticdem_v4.1_32m_water_3338.tif`. The
  `elevation` variable is not masked.
- Filled voids are smooth surfaces, so their PISR is closer to the flat-ground value. Interpolated cells are
  flagged in `dem/arcticdem_v4.1_32m_dem_3338_fillmask.tif`.
- PISR is a clear-sky potential; it ignores clouds, cast shadows, and ground reflection by design.

## Citation and acknowledgment (source data)

Porter, C., Howat, I., Noh, M.J., Husby, E., Khuvis, S., Danish, E., Tomko, K., Gardiner, J., Negrete, A.,
Yadav, B., Klassen, J., Kelleher, C., Cloutier, M., Bakker, J., Enos, J., Arnold, G., Bauer, G., and Morin, P.,
2023, "ArcticDEM – Mosaics, Version 4.1", https://doi.org/10.7910/DVN/3VDC4W, Harvard Dataverse, V1.

DEMs provided by the Polar Geospatial Center under NSF-OPP awards 1043681, 1559691, 1542736, 1810976, and 2129685.
