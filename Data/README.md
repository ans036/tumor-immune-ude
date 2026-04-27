# Data Source Attribution

The two CSV files in this folder are obtained **unmodified** from the GitHub repository accompanying Atsou et al.'s tumor-growth modeling work:

- **Source repository:** https://github.com/kevin-atsou/tumorgrowth/tree/main/data
- **License:** Creative Commons Zero v1.0 Universal (CC0)
- **Files:**
  - `tumor_time_to_event_data.csv` — longitudinal tumor-volume + immune-cell measurements
  - `tumor_volume_vs_Im_cells_rate.csv` — paired tumor-volume vs. immune killing-rate measurements

The data were generated from in vivo experiments using the mSCC38 squamous cell carcinoma cell line in FVB/N wild-type mice. Detailed experimental protocols are described in Atsou et al. (2022).

## Citation

If you use this data, please cite the original sources:

**Atsou, K., Anjuère, F., Braud, V. M., & Goudon, T. (2021).** A size and space structured model of tumor growth describes a key role for protumor immune cells in breaking equilibrium states in tumorigenesis. *PLOS ONE*, 16(11), e0259291. https://doi.org/10.1371/journal.pone.0259291

**Atsou, K., Khou, S., Anjuère, F., Braud, V. M., & Goudon, T. (2022).** Analysis of the equilibrium phase in immune-controlled tumors provides hints for designing better strategies for cancer treatment. *Frontiers in Oncology*, 12, 878827. https://doi.org/10.3389/fonc.2022.878827

## Column structure

### `tumor_time_to_event_data.csv`

The original CSV uses generic column names (`A,B,C,D,E`). They correspond to:

| Column | Meaning |
|---|---|
| A (KineticID) | Immune-cell condition: `C2`, `C3`, `C4` |
| B (TumorID) | Tumor replicate / scenario: `T1`–`T5` |
| C (Time) | Observation day |
| D (TumorVolume) | Tumor volume (mm³) |
| E (ImmuneCellCount) | Concurrent immune cell measurement |

### `tumor_volume_vs_Im_cells_rate.csv`

Headerless, two columns: `TumorVolume, Im_cells_rate` — paired measurements used for physics-informed rank-correlation constraints in our UDE training.
