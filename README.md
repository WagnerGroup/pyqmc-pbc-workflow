Start optimization by doing 
```
snakemake -j1 h2_1.4/hf_2_2_2/vdz_0.2/1_0_0_0_2_0_0_0_1/opt_0-0-0_mf_fixed_0_400.chk
```

The form is 
```
{system}/{functional}_{nx}_{ny}_{nz}/{basis}_{exp_to_discard}/{supercell}/{method}_{trialwf}_{orbitals}_{state}_{nconfig}.chk
```
