import functions
import concurrent
import numpy as np
import pyqmc
qmc_threads=2
import json

#The sequence of numbers of configurations to use in optimization. 
#You should check that the energy does not change as you increase this 
#number.
nconfigs = [400,1600,3200,6400,12800,25600] 

rule MEAN_FIELD:
    input: "{dir}/system.json"
    output: "{dir}/{functional}_{nx}_{ny}_{nz}/{basis}_{exp_to_discard}/mf.chk"
    resources:
        walltime="4:00:00", partition="qmchamm"
    run:
        kmesh = (int(wildcards.nx),int(wildcards.ny), int(wildcards.nz))
        functions.mean_field(output[0],kmesh=kmesh, exp_to_discard=float(wildcards.exp_to_discard), settings=json.load(open(input[0])), basis=wildcards.basis, functional=wildcards.functional)


def opt_dependency(wildcards):
    d={}
    basedir = f"{wildcards.dir}/"
    nconfig = int(wildcards.nconfig)
    ind = nconfigs.index(nconfig)
    if hasattr(wildcards,'hci_tol'):
        startingwf = f'hci{wildcards.hci_tol}'
    else:
        startingwf = "mf"

    if hasattr(wildcards, 'hci_tol'):
        basefile = basedir+f"opt_hci{wildcards.hci_tol}_{wildcards.determinant_cutoff}_{wildcards.orbitals}_"
    else: 
        basefile = basedir+f"opt_mf_{wildcards.orbitals}_"

    if ind > 0:
        d['start_from'] = basefile+f"{wildcards.statenumber}_{nconfigs[ind-1]}.chk"
    for i in range(int(wildcards.statenumber)):
        d[f'anchor_wf{i}'] = basefile+f"{i}_{nconfigs[-1]}.chk"
    return d

def convert_superdir(superdir):
    return np.array([int(x) for x in superdir.split('_')]).reshape(3,3)

rule OPTIMIZE_MF:
    input: unpack(opt_dependency), mf = "{dir}/mf.chk"
    output: "{dir}/{superdir}/opt_mf_{orbitals}_{statenumber}_{nconfig}.chk"
    resources:
        walltime="72:00:00", partition="qmchamm"
    run:
        n = int(wildcards.statenumber)
        start_from = None
        if hasattr(input, 'start_from'):
            start_from=input.start_from
        if wildcards.orbitals=='orbitals':
            slater_kws={'optimize_orbitals':True}
        elif wildcards.orbitals=='fixed':
            slater_kws={'optimize_orbitals':False}
        elif wildcards.orbitals=='large':
            slater_kws={'optimize_orbitals':True, 'optimize_zeros':False}
        else:
            raise Exception("Did not expect",wildcards.orbitals)
        S = convert_superdir(wildcards.superdir)
        if n==0:
            anchor_wfs=None
        else: 
            anchor_wfs = [input[f'anchor_wf{i}'] for i in range(n)]            

        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyqmc.OPTIMIZE(input.mf, output[0], anchors = anchor_wfs, start_from=start_from, nconfig=int(wildcards.nconfig), slater_kws=slater_kws, 
                            S=S, client=client, npartitions=qmc_threads)



rule VMC:
    input: mf = "{dir}/mf.chk", opt = "{dir}/{superdir}/opt_{variables}.chk"
    output: "{dir}/{superdir}/vmc_{variables}.chk"
    threads: qmc_threads
    resources:
        walltime="24:00:00", partition="qmchamm"
    run:
        S = convert_superdir(wildcards.superdir)
        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyqmc.VMC(input.mf, output[0], start_from=input.opt, nconfig=8000, client=client, npartitions=qmc_threads, S=S, vmc_kws=dict(nblocks=80))


rule DMC:
    input: mf = "{dir}/mf.chk", opt = "{dir}/{superdir}/opt_{variables}.chk"
    output: "{dir}/{superdir}/dmc_{variables}_{tstep}.chk"
    threads: qmc_threads
    resources:
        walltime="24:00:00", partition="qmchamm"
    run:
        multideterminant = None
        startingwf = input.opt.split('/')[-1].split('_')[1]
        if 'hci' in startingwf:
            multideterminant = wildcards.dir+"/"+startingwf+".chk"
        tstep = float(wildcards.tstep)
        nsteps = int(30/tstep)
        S = convert_superdir(wildcards.superdir)
        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyqmc.DMC(input.mf,  output[0], S=S, start_from=input.opt, dmc_kws=dict(tstep=tstep, nsteps=nsteps), nconfig=8000, client=client, npartitions=qmc_threads)
