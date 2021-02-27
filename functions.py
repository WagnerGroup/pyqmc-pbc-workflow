import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pyscf
import pyqmc
import h5py
from functools import partial

def save_scf_iteration(chkfile, envs):
    cycle = envs['cycle']
    info = {'mo_energy':envs['mo_energy'],
            'e_tot'   : envs['e_tot']}
    pyscf.scf.chkfile.save(chkfile, 'iteration/%d' % cycle, info)



def mean_field(chkfile, functional, kmesh=(2,2,2),exp_to_discard=0.2, basis='vdz', settings=None):
    import pyscf.pbc.scf as scf
    #mol = pyscf.pbc.gto.M(basis=f'ccecpccp{basis}', ecp='ccecp', unit='bohr', **settings)
    mol = pyscf.pbc.gto.M(basis=f'bfd{basis}', ecp='bfd_pp', unit='bohr', **settings)
    kpts = mol.make_kpts(kmesh)
    mol.exp_to_discard=exp_to_discard
    mol.build()

    if functional=='hf':
        mf = scf.KROHF(mol, kpts)
    elif functional=='uhf':
        mf = scf.KUHF(mol, kpts)
    else:
        mf = scf.KROKS(mol, kpts)
        mf.xc=functional
    mf.callback = partial(save_scf_iteration,chkfile)
    mf.chkfile=chkfile
    mf.kernel()

    if functional=='uhf':
        mo1 = mf.stability()[0]
        rdm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf.chkfile=chkfile
        mf.callback = partial(save_scf_iteration,chkfile)
        mf = mf.run(rdm1)
    


def construct_excited_state(chkfile, gs_wf, es_wf):
    mol, mf = pyqmc.recover_pyscf(chkfile)
    wf = {}
    with h5py.File(gs_wf,'r') as f:
        for k in f['wf/'].keys():
            wf[k]=f[f'wf/{k}'][...]
    print(wf.keys())
    print(mf.mo_coeff[0].shape, wf['wf1mo_coeff_alpha'].shape)
    wf['wf1mo_coeff_alpha'][:,2] = mf.mo_coeff[0][:,2]