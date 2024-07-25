import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import needlets_analysis as need
import seaborn as sns
###################################

nside = 128

lmax = 256
nsim = 1000
jmax= 12

cl_theory = np.loadtxt('EUCLID_fiducial_lmin0.dat')
ell_theory = cl_theory[0]
cl_theory_tt = cl_theory[1]
cl_theory_tg = cl_theory[2]
cl_theory_gg = cl_theory[3]
Nll = np.ones(cl_theory_gg.shape[0])/354543085.80126834


mask_eu = hp.read_map(f'mask_rsd2022g-wide-footprint-year-6-equ-order-13-moc_ns0{nside}_G_filled_2deg2.fits')
mask_pl = hp.read_map(f'mask_temp_ns{nside}.fits')

need_theory = need.NeedletTheory(jmax, lmax)
D = need_theory.get_D_parameter()
b_need = need_theory.get_bneed(jmax, lmax)
jvec = need_theory.get_jvec()

fig, ax1  = plt.subplots(1,1,figsize=(5.3,4), dpi=100) 
for i in range(1,jmax):
    ax1.plot(b_need[i]*b_need[i], label = 'j='+str(i) )
ax1.set_xscale('log')
ax1.set_xlim([0.40, 350 ])
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$w^{2}(\frac{\ell}{D^{j}})$')
ax1.legend(loc='upper left', fontsize=9)
plt.tight_layout()


ell_binning=need_theory.ell_binning(jmax, lmax)
fig = plt.figure()
plt.suptitle(r'$D = %1.2f $' %D +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
ax = fig.add_subplot(1, 1, 1)
for i in range(0,jmax+1):
    ell_range = ell_binning[i][ell_binning[i]!=0]
    plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
    plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))

ax.set_xlabel(r'$\ell$')
ax.legend(loc='right', ncol=2)
plt.tight_layout()


#####
Mll_pl_eu  = np.loadtxt('kernel_Euclid_Planck_TTGG_lmax256.dat')
Mll_comb  = np.loadtxt('kernel_Euclid_Planck_TGTG_lmax256.dat')


gammaJ_tg = need_theory.gammaJ(cl_theory_tg, Mll_pl_eu, lmax)
delta_gammaj = need_theory.variance_gammaj(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll_1x2=Mll_comb, Mll=Mll_pl_eu,  lmax=lmax, noise_gal_l=Nll)

# Covariances

fig = plt.figure()

plt.suptitle(r'$D = %1.2f $' %D +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(nside) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:jmax], np.diag(delta_gammaj)[1:jmax], 'o')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xticks(jvec[1:jmax])
ax.set_xticklabels(jvec[1:jmax])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$(\Gamma_j)^2$')

fig.tight_layout()


# GammaJ TG

fig = plt.figure()

plt.suptitle(r'$D = %1.2f $' %D +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax) + r'$ ,~N_{\mathrm{side}} =$'+str(nside) + r',$~N_{\mathrm{sim}} = $'+str(nsim))

ax = fig.add_subplot(1, 1, 1)

ax.plot(jvec[1:jmax], gammaJ_tg[1:jmax], 'o')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xticks(jvec[1:jmax])
ax.set_xticklabels(jvec[1:jmax])
ax.set_xlabel(r'$j$')
ax.set_ylabel(r'$\tilde{\Gamma}^{\mathrm{\,TG}}_j$')

fig.tight_layout()

plt.show()