import numpy as np
# from mll import mll


class NeedletTheory(object):
    """ class to compute theoretical quantities related to needlet, i.e. needlet power spectrum
        given an angular power spectrum Cl
    """
    def __init__(self,jmax,lmax, npoints=1000):
        """ 
        * jmax    = Number of bin 
        * lmax    = Maximum multipole for the analysis
        * B       = Needlet width parameter
        * npoints = Number of points to sample the integrals
        """
        self.jmax = jmax
        self.lmax =lmax
        self.D = self.get_D_parameter()
        self.npoints = npoints
        self.norm = self.get_normalization()
        self.jvec = np.arange(jmax+1)

    def get_D_parameter(self):
        """
        Returns the D parameter for needlets
        """
        return np.power(self.lmax, 1./self.jmax)

    def ell_binning(self, jmax, lmax, ell):
        """
        Returns the binning scheme in  multipole space
        """
        #assert(np.floor(self.B**(jmax+1)) <= ell.size-1) 
        ell  = np.arange(lmax+1)*np.ones((jmax+1, lmax+1))
        bjl  = np.zeros((jmax+1,lmax+1))
        ellj =np.zeros((jmax+1, lmax+1))
        #delta_gammaj = np.zeros((jmax+1, jmax+1))
        for j in range(jmax+1):
            b2 = self.b_need(ell[j]/self.D**j)**2
            b2[np.isnan(b2)] = 0.
            b2[b2!=0] = 1.
            bjl[j,:] = b2
            ellj[j,:] = ell[j,:]*bjl[j,:]
        return ellj 
    
    def fl_j(self, jmax, lmax):
        ell_binning=self.ell_binning(jmax, lmax)
        l_j = np.zeros(jmax+1, dtype=int)
    
        for j in range(1,jmax+1):
            ell_range = ell_binning[j][ell_binning[j]!=0]
            if ell_range.shape[0] == 1:
                l_j[j] = ell_range
            else:
                l_j[j] = int(ell_range[int(np.ceil((len(ell_range))/2))])
        return l_j

    def f_need(self, t):
        """
        Standard needlets f function
        @see arXiv:0707.0844
        """
        good_idx = np.logical_and(-1. < t, t < 1.)
        f1 = np.zeros(len(t))
        f1[good_idx] = np.exp(-1./(1.-(t[good_idx]*t[good_idx])))
        return f1

    def get_normalization(self):
        """
        Evaluates the normalization of the standard needlets function
        @see arXiv:0707.0844
        """
        from scipy.integrate import simps
        
        t = np.linspace(-1,1,self.npoints)
        return simps(self.f_need(t), t)

    def psi_need(self, u):
        """
        Standard needlets Psi function
        @see arXiv:0707.0844
        """
        from scipy.integrate import simps
           
        # u_ = np.linspace(-1.,u,self.npoints)
        return [simps(self.f_need(np.linspace(-1.,u_,self.npoints)), np.linspace(-1.,u_,self.npoints))/self.norm for u_ in u]

    def phi_need(self, t):
        """
        Standard needlets Phi function
        @see arXiv:0707.0844
        """
        from scipy.integrate import simps

        left_idx = np.logical_and(0 <= t, t <= 1./self.D)
        cent_idx = np.logical_and(1./self.D < t, t < 1.)
        rite_idx = t > 1.

        phi = np.zeros(len(t))
        phi[left_idx] = 1.
        phi[cent_idx] = self.psi_need(1.-2.*self.D/(self.D-1.)*(t[cent_idx]-1./self.D))
        phi[rite_idx] = 0.

        return phi

    def b_need(self, xi):
        """
        Standard needlets windows function
        @see arXiv:0707.0844
        """
        return np.sqrt(np.abs(self.phi_need(xi/self.D)-self.phi_need(xi)))

    def cl2betaj(self, jmax,lmax, cl):
        """
        Returns needlet power spectrum \beta_j given an angular power spectrum Cl.
        @see eq 2.17 https://arxiv.org/abs/1607.05223
        """

        #assert(np.floor(self.B**(jmax+1)) <= cl.size-1) 
        #print( np.floor(self.B**(jmax+1)), cl.size-1)
        
        ell = np.arange(0, lmax+1)
        betaj = np.zeros(jmax+1)
        bjl = np.zeros((jmax+1, lmax+1))
        for j in range(jmax+1):
            b2 = self.b_need(ell/self.D**j)**2
            b2[np.isnan(b2)] = 0.
            bjl[j, :] = b2
            betaj[j] = np.sum(b2*(2.*ell+1.)/4./np.pi*cl[ell])
        
        return betaj

    def delta_beta_j(self, jmax, lmax, cltg, cltt, clgg, noise_gal_l=None):
            """
                noise_gal_l = shot noise power spectrum, noise_gal_l.shape = clgg.shape
                Returns the \delta beta_j (variance) 
                @see eq 2.19 https://arxiv.org/abs/1607.05223
                
            """
            delta_beta_j_squared = np.zeros(jmax+1)
            if noise_gal_l is not None:
                clgg_tot = clgg+noise_gal_l
            else:
                clgg_tot = clgg
    
            #for j in range(jmax+1):
            #    l_min = np.floor(self.D**(j-1))
            #    l_max = np.floor(self.D**(j+1))
            #    if l_max > cltg.size-1:
            #        l_max=cltg.size-1
            #    ell = np.arange(l_min,l_max+1, dtype=np.int)
            #    delta_beta_j_squared[j] = np.sum(((2*ell+1)/(16*np.pi**2))*(self.b_need(ell/self.D**j)**4)*(cltg[ell]**2 + cltt[ell]*clgg_tot[ell]))
#
            ell = np.arange(0,lmax+1)
            bjl = np.zeros((jmax+1, lmax+1))
            for j in range(jmax+1):
                b2 = self.b_need(ell/self.D**j)**2
                b2[np.isnan(b2)] = 0.
                bjl[j, :] = b2
                delta_beta_j_squared[j]= np.sum(((2*ell+1)/(16*np.pi**2))*(b2**2)*(cltg[ell]**2 + cltt[ell]*clgg_tot[ell]))
            return np.sqrt(delta_beta_j_squared)

    def betaj_master(self, Mll_1x2, cl, lmax):

        """
        Returns the \Gamma_j vector from Domenico's notes

        Notes
        -----
        gamma_lj.shape = (lmax+1, jmax+1)
        """
        ### Original BDC: ###
        # Mll  = self.get_Mll(wl, lmax=lmax)

        ell  = np.arange(0, lmax+1, dtype=np.int)
        bjl  = self.b_values**2
        return (bjl*(2*ell+1.)*np.dot(Mll_1x2, cl[:lmax+1])).sum(axis=1)/(4*np.pi)

    def variance_betaj_master(self, Mll, Mll_1x2,cltg,cltt, clgg, lmax, noise_gal_l=None):
        """
        Returns the Cov(\Gamma_j, \Gamma_j') 
        Notes
        -----
        Cov(gamma)_jj'.shape = (jmax+1, jmax+1)
        """
        if noise_gal_l is not None:
            clgg_tot = clgg+noise_gal_l
        else:
            clgg_tot = clgg

        #Original
        #Mll  = self.get_Mll(wl, lmax=lmax)
        ell  = np.arange(lmax+1, dtype=int)
        bjl  = self.b_values**2*(2*ell+1.) #np.zeros((jmax+1,lmax+1))

        covll = np.zeros((lmax+1, lmax+1))
        for ell1 in range(lmax+1):
            for ell2 in range(lmax+1):
                covll[ell1,ell2] = (Mll_1x2[ell1,ell2]*(cltg[ell1]*cltg[ell2])+Mll[ell1,ell2]*(np.sqrt(cltt[ell1]*cltt[ell2]*clgg_tot[ell1]*clgg_tot[ell2])))/(2.*ell1+1)
        delta_gammaj = np.dot(bjl, np.dot(covll, bjl.T))
        return delta_gammaj/(4*np.pi)**2

    def pseudo_cl(self, cltg, lmax, wl):
        Mll  = self.get_Mll(wl, lmax=lmax)
        return np.dot(Mll, cltg[:lmax+1])
    
    def cov_cl(self, cltg,cltt, clgg, Mll,  Mll_1x2, lmax, noise_gal_l=None):
        """
        Returns the Cov(Pseudo-C_\ell, Pseudo-C_\ell') 
        Notes
        -----
        Cov(Pseudo-C_\ell, Pseudo-C_\ell') .shape = (lmax+1, lmax+1)
        """
        if noise_gal_l is not None:
            clgg_tot = clgg+noise_gal_l
        else:
            clgg_tot = clgg
        ell= np.arange(lmax+1)
        covll = np.zeros((ell.shape[0],ell.shape[0]))
        for l,ell1 in enumerate(ell):
            for ll,ell2 in enumerate(ell):
                covll[l,ll] = (Mll_1x2[l,ll]*(cltg[l]*cltg[ll])+Mll[l,ll]*(np.sqrt(cltt[l]*cltt[ll]*clgg_tot[l]*clgg_tot[ll])))/(2.*ell1+1)
        return covll

    
######################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os
    import healpy as hp
    from matplotlib import rc, rcParams, gridspec


    def GetNlgg(counts, dim=None,mask=None, lmax=None, return_ngal=False):
        """
        Returns galaxy shot-noise spectra given a number counts Healpix map. 
        If return_ngal is True, it returns also the galaxy density in gal/ster.

        Note
        ----
        1. Use only binary mask.
        2. If mask is not None, yielded spectrum is not pseudo
        """
        counts = np.asarray(counts)

        if lmax is None: lmax = hp.npix2nside(counts.size) * 2
        if mask is not None: 
            mask = np.asarray(mask)
            fsky = np.mean(mask)
        else: 
            mask = 1.
            fsky = 1.

        N_tot = np.sum(counts * mask)
        if dim=='ster':
            ngal  = N_tot / fsky
        else:
            ngal  = N_tot / 4. / np.pi / fsky

        if return_ngal:
            return np.ones(lmax+1) / ngal, ngal
        else:
            return np.ones(lmax+1)/ ngal

    jmax=5
    lmax=256
    nside = 128
    nsim = 1000
    ngal_ster=35454308.580126834
    need_theory = NeedletTheory(lmax=lmax,jmax=jmax)
    print(f'D={need_theory.D}')

    cl_theory = np.loadtxt('/home/bianca/Documents/xcmbneed/src/spectra/inifiles/EUCLID_fiducial_lmin0.dat')
    ell_theory = cl_theory[0]
    cl_theory_tt = cl_theory[1]
    cl_theory_tg = cl_theory[2]
    cl_theory_gg = cl_theory[3]
    Nll= np.ones(cl_theory_gg.shape[0])/ngal_ster
    fsky=0.36

    beta_sims = np.loadtxt('/home/bianca/Documents/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/betaj_sims_TS_galT_jmax5_B_3.03_nside128_fsky0.36.dat')
    cov_sims = np.loadtxt('/home/bianca/Documents/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/cov_TS_galT_jmax5_B_3.03_nside128_fsky0.36.dat')
    beta_sims_mean = np.mean(beta_sims, axis = 0)

    betatg = need_theory.cl2betaj(jmax, cl=cl_theory_tg)
    delta = need_theory.delta_beta_j(jmax=jmax, cltg=cl_theory_tg, cltt=cl_theory_tt,clgg=cl_theory_gg, noise_gal_l=Nll[:cl_theory_gg.shape[0]])


    ell_bin = need_theory.ell_binning(jmax=jmax, lmax=lmax, ell =ell_theory)
    fig = plt.figure()
    plt.suptitle(r'$D = %1.2f $' %need_theory.D +r'$ ,~j_{\mathrm{max}} =$'+str(jmax) + r'$ ,~\ell_{\mathrm{max}} =$'+str(lmax))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(1,jmax+1):
        ell_range = ell_bin[i][ell_bin[i]!=0]
        plt.plot(ell_range, i*ell_range/ell_range, label= f'j={i}')
        plt.text(ell_range[0], i, r'$\ell_{min}=%d,\,\ell_{max}=%d$'%(ell_range[0],ell_range[-1]))
    
    ax.set_xlabel(r'$\ell$')
    ax.legend(loc='right', ncol=2)
    plt.tight_layout()
    plt.show()

    beta_sims_full_sky = np.loadtxt('/home/bianca/Documents/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/betaj_sims_TS_galT_jmax5_B_3.03_nside128.dat')
    cov_betaj_full_sky = np.loadtxt('/home/bianca/Documents/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/cov_TS_galT_jmax5_B_3.03_nside128.dat')
    beta_sims_mean_full_sky = np.mean(beta_sims_full_sky, axis=0)

    fig = plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.5], wspace=0)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    plt.suptitle(r'$D = %1.2f $' %need_theory.D + r' $\ell_{max} =$'+str(lmax) + r' $j_{max} = $'+str(jmax)+ r' $n_{side} =$'+str(nside) + r' $n_{sim} =$'+str(nsim) )

    ax0.plot(need_theory.jvec[1:jmax+1], betatg[1:jmax+1], label=r'$\beta_j^{Tgal}$')
    ax0.errorbar(need_theory.jvec[1:jmax+1], beta_sims_mean_full_sky[1:jmax+1], yerr=delta[1:jmax+1]/np.sqrt(nsim), fmt='o',ms=5,capthick=5, label=r'Variance on the mean from theory')
    ax0.errorbar(need_theory.jvec[1:jmax+1], beta_sims_mean_full_sky[1:jmax+1], yerr=np.sqrt(np.diag(cov_betaj_full_sky)[1:jmax+1])/(np.sqrt(nsim)), fmt='o',ms=5,capthick=5, label=r'Variance on the mean from sim')

    difference = (beta_sims_mean_full_sky -betatg)/betatg     
    ax1.errorbar(need_theory.jvec[1:jmax+1], difference[1:jmax+1], yerr=delta[1:jmax+1]/(betatg[1:jmax+1]*np.sqrt(nsim)), fmt='o',ms=5,capthick=5, label=r'Variance in the mean from theory')
    ax1.errorbar(need_theory.jvec[1:jmax+1], difference[1:jmax+1], yerr=np.sqrt(np.diag(cov_betaj_full_sky)[1:jmax+1])/(betatg[1:jmax+1]*np.sqrt(nsim) ), fmt='o',ms=5,capthick=5, label=r'Variance on the mean from sim')
    ax1.axhline(ls='--', color='k')
    plt.legend()
    ax1.set_ylabel(r'$(\langle \beta_j^{Tgal} \rangle - \beta_j^{Tgal, th})/\beta_j^{Tgal, th}$')

    ax0.legend(loc='best')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_xlabel(r'$j$')
    ax0.set_ylabel(r'$\beta_j^{Tgal}$')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    #MASK
    Mll= np.loadtxt('/home/bianca/Downloads/Mll_Needlets_wl_EuclidxCOMM2018Planck.dat')
    betaj_master_theory_Bianca = np.loadtxt('/home/bianca/Documents/xcmbneed/src/output_needlet_TG/EUCLID/Mask_noise/TG_128_nsim1000_nuova_mask/gammaj_theory_B3.03_jmax5_nside128_lmax256_nsim1000_fsky0.36.dat')
    fsky=0.36

    variance_betaj_master = need_theory.variance_betaj_master(cltg=cl_theory_tg,cltt=cl_theory_tt, clgg=cl_theory_gg, Mll=Mll, jmax=jmax, lmax=lmax, noise_gal_l=Nll[:cl_theory_gg.shape[0]])
    betaj_master_tg = need_theory.betaj_master(Mll=Mll, cl=cl_theory_tg, jmax=jmax, lmax=lmax)

    #fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,7))
    #ax1.set_title('Analitycal Covariance Needlet')
    #plt1=ax1.imshow(variance_betaj_master, cmap = 'viridis')
    #ax2.set_title('Simulated Covariance Needlet')
    #plt2=ax2.imshow(cov_sims, cmap= 'viridis')
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.6)
    #cax = plt.axes([0.85, 0.09, 0.045, 0.5])
    #plt.colorbar(plt1, ax=ax1, cax=cax)
    #plt.colorbar(plt2, ax=ax2, cax=cax)
    #ax1.set_xlabel('j')
    #ax1.set_ylabel('j')
    #ax2.set_xlabel('j')
    #ax2.set_ylabel('j')
    #plt.show()
  

    #fig = plt.figure(figsize=(10,7))
#
    #plt.suptitle(r'$D = %1.2f $' %need_theory.D + r' $\ell_{max} =$'+str(lmax) + r' $j_{max} = $'+str(jmax)+f' fsky={fsky:0.2f}')
    #ax = fig.add_subplot(1, 1, 1)
    #ax.plot(need_theory.jvec, betaj_master_tg,color='firebrick', marker='o',ms=5,label=r'$\tilde{\beta_j}$')
    #ax.legend(loc='best')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #ax.set_xlabel(r'$j$')
    #ax.set_ylabel(r'$\tilde{\beta}_j^{Tgal}$')
    #fig.tight_layout()
    

    fig = plt.figure(figsize=(10,7))
    
    plt.suptitle('Variance comparison - Cut sky')

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    plt.suptitle(r'$D = %1.2f $' %need_theory.D + r' $\ell_{max} =$'+str(lmax) + r' $j_{max} = $'+str(jmax)+ r' $n_{side} =$'+str(nside) + r' $n_{sim} =$'+str(nsim) +f' fsky={fsky:0.2f}')

    ax0.plot(need_theory.jvec[1:jmax+1], betaj_master_tg[1:jmax+1],'k', label=r'$\tilde{\beta}_j^{Tgal}$')
    ax0.errorbar(need_theory.jvec[1:jmax+1], beta_sims_mean[1:jmax+1], yerr=np.sqrt(np.diag(variance_betaj_master)[1:jmax+1])/(np.sqrt(nsim)), color='firebrick', fmt='o',ms=5,capthick=5, label=r'Variance on the mean from theory')
    ax0.errorbar(need_theory.jvec[1:jmax+1], beta_sims_mean[1:jmax+1], yerr=np.sqrt(np.diag(cov_sims)[1:jmax+1])/(np.sqrt(nsim)), color='seagreen', fmt='o',ms=5,capthick=5, label=r'Variance on the mean from sim')

    difference = (beta_sims_mean -betaj_master_tg)/betaj_master_tg     
    #ax1.errorbar(myanalysis.jvec[1:jmax], difference[1:jmax],yerr=delta_noise[1:jmax]/(gammaJ_tg[1:jmax]/(4*np.pi)*np.sqrt(nsim) ),color='seagreen', fmt='o',  ms=10,capthick=5, label=r'Variance from theory')
    ax1.errorbar(need_theory.jvec[1:jmax+1], difference[1:jmax+1], yerr=np.sqrt(np.diag(variance_betaj_master)[1:jmax+1])/(betaj_master_tg[1:jmax+1]*np.sqrt(nsim) ), color='firebrick', fmt='o',ms=5,capthick=5, label=r'Variance in the mean from theory')
    ax1.errorbar(need_theory.jvec[1:jmax+1], difference[1:jmax+1], yerr=np.sqrt(np.diag(cov_sims)[1:jmax+1])/(betaj_master_tg[1:jmax+1]*np.sqrt(nsim) ), color='seagreen', fmt='o',ms=5,capthick=5, label=r'Variance on the mean from sim')
    ax1.axhline(ls='--', color='k')
    ax1.set_ylabel(r'$(\langle \tilde{\beta}_j^{Tgal} \rangle - \tilde{\beta}_j^{Tgal, th})/\beta_j^{Tgal, th}$')

    ax0.legend(loc='best')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_xlabel(r'$j$')
    ax0.set_ylabel(r'$\tilde{\beta}_j^{Tgal}$')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    ############
    #confronto theory Bianca 
    fig, ax = plt.subplots(1,2, figsize = (10,4))

    ax[0].scatter(np.arange(jmax+1), betaj_master_theory_Bianca,s = 80, label = 'Theory BDC')
    ax[0].scatter(np.arange(jmax+1), betaj_master_tg,  label= 'Theory VC')
    ax[0].scatter(np.arange(jmax+1), beta_sims_mean, label = 'Mean of sims.')


    # ax[0].set_xlim(0.1,11.5)
    # ax[0].set_ylim(-0.480,0.2)
    ax[0].set_ylabel(r"$\beta_{j}$", size = 16)
    ax[0].set_xlabel("$j$", size = 16)
    ax[0].legend(fontsize = 12, frameon = True, framealpha = 1, loc = 'center left')

    ax[1].axhline(c='k', ls='--')
    ax[1].scatter(np.arange(jmax+1),((betaj_master_tg/ betaj_master_theory_Bianca)-1)*100)
    ax[1].scatter(np.arange(6),((beta_sims_mean/ betaj_master_tg)-1)*100)

    print(betaj_master_tg,betaj_master_theory_Bianca)

    ax[1].set_xlabel("$j$", size = 16)
    ax[1].set_ylabel(r"$\Delta(\%)$ wrt Th. BDC", size = 16)
    plt.show()
