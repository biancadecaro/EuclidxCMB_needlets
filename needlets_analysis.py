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
        Returns the \Gamma_j vector

        Notes
        Mll_1x2 = Mll matrix for the TG term
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
        Mll = Mll matrix for the TTGG term
        Mll_1x2 = Mll matrix for the TG term
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


    def pseudo_cl(self, cltg, lmax, Mll_1x2):
        """
        Returns the Pseudo Cl vector

        Notes
        Mll_1x2 = Mll matrix for the TG term
        """
        return np.dot(Mll_1x2, cltg[:lmax+1])
    
    def cov_cl(self, cltg,cltt, clgg, Mll,  Mll_1x2, lmax, noise_gal_l=None):
        """
        Returns the Cov(Pseudo-C_\ell, Pseudo-C_\ell') 
        Notes
        Mll = Mll matrix for the TTGG term
        Mll_1x2 = Mll matrix for the TG term
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

 
  
