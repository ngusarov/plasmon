
def simulate(N, Cc_first,Cc_last, Cc_middle, Cg_first, Cg_last, Cg_middle, L_um_SQUID, d_length):
    global MODE_IDX, FREQ1, FREQ2, KAPPA, T1_PURCELL, TPHI_CHARGE, G_COUPLING




    import numpy as np
    import scipy.constants as const

    import scqubits as scq
    hbar = const.hbar


    # 1. Define physical constants and conversion factors
    h = const.h                      # Planck constant in J*s
    e = const.e                      # Elementary charge in C
    Phi0 = h / (2 * e)               # Magnetic flux quantum in Wb
    GHz = 1e9                        # Conversion for Hz to GHz
    fF_to_F = 1e-15                  # femtoFarad to Farad

    # 2. Define input parameters
    fp_target_GHz = 30          # Target plasma frequency in GHz
    wp_target = 2 * np.pi * fp_target_GHz * GHz # Target angular frequency in rad/s

    # Junction dimensions
    W_um = 0.260                       # Width in um (260 nm)
    L_um = 2.8                        # Length in um 2.8 is the largest
    # N = 90              # number of cells
    # for N in [120]:

    C_area_fF_per_um2 = 45.0         # Capacitance per unit area in fF/um^2 (Assumed)

    # 3. Calculate Junction Capacitance (C_J)
    Area_um2 = W_um * L_um
    CJ = C_area_fF_per_um2 * Area_um2 * fF_to_F # Junction capacitance C in Farads (F)

    # 4. Calculate Josephson Inductance (L_J)
    # Formula: wp = 1/sqrt(LJ * CJ) => LJ = 1 / (wp^2 * CJ)
    LJ = 1.0 / (wp_target**2 * CJ)  # Josephson Inductance L in Henries (H)

    # 5. Calculate Josephson Energy (E_J)
    # Formula: EJ = (Phi0 / (2*pi))^2 / LJ
    EJ = (Phi0 / (2 * np.pi))**2 / LJ # Josephson Energy EJ in Joules (J)
    EJ_GHz = EJ / h  / GHz              # Josephson Energy EJ in GHz

    # 6. Calculate Charging Energy (E_C)
    # Formula: EC = e^2 / (2 * (2 * CJ))
    EC = e**2 / (2 * CJ)             # Charging Energy EC in Joules (J)
    EC_GHz = EC / h  / GHz                  # Charging Energy EC in GHz


    # Cg_over_CJ = 0.10       # stray-to-ground ratio
    Cg = 4.5e-18 ###Cg_over_CJ * CJ

    # --- Chain design targets (edit here) ---
    a = W_um * 1e-6                # cell pitch [m] (for bookkeeping; not critical here)


    #########################################
    R = 50.0                       # target wave impedance [Ohm]
    ########################################


    # Qubit junction dimensions
    W_um = 0.260                       # Width in um (260 nm)
    L_um = L_um_SQUID                         # Length in um
    # d_length = d_length

    L1_um = L_um*(1-d_length)
    L2_um = L_um*(1+d_length)

    Area_um2 = W_um * L_um
    # CJ_qubit = C_area_fF_per_um2 * Area_um2 * fF_to_F # Junction capacitance C in Farads (F)

    CJ1 = C_area_fF_per_um2 * W_um * L1_um * fF_to_F
    CJ2 = C_area_fF_per_um2 * W_um * L2_um * fF_to_F

    LJ1 = 1.0 / (wp_target**2 * CJ1)  # Josephson Inductance L in Henries (H)
    LJ2 = 1.0 / (wp_target**2 * CJ2)  # Josephson Inductance L in Henries (H)

    # LJ_qubit = 1.0 / (wp_target**2 * CJ_qubit)  # Josephson Inductance L in Henries (H)

    EJ1 = (Phi0 / (2 * np.pi))**2 / LJ1 # Josephson Energy EJ in Joules (J) 
    EJ1_GHz = EJ1 / h  / GHz              # Josephson Energy EJ in GHz

    EJ2 = (Phi0 / (2 * np.pi))**2 / LJ2 # Josephson Energy EJ in Joules (J) 
    EJ2_GHz = EJ2 / h  / GHz              # Josephson Energy EJ in GHz

    # EJ_qubit = 2*(Phi0 / (2 * np.pi))**2 / LJ_qubit # Josephson Energy EJ in Joules (J) # factor of 2 for the SQUID
    # EJ_qubit_GHz = EJ_qubit / h  / GHz              # Josephson Energy EJ in GHz

    EJ_qubit_GHz = (EJ1_GHz + EJ2_GHz) # total EJ of the SQUID is the sum of the two junctions

    EC1 = e**2 / (2 * CJ1)             # Charging Energy EC in Joules (J) 
    EC1_GHz = EC1 / h  / GHz                  # Charging Energy EC in GHz
    EC2 = e**2 / (2 * CJ2)             # Charging Energy EC in Joules (J)
    EC2_GHz = EC2 / h  / GHz                  # Charging Energy EC in GHz

    EC_qubit = e**2 / (2 * (CJ1 + CJ2))             # Charging Energy EC in Joules (J)
    EC_qubit_GHz = EC_qubit / h  / GHz                  # Charging Energy EC in GHz

    # EC_qubit = e**2 / (2 * CJ_qubit * 2)             # Charging Energy EC in Joules (J) # factor of 2 for the SQUID since the two junctions are in parallel, so total C is 2*CJ_qubit
    # EC_qubit_GHz = EC_qubit / h  / GHz                  # Charging Energy EC in GHz


    d = (EJ2 - EJ1)/(EJ2 + EJ1)

    Phi_ext = 0.5*Phi0


    Cc = 0.16e-15 #0.1e-15             # [F]

    N_SQUID = N-2 #N//2


    qubit_params = {
        'EJ_GHz': EJ_qubit_GHz,
        'EC_GHz': EC_qubit_GHz,
        'd': d,
        'ng': 0.0,
        'ncut': 20,
        'truncated_dim': 10,
        'CJ': CJ1 + CJ2, # total capacitance of the SQUID (two junctions in parallel)
    }

    print("true d = ", d)


    ###########################################

    MODE_IDX = 0
    FREQ1, FREQ2, KAPPA, T1_PURCELL, G_COUPLING = None, None, None, None, None


    Cc_array = np.full(N+1, Cc_middle) #[::2]  # Take every other element to match the islands (assuming interleaved)
    Cg_array = np.full(N+1, Cg_middle) #[::2]  # Take every other element to match the islands (assuming interleaved)

    Cc_array[0], Cc_array[-1] = Cc_first, Cc_last
    Cg_array[0], Cg_array[-1] = Cg_first, Cg_last

    ###########################################


    #########################################################################################


    import numpy as np
    import scipy.constants as const
    import matplotlib.pyplot as plt
    from numpy.linalg import inv, eig

    # ======================================================================
    # 0. CONSTANTS
    # ======================================================================

    h = const.h
    e = const.e
    pi = np.pi
    Phi0 = h / (2 * e)   # flux quantum [Wb]


    # ======================================================================
    # 1. SQUID + TRANSMON BASICS  (USED FOR PURCELL, NOT FOR MATRIX MODES)
    # ======================================================================

    def squid_EJ_eff(EJ, d, Phi_ext):
        EJ1 = EJ * (1 + d)
        EJ2 = EJ * (1 - d)
        return np.sqrt(EJ1**2 + EJ2**2 + 2 * EJ1 * EJ2 * np.cos(2*pi*Phi_ext/Phi0))

    def squid_LJ(EJ_eff):
        return Phi0**2 / (4 * pi**2 * EJ_eff)

    def squid_plasma_freq(EJ_eff, Cq):
        Lj = squid_LJ(EJ_eff)
        return 1.0 / np.sqrt(Lj * Cq)       # [rad/s]


    # ======================================================================
    # 2. ENVIRONMENT ADMITTANCE (CORRECT NEW MODEL)
    # ======================================================================

    def Y_feedline(w, Cc, R):
        """
        Scalar feedline termination seen through a coupling capacitor:
        Z = R + 1/(j w Cc)  ->  Y = 1/Z

        NOTE: Signature unchanged. If you pass an array, it will return an array
        (elementwise), which is convenient for stamping per-node shunts.
        """
        Cc = np.asarray(Cc, dtype=float)  # allows scalar or array
        x = Cc * R * w
        return (Cc**2 * R * w**2 + 1j * Cc * w) / (1 + x**2)

    def build_chain_admittance_matrix(N, L_sec, C_sec, Cg, Cc, R, w,
                                    break_middle=True, j_mid=None):
        n_nodes = N + 1
        Y = np.zeros((n_nodes, n_nodes), dtype=complex)

        Y_sec = 1j*w*C_sec + 1/(1j*w*L_sec)
        if j_mid is None:
            j_mid = N//2
        mid_left, mid_right = j_mid, j_mid + 1

        # LC branches except the middle one (reserved for qubit)
        for j in range(N):
            if break_middle and j == j_mid:
                continue
            i1, i2 = j, j+1
            Y[i1,i1] += Y_sec
            Y[i2,i2] += Y_sec
            Y[i1,i2] -= Y_sec
            Y[i2,i1] -= Y_sec

        # -----------------------------
        # UPDATED shunt Cg:
        #   - scalar: applied to all nodes
        #   - array-like (can be shorter than n_nodes): missing entries -> 0
        # -----------------------------
        if np.isscalar(Cg):
            Y_shunt = 1j*w*float(Cg)
            if Y_shunt != 0 or True:  # even if Y_shunt=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                for k in range(1, n_nodes):
                    Y[k,k] += Y_shunt
        else:
            Cg_arr = np.asarray(Cg, dtype=float).ravel()
            n_hfss = int(Cg_arr.size)
            for k in range(1, n_nodes):
                if k < n_hfss:
                    cgk = Cg_arr[k]
                    if cgk != 0.0 or True:  # even if cgk=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                        Y[k,k] += 1j*w*cgk
                # else: cgk=0 -> do nothing

        # -----------------------------
        # UPDATED feedline coupling via Cc:
        #   - scalar: keep old behavior (only at node 0)
        #   - array-like (can be shorter than n_nodes): stamp Y_feedline at each node k
        #     missing entries -> 0
        # -----------------------------
        if np.isscalar(Cc):
            cc = float(Cc)
            if cc > 0 or True:  # even if cc=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                Y[0,0] += Y_feedline(w, cc, R)
        else:
            Cc_arr = np.asarray(Cc, dtype=float).ravel()
            n_hfss = int(Cc_arr.size)
            for k in range(n_nodes):
                if k < n_hfss:
                    cck = Cc_arr[k]
                    if cck > 0.0 or True:  # even if cck=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                        Y[k,k] += Y_feedline(w, cck, R)
                # else: cck=0 -> do nothing

        return Y, mid_left, mid_right

    def Y_env_between_middle_nodes(N, L_sec, C_sec, Cg, Cc, R, w, j_mid=None):
        Ymat, iL, iR = build_chain_admittance_matrix(N, L_sec, C_sec, Cg, Cc, R, w, break_middle=True, j_mid=j_mid)
        n = Ymat.shape[0]
        I = np.zeros(n, dtype=complex)
        I[iL] = 1
        I[iR] = -1
        V = np.linalg.solve(Ymat, I)
        Zenv = V[iL] - V[iR]
        return 1/Zenv

    def Y_env_spectrum(N, L_sec, C_sec, Cg, Cc, R, freqs, j_mid=None):
        out = []
        for f in freqs:
            w = 2*pi*f
            out.append(Y_env_between_middle_nodes(N, L_sec, C_sec, Cg, Cc, R, w, j_mid=j_mid))
        return np.array(out)



    # ======================================================================
    # 3. PURCELL T1 USING Y_env
    # ======================================================================



    def make_tunable_transmon(qubit_params, flux, ng=None):
        """
        Helper constructor for scqubits TunableTransmon.
        flux in Weber; internally converted to Phi/Phi0.
        """
        return scq.TunableTransmon(
            EJmax=qubit_params['EJ_GHz'],
            EC=qubit_params['EC_GHz'],
            ng=qubit_params['ng'] if ng is None else ng,
            d=qubit_params['d'],
            ncut=qubit_params['ncut'],
            truncated_dim=qubit_params['truncated_dim'],
            flux=flux / Phi0,
        )


    def transmon_observables(qubit_params, flux, ng=None):
        """
        Returns:
        w01   [rad/s]
        phi01 [dimensionless, via sin(phi) ~ phi]
        n01   [dimensionless charge matrix element]
        evals [GHz]
        evecs
        """
        qubit = make_tunable_transmon(qubit_params, flux, ng=ng)
        evals, evecs = qubit.eigensys(evals_count=3)

        # 0->1 transition in angular frequency
        w01 = 2 * pi * (evals[1] - evals[0]) * 1e9

        # small-phase proxy
        sinphi = qubit.sin_phi_operator()
        phi01 = (evecs[:, 0].conj().T @ sinphi @ evecs[:, 1]).item()

        # charge operator
        n_op = qubit.n_operator()
        n01 = (evecs[:, 0].conj().T @ n_op @ evecs[:, 1]).item()

        return w01, phi01, n01, evals, evecs


    def transmon_charge_dispersion(qubit_params, flux):
        """
        Peak-to-peak 0->1 charge dispersion in [rad/s], estimated from ng = 0 and 0.5.
        This is usually a better measure of charge-noise sensitivity than a local derivative.
        """
        w_ng0, _, _, _, _ = transmon_observables(qubit_params, flux, ng=0.0)
        w_ng05, _, _, _, _ = transmon_observables(qubit_params, flux, ng=0.5)

        delta_omega_01 = np.abs(w_ng0 - w_ng05)
        return delta_omega_01

    def EC_Joules_from_params(qubit_params):
        """
        Convert EC from GHz units to Joules.
        """
        return qubit_params['EC_GHz'] * h * 1e9


    def gamma_charge_T1_from_Sng(qubit_params, n01, Sng_w01):
        """
        Charge-noise-induced relaxation rate from high-frequency charge-noise PSD.

        Parameters
        ----------
        Sng_w01 : float
            PSD of ng fluctuations at omega01, expressed as a PSD with respect
            to angular frequency omega. Treat as phenomenological input.

        Returns
        -------
        Gamma_1_ng [1/s], T1_ng [s]
        """
        ECJ = EC_Joules_from_params(qubit_params)
        Gamma = ((8 * ECJ) / hbar)**2 * (np.abs(n01)**2) * Sng_w01
        if Gamma <= 0:
            return 0.0, np.inf
        return Gamma, 1.0 / Gamma


    def gamma_charge_dephasing_from_dispersion(delta_omega_01, sigma_ng_eff=1.0):
        """
        Charge-noise dephasing estimate from full charge dispersion.

        Parameters
        ----------
        delta_omega_01 : float
            Full 0->1 charge dispersion in rad/s:
                |omega01(ng=0) - omega01(ng=0.5)|

        sigma_ng_eff : float
            Effective fraction of the full dispersion sampled by slow ng fluctuations.
            Use:
            - 1.0 for a rough upper-scale estimate
            - smaller values (e.g. 0.01, 0.1) for a softer phenomenological estimate

        Returns
        -------
        Gamma_phi_ng [1/s], Tphi_ng [s]
        """
        Gamma_phi = np.abs(delta_omega_01) * sigma_ng_eff
        if Gamma_phi <= 0:
            return 0.0, np.inf
        return Gamma_phi, 1.0 / Gamma_phi


    def combine_T1_Tphi(T1_purcell=np.inf, Gamma1_extra=0.0, Gamma_phi=0.0):
        """
        Combine Purcell T1 with extra relaxation and dephasing channels.

        Returns:
        Gamma1_tot, T1_tot, Gamma2_tot, T2_tot
        """
        Gamma1_p = 0.0 if not np.isfinite(T1_purcell) else 1.0 / T1_purcell
        Gamma1_tot = Gamma1_p + Gamma1_extra
        T1_tot = np.inf if Gamma1_tot <= 0 else 1.0 / Gamma1_tot

        Gamma2_tot = Gamma1_tot / 2.0 + Gamma_phi
        T2_tot = np.inf if Gamma2_tot <= 0 else 1.0 / Gamma2_tot

        return Gamma1_tot, T1_tot, Gamma2_tot, T2_tot


    def Sng_white(S0):
        """
        White charge-noise PSD with respect to angular frequency omega.
        """
        return S0


    def Sng_zero():
        """
        No high-frequency charge noise.
        """
        return 0.0



    def purcell_gamma_T1(CJ, Cq, Yenv):
        ReY = np.real(Yenv)
        if ReY <= 0:
            return 0.0, np.inf
        Gamma = ReY / (2*CJ+Cq)
        return Gamma, 1/Gamma


    def purcell_gamma_T1_qub(omega_q, phi01, Yenv):

        prefactor = hbar / (4 * e**2)  # ~1.0e3 in SI units (Ohms)   

        ReY = np.real(Yenv)
        if ReY <= 0:
            return 0.0, np.inf
        Gamma = prefactor * 2 * omega_q * (np.abs(phi01)**2) * ReY
        return Gamma, 1/Gamma

    # ======================================================================
    # 4. ORIGINAL MATRIX-CHAIN MODEL (FOR MODE SPECTRA ONLY)
    # ======================================================================

    def build_chain_plain(
        N,
        LJ,
        CJ,
        Cg,   # scalar OR array-like from HFSS (can be shorter than N)
        Cc,   # scalar OR array-like from HFSS (can be shorter than N)
        R,
    ):
        """
        Topology (unchanged):
        n = N + 2
            node 0 : strip/port (50Ω environment)
            node 1 : left pad / chain-end node used by LJ/CJ stamps
            nodes 2..N+1 : remaining chain islands

        JJ stamps (UNCHANGED):
        for j=1..N connect (j, j+1) with LJ and CJ.

        Updated only Cg & Cc handling:
        - Cg:
            * scalar: applied to nodes k=2..N+1
            * array-like: interpreted as per-node values for k=2..N+1 (length may be < N)
                If len(Cg_arr) < N, remaining nodes get 0.
        - Cc:
            * scalar: same capacitor value between node 0 and each node k=2..N+1
            * array-like: distributed coupling caps between node 0 and node k=2..N+1 (length may be < N)
                If len(Cc_arr) < N, remaining nodes get 0.

        Note: We do NOT use HFSS diagonals; we only stamp the selected mutuals (as caps).
        """
        n = N + 2
        K = np.zeros((n, n), dtype=float)
        C = np.zeros((n, n), dtype=float)
        G = np.zeros((n, n), dtype=float)

        # -----------------------------
        # JJ inductors & capacitors (UNCHANGED)
        # -----------------------------
        valL = 1.0 / LJ
        for j in range(1, N + 1):
            i1, i2 = j, j + 1
            K[i1, i1] += valL; K[i2, i2] += valL
            K[i1, i2] -= valL; K[i2, i1] -= valL

            C[i1, i1] += CJ;  C[i2, i2] += CJ
            C[i1, i2] -= CJ;  C[i2, i1] -= CJ

        # -----------------------------
        # Helper: fetch per-island value with zero-padding
        # -----------------------------
        def _as_array_or_none(x):
            return None if np.isscalar(x) else np.asarray(x, dtype=float).ravel()

        Cg_arr = _as_array_or_none(Cg)
        Cc_arr = _as_array_or_none(Cc)

        # -----------------------------
        # Shunt Cg on nodes k=2..N+1
        # -----------------------------
        if Cg_arr is None:
            cg = float(Cg)
            if cg != 0.0 or True:  # even if cg=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                for k in range(1, N + 2):
                    C[k, k] += cg
        else:
            n_hfss = int(Cg_arr.size)
            # if shorter, missing nodes are treated as 0 automatically
            for idx, k in enumerate(range(1, N + 2)):  # idx=0..N-1
                if idx < n_hfss:
                    cg = Cg_arr[idx]
                    if cg != 0.0 or True:  # even if cg=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                        C[k, k] += cg
                # else: cg=0, do nothing

        # -----------------------------
        # Port coupling Cc between node 0 and nodes k=2..N+1
        # -----------------------------
        if Cc_arr is None:
            cc = float(Cc)
            if cc != 0.0 or True:  # even if cc=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                # for k in range(2, N + 2):
                C[0, 0] += cc
                C[1, 1] += cc
                C[0, 1] -= cc
                C[1, 0] -= cc
        else:
            n_hfss = int(Cc_arr.size)
            for idx, k in enumerate(range(1, N + 2)):  # idx=0..N-1
                if idx < n_hfss:
                    cc = Cc_arr[idx]
                    if cc != 0.0 or True:  # even if cc=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                        C[0, 0] += cc
                        C[k, k] += cc
                        C[0, k] -= cc
                        C[k, 0] -= cc
                # else: cc=0, do nothing
                        # print(idx, k, cc/fF_to_F)

        # -----------------------------
        # Dissipation at port node
        # -----------------------------
        G[0, 0] = 1.0 / R

        return K, C, G


    def build_chain_squid(N, LJ, CJ, Cg, Cc, R, qubit_params, Phi_ext, N_SQ=None):
        if N_SQ is None:
            N_SQ = N // 2

        w01, _,_,_,_ = transmon_observables(qubit_params, Phi_ext)

        CJ_s =qubit_params['CJ'] # total capacitance of the SQUID (two junctions in parallel)
        LJ_s = 1.0 / (w01**2 * CJ_s)

        n = N + 2
        K = np.zeros((n, n), dtype=float)
        C = np.zeros((n, n), dtype=float)
        G = np.zeros((n, n), dtype=float)

        # --------------------------------------------------
        # Inductors & capacitors, with one junction replaced by SQUID
        # (UNCHANGED logic for LJ/CJ vs LJ_s/CJ_s)
        # --------------------------------------------------
        for j in range(1, N + 1):
            # if j<N:    
            i1, i2 = j, j + 1

            if j == N_SQ:
                valL = 1.0 / LJ_s
                CJ_eff = CJ_s
            else:
                valL = 1.0 / LJ
                CJ_eff = CJ

            K[i1, i1] += valL; K[i2, i2] += valL
            K[i1, i2] -= valL; K[i2, i1] -= valL

            C[i1, i1] += CJ_eff; C[i2, i2] += CJ_eff
            C[i1, i2] -= CJ_eff; C[i2, i1] -= CJ_eff
        #     else:
        #         # Last junction replaced by a large island
        #         i1, i2 = N, N+1
                
        #         C_last_to_prelast = 0.16e-15
        #         CJ_eff = C_last_to_prelast

        #         C[i1, i1] += CJ_eff; C[i2, i2] += CJ_eff
        #         C[i1, i2] -= CJ_eff; C[i2, i1] -= CJ_eff

        # # last node to feed (large pad capacitance)
        # C_last_to_gnd = 9.17e-15
        # C[N+1, N+1] += C_last_to_gnd
        # C[0, 0] += C_last_to_gnd
        # C[0, N+1] -= C_last_to_gnd
        # C[N+1, 0] -= C_last_to_gnd

        # # last node to other nodes (stray capacitances)
        # C_last_to_others = 0.05e-15
        # for k in range(2, N + 1):
        #     C[k, k] += C_last_to_others
        #     C[k, N+1] -= C_last_to_others
        #     C[N+1, k] -= C_last_to_others
        #     C[N+1, N+1] += C_last_to_others

        # # last node to 1st island (stray capacitance)
        # C_last_to_first = 0.58e-15
        # C[1, 1] += C_last_to_first
        # C[1, N+1] -= C_last_to_first
        # C[N+1, 1] -= C_last_to_first
        # C[N+1, N+1] += C_last_to_first

        # --------------------------------------------------
        # UPDATED Cg handling (scalar OR array-like, can be shorter than N)
        # Applies to nodes k = 2..N+1 (N nodes)
        # Missing entries (if HFSS shorter) => treated as 0
        # --------------------------------------------------
        if np.isscalar(Cg):
            cg = float(Cg)
            if cg != 0.0 or True:  # even if cg=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                for k in range(1, N + 2):
                    C[k, k] += cg
        else:
            Cg_arr = np.asarray(Cg, dtype=float).ravel()
            n_hfss = int(Cg_arr.size)
            for idx, k in enumerate(range(1, N + 2)):  # idx=0..N-1
                if idx < n_hfss:
                    cg = Cg_arr[idx]
                    if cg != 0.0 or True:  # even if cg=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                        C[k, k] += cg
                # else: cg=0 -> do nothing

        # --------------------------------------------------
        # UPDATED Cc handling (scalar OR array-like, can be shorter than N)
        # Now: distributed coupling caps between port node 0 and nodes k = 2..N+1
        # Missing entries (if HFSS shorter) => treated as 0
        # --------------------------------------------------
        if np.isscalar(Cc):
            cc = float(Cc)
            if cc != 0.0 or True:  # even if cc=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                # for k in range(2, N + 2):
                C[0, 0] += cc
                C[1, 1] += cc
                C[0, 1] -= cc
                C[1, 0] -= cc
        else:
            Cc_arr = np.asarray(Cc, dtype=float).ravel()
            n_hfss = int(Cc_arr.size)
            for idx, k in enumerate(range(1, N + 2)):  # idx=0..N-1
                if idx < n_hfss:
                    cc = Cc_arr[idx]
                    if cc != 0.0 or True:  # even if cc=0, we can still add it without harm; this simplifies the logic and avoids missing any nonzero entries due to a strict check
                        C[0, 0] += cc
                        C[k, k] += cc
                        C[0, k] -= cc
                        C[k, 0] -= cc
                # else: cc=0 -> do nothing

        # --------------------------------------------------
        # Dissipation at port node
        # --------------------------------------------------
        G[0, 0] = 1.0 / R

        return K, C, G, LJ_s, CJ_s, N_SQ



    def s21(K,C,G,R,freqs):
        n = C.shape[0]
        S = []
        b = np.zeros(n); b[0]=1/R
        for f in freqs:
            w = 2*np.pi*f
            M = K - w*w*C + 1j*w*G
            phi = np.linalg.solve(M,b)
            S.append(phi[-1])
        return np.array(S)

    def s21_lossy(K, C, G, R, freqs, kappa_i=0, gamma_i=0):
        """
        Adds internal losses to:
            node 1 = resonator:  kappa_i
            node 2 = qubit:      gamma_i
        """

        n = C.shape[0]
        S_phi0 = []
        S_phi1 = []
        S_phi2 = []

        for f in freqs:
            w = 2*np.pi*f

            # copy G each iteration (never modify input)
            G_eff = G.copy()

            # Add internal loss conductances:
            # physically, G has units of siemens (Ohm^-1)
            # damping term is (i*ω)*G in the equations.
            G_eff[1,1] += kappa_i / w       # resonator intrinsic loss
            G_eff[N//2,N//2] += gamma_i / w       # qubit intrinsic loss

            # Standard solve
            b = np.zeros(n)
            b[0] = 1/R

            M = K - w*w*C + 1j*w*G_eff
            phi = np.linalg.solve(M, b)
            S_phi0.append(phi[0])  # measure at port node
            S_phi1.append(phi[1])  # measure at resonator node
            S_phi2.append(phi[N//2])  # measure at resonator node (or phi[-1] if you want qubit node)

        return np.array(S_phi0), np.array(S_phi1), np.array(S_phi2)


    def build_A(K,C,G):
        n = C.shape[0]
        Cinv = inv(C)
        zero = np.zeros((n,n))
        I = np.eye(n)
        A = np.block([[zero, I], [-Cinv@K, -Cinv@G]])
        return A

    def compute_modes(A):
        lam, vec = eig(A)
        mask = np.imag(lam) > 0
        lam = lam[mask]
        vec = vec[:,mask]
        order = np.argsort(np.imag(lam))
        lam = lam[order]
        vec = vec[:,order]

        omega = np.imag(lam)
        freqs = omega/(2*pi)
        Q = omega / (2*(-np.real(lam)))
        return freqs, Q, lam, vec


    def compare_modes(freqs_plain, freqs_squid, f_q):
        plt.figure(figsize=(6,4))
        plt.plot(freqs_plain/1e9, '.', label='Plain chain')
        plt.plot(freqs_squid/1e9, 'x', label='Chain + SQUID')
        plt.axhline(f_q/1e9, color='r', linestyle='--', label='Bare qubit freq')
        plt.xlabel("Mode index")
        plt.ylabel("Frequency [GHz]")
        plt.legend()
        plt.title("Mode Frequencies (Plain vs SQUID)")
        plt.tight_layout()
        plt.show()

    # ======================================================================
    # 5. MAIN SCRIPT (INTEGRATED)
    # ======================================================================


    # Flux sweep
    Phi_list = np.linspace(0, 0.5*Phi0, 201)

    # Frequency span for Y_env
    f_span = np.linspace(1e9, 31e9, 2000)


    # ======================================================================
    # 5.1 COMPUTE MATRIX MODES (REFERENCE FOR PLOTS)
    # ======================================================================

    print("Computing matrix-chain modes (plain)...")
    # Kp,Cp,Gp = build_chain_plain(N, LJ, CJ, Cg, Cc, R)
    Kp,Cp,Gp = build_chain_plain(N, LJ, CJ, Cg_array, Cc_array, R)
    Ap = build_A(Kp,Cp,Gp)
    freq_plain, Q_plain, evals_plain, evecs_plain = compute_modes(Ap)

    print("Computing matrix-chain modes (with SQUID)...")
    Ks,Cs,Gs, LJ_s, CJ_s, idx_squid = build_chain_squid(
        N, LJ, CJ, Cg_array, Cc_array, R, qubit_params, Phi_list[-1], N_SQ=N_SQUID
    )
    As = build_A(Ks,Cs,Gs)
    freq_squid, Q_squid, evals_squid, evecs_squid = compute_modes(As)


    #############################
    # MODE ANALYSIS
    #############################
    # ============================================================
    # DROP-IN: add kappa (and kappa/2pi in MHz) to the mode-shape plots
    # Works with your existing freq[], Q[] arrays (from compute_modes).
    # ============================================================

    import numpy as np
    import matplotlib.pyplot as plt

    def _C_normalize_modes(V_nodes, C_nodes):
        V = np.array(V_nodes, dtype=complex, copy=True)
        for k in range(V.shape[1]):
            vk = V[:, k]
            nk = np.sqrt(np.abs(vk.conj().T @ (C_nodes @ vk)))
            if nk > 0:
                V[:, k] = vk / nk
        return V

    def _mode_participation(v, C):
        Cdiag = np.real(np.diag(C))
        p = (np.abs(v)**2) * np.maximum(Cdiag, 0.0)
        s = p.sum()
        return p / s if s > 0 else p

    def _extract_node_block_from_state_evecs(evecs, n_nodes, which="first"):
        V = np.array(evecs, dtype=complex)
        if V.shape[0] == n_nodes:
            return V
        if V.shape[0] == 2 * n_nodes:
            return V[:n_nodes, :] if which == "first" else V[-n_nodes:, :]
        raise ValueError(f"Unexpected evecs shape {V.shape}; expected {n_nodes} or {2*n_nodes} rows.")

    def analyze_and_plot_modes(tag, C, freq, Q, evecs, idx_squid=None,
                            n_show=6, remove_nodes=(0,), start_mode=0,
                            state_node_block="first"):
        """
        Adds kappa on plot titles:
        kappa = omega / Q  (rad/s)
        kappa/2pi = f / Q  (Hz)
        Uses freq in Hz (as returned by your compute_modes).
        """
        global MODE_IDX, FREQ1, FREQ2, KAPPA

        n = C.shape[0]

        # Extract node-component eigenvectors if evecs are for state-space A
        Vnodes_full = _extract_node_block_from_state_evecs(evecs, n_nodes=n, which=state_node_block)

        keep = np.ones(n, dtype=bool)
        for rn in remove_nodes:
            if 0 <= rn < n:
                keep[rn] = False

        Ck = C[np.ix_(keep, keep)]
        Vk = Vnodes_full[keep, :]
        Vk = _C_normalize_modes(Vk, Ck)

        nodes_kept = np.where(keep)[0]
        mode_ids = list(range(start_mode, min(start_mode + n_show, Vk.shape[1])))

        squid_plot_idx = None
        if idx_squid is not None and 0 <= idx_squid < n and keep[idx_squid]:
            squid_plot_idx = np.where(nodes_kept == idx_squid)[0][0]

        for m in mode_ids:
            v = Vk[:, m]

            # Choose a consistent global phase for pretty plots
            imax = np.argmax(np.abs(v))
            phase = np.exp(-1j * np.angle(v[imax])) if np.abs(v[imax]) > 0 else 1.0
            v = v * phase

            v_plot = np.real(v)
            p = _mode_participation(v, Ck)

            f_Hz = float(freq[m]) if np.isfinite(freq[m]) else np.nan
            Qm = float(Q[m]) if np.isfinite(Q[m]) else np.nan

            # kappa from Q and freq (no need for evals):
            # kappa/2pi (Hz) = f / Q
            # kappa (rad/s) = 2pi f / Q
            if np.isfinite(f_Hz) and np.isfinite(Qm) and Qm > 0:
                kappa_over_2pi_Hz = f_Hz / Qm
                kappa_over_2pi_MHz = kappa_over_2pi_Hz / 1e6
            else:
                kappa_over_2pi_MHz = np.nan

            fGHz = f_Hz / 1e9 if np.isfinite(f_Hz) else np.nan

            print(f"{tag} mode {m}: f={fGHz:.2f},{kappa_over_2pi_MHz:.3g}, MHz")

            if m==0 and (fGHz < 0.5):
                MODE_IDX += 1
                print(f"  Warning: mode {m} has out-of-range frequency {fGHz:.2f} GHz; check if it's a physical mode or a numerical artifact.")

            elif m == MODE_IDX:
                FREQ1 = fGHz
                KAPPA = kappa_over_2pi_MHz
            elif m == MODE_IDX+1:
                FREQ2 = fGHz

            # # --- Mode shape plot ---
            # plt.figure()
            # plt.plot(v_plot, marker="o")
            # if squid_plot_idx is not None:
            #     plt.axvline(squid_plot_idx, linestyle="--")
            # plt.title(
            #     f"{tag} mode {m}: f={fGHz:.4f} GHz, Q={Qm:.3g}, κ/2π={kappa_over_2pi_MHz:.3g} MHz"
            # )
            # plt.xlabel("Node index (after removal)")
            # plt.ylabel("Re(node amplitude), C-normalized")
            # plt.grid(True)
            # plt.show()

            # # --- Participation plot ---
            # plt.figure()
            # plt.plot(p, marker="o")
            # if squid_plot_idx is not None:
            #     plt.axvline(squid_plot_idx, linestyle="--")
            # plt.title(
            #     f"{tag} mode {m}: participation, κ/2π={kappa_over_2pi_MHz:.3g} MHz"
            # )
            # plt.xlabel("Node index (after removal)")
            # plt.ylabel("p_i (normalized)")
            # plt.grid(True)
            # plt.show()
    # =======================
    # Use it (drop-in)
    # =======================
    # analyze_and_plot_modes(
    #     tag="PLAIN",
    #     C=Cp,
    #     freq=freq_plain, Q=Q_plain,
    #     evecs=evecs_plain,
    #     idx_squid=None,
    #     n_show=6,
    #     remove_nodes=(0,),
    #     start_mode=0,
    #     state_node_block="first"   # if shapes look wrong, switch to "last"
    # )

    analyze_and_plot_modes(
        tag="SQUID",
        C=Cs,
        freq=freq_squid, Q=Q_squid,
        evecs=evecs_squid,
        idx_squid=idx_squid,
        n_show=4,
        remove_nodes=(0,),
        start_mode=0,
        state_node_block="first"   # if shapes look wrong, switch to "last"
    )
    # np.save("modes_plain.npy", evals_plain)
    # np.save("modes_squid.npy", evals_squid)




    EJ_eff = squid_EJ_eff(EJ, d, Phi_list[-1])
    wq = squid_plasma_freq(EJ_eff, 2*CJ)   # Cq = 2*CJ here
    # compare_modes(freq_plain, freq_squid, wq/(2*pi))

    # ======================================================================
    # 5.2 ENVIRONMENT ADMITTANCE SPECTRUM
    # ======================================================================

    # print("Computing Y_env(f)...")
    # try:
    #     Y_env = Y_env_spectrum(N, LJ, CJ, Cg_array, Cc_array, R, f_span, j_mid=N_SQUID)
    # except Exception as e:
    #     print(f"Error computing Y_env: {e}")
    #     Y_env = np.zeros_like(f_span, dtype=complex)

    # plt.figure(figsize=(7,6))
    # plt.subplot(2,1,1)
    # plt.semilogy(f_span*1e-9, np.abs(Y_env))
    # # for f in freq_plain:
    # #     plt.axvline(f*1e-9, color='gray', alpha=0.2)
    # for f in freq_squid:
    #     plt.axvline(f*1e-9, color='red', alpha=0.2)
    # plt.title("|Y_env| with chain modes (gray=plain, red=SQUID)")
    # plt.ylabel("|Y_env| [S]")

    # plt.subplot(2,1,2)
    # plt.plot(f_span*1e-9, np.real(Y_env))
    # # for f in freq_plain:
    # #     plt.axvline(f*1e-9, color='gray', alpha=0.2)
    # for f in freq_squid:
    #     plt.axvline(f*1e-9, color='red', alpha=0.2)
    # plt.yscale("log")
    # plt.xlabel("Frequency [GHz]")
    # plt.ylabel("Re[Y_env] [S]")
    # plt.tight_layout()
    # plt.show()


    # ======================================================================
    # 5.3 QUBIT FREQUENCY & PURCELL T1(Φ)
    # ======================================================================

    f_q_list = []
    f_q_list_scq = []
    # T1_list = []
    T1_list_qub = []

    T1_charge_list = []
    Tphi_charge_list = []
    T1_total_list = []
    T2_total_list = []
    n01_list = []
    domega_dng_list = []

    # for Phi in Phi_list:
        # w01, phi01 = transmon_freq_and_phi01(qubit_params, Phi)
        # f_q_list_scq.append(w01/(2*pi))

        # # Using our simple SQUID formulas for comparison
        # EJ_eff = squid_EJ_eff(EJ, d, Phi)
        # wq = squid_plasma_freq(EJ_eff, CJ*2)
        # fq = wq/(2*pi)
        # f_q_list.append(fq)

        # try:
        #     Yenv_q = Y_env_between_middle_nodes(N, LJ, CJ, Cg_array, Cc_array, R, wq, j_mid=N_SQUID)
        #     _, T1_qub = purcell_gamma_T1_qub(wq, phi01, Yenv_q)
        #     _, T1 = purcell_gamma_T1(CJ, CJ*2, Yenv_q)
        # except Exception as e:
        #     print(f"Error computing T1 for Phi={Phi}: {e}")
        #     T1_qub = 0
        #     T1 = 0

        # T1_qub = np.clip(T1_qub, 0, 1e3)   # clip to max 1e6 s for plotting
        # T1 = np.clip(T1, 0, 1e3)   # clip to max 1e6 s for plotting
        
        # T1_list_qub.append(T1_qub)
        # T1_list.append(T1)


    for Phi in Phi_list:
        # SQUID/transmon observables from scqubits
        w01, phi01, n01, evals_q, evecs_q = transmon_observables(qubit_params, Phi)
        domega_dng = transmon_charge_dispersion(qubit_params, Phi)

        f_q_list_scq.append(w01 / (2*pi))
        n01_list.append(np.abs(n01))
        domega_dng_list.append(domega_dng)

        # Simple SQUID formula for comparison only
        EJ_eff = squid_EJ_eff(EJ, d, Phi)
        wq = squid_plasma_freq(EJ_eff, CJ*2)
        fq = wq / (2*pi)
        f_q_list.append(fq)

        try:
            # Purcell from the chain environment
            Yenv_q = Y_env_between_middle_nodes(N, LJ, CJ, Cg_array, Cc_array, R, w01, j_mid=N_SQUID)
            Gamma_purcell, T1_purcell = purcell_gamma_T1_qub(w01, phi01, Yenv_q)
        except Exception as e:
            print(f"Error computing Yenv / Purcell for Phi={Phi}: {e}")
            Gamma_purcell, T1_purcell = 0.0, np.inf

        # -----------------------------
        # Charge-noise assumptions
        # -----------------------------
        # High-frequency PSD at omega01: unknown -> phenomenological fit parameter
        Sng_w01 = Sng_white(1e-23)          # offset-charge noise scaling roughly as 1 / 𝑓 𝛼 1/f α with 𝛼 ≈ 1.93 α≈1.93 and 𝑆 𝑞 ( 1 H z ) = 2.9 × 10 − 4 𝑒 2 / H z S q ​ (1Hz)=2.9×10 −4 e 2 /Hz

        # Slow rms offset-charge fluctuation for dephasing
        sigma_ng = 0.5              # example placeholder; treat as fit parameter

        Gamma1_ng, T1_ng = gamma_charge_T1_from_Sng(qubit_params, n01, Sng_w01)
        Gamma_phi_ng, Tphi_ng = gamma_charge_dephasing_from_dispersion(domega_dng, sigma_ng_eff=sigma_ng)

        # Combine
        Gamma1_tot, T1_tot, Gamma2_tot, T2_tot = combine_T1_Tphi(
            T1_purcell=T1_purcell,
            Gamma1_extra=Gamma1_ng,
            Gamma_phi=Gamma_phi_ng
        )

        # Clip for plotting
        T1_purcell = np.clip(T1_purcell, 0, 1e3)
        T1_ng = np.clip(T1_ng, 0, 1e3)
        Tphi_ng = np.clip(Tphi_ng, 0, 1e3)
        T1_tot = np.clip(T1_tot, 0, 1e3)
        T2_tot = np.clip(T2_tot, 0, 1e3)

        T1_list_qub.append(T1_purcell)
        T1_charge_list.append(T1_ng)
        Tphi_charge_list.append(Tphi_ng)
        T1_total_list.append(T1_tot)
        T2_total_list.append(T2_tot)


    f_q_list = np.array(f_q_list)
    f_q_list_scq = np.array(f_q_list_scq)
    # T1_list = np.array(T1_list)
    T1_list_qub = np.array(T1_list_qub)

    T1_charge_list = np.array(T1_charge_list)
    Tphi_charge_list = np.array(Tphi_charge_list)
    T1_total_list = np.array(T1_total_list)
    T2_total_list = np.array(T2_total_list)
    n01_list = np.array(n01_list)
    domega_dng_list = np.array(domega_dng_list)


    # plt.figure(figsize=(7,6))

    # plt.subplot(2,1,1)
    # plt.plot(Phi_list/Phi0, f_q_list*1e-9, label='simple SQUID')
    # plt.plot(Phi_list/Phi0, f_q_list_scq*1e-9, linestyle='--', label='scqubits')
    # for f in freq_squid:
    #     plt.axhline(f*1e-9, color='red', alpha=0.15)
    # plt.legend()
    # plt.ylabel("Qubit f(Φ) [GHz]")
    # plt.title("Qubit frequency & chain+SQUID modes (red lines)")

    # plt.subplot(2,1,2)
    # # plt.semilogy(Phi_list/Phi0, T1_list, label='simple SQUID')
    # plt.semilogy(Phi_list/Phi0, T1_list_qub, linestyle='--', label='scqubits')
    # # for f in freq_squid:
    # #     plt.axhline(1e-6, color='red', alpha=0.05)   # Just visual grid
    # plt.xlabel("Φ/Φ0")
    # plt.ylabel("Purcell T1(Φ) [s]")
    # plt.tight_layout()
    # plt.show()



    # plt.figure(figsize=(7,8))

    # plt.subplot(3,1,1)
    # plt.plot(Phi_list/Phi0, f_q_list_scq*1e-9, label='scqubits')
    # for f in freq_squid:
    #     plt.axhline(f*1e-9, color='red', alpha=0.15)
    # plt.ylabel("f01 [GHz]")
    # plt.legend()
    # plt.title("SQUID frequency and chain modes")

    # plt.subplot(3,1,2)
    # plt.semilogy(Phi_list/Phi0, T1_list_qub, label='Purcell T1')
    # plt.semilogy(Phi_list/Phi0, T1_charge_list, label='Charge-noise T1')
    # plt.semilogy(Phi_list/Phi0, T1_total_list, label='Total T1', linewidth=2)
    # plt.ylabel("T1 [s]")
    # plt.legend()

    # plt.subplot(3,1,3)
    # plt.semilogy(Phi_list/Phi0, Tphi_charge_list, label='Charge-noise Tphi')
    # plt.semilogy(Phi_list/Phi0, T2_total_list, label='Total T2', linewidth=2)
    # plt.xlabel("Phi / Phi0")
    # plt.ylabel("Time [s]")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()


    # plt.figure(figsize=(7,4))
    # plt.plot(Phi_list/Phi0, n01_list, label='|n01|')
    # plt.plot(Phi_list/Phi0, np.abs(domega_dng_list)/(2*pi*1e9), label='|d f01 / dng| [GHz/ng]')
    # plt.xlabel("Phi / Phi0")
    # plt.legend()
    # plt.title("Charge-noise sensitivity indicators")
    # plt.tight_layout()
    # plt.show()



    print("Purcell T1", round(T1_list_qub[-1]*1e6, 2), ", µs at Φ =", round(Phi_list[-1]/Phi0, 2), "Φ0")

    T1_PURCELL = T1_list_qub[-1]*1e6

    print("Charge-noise Tphi", round(Tphi_charge_list[-1]*1e6, 2), ", µs at Φ =", round(Phi_list[-1]/Phi0, 2), "Φ0")

    TPHI_CHARGE = Tphi_charge_list[-1]*1e6


    ## =======================================================================

    return FREQ1, FREQ2, KAPPA, T1_PURCELL, TPHI_CHARGE,G_COUPLING




from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# USER SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_CSV = SCRIPT_DIR / "chain_resonator_purcell_charge_nonsym.csv"


# ============================================================
# HELPERS
# ============================================================

def append_result_row(out_csv: Path, row_dict: dict):
    row_df = pd.DataFrame([row_dict])
    header = not out_csv.exists()
    row_df.to_csv(out_csv, mode="a", header=header, index=False)


# ============================================================
# MAIN SWEEP
# ============================================================

if __name__ == "__main__":
    N = 50
    fstLidL = 21
    Cc_first,Cc_last, Cc_middle, Cg_first, Cg_last, Cg_middle = 6.41126806708267e-15,8.850276669143131e-17,4.7066593819801306e-17,2.67008927453172e-16,3.5482902900488005e-17,1.85398480144286e-17


    # Sweep grids
    # 0.1 to 2.8 inclusive, step 0.1
    L_um_SQUID_vals = np.round(np.arange(0.1, 2.8 + 1e-12, 0.1), 10)

    # 0.001 to 0.5, logarithmically spaced, 30 points
    d_length_vals = np.logspace(np.log10(0.001), np.log10(0.5), 30)

    total_points = len(L_um_SQUID_vals) * len(d_length_vals)
    print(f"Total sweep points: {total_points}")

    # Optional: remove old CSV if you want a clean run every time
    # if OUT_CSV.exists():
    #     OUT_CSV.unlink()

    with tqdm(total=total_points, desc="Sweeping") as pbar:
        for L_um_SQUID in L_um_SQUID_vals:
            for d_length in d_length_vals:
                try:
                    FREQ1, FREQ2, KAPPA, T1_PURCELL, TPHI_CHARGE, G_COUPLING = simulate(
                        N,
                        Cc_first, Cc_last, Cc_middle,
                        Cg_first, Cg_last, Cg_middle,
                        L_um_SQUID, d_length
                    )

                    row = {
                        "N": N,
                        "fstLidL": fstLidL,
                        "Cc_first": Cc_first,
                        "Cc_last": Cc_last,
                        "Cc_middle": Cc_middle,
                        "Cg_first": Cg_first,
                        "Cg_last": Cg_last,
                        "Cg_middle": Cg_middle,
                        "L SQUID um": L_um_SQUID,
                        "d": d_length,
                        "FREQ1 GHz": FREQ1,
                        "FREQ2 GHz": FREQ2,
                        "KAPPA MHz": KAPPA,
                        "T1_PURCELL us": T1_PURCELL,
                        "TPHI_CHARGE us": TPHI_CHARGE,
                        "G_COUPLING MHz": G_COUPLING,
                    }

                    append_result_row(OUT_CSV, row)

                except Exception as e:
                    print(f"\nFailed at L_um_SQUID={L_um_SQUID}, d={d_length}: {e}")

                pbar.update(1)

    print(f"\nSaved results to: {OUT_CSV}")
