

def simulate(N, Cc_first,Cc_last, Cc_middle, Cg_first, Cg_last, Cg_middle):
    global MODE_IDX, FREQ1, FREQ2, KAPPA, T1_PURCELL, G_COUPLING
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
    # N = 30              # number of cells
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
    L_um = 2.4                         # Length in um

    Area_um2 = W_um * L_um
    CJ_qubit = C_area_fF_per_um2 * Area_um2 * fF_to_F # Junction capacitance C in Farads (F)
    CJ_qubit = CJ_qubit # for one junction;

    LJ_qubit = 1.0 / (wp_target**2 * CJ_qubit)  # Josephson Inductance L in Henries (H)

    EJ_qubit = 2*(Phi0 / (2 * np.pi))**2 / LJ_qubit # Josephson Energy EJ in Joules (J) # factor of 2 for the SQUID
    EJ_qubit_GHz = EJ_qubit / h  / GHz              # Josephson Energy EJ in GHz

    EC_qubit = e**2 / (2 * CJ_qubit * 2)             # Charging Energy EC in Joules (J) # factor of 2 for the SQUID since the two junctions are in parallel, so total C is 2*CJ_qubit
    EC_qubit_GHz = EC_qubit / h  / GHz                  # Charging Energy EC in GHz


    d = 0.03

    Phi_ext = 0.5*Phi0


    Cc = 0.16e-15 #0.1e-15             # [F]

    N_SQUID = N-2#N//2


    qubit_params = {
        'EJ_GHz': EJ_qubit_GHz,
        'EC_GHz': EC_qubit_GHz,
        'd': d,
        'ng': 0.0,
        'ncut': 20,
        'truncated_dim': 10,
        'CJ': CJ_qubit*2, # total capacitance of the SQUID (two junctions in parallel)
    }


    ###########################################

    MODE_IDX = 1
    FREQ1, FREQ2, KAPPA, T1_PURCELL, G_COUPLING = None, None, None, None, None


    Cc_array = np.full(N+1, Cc_middle) #[::2]  # Take every other element to match the islands (assuming interleaved)
    Cg_array = np.full(N+1, Cg_middle) #[::2]  # Take every other element to match the islands (assuming interleaved)

    Cc_array[0], Cc_array[-1] = Cc_first, Cc_last
    Cg_array[0], Cg_array[-1] = Cg_first, Cg_last

    ###########################################

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


    def transmon_freq_and_phi01(qubit_params, flux):
        """
        Use scqubits TunableTransmon to get:
        - ωq(Φ) from 0→1 transition
        - approximate φ_01(Φ) via sin(phi) operator (small-phase limit)
        """
        qubit = scq.TunableTransmon(
            EJmax=qubit_params['EJ_GHz'],
            EC=qubit_params['EC_GHz'],
            ng=qubit_params['ng'],
            d=qubit_params['d'],
            ncut=qubit_params['ncut'],
            truncated_dim=qubit_params['truncated_dim'],
            flux=flux/Phi0,
        )
        evals, evecs = qubit.eigensys(evals_count=2)
        # evals in GHz; convert to angular frequency
        w01 = 2 * pi * (evals[1] - evals[0]) * 1e9

        # Use sin(phi) operator; in small-phase regime sin φ ≈ φ
        sinphi = qubit.sin_phi_operator()        # matrix in charge basis
        sinphi01 = (evecs[:, 0].conj().T @ sinphi @ evecs[:, 1]).item()
        phi01 = sinphi01  # dimensionless φ_01 ≈ <0|sin φ|1>

        return w01, phi01


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

        w01, _ = transmon_freq_and_phi01(qubit_params, Phi_ext)

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
                MODE_IDX = 2
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

    print("Computing Y_env(f)...")
    try:
        Y_env = Y_env_spectrum(N, LJ, CJ, Cg_array, Cc_array, R, f_span, j_mid=N_SQUID)
    except Exception as e:
        print(f"Error computing Y_env: {e}")
        Y_env = np.zeros_like(f_span, dtype=complex)

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
    T1_list = []
    T1_list_qub = []

    for Phi in Phi_list:
        w01, phi01 = transmon_freq_and_phi01(qubit_params, Phi)
        f_q_list_scq.append(w01/(2*pi))

        # Using our simple SQUID formulas for comparison
        EJ_eff = squid_EJ_eff(EJ, d, Phi)
        wq = squid_plasma_freq(EJ_eff, CJ*2)
        fq = wq/(2*pi)
        f_q_list.append(fq)

        try:
            Yenv_q = Y_env_between_middle_nodes(N, LJ, CJ, Cg_array, Cc_array, R, wq, j_mid=N_SQUID)
            _, T1_qub = purcell_gamma_T1_qub(wq, phi01, Yenv_q)
            _, T1 = purcell_gamma_T1(CJ, CJ*2, Yenv_q)
        except Exception as e:
            print(f"Error computing T1 for Phi={Phi}: {e}")
            T1_qub = 0
            T1 = 0

        T1_qub = np.clip(T1_qub, 0, 1e3)   # clip to max 1e6 s for plotting
        T1 = np.clip(T1, 0, 1e3)   # clip to max 1e6 s for plotting
        
        T1_list_qub.append(T1_qub)
        T1_list.append(T1)

    f_q_list = np.array(f_q_list)
    f_q_list_scq = np.array(f_q_list_scq)
    T1_list = np.array(T1_list)
    T1_list_qub = np.array(T1_list_qub)



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
    # plt.semilogy(Phi_list/Phi0, T1_list, label='simple SQUID')
    # plt.semilogy(Phi_list/Phi0, T1_list_qub, linestyle='--', label='scqubits')
    # # for f in freq_squid:
    # #     plt.axhline(1e-6, color='red', alpha=0.05)   # Just visual grid
    # plt.xlabel("Φ/Φ0")
    # plt.ylabel("Purcell T1(Φ) [s]")
    # plt.tight_layout()
    # plt.show()

    print("Purcell T1", round(T1_list_qub[-1]*1e6, 2), ", µs at Φ =", round(Phi_list[-1]/Phi0, 2), "Φ0")

    T1_PURCELL = T1_list_qub[-1]*1e6


    # ======================================================================


    from tqdm import tqdm
    ###############################################################################
    # MAIN SIMULATION LOOP
    ###############################################################################

    # Phi_list = np.linspace(0, 0.5*Phi0, 1001)
    # f_span = np.linspace(8e9, 31e9, 5000)

    Phi_list = np.linspace(0.1*Phi0, 0.5*Phi0, 801)
    f_span = np.linspace(FREQ1*1e9-3e9, FREQ1*1e9+3e9, 500)
    # f_span = np.linspace(10e9, 31e9, 500)


    S_phi0 = np.zeros((len(Phi_list), len(f_span)), dtype=complex)
    S_phi1 = np.zeros((len(Phi_list), len(f_span)), dtype=complex)
    S_phi2 = np.zeros((len(Phi_list), len(f_span)), dtype=complex)

    f_q_list = []
    f_q_list_scq = []

    for i, Phi in enumerate(tqdm(Phi_list)):

        w01, phi01 = transmon_freq_and_phi01(qubit_params, Phi)
        f_q_list_scq.append(w01/(2*pi))

        # Using our simple SQUID formulas for comparison
        EJ_eff = squid_EJ_eff(EJ, d, Phi)
        wq = squid_plasma_freq(EJ_eff, CJ*2)
        fq = wq/(2*pi)
        f_q_list.append(fq)

        Ks,Cs,Gs, LJ_s, CJ_s, idx_squid = build_chain_squid(
            N, LJ, CJ, Cg_array, Cc_array, R, qubit_params, Phi, N_SQ=N_SQUID
        )

        kappa_i = 0#2*np.pi*1e3
        gamma_i = 0#2*np.pi*2e6

        S_phi0[i,:], S_phi1[i,:], S_phi2[i,:] = s21_lossy(
            Ks, Cs, Gs,
            R,
            f_span,
            kappa_i=kappa_i,
            gamma_i=gamma_i
        )


    f_q_list_scq = np.array(f_q_list_scq, dtype=float)
    f_q_list = np.array(f_q_list, dtype=float)


    # =======================================================================


    from scipy.signal import find_peaks

    # ###############################################################################
    # # 1) Bare first chain mode (ignore spurious near 0 Hz)
    # ###############################################################################

    # # Rebuild plain chain and its modes (fast; keeps everything self-contained)
    # Kp, Cp, Gp = build_chain_plain(N, LJ, CJ, Cg, Cc, R)
    # Ap = build_A(Kp, Cp, Gp)
    # freq_plain, Q_plain, _, _ = compute_modes(Ap)   # [Hz]

    freq_plain = freq_squid  # use modes with SQUID for better accuracy

    # Ignore spurious ultra-low mode, keep first mode above some threshold
    f_min_th = 1e9   # 1 GHz threshold; adjust if needed
    mask_valid = freq_plain > f_min_th
    if not np.any(mask_valid):
        raise RuntimeError("No chain mode found above f_min_th; adjust threshold.")

    f_chain0 = freq_plain[mask_valid][1]   # bare 0th chain mode [Hz]

    print(f"Bare first chain mode (above {f_min_th/1e9:.1f} GHz): "
        f"f_chain0 = {f_chain0/1e9:.3f} GHz")

    ###############################################################################
    # 2) Flux where qubit (scqubits) crosses the bare chain mode
    ###############################################################################

    # f_q_list_scq is already in Hz from your loop
    diff = np.abs(f_q_list_scq - f_chain0)
    idx_cross = int(np.argmin(diff))          # index in Phi_list where crossing is closest
    print(f"Index of closest crossing: {idx_cross}")
    print(Phi_list.shape, f_q_list_scq.shape)
    Phi_cross = Phi_list[idx_cross]           # [Wb]
    Phi_cross_over_Phi0 = Phi_cross / Phi0
    f_q_cross = f_q_list_scq[idx_cross]       # [Hz]

    print(f"Closest qubit–chain crossing at:")
    print(f"  Phi ≈ {Phi_cross_over_Phi0:.3f} Φ0")
    print(f"  f_q(Phi) ≈ {f_q_cross/1e9:.3f} GHz")

    ###############################################################################
    # 3) On this flux slice, find upper/lower hybridized peaks of the first mode
    ###############################################################################

    # Take |S21| at node 2 (S_phi2) at this flux
    S_line = np.abs(S_phi2[idx_cross, :])   # shape [len(f_span)]

    # Focus on a frequency window around the bare chain mode
    window_half_width = 2e9  # 2 GHz window on each side; adjust if needed
    f_low = f_chain0 - window_half_width
    f_high = f_chain0 + window_half_width

    freq_mask = (f_span >= f_low) & (f_span <= f_high)
    freq_idx = np.where(freq_mask)[0]

    if len(freq_idx) < 10:
        raise RuntimeError("Window around chain mode too small or out of f_span; adjust window.")

    S_win = S_line[freq_mask]
    f_win = f_span[freq_mask]

    # Find peaks in this window
    peaks, properties = find_peaks(S_win, height=np.max(S_win)*0.1)  # 10% height threshold
    if len(peaks) < 2:
        # fallback: take the two highest points as 'peaks'
        sort_idx = np.argsort(S_win)[-2:]
        peaks = np.sort(sort_idx)
    else:
        # take two highest peaks among those found
        peak_heights = S_win[peaks]
        best_two = np.argsort(peak_heights)[-2:]   # indices in 'peaks'
        peaks = np.sort(peaks[best_two])

    # Frequencies of the two hybridized peaks
    f_minus = f_win[peaks[0]]   # lower branch [Hz]
    f_plus  = f_win[peaks[1]]   # upper branch [Hz]

    # Splitting and coupling
    delta_f = f_plus - f_minus      # total splitting [Hz]
    g_Hz = delta_f / 2.0            # g in Hz
    g_rad_s = 2 * np.pi * g_Hz      # g in rad/s
    g_over_2pi_MHz = g_Hz / 1e6     # g/2π in MHz

    print("\n===== Extracted coupling g from avoided crossing =====")
    print(f"  Lower peak f- = {f_minus/1e9:.3f} GHz")
    print(f"  Upper peak f+ = {f_plus/1e9:.3f} GHz")
    print(f"  Splitting Δf = {delta_f/1e9:.3f} GHz")
    print(f"  g = Δf/2 = {g_Hz/1e9:.3f} GHz = {g_over_2pi_MHz:.2f} MHz  (g/2π)")


    G_COUPLING = g_over_2pi_MHz


    return FREQ1, FREQ2, KAPPA, T1_PURCELL,G_COUPLING


import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


CAP_DIR = Path(r"C:\Users\gusarov\Desktop\Plasmon_Lumped_chain+trans_CapCalc_nonsym2")
CAP_CSV = Path("capacitance_catalog_nonsym2.csv")
OUT_CSV = Path("chain_resonator_data_nonsym2.csv")

VALID_CAP_NAMES = {
    "Cc_first", "Cc_last", "Cc_middle",
    "Cg_first", "Cg_last", "Cg_middle",
}


def parse_filename(path: Path):
    """
    Examples:
        Cc_first_nonsym_N80.xlsx
        Cg_last_nonsym_N90.csv
    Returns:
        (cap_name, N) or None
    """
    stem = path.stem

    m_cap = re.search(r"(Cc_first|Cc_last|Cc_middle|Cg_first|Cg_last|Cg_middle)", stem)
    m_N = re.search(r"_N(\d+)\b", stem, flags=re.IGNORECASE)

    if m_cap is None or m_N is None:
        return None

    cap_name = m_cap.group(1)
    N = int(m_N.group(1))

    if cap_name not in VALID_CAP_NAMES:
        return None

    return cap_name, N


def read_cap_file(path: Path, cap_name: str, N: int) -> pd.DataFrame:
    """
    Read one capacitance file and return long-format dataframe:
        N, fstLidL, cap_name, value
    Supports .xlsx/.xls/.csv/.txt
    """
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        # first try tab-separated
        try:
            df = pd.read_csv(path, sep="\t", engine="python")
            if df.shape[1] < 2:
                raise ValueError("Too few columns with tab parser")
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Find fstLidL column
    lid_col = None
    for c in df.columns:
        c_low = c.lower().replace(" ", "")
        if "fstlidl" in c_low:
            lid_col = c
            break

    if lid_col is None:
        raise ValueError(f"Could not find fstLidL column in {path.name}. Columns: {list(df.columns)}")

    # Find capacitance column = anything that's not freq or fstLidL
    cap_col = None
    for c in df.columns:
        c_low = c.lower()
        if c == lid_col:
            continue
        if "freq" in c_low:
            continue
        cap_col = c
        break

    if cap_col is None:
        raise ValueError(f"Could not find capacitance column in {path.name}. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "N": N,
        "fstLidL": pd.to_numeric(df[lid_col], errors="coerce"),
        "cap_name": cap_name,
        "value": pd.to_numeric(df[cap_col], errors="coerce"),
    })

    out = out.dropna(subset=["fstLidL", "value"]).copy()

    # raw files are in fF and negative by Maxwell convention
    # convert to positive Farads like in your previous CSV
    out["value"] = np.abs(out["value"]) * 1e-15

    out["N"] = out["N"].astype(int)
    out["fstLidL"] = out["fstLidL"].astype(float)

    return out


def build_capacitance_catalog(cap_dir: Path) -> pd.DataFrame:
    if not cap_dir.exists():
        raise FileNotFoundError(f"Folder does not exist: {cap_dir}")

    all_files = [p for p in cap_dir.iterdir() if p.is_file()]
    recognized = []
    skipped = []

    for p in sorted(all_files):
        parsed = parse_filename(p)
        if parsed is None:
            skipped.append(p.name)
        else:
            cap_name, N = parsed
            recognized.append((p, cap_name, N))

    print(f"Found {len(all_files)} files in total.")
    print(f"Recognized {len(recognized)} capacitance files.")

    if skipped:
        print("\nSkipped files:")
        for s in skipped:
            print("  ", s)

    if not recognized:
        raise RuntimeError("No valid capacitance files found.")

    dfs = []
    for path, cap_name, N in recognized:
        try:
            df_one = read_cap_file(path, cap_name, N)
            dfs.append(df_one)
            print(f"Loaded {path.name}: {len(df_one)} rows")
        except Exception as e:
            print(f"Failed to read {path.name}: {e}")

    if not dfs:
        raise RuntimeError("All files failed to read.")

    long_df = pd.concat(dfs, ignore_index=True)

    print("\nLong-format preview:")
    print(long_df.head(10).to_string(index=False))

    # Pivot once instead of repeated merges
    merged = (
        long_df
        .pivot_table(
            index=["N", "fstLidL"],
            columns="cap_name",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # remove columns name left by pivot
    merged.columns.name = None

    merged = merged.sort_values(["N", "fstLidL"]).reset_index(drop=True)

    print("\nMerged capacitance catalog preview:")
    print(merged.head(20).to_string(index=False))

    required = [
        "Cc_first", "Cc_last", "Cc_middle",
        "Cg_first", "Cg_last", "Cg_middle",
    ]

    missing_cols = [c for c in required if c not in merged.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required capacitance columns after pivot: {missing_cols}")

    missing_mask = merged[required].isna().any(axis=1)
    if missing_mask.any():
        print("\nWARNING: Incomplete rows detected:")
        print(merged.loc[missing_mask, ["N", "fstLidL"] + required].to_string(index=False))

    return merged
# ============================================================
# YOUR SIMULATION FUNCTION
# Replace this with your actual implementation/import.
# Must return:
#   FREQ1, FREQ2, KAPPA, T1_PURCELL, G_COUPLING
# ============================================================

# ============================================================
# RUN SIMULATIONS AND APPEND TO CSV
# - resumes if OUT_CSV already exists
# - writes after each successful simulation
# - uses tqdm
# ============================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
from tqdm import tqdm


def load_done_pairs(out_csv):
    if not out_csv.exists():
        return set()

    df_done = pd.read_csv(out_csv)
    if not {"N", "fstLidL"}.issubset(df_done.columns):
        return set()

    return set(zip(df_done["N"].astype(int), df_done["fstLidL"].astype(float)))


def append_result_row(out_csv, row_dict):
    row_df = pd.DataFrame([row_dict])
    header = not out_csv.exists()
    row_df.to_csv(out_csv, mode="a", header=header, index=False)


def worker_simulate(row_dict):
    """
    Top-level worker function so it is picklable on Windows.
    """
    N = int(row_dict["N"])
    fstLidL = float(row_dict["fstLidL"])

    Cc_first = float(row_dict["Cc_first"])
    Cc_last = float(row_dict["Cc_last"])
    Cc_middle = float(row_dict["Cc_middle"])
    Cg_first = float(row_dict["Cg_first"])
    Cg_last = float(row_dict["Cg_last"])
    Cg_middle = float(row_dict["Cg_middle"])

    FREQ1, FREQ2, KAPPA, T1_PURCELL, G_COUPLING = simulate(
        N, Cc_first, Cc_last, Cc_middle, Cg_first, Cg_last, Cg_middle
    )

    return {
        "N": N,
        "fstLidL": fstLidL,
        "Cc_first": Cc_first,
        "Cc_last": Cc_last,
        "Cc_middle": Cc_middle,
        "Cg_first": Cg_first,
        "Cg_last": Cg_last,
        "Cg_middle": Cg_middle,
        "FREQ1": FREQ1,
        "FREQ2": FREQ2,
        "KAPPA": KAPPA,
        "T1_PURCELL": T1_PURCELL,
        "G_COUPLING": G_COUPLING,
    }


def run_all_simulations_parallel(cap_df, out_csv, max_workers=None):
    required = [
        "Cc_first", "Cc_last", "Cc_middle",
        "Cg_first", "Cg_last", "Cg_middle",
    ]

    sim_df = cap_df.dropna(subset=required).copy()
    sim_df = sim_df.sort_values(["N", "fstLidL"]).reset_index(drop=True)

    done = load_done_pairs(out_csv)

    todo = []
    for _, row in sim_df.iterrows():
        key = (int(row["N"]), float(row["fstLidL"]))
        if key not in done:
            todo.append({
                "N": int(row["N"]),
                "fstLidL": float(row["fstLidL"]),
                "Cc_first": float(row["Cc_first"]),
                "Cc_last": float(row["Cc_last"]),
                "Cc_middle": float(row["Cc_middle"]),
                "Cg_first": float(row["Cg_first"]),
                "Cg_last": float(row["Cg_last"]),
                "Cg_middle": float(row["Cg_middle"]),
            })

    print(f"\nTotal valid configs: {len(sim_df)}")
    print(f"Already done: {len(done)}")
    print(f"Remaining: {len(todo)}")

    if len(todo) == 0:
        return

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    print(f"Using {max_workers} worker processes")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(worker_simulate, job): job for job in todo
        }

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Simulating"):
            job = future_to_job[future]
            try:
                result = future.result()
                append_result_row(out_csv, result)
            except Exception as e:
                print(f"\nFailed at N={job['N']}, fstLidL={job['fstLidL']}: {e}")
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cap_df = build_capacitance_catalog(CAP_DIR)
    cap_df.to_csv(CAP_CSV, index=False)
    print(f"\nSaved capacitance catalog to: {CAP_CSV.resolve()}")

    run_all_simulations_parallel(cap_df, OUT_CSV, max_workers=6)

    print(f"\nSaved/updated results in: {OUT_CSV.resolve()}")