import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import astropy.constants as const
import astropy.units as u
import numpy as np
from contop import lte_h_ion_fracs, continuum_opacity
from voigt import voigt_H_re as voigt
import jax_dataclasses as jdc
import lightweaver as lw

HC = const.h.value * const.c.value
NM_TO_M = u.Unit('nm').to('m')
M_TO_NM = u.Unit('m').to('nm')
E_RYD = const.Ryd.to('J', equivalencies=u.spectral()).value
Q_ELE = u.eV.to(u.J)
EPS_0 = const.eps0.value
M_ELE = const.m_e.value
K_B = const.k_B.value
K_B_EV = const.k_B.to('eV / K').value
K_B_U = (const.k_B / const.u).value
INV_C = 1.0 / const.c.value
INV_FOURPI_C = 1.0 / (4.0 * np.pi * const.c.value)
HC_FOURPI_KJ_NM = (const.h * const.c).to('kJ nm').value / (4.0 * np.pi)
SQRT_PI = np.sqrt(np.pi)
SAHA_CONST = ((2 * jnp.pi * const.m_e.value * const.k_B.value) / const.h.value**2)**1.5
TWOHC2_NM5 = 2.0 * (const.h.to('kJ s') * const.c**2 / (1e-9**4)).value
HC_KB_NM = (const.h * const.c / const.k_B).to('K nm').value

@jdc.pytree_dataclass
class AtomicData:
    mass: jax.Array
    elem: jax.Array
    stage: jax.Array
    abund: jax.Array
    lambda0: jax.Array
    log_grad: jax.Array
    log_gs: jax.Array
    log_gw: jax.Array
    gi: jax.Array
    gj: jax.Array
    ei: jax.Array
    ej: jax.Array
    Aji: jax.Array


# https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/database/atomllapi.py
def air_to_vac(wlair):
    """Convert wavelengths [AA] in air into those in vacuum.

    * See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Args:
        wlair:  wavelengthe in air [Angstrom]
        n:  Refractive Index in dry air at 1 atm pressure and 15ºC with 0.045% CO2 by volume (Birch and Downs, 1994, Metrologia, 31, 315)

    Returns:
        wlvac:  wavelength in vacuum [Angstrom]
    """
    s = 1e4 / wlair
    n = (
        1.0
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s * s)
        + 0.0001599740894897 / (38.92568793293 - s * s)
    )
    wlvac = wlair * n
    return wlvac

# Modified from https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/database/atomllapi.py
def read_kurucz(kuruczf):
    """Input Kurucz line list (http://kurucz.harvard.edu/linelists/)

    Args:
        kuruczf: file path

    Returns:
        AtomicData, containing some of:
        A:  Einstein coefficient in [s-1]
        wavelength:  vacuum transition wavelength in [nm]
        elower: lower excitation potential [eV]
        eupper: upper excitation potential [eV]
        glower: lower statistical weight
        gupper: upper statistical weight
        jlower: lower J (rotational quantum number, total angular momentum)
        jupper: upper J
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly)
        gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta: log of gamma of Stark damping (s-1 / m-3)
        gamvdW:  log of (van der Waals damping constant / neutral hydrogen number) (s-1 / m-3)
    """
    ccgs = 29979245800.0  # c in cgs
    ecgs = 4.80320450e-10  # [esu]=[dyn^0.5*cm] #elementary charge
    mecgs = 9.10938356e-28  # [g] !electron mass
    with open(kuruczf) as f:
        lines = f.readlines()
    (
        wlnmair,
        loggf,
        species,
        elower,
        jlower,
        labellower,
        eupper,
        jupper,
        labelupper,
        gamRad,
        gamSta,
        gamvdW,
        ref,
        NLTElower,
        NLTEupper,
        isonum,
        hyperfrac,
        isonumdi,
        isofrac,
        hypershiftlower,
        hypershiftupper,
        hyperFlower,
        hypernotelower,
        hyperFupper,
        hypternoteupper,
        strenclass,
        auto,
        landeglower,
        landegupper,
        isoshiftmA,
    ) = (
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.array([""] * len(lines), dtype=object),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
        np.zeros(len(lines)),
    )
    ielem, iion = np.zeros(len(lines), dtype=int), np.zeros(len(lines), dtype=int)

    for i, line in enumerate(lines):
        wlnmair[i] = float(line[0:11])
        loggf[i] = float(line[11:18])
        species[i] = str(line[18:24])
        ielem[i] = int(species[i].split(".")[0])
        iion[i] = int(species[i].split(".")[1]) + 1
        elower[i] = float(line[24:36])
        jlower[i] = float(line[36:41])
        eupper[i] = float(line[52:64])
        jupper[i] = float(line[64:69])
        gamRad[i] = float(line[80:86])
        gamSta[i] = float(line[86:92])
        gamvdW[i] = float(line[92:98])

    elower_inverted = np.where((eupper - elower) > 0, elower, eupper)
    eupper_inverted = np.where((eupper - elower) > 0, eupper, elower)
    jlower_inverted = np.where((eupper - elower) > 0, jlower, jupper)
    jupper_inverted = np.where((eupper - elower) > 0, jupper, jlower)
    elower = elower_inverted
    eupper = eupper_inverted
    jlower = jlower_inverted
    jupper = jupper_inverted

    wlaa = np.where(wlnmair < 200, wlnmair * 10, air_to_vac(wlnmair * 10))
    wl_vac = np.where(wlnmair < 200, wlnmair, air_to_vac(wlnmair * 10) * 0.1)
    nu_lines = 1e8 / wlaa[::-1]  # [cm-1]<-[AA]
    elower = (elower << u.Unit('cm-1')).to('eV', equivalencies=u.spectral()).value
    eupper = (eupper << u.Unit('cm-1')).to('eV', equivalencies=u.spectral()).value
    glower = jlower * 2 + 1
    gupper = jupper * 2 + 1
    A = (
        10**loggf
        / gupper
        * (ccgs * nu_lines) ** 2
        * (8 * np.pi**2 * ecgs**2)
        / (mecgs * ccgs**3)
    )
    gamSta = gamSta - 6.0
    gamvdW = gamvdW - 6.0

    return AtomicData(
        mass=jnp.array([lw.PeriodicTable[lw.Element(Z=z)].mass for z in ielem]),
        elem=jnp.array(ielem),
        stage=jnp.array(iion),
        abund=jnp.array([lw.DefaultAtomicAbundance[lw.Element(Z=z)] for z in ielem]),
        lambda0=jnp.array(wl_vac),
        log_grad=jnp.array(gamRad),
        log_gs=jnp.array(gamSta),
        log_gw=jnp.array(gamvdW),
        gi=jnp.array(glower),
        gj=jnp.array(gupper),
        ei=jnp.array(elower),
        ej=jnp.array(eupper),
        Aji=jnp.array(A)
    )

    # return (
    #     A,
    #     wl_vac,
    #     elower,
    #     eupper,
    #     glower,
    #     gupper,
    #     jlower,
    #     jupper,
    #     ielem,
    #     iion,
    #     gamRad,
    #     gamSta,
    #     gamvdW,
    # )


def thermal_vel(mass, temperature):
    r"""
    /** Compute mean thermal velocity
    * \param mass [u]
    * \param temperature [K]
    * \return mean thermal velocity [m/s]
    */
    """
    return jnp.sqrt(2.0 * temperature / mass * K_B_U)

def doppler_width(lambda0, mass, temperature, vturb):
    r"""
    /** Compute doppler width
    * \param lambda0 wavelength [nm]
    * \param mass [u]
    * \param temperature [K]
    * \param vturb microturbulent velocity [m / s]
    * \return Doppler width [nm]
    */
    """
    return lambda0 * jnp.sqrt(2.0 * K_B_U * temperature / mass + vturb**2) * INV_C

def damping_from_gamma(gamma, wave, dop_width):
    r"""
    /** Compute damping coefficient for Voigt profile from gamma
    * \param gamma [rad / s]
    * \param wave [nm]
    * \param dop_width [nm]
    * \return gamma / (4 pi dnu_D) = gamma / (4 pi dlambda_D) * wave**2
    */
    """
    # NOTE(cmo): Extra 1e-9 to convert c to nm / s (all lengths here in nm)
    return INV_FOURPI_C * gamma * 1e-9 * wave**2 / dop_width

def gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi):
    """
    Compute damping gamma from the line parameters and atmospheric parameters
    """
    grad = 10**log_grad
    gstark = 10**log_gs * ne
    gvdw = 10**log_gw * nhi

    gamma = grad + gstark + gvdw
    return gamma

def planck(wave, temperature):
    """
    Compute the Planck function (in kW/(nm m2 sr))

    wave : float
        Wavelength [nm]
    temperature : float
        Temperature [K]
    """
    return TWOHC2_NM5 / (wave**5 * (jnp.exp(HC_KB_NM / (wave * temperature)) - 1.0))

def emis_opac_line(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
):
    """
    Compute emissivity/opacity for a single LTE line
    """
    nhi, nhii = lte_h_ion_fracs(temperature, ne, nhtot)
    dop_width = doppler_width(lambda0, mass, temperature, vturb)
    gamma = gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi)
    adamp = damping_from_gamma(gamma, wave, dop_width)
    v = ((wave - lambda0) + (vel * lambda0) * INV_C) / dop_width
    p = voigt(adamp, v) / (SQRT_PI * dop_width)

    hnu_4pi = HC_FOURPI_KJ_NM / wave
    Uji = hnu_4pi * Aji * p
    Sfn = planck(wave, temperature)
    nj = fei_pop_i(abund, temperature, ne, nhtot, gj, ej)
    eta = nj * Uji
    chi = eta / Sfn
    return eta, chi

def emis_opac(adata: AtomicData, wave, temperature, ne, nhtot, vel, vturb):
    """
    Compute total emissivity/opacity for a atmospheric point

    adata : AtomicData
        The dataclass containing the atomic data
    wave : float
        The wavelength at which to compute [nm]
    temperature : float
        Temperature [K]
    ne : float
        Electron density [m-3]
    nhtot : float
        Total H density [m-3]
    vel : float
        LOS velocity [m/s]
    vturb : float
        microturbulent velocity [m/s]

    Returns
    -------
    Total emissivity, total opacity, from continuum and all lines.
    """
    chi_cont = continuum_opacity(wave, temperature, ne, nhtot)
    B_planck = planck(wave, temperature)
    eta_cont = chi_cont * B_planck

    axis_spec = (*[0]*9, *[None]*6)
    line_eta, line_chi = jax.vmap(emis_opac_line, in_axes=axis_spec)(
        adata.mass,
        adata.abund,
        adata.lambda0,
        adata.log_grad,
        adata.log_gs,
        adata.log_gw,
        adata.gj,
        adata.ej,
        adata.Aji,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb
    )

    # line_eta = 0.0
    # line_chi = 0.0
    # for i in range(adata.mass.shape[0]):
    #     e, c = emis_opac_line(
    #         adata.mass[i],
    #         adata.abund[i],
    #         adata.lambda0[i],
    #         adata.log_grad[i],
    #         adata.log_gs[i],
    #         adata.log_gw[i],
    #         adata.gj[i],
    #         adata.ej[i],
    #         adata.Aji[i],
    #         wave,
    #         temperature,
    #         ne,
    #         nhtot,
    #         vel,
    #         vturb
    #     )
    #     line_eta = line_eta + e
    #     line_chi = line_chi + c

    eta = eta_cont + line_eta.sum()
    chi = chi_cont + line_chi.sum()
    return eta, chi



def Q_FeI(T):
    a = jnp.array([
        -1.15609527e3,
        7.46597652e2,
        -1.92865672e2,
        2.49658410e1,
        -1.61934455e0,
        4.21182087e-2
    ])
    lnQ = 0.0
    for i in range(a.shape[0]):
        lnQ = lnQ + a[i] * jnp.log(T)**i
    return jnp.exp(lnQ)

def Q_FeII(T):
    a = jnp.array([
        2.71692895e2,
        -1.52697440e2,
        3.36119665e1,
        -3.56415427e0,
        1.80193259e-1,
        -3.38654879e-3
    ])
    lnQ = 0.0
    for i in range(a.shape[0]):
        lnQ = lnQ + a[i] * jnp.log(T)**i
    return jnp.exp(lnQ)

def Q_FeIII(T):
    a = jnp.array([
        5.42788652e2,
        -3.26170152e2,
        7.77054463e1,
        -9.12244699e0,
        5.27184053e-1,
        -1.19689432e-2
    ])
    lnQ = 0.0
    for i in range(a.shape[0]):
        lnQ = lnQ + a[i] * jnp.log(T)**i
    return jnp.exp(lnQ)

def fe_pops(abund, temperature, ne, nhtot):
    """
    Compute the ion fractions of Fe I, II and III using the partition functions of Irwin 1981

    ionisation potentials from https://srd.nist.gov/jpcrdreprint/1.555659.pdf

    abund : float
        The decimal abundance of Fe relative to H
    temperature : float
        The temperature [K]
    ne : float
        The electron density [m-3]
    nhtot : float
        Total H density [m-3]
    """
    ionpot = (7.870, 16.1879) # eV
    saha = 2.0 * SAHA_CONST * temperature**1.5 / ne
    kBT = K_B_EV * temperature
    n1_n0 = Q_FeII(temperature) / Q_FeI(temperature) * saha * jnp.exp(-ionpot[0] / kBT)
    n2_n1 = Q_FeIII(temperature) / Q_FeII(temperature) * saha * jnp.exp(-ionpot[1] / kBT)

    nfei = (abund * nhtot) / (1.0 + n1_n0 + n2_n1)
    nfeii = n1_n0 * nfei
    nfeiii = n2_n1 * nfeii

    return nfei, nfeii, nfeiii

def fei_pop_i(abund, temperature, ne, nhtot, gi, ei):
    """
    Compute the population of a level i of Fe I with statistical weight gi and energy ei

    abund : float
        The decimal abundance of Fe relative to H
    temperature : float
        The temperature [K]
    ne : float
        The electron density [m-3]
    nhtot : float
        Total H density [m-3]
    gi : float
        Statistical weight g
    ei : float
        Energy level in [eV]
    """
    nfei, nfeii, nfeiii = fe_pops(abund, temperature, ne, nhtot)
    kBT = K_B_EV * temperature
    ni = nfei * gi / Q_FeI(temperature) * jnp.exp(-ei / kBT)
    return ni



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    # plt.figure()
    kd = read_kurucz("kurucz_6301_6302.linelist")

    # wave = np.linspace(50, 1000.0, 100)
    # b = planck(wave, 5000)
    # import lightweaver as lw
    # b_lw = (lw.planck(5000, wave) << u.Unit('W/(m2 Hz sr)')).to('kW/(m2 nm sr)', equivalencies=u.spectral_density(wav=wave << u.nm)).value
    # plt.figure()
    # plt.plot(wave, b)
    # plt.plot(wave, b_lw, '--')

    emis_opac(kd, 630.15, 5000, 1e22, 1e25, 0.0, 2e3)
