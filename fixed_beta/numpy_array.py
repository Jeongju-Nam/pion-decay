from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIRECTORY = Path(__file__).resolve().parent/"figures"
FIGURE_DIRECTORY.mkdir(parents = True, exist_ok = True)

PION_MASS_PDG = 0.1349768

NUMBER_OF_EVENTS = 100_000

RANDOM_SEED = 12345

BETA = 0.9


def pion_decay_fixed_beta():
    rng = np.random.default_rng(RANDOM_SEED)

    # Pion decay in the pion rest frame
    # Each photon carries half of the pion rest energy
    photon_energy_rest = PION_MASS_PDG/2
    photon_momentum_rest = photon_energy_rest

    # Isotropic decay directions
    cos_theta_rest = rng.uniform(-1, 1, NUMBER_OF_EVENTS)
    sin_theta_rest = np.sqrt(1 - cos_theta_rest**2)
    phi_rest = rng.uniform(0, 2*np.pi, NUMBER_OF_EVENTS)

    # Four-momentum of photon 1
    energy_1_rest = np.full(NUMBER_OF_EVENTS, photon_energy_rest)
    px_1_rest = photon_momentum_rest*sin_theta_rest*np.cos(phi_rest)
    py_1_rest = photon_momentum_rest*sin_theta_rest*np.sin(phi_rest)
    pz_1_rest = photon_momentum_rest*cos_theta_rest

    # Four-momentum of photon 2
    energy_2_rest = np.full(NUMBER_OF_EVENTS, photon_energy_rest)
    px_2_rest = -px_1_rest
    py_2_rest = -py_1_rest
    pz_2_rest = -pz_1_rest

    # Total four-momentum
    total_energy_rest = energy_1_rest + energy_2_rest
    total_px_rest = px_1_rest + px_2_rest
    total_py_rest = py_1_rest + py_2_rest
    total_pz_rest = pz_1_rest + pz_2_rest

    # Pion kinematics in the lab frame
    pion_beta_lab = BETA
    pion_gamma_lab = 1/np.sqrt(1 - pion_beta_lab**2)

    pion_energy_lab = pion_gamma_lab*PION_MASS_PDG
    pion_pz_lab = pion_gamma_lab*pion_beta_lab*PION_MASS_PDG

    # Lorentz boost to the lab frame
    # Four-momentum of photon 1
    energy_1_lab = pion_gamma_lab*(energy_1_rest + pion_beta_lab*pz_1_rest)
    px_1_lab = px_1_rest
    py_1_lab = py_1_rest
    pz_1_lab = pion_gamma_lab*(pz_1_rest + pion_beta_lab*energy_1_rest)

    # Four-momentum of photon 2
    energy_2_lab = pion_gamma_lab*(energy_2_rest + pion_beta_lab*pz_2_rest)
    px_2_lab = px_2_rest
    py_2_lab = py_2_rest
    pz_2_lab = pion_gamma_lab*(pz_2_rest + pion_beta_lab*energy_2_rest)

    # Total four-momentum
    total_energy_lab = energy_1_lab + energy_2_lab
    total_px_lab = px_1_lab + px_2_lab
    total_py_lab = py_1_lab + py_2_lab
    total_pz_lab = pz_1_lab + pz_2_lab

    # Photon kinematics
    pt_1_lab = np.sqrt(px_1_lab**2 + py_1_lab**2)
    pt_2_lab = np.sqrt(px_2_lab**2 + py_2_lab**2)

    theta_1_lab = np.arctan2(pt_1_lab, pz_1_lab)
    theta_2_lab = np.arctan2(pt_2_lab, pz_2_lab)

    theta_1_lab_degree = np.degrees(theta_1_lab)
    theta_2_lab_degree = np.degrees(theta_2_lab)

    photon_angles_lab = np.concatenate([theta_1_lab_degree, theta_2_lab_degree])

    print()
    print("Pion decay in the pion rest frame")
    print(f"{'Pion mass':<24}: {PION_MASS_PDG:.7f} GeV")
    print(f"{'Photon energy':<24}: {photon_energy_rest:.7f} GeV")

    print()
    print("Four-momentum conservation in the pion rest frame")
    print(f"{'Maximum energy error':<24}: {np.max(np.abs(total_energy_rest - PION_MASS_PDG)):.2e} GeV")
    print(f"{'Maximum total px':<24}: {np.max(np.abs(total_px_rest)):.2e} GeV")
    print(f"{'Maximum total py':<24}: {np.max(np.abs(total_py_rest)):.2e} GeV")
    print(f"{'Maximum total pz':<24}: {np.max(np.abs(total_pz_rest)):.2e} GeV")

    print()
    print("Pion kinematics in the lab frame")
    print(f"{'Pion beta':<24}: {pion_beta_lab:.4f}")
    print(f"{'Pion gamma':<24}: {pion_gamma_lab:.4f}")
    print(f"{'Pion energy':<24}: {pion_energy_lab:.4f} GeV")
    print(f"{'Pion pz':<24}: {pion_pz_lab:.4f} GeV")

    print()
    print("Four-momentum conservation in the lab frame")
    print(f"{'Maximum energy error':<24}: {np.max(np.abs(total_energy_lab - pion_energy_lab)):.2e} GeV")
    print(f"{'Maximum total px':<24}: {np.max(np.abs(total_px_lab)):.2e} GeV")
    print(f"{'Maximum total py':<24}: {np.max(np.abs(total_py_lab)):.2e} GeV")
    print(f"{'Maximum pz error':<24}: {np.max(np.abs(total_pz_lab - pion_pz_lab)):.2e} GeV")

    # Photon angular distribution in the pion rest frame
    fig, ax = plt.subplots(figsize = (8, 6))

    ax.hist(
        cos_theta_rest,
        bins = 50,
        range = (-1, 1),
        density = True,
        histtype = "step",
        linewidth = 1.5,
        label = r"$\pi^0 \rightarrow \gamma\gamma$"
    )
    ax.set_xlabel(r"$\cos\theta^*$")
    ax.set_ylabel("Normalized number of photons")
    ax.legend(frameon = False)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY/"photon_angular_distribution_in_pion_rest_frame.png",
        bbox_inches = "tight"
    )
    plt.close(fig)

    # Photon angular distribution in the lab frame
    fig, ax = plt.subplots(figsize = (8, 6))

    ax.hist(
        photon_angles_lab,
        bins = 90,
        range = (0, 180),
        density = True,
        histtype = "step",
        linewidth = 1.5,
        label = rf"$\beta_\pi = {BETA}$"
    )
    ax.set_xlabel(r"Photon polar angle $\theta_{\mathrm{lab}}$ [degree]")
    ax.set_ylabel("Normalized number of photons")
    ax.legend(frameon = False)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIRECTORY/"photon_angular_distribution_in_lab_frame.png",
        bbox_inches = "tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    pion_decay_fixed_beta()
