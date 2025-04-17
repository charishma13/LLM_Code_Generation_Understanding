import numpy as np
import pandas as pd
from scipy.special import erfc,erf
import math 

# defining all variables
atom_properties = {
    'O': {'type': 'O', 'sigma': 3.165558, 'epsilon': 78.197431, 'charge': -0.8476, 'num_particles': 1},
    'H': {'type': 'H', 'sigma': 0.000000, 'epsilon': 0.000000, 'charge': 0.4238, 'num_particles': 2},
}

# Trying for small configurations first
file_paths = [
    #    '../data/spce_sample_config_periodic4.txt',
    #    '../data/spce_sample_config_periodic2.txt',
    #    '../data/spce_sample_config_periodic3.txt',
        '../data/spce-sample-non-cuboid-configurations/spce_triclinic_sample_periodic3.txt'
    ]

NIST_TRICLINIC_SPC_E_Water = {'Configuration': [1, 2, 3, 4],
 'M (number of SPC/E molecules)': [400, 300, 200, 100],
 'Cell Type': ['Triclinic', 'Monoclinic', 'Triclinic', 'Monoclinic'],
 'Cell Side Lengths [a, b, c] (Å)': ['[30 Å, 30 Å, 30 Å]',
  '[27 Å, 30 Å, 36 Å]',
  '[30 Å, 30 Å, 30 Å]',
  '[36 Å, 36 Å, 36 Å]'],
 'Cell Angles [α, β, γ] (degrees)': ['[100°, 95°, 75°]',
  '[90°, 75°, 90°]',
  '[85°, 75°, 80°]',
  '[90°, 60°, 90°]'],
 'Number of Wave Vectors': [831, 1068, 838, 1028],
 'Edisp/kB (K)': [111992.0, 43286.0, 14403.3, 25025.1],
 'ELRC/kB (K)': [-4109.19, -2105.61, -1027.3, -163.091],
 'Ereal/kB (K)': [-727219.0, -476902.0, -297129.0, -171462.0],
 'Efourier/kB (K)': [44677.0, 44409.4, 28897.4, 22337.2],
 'Eself/kB (K)': [-11582000.0, -8686470.0, -5790980.0, -2895490.0],
 'Eintra/kB (K)': [11435400.0, 8576520.0, 5717680.0, 2858840.0],
 'Etotal/kB (K)': [-721254.0, -501259.0, -328153.0, -160912.0]}

# Data processing




def extracting_positions(input_file):
        # Extract the positions from the .xyz file
    with open(input_file, "r") as file:
        lines = file.readlines()

    data_lines = lines[2:]

    data_list = []
    for line in data_lines:
        stripped_line = line.strip()
        parts = stripped_line.split()
        if len(parts) >= 5:  
            try:
                x, y, z = map(float, parts[1:4])
                atom_type = parts[4]
                data_list.append([x, y, z, atom_type])
            except ValueError:
                continue  

    # Create a DataFrame with all configurations
    columns = ["X", "Y", "Z", "Atom Type"]
    configuration = pd.DataFrame(data_list, columns=columns)

    configuration.index = range(1, len(configuration) + 1)

    configuration["Molecule"] = ((configuration.index - 1) // 3) + 1
    
    return configuration

# create the target dataframes
def parse_cell_sides(sides_str):
    sides_str = sides_str.strip().strip('[]')
    parts = sides_str.split(',')
    values = []
    for p in parts:
        p = p.strip()
        p = p.replace('Å', '')
        values.append(float(p))
    return values

def parse_cell_angles(angles_str):
    angles_str = angles_str.strip().strip('[]')
    parts = angles_str.split(',')
    values = []
    for p in parts:
        p = p.strip()
        p = p.replace('°', '')
        values.append(float(p))
    return values

def creating_dataframes(file_paths, atom_properties, NIST_SPC_E_Water):
    NIST_SPC_E_Water = pd.DataFrame(NIST_SPC_E_Water)
    NIST_SPC_E_Water['Sum of energies'] = (
        NIST_SPC_E_Water['Edisp/kB (K)'] + NIST_SPC_E_Water['ELRC/kB (K)'] +
        NIST_SPC_E_Water['Ereal/kB (K)'] + NIST_SPC_E_Water['Efourier/kB (K)'] +
        NIST_SPC_E_Water['Eself/kB (K)'] + NIST_SPC_E_Water['Eintra/kB (K)']
    )

    force_field = pd.DataFrame(atom_properties).from_dict(atom_properties, orient='index')
    system = pd.DataFrame(file_paths, columns=["file_paths"])
    system['configuration #'] = (
        system['file_paths']
        .str.extract(r'(\d+)', expand=False)
        .fillna('0')
        .astype(int)
    )

    def get_abcs_and_angles(conf_num):
        if conf_num in NIST_SPC_E_Water["Configuration"].values:
            row = NIST_SPC_E_Water.loc[NIST_SPC_E_Water["Configuration"] == conf_num]
            sides_str = row['Cell Side Lengths [a, b, c] (Å)'].values[0]
            angles_str = row['Cell Angles [α, β, γ] (degrees)'].values[0]
            sides = parse_cell_sides(sides_str)
            angles = parse_cell_angles(angles_str)
            return pd.Series({
                "number of particles": float(row["M (number of SPC/E molecules)"].values[0]),
                "a": sides[0],
                "b": sides[1],
                "c": sides[2],
                "alpha_deg": angles[0],
                "beta_deg": angles[1],
                "gamma_deg": angles[2]
            })
        else:
            return pd.Series({
                "number of particles": 0.0,
                "a": 20.0,
                "b": 20.0,
                "c": 20.0,
                "alpha_deg": 90.0,
                "beta_deg": 90.0,
                "gamma_deg": 90.0
            })

    system = pd.concat([
        system,
        system["configuration #"].apply(get_abcs_and_angles)
    ], axis=1)

    system['cutoff'] = 10.0
    system['alpha'] = system.apply(
        lambda row: 5.6 / min(row['a'], row['b'], row['c']) if (row['a'] * row['b'] * row['c']) != 0 else 0.28,
        axis=1
    )

    system['kmax'] = 5
    system['ε0'] = float(8.854187817E-12)
    system['kB'] = float(1.3806488E-23)

    return system, force_field, NIST_SPC_E_Water

# pairwise dispersion energy functions

# Minimum Image Distance function and Pair Dispersion Energy calculation (Code 3)
def minimum_image_distance(r_ij, cell_length):
    # Apply the minimum image convention to distances.
    return r_ij - cell_length * np.round(r_ij / cell_length)

# pairwise dispersion energy operation
def build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0

    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma) if abs(math.sin(gamma)) > 1e-14 else 1e-14
    cy = c * ((math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / sin_gamma)
    cz = math.sqrt(c**2 - cx**2 - cy**2)

    h = np.array([
        [ax, bx, cx],
        [ay, by, cy],
        [az, bz, cz]
    ], dtype=float)

    h_inv = np.linalg.inv(h)
    return h, h_inv

def min_image_distance_triclinic(r_ij, h, h_inv):
    frac = h_inv.dot(r_ij)
    frac -= np.round(frac)
    return h.dot(frac)

def pair_dispersion_energy(system_data, configuration, force_field):
    positions = configuration[['X', 'Y', 'Z']].values
    atom_types = configuration['Atom Type'].values
    cutoff = system_data['cutoff']

    a = system_data['a']
    b = system_data['b']
    c = system_data['c']
    alpha_deg = system_data['alpha_deg']
    beta_deg = system_data['beta_deg']
    gamma_deg = system_data['gamma_deg']

    h, h_inv = build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)

    total_dispersion_energy = 0.0
    num_atoms = len(positions)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            r_ij = positions[i] - positions[j]
            r_ij = min_image_distance_triclinic(r_ij, h, h_inv)
            distance = np.linalg.norm(r_ij)

            if 0 < distance < cutoff:
                type_i = atom_types[i]
                type_j = atom_types[j]
                if type_i not in force_field.index or type_j not in force_field.index:
                    continue

                epsilon_i = force_field.loc[type_i, 'epsilon']
                epsilon_j = force_field.loc[type_j, 'epsilon']
                sigma_i = force_field.loc[type_i, 'sigma']
                sigma_j = force_field.loc[type_j, 'sigma']

                epsilon_ij = math.sqrt(epsilon_i * epsilon_j)
                sigma_ij = 0.5 * (sigma_i + sigma_j)
                s_over_r = sigma_ij / distance
                lj_pot = 4.0 * epsilon_ij * (s_over_r**12 - s_over_r**6)
                total_dispersion_energy += lj_pot

    return total_dispersion_energy

# Compute LRC energy
def build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0

    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma) if abs(math.sin(gamma)) > 1e-14 else 1e-14
    cy = c * ((math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / sin_gamma)
    cz = math.sqrt(c**2 - cx**2 - cy**2)

    h = np.array([
        [ax, bx, cx],
        [ay, by, cy],
        [az, bz, cz]
    ], dtype=float)

    h_inv = np.linalg.inv(h)
    return h, h_inv

def compute_lrc_energy(system_row, force_field):
    U_lrc_total = 0.0
    cutoff = system_row['cutoff']

    h, _ = build_triclinic_matrix(system_row['a'], system_row['b'], system_row['c'],
                                  system_row['alpha_deg'], system_row['beta_deg'], system_row['gamma_deg'])
    volume = abs(np.linalg.det(h))

    for atom_type, atom_data in force_field.iterrows():
        num_particles = system_row['number of particles'] * atom_data['num_particles']
        rho = num_particles / volume
        sigma = atom_data['sigma']
        epsilon = atom_data['epsilon']

        if cutoff > 0:
            sc3 = (sigma / cutoff)**3
            sc9 = sc3**3
        else:
            sc3, sc9 = 0.0, 0.0

        U_lrc = (8.0 / 3.0)*math.pi*rho*epsilon*(sigma**3)*(sc9/3.0 - sc3)
        U_lrc *= num_particles
        U_lrc_total += U_lrc

    return U_lrc_total

def build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0

    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma) if abs(math.sin(gamma)) > 1e-14 else 1e-14
    cy = c * ((math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / sin_gamma)
    cz = math.sqrt(c**2 - cx**2 - cy**2)

    h = np.array([
        [ax, bx, cx],
        [ay, by, cy],
        [az, bz, cz]
    ], dtype=float)

    h_inv = np.linalg.inv(h)
    return h, h_inv

def min_image_distance_triclinic(r_ij, h, h_inv):
    frac = h_inv.dot(r_ij)
    frac -= np.round(frac)
    return h.dot(frac)

def compute_real_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data["alpha"]
    cutoff = system_data["cutoff"]

    a = system_data['a']
    b = system_data['b']
    c = system_data['c']
    alpha_deg = system_data['alpha_deg']
    beta_deg = system_data['beta_deg']
    gamma_deg = system_data['gamma_deg']

    h, h_inv = build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    mol_ids = configuration["Molecule"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types])

    real_energy = 0.0
    n_atoms = len(positions)

    for j in range(n_atoms - 1):
        for l in range(j + 1, n_atoms):
            if mol_ids[j] == mol_ids[l]:
                continue
            r_ij = positions[l] - positions[j]
            r_ij = min_image_distance_triclinic(r_ij, h, h_inv)
            distance = np.linalg.norm(r_ij)
            if 1e-14 < distance < cutoff:
                q_j = charges[j]*e_charge
                q_l = charges[l]*e_charge
                r_m = distance*1e-10
                real_energy += coulomb_factor*(q_j*q_l / r_m)*erfc(alpha*distance)

    return real_energy

def build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0

    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma) if abs(math.sin(gamma)) > 1e-14 else 1e-14
    cy = c * ((math.cos(alpha) - math.cos(beta)*math.cos(gamma)) / sin_gamma)
    cz = math.sqrt(c**2 - cx**2 - cy**2)

    h = np.array([
        [ax, bx, cx],
        [ay, by, cy],
        [az, bz, cz]
    ], dtype=float)

    h_inv = np.linalg.inv(h)
    return h, h_inv

def compute_fourier_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data["alpha"]
    kmax = system_data["kmax"]

    a = system_data['a']
    b = system_data['b']
    c = system_data['c']
    alpha_deg = system_data['alpha_deg']
    beta_deg = system_data['beta_deg']
    gamma_deg = system_data['gamma_deg']

    h, _ = build_triclinic_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)
    V_m = abs(np.linalg.det(h))*(1e-10**3)

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types])
    charges_c = charges*e_charge

    positions_m = positions*1e-10
    prefactor = coulomb_factor/(2.0*V_m)

    alpha_m = alpha*1e10
    fourier_energy = 0.0

    max_sq = kmax*kmax + 2

    L_avg = (a + b + c)/3.0
    kfactor = 2.0*math.pi/(L_avg*1e-10)

    for kx in range(-kmax, kmax+1):
        for ky in range(-kmax, kmax+1):
            for kz in range(-kmax, kmax+1):
                if (kx == 0 and ky == 0 and kz == 0):
                    continue
                if (kx*kx + ky*ky + kz*kz) >= max_sq:
                    continue

                kx_m = kfactor*kx
                ky_m = kfactor*ky
                kz_m = kfactor*kz
                k_sq = kx_m**2 + ky_m**2 + kz_m**2
                if k_sq < 1e-14:
                    continue

                real_part = 0.0
                imag_part = 0.0
                for j, (xj, yj, zj) in enumerate(positions_m):
                    kr = kx_m*xj + ky_m*yj + kz_m*zj
                    real_part += charges_c[j]*math.cos(kr)
                    imag_part += charges_c[j]*math.sin(kr)

                sk_sq = real_part**2 + imag_part**2
                exp_factor = math.exp(-k_sq/(4.0*(alpha_m**2)))

                term = prefactor*(4.0*math.pi/k_sq)*exp_factor*sk_sq
                fourier_energy += term

    return fourier_energy


def compute_self_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data['alpha']
    atom_types = configuration['Atom Type'].values
    charges = np.array([force_field.loc[t, 'charge'] for t in atom_types])
    charges_c = charges*e_charge

    alpha_m = alpha*1e10
    sum_q2 = np.sum(charges_c**2)

    # standard Ewald self correction
    self_energy = -coulomb_factor*(alpha_m/math.sqrt(math.pi))*sum_q2

    return self_energy


def compute_intra_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data['alpha']

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration['Atom Type'].values
    mol_ids = configuration['Molecule'].values
    charges = np.array([force_field.loc[t, 'charge'] for t in atom_types])

    unique_mols = np.unique(mol_ids)
    intra_energy = 0.0

    for m_id in unique_mols:
        idxs = np.where(mol_ids == m_id)[0]
        for i in range(len(idxs) - 1):
            for j in range(i + 1, len(idxs)):
                idx_i = idxs[i]
                idx_j = idxs[j]
                dx = positions[idx_j, 0] - positions[idx_i, 0]
                dy = positions[idx_j, 1] - positions[idx_i, 1]
                dz = positions[idx_j, 2] - positions[idx_i, 2]
                r = math.sqrt(dx**2 + dy**2 + dz**2)
                if r > 1e-14:
                    q_i = charges[idx_i]*e_charge
                    q_j = charges[idx_j]*e_charge
                    r_m = r*1e-10
                    erf_val = erf(alpha*r)
                    # negative sign for intramolecular.
                    val = coulomb_factor*(q_i*q_j/r_m)*erf_val
                    intra_energy -= val

    return intra_energy


# DataFrame Descriptions:

# 1. NIST_TRICLINIC_SPC_E_Water DataFrame:
#    - Contains thermodynamic properties of SPC/E water configurations in both triclinic and monoclinic cells.
#    - Columns:
#        - 'Configuration' (int): Configuration ID (1-4).
#        - 'M (number of SPC/E molecules)' (int): Number of SPC/E molecules in the system.
#        - 'Cell Type' (str): Type of the cell (e.g., Triclinic, Monoclinic).
#        - 'Cell Side Lengths [a, b, c] (Å)' (list of strings): Cell side lengths in Ångströms.
#        - 'Cell Angles [α, β, γ] (degrees)' (list of strings): Cell angles in degrees.
#        - 'Number of Wave Vectors' (int): The number of wave vectors for each configuration.
#        - 'Edisp/kB (K)' (float): Dispersion energy in Kelvin.
#        - 'ELRC/kB (K)' (float): Long-range correction energy in Kelvin.
#        - 'Ereal/kB (K)' (float): Real energy in Kelvin.
#        - 'Efourier/kB (K)' (float): Fourier transform energy in Kelvin.
#        - 'Eself/kB (K)' (float): Self-interaction energy in Kelvin.
#        - 'Eintra/kB (K)' (float): Intra-molecular energy in Kelvin.
#        - 'Etotal/kB (K)' (float): Total energy in Kelvin.

# 2. force_field DataFrame:
#    - Contains force field parameters for SPC/E water, specifically for oxygen ('O') and hydrogen ('H').
#    - Columns:
#        - 'type' (str): Atom type ('O' or 'H').
#        - 'sigma' (float): Lennard-Jones parameter (Å).
#        - 'epsilon' (float): Lennard-Jones well depth (K).
#        - 'charge' (float): Partial charge (e).
#        - 'num_particles' (int): Number of particles per molecule.

# 3. system DataFrame:
#    - Contains metadata about each system configuration.
#    - Columns:
#        - 'file_paths' (str): File names containing atomic configurations.
#        - 'configuration #' (int): Extracted configuration number (1-4).
#        - 'number of particles' (float): Number of molecules (from 'NIST_SPC_E_Water').
#        - 'box length' (float): Box dimensions (from 'NIST_SPC_E_Water').
#        - 'cutoff' (int): Fixed cutoff distance for interactions (10 Å).
#        - 'alpha' (float): Ewald summation parameter (5.6 / min(a, b, c)).
#        - 'kmax' (int): Maximum wave vector index (5); also, only include k for which k² < kmax² + 2, i.e., k² < 27.
#        - 'ε0' (float): Permittivity of Vacuum (8.854187817E-12 C²/(J m)).
#        - 'kB' (float): Boltzmann Constant (1.3806488E-23 J/K).

# 4. configuration DataFrame (from 'extracting_positions'):
#    - Created per file, containing atomic positions.
#    - Columns:
#        - 'X' (float): Atom coordinates in Ångströms.
#        - 'Y' (float): Atom coordinates in Ångströms.
#        - 'Z' (float): Atom coordinates in Ångströms.
#        - 'Atom Type' (str): Type of atom ('O' or 'H').
#        - 'Molecule' (int): Molecule index assigned based on position.


system, force_field, NIST_TRICLINIC_SPC_E_Water = creating_dataframes(file_paths, atom_properties,NIST_TRICLINIC_SPC_E_Water)

# Computing energies storing in results
results = pd.DataFrame()

results['Number of Particles'] = system['number of particles'].astype(int)

# Calculate LRC energy for all system configurations
results['lrc_Energies'] = system.apply(
    lambda row: compute_lrc_energy(row, force_field), axis=1
)

# Calculate pairwise energy for all system configurations
results['dispersion_energies'] = system['file_paths'].apply(
    lambda file_path: pair_dispersion_energy(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path), 
        force_field
    )
)

# Calculate pairwise energy for all system configurations
results['real_energies'] = system['file_paths'].apply(
    lambda file_path: compute_real_energies(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path),
        force_field
    )
)

# Calculate pairwise energy for all system configurations
results['fourier_energies'] = system['file_paths'].apply(
    lambda file_path: compute_fourier_energies(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path),
        force_field
    )
)

# Calculate pairwise energy for all system configurations
results['self_energies'] = system['file_paths'].apply(
    lambda file_path: compute_self_energies(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path),
        force_field
    )
)

# Calculate pairwise energy for all system configurations
results['intra_energies'] = system['file_paths'].apply(
    lambda file_path: compute_intra_energies(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path),
        force_field
    )
)

def compare_LJ_coulomb_energy(df1, df2, tolerance=1e-4):
    df_merged = df1.merge(df2, left_on='Number of Particles', right_on='M (number of SPC/E molecules)', how='left')

    matched_real = matched_fourier = matched_self = matched_intra = 0
    matched_dispersion = matched_lrc = 0
    not_matched_real = not_matched_fourier = not_matched_self = not_matched_intra = 0
    not_matched_dispersion = not_matched_lrc = 0
    l2_dispersion = l2_lrc = l2_real = 0.0
    l2_fourier = l2_self = l2_intra = 0.0

    real_energy_output = []
    fourier_energy_output = []
    self_energy_output = []
    intra_energy_output = []
    lrc_energy_output = []
    dispersion_energy_output = []

    for idx, row in df_merged.iterrows():
        real_energy = row['real_energies']
        fourier_energy = row['fourier_energies']
        self_energy = row['self_energies']
        intra_energy = row['intra_energies']
        num_molecules = row['Number of Particles']
        lrc_energy = row['lrc_Energies']
        dispersion_energy = row['dispersion_energies']

        if pd.isna(row['Ereal/kB (K)']):
            continue
        nist_real_energy = float(row['Ereal/kB (K)'])
        nist_fourier_energy = float(row['Efourier/kB (K)'])
        nist_self_energy = float(row['Eself/kB (K)'])
        nist_intra_energy = float(row['Eintra/kB (K)'])
        nist_lrc_energy = float(row['ELRC/kB (K)'])
        nist_dispersion_energy = float(row['Edisp/kB (K)'])

        match_real = np.isclose(real_energy, nist_real_energy, atol=tolerance)
        match_fourier = np.isclose(fourier_energy, nist_fourier_energy, atol=tolerance)
        match_self = np.isclose(self_energy, nist_self_energy, atol=tolerance)
        match_intra = np.isclose(intra_energy, nist_intra_energy, atol=tolerance)
        match_dispersion = np.isclose(dispersion_energy, nist_dispersion_energy, atol=tolerance)
        match_lrc = np.isclose(lrc_energy, nist_lrc_energy, atol=tolerance)

        matched_real += int(match_real)
        not_matched_real += int(not match_real)
        matched_fourier += int(match_fourier)
        not_matched_fourier += int(not match_fourier)
        matched_self += int(match_self)
        not_matched_self += int(not match_self)
        matched_intra += int(match_intra)
        not_matched_intra += int(not match_intra)
        matched_dispersion += int(match_dispersion)
        not_matched_dispersion += int(not match_dispersion)
        matched_lrc += int(match_lrc)
        not_matched_lrc += int(not match_lrc)

        dispersion_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {dispersion_energy:.4E}, NIST: {nist_dispersion_energy:.4E}, Match: {match_dispersion}")
        lrc_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {lrc_energy:.4E}, NIST: {nist_lrc_energy:.4E}, Match: {match_lrc}")
        real_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {real_energy:.4E}, NIST: {nist_real_energy:.4E}, Match: {match_real}")
        fourier_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {fourier_energy:.4E}, NIST: {nist_fourier_energy:.4E}, Match: {match_fourier}")
        self_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {self_energy:.4E}, NIST: {nist_self_energy:.4E}, Match: {match_self}")
        intra_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {intra_energy:.4E}, NIST: {nist_intra_energy:.4E}, Match: {match_intra}")

        l2_dispersion = np.square(dispersion_energy - nist_dispersion_energy)
        l2_lrc = np.square(lrc_energy - nist_lrc_energy)
        l2_real = np.square(real_energy - nist_real_energy)
        l2_fourier = np.square(fourier_energy - nist_fourier_energy)
        l2_self = np.square(self_energy - nist_self_energy)
        l2_intra = np.square(intra_energy - nist_intra_energy)

    print()
    print("Lennard-Jones Pair Dispersion Energy Comparison:")
    print(*dispersion_energy_output)
    print("Lennard-Jones long-range corrections Energy Comparison:")
    print(*lrc_energy_output)
    print("Real Energy Comparison:")
    print(*real_energy_output)
    print("Fourier Energy Comparison:")
    print(*fourier_energy_output)
    print("Self Energy Comparison:")
    print(*self_energy_output)
    print("Intra Energy Comparison:")
    print(*intra_energy_output)
    print()
    print(f"Count of correct pairwise answers: {matched_dispersion}")
    print(f"Count of incorrect pairwise answers: {not_matched_dispersion}")
    print(f"Count of correct LRC answers: {matched_lrc}")
    print(f"Count of incorrect LRC answers: {not_matched_lrc}")
    print(f"Count of correct Real Energy answers: {matched_real}")
    print(f"Count of incorrect Real Energy answers: {not_matched_real}")
    print(f"Count of correct Fourier Energy answers: {matched_fourier}")
    print(f"Count of incorrect Fourier Energy answers: {not_matched_fourier}")
    print(f"Count of correct Self Energy answers: {matched_self}")
    print(f"Count of incorrect Self Energy answers: {not_matched_self}")
    print(f"Count of correct Intra Energy answers: {matched_intra}")
    print(f"Count of incorrect Intra Energy answers: {not_matched_intra}")
    print()
    print(f"L2 Value Comparison of Squared Differences Between Computed and NIST Energy Values with tolerance: {tolerance}")
    print(f"L2 value for Dispersion: {np.sqrt(np.sum(l2_dispersion))}")
    print(f"L2 value for LRC: {np.sqrt(np.sum(l2_lrc))}")
    print(f"L2 value for Real Energy: {np.sqrt(np.sum(l2_real))}")
    print(f"L2 value for Fourier Energy: {np.sqrt(np.sum(l2_fourier))}")
    print(f"L2 value for Self Energy: {np.sqrt(np.sum(l2_self))}")
    print(f"L2 value for Intra Energy: {np.sqrt(np.sum(l2_intra))}")
    total_correct = matched_real + matched_fourier + matched_self + matched_intra + matched_dispersion + matched_lrc
    total_incorrect = not_matched_real + not_matched_fourier + not_matched_self + not_matched_intra + not_matched_dispersion + not_matched_lrc
    print(f"Total correct answers: {total_correct}")
    print(f"Total incorrect answers: {total_incorrect}")


# calling compare_coulomb_energy function
compare_LJ_coulomb_energy(results, NIST_TRICLINIC_SPC_E_Water)