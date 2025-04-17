import numpy as np
import pandas as pd
from scipy.special import erfc,erf
import math 
import numpy as np

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
 'Etotal/kB (K)': [-721254.0, -501259.0, -328153.0, -160912.0]

}


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

# Function to compute the triclinic box matrix
def triclinic_box_matrix(a, b, c, alpha, beta, gamma):
    # Convert degrees to radians
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    
    # Compute components of the box matrix
    v_x = [a, 0.0, 0.0]
    v_y = [b * np.cos(gamma_r), b * np.sin(gamma_r), 0.0]
    
    cx = c * np.cos(beta_r)
    cy = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    
    v_z = [cx, cy, cz]

    return np.array([v_x, v_y, v_z]).T  # 3x3 box matrix

# Function to convert scaled coordinates to Cartesian coordinates
def scaled_to_cartesian(scaled_coords, box_matrix):
    return np.dot(scaled_coords, box_matrix.T)


def minimum_image_distance(r_ij, cell_length):
    # Apply the minimum image convention to distances.
    return r_ij - cell_length * np.round(r_ij / cell_length)

# Create the target dataframes
def creating_dataframes(file_paths, atom_properties, NIST_SPC_E_Water):
    # Create the NIST_SPC_E_Water dataframe
    NIST_SPC_E_Water = pd.DataFrame(NIST_SPC_E_Water)
    NIST_SPC_E_Water['Sum of energies'] = (
        NIST_SPC_E_Water['Edisp/kB (K)'] + NIST_SPC_E_Water['ELRC/kB (K)'] +
        NIST_SPC_E_Water['Ereal/kB (K)'] + NIST_SPC_E_Water['Efourier/kB (K)'] +
        NIST_SPC_E_Water['Eself/kB (K)'] + NIST_SPC_E_Water['Eintra/kB (K)']
    )

    # Creating the force_field dataframe
    force_field = pd.DataFrame(atom_properties).from_dict(atom_properties, orient='index')

    # Create the system dataframe containing some variables
    system = pd.DataFrame(file_paths, columns=["file_paths"])

    # Use '(\d+)' and expand=False to get a single Series, fill missing values before converting
    system['configuration #'] = (
        system['file_paths']
        .str.extract(r'(\d+)', expand=False)
        .fillna('0')
        .astype(int)
    )

    # Now we add the triclinic box matrix and box length for each configuration
    def get_box_matrix(config):
        # Extract the box dimensions and angles
        cell_lengths = NIST_SPC_E_Water.loc[NIST_SPC_E_Water['Configuration'] == config, 'Cell Side Lengths [a, b, c] (Å)'].values[0]
        cell_angles = NIST_SPC_E_Water.loc[NIST_SPC_E_Water['Configuration'] == config, 'Cell Angles [α, β, γ] (degrees)'].values[0]

        # Clean up and convert the cell_lengths and cell_angles strings into lists of floats
        cell_lengths = [float(x.strip().replace('Å', '')) for x in cell_lengths.strip('[]').split(',')]
        cell_angles = [float(x.strip().replace('°', '')) for x in cell_angles.strip('[]').split(',')]
        
        # Check if cell_lengths has exactly 3 values
        if len(cell_lengths) != 3:
            raise ValueError(f"Cell side lengths are not correctly formatted: {cell_lengths}")
        
        # Extract individual values for a, b, c
        a, b, c = cell_lengths
        alpha, beta, gamma = cell_angles
        
        # Compute the triclinic box matrix
        box_matrix = triclinic_box_matrix(a, b, c, alpha, beta, gamma)

        # Compute the box length (diagonal of the triclinic box)
        alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
        box_length = np.sqrt(
            a**2 + b**2 + c**2 + 2 * a * b * np.cos(gamma_r) + 2 * a * c * np.cos(beta_r) + 2 * b * c * np.cos(alpha_r)
        )

        return box_matrix, box_length

    # Apply the get_box_matrix function
    system[['box_matrix', 'box_length']] = system['configuration #'].apply(
        lambda x: pd.Series(get_box_matrix(x))
    )

    # Compute the 'number of particles' from the NIST data
    system["number of particles"] = system["configuration #"].apply(
        lambda x: float(
            NIST_SPC_E_Water.loc[
                NIST_SPC_E_Water["Configuration"] == x,
                "M (number of SPC/E molecules)"
            ].values[0]
        ) if x in NIST_SPC_E_Water["Configuration"].values else 0.0
    )


    system['cutoff'] = 10

    # Fix alpha so that alpha = 5.6 / L
    system['alpha'] = system.apply(
        lambda row: 5.6 / row['box_length'] if row['box_length'] != 0 else 0.28,
        axis=1
    )

    system['kmax'] = 5
    system['ε0'] = float(8.854187817E-12)
    system['kB'] = float(1.3806488E-23)

    return system, force_field, NIST_SPC_E_Water


# pairwise dispersion energy operation
def pair_dispersion_energy(system_data, configuration, force_field):
    """
    Compute the total pair dispersion energy for a system of particles.
    
    Parameters:
    - system_row: A row of the system DataFrame containing simulation properties.
    - configuration: DataFrame with atom positions and types.
    - force_field: DataFrame with force field parameters for atom types.

    Returns:
    - total_dispersion_energy: float, the total pair dispersion energy.
    """
    positions = configuration[['X', 'Y', 'Z']].values
    atom_types = configuration['Atom Type'].values
    cutoff = system_data['cutoff']
    cell_length = system_data['box_length']
    num_atoms = len(positions)

    total_dispersion_energy = 0.0

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            r_ij = positions[i] - positions[j]
            r_ij = minimum_image_distance(r_ij, cell_length)
            distance = np.linalg.norm(r_ij)

            if 0 < distance < cutoff:
                type_i, type_j = atom_types[i], atom_types[j]

                if type_i not in force_field.index or type_j not in force_field.index:
                    continue

                epsilon_i = force_field.loc[type_i, 'epsilon']
                epsilon_j = force_field.loc[type_j, 'epsilon']
                sigma_i = force_field.loc[type_i, 'sigma']
                sigma_j = force_field.loc[type_j, 'sigma']

                epsilon_ij = np.sqrt(epsilon_i * epsilon_j)
                sigma_ij = (sigma_i + sigma_j) / 2.0
                s_over_r = sigma_ij / distance

                # Lennard-Jones potential
                potential_energy = 4 * epsilon_ij * (s_over_r**12 - s_over_r**6)
                total_dispersion_energy += potential_energy
                
    return total_dispersion_energy

# Compute LRC energy
def compute_lrc_energy(system_row, force_field):
    """
    Compute the Long-Range Correction (LRC) to the Lennard-Jones potential energy for a single system.
    """
    U_lrc_total = 0

    # Iterate over atom types in the force field
    for atom_type, atom_data in force_field.iterrows():
        num_particles = system_row['number of particles'] * atom_data['num_particles']
        
        # Calculate the system's volume
        volume = system_row['box_length'] ** 3
        rho = num_particles / volume

        # Compute the LRC energy for each particle
        total_lrc_energy = 0.0

        # Get epsilon and sigma for each particle
        sigma = atom_data['sigma']
        epsilon = atom_data['epsilon']

        # Apply cutoff and compute LRC energy for each particle
        sigma_by_cutoff_3 = (sigma / system_row['cutoff']) ** 3
        sigma_by_cutoff_9 = sigma_by_cutoff_3 ** 3

        # LRC energy per particle
        U_lrc_per_particle = (8 / 3) * np.pi * rho * epsilon * sigma**3 * (sigma_by_cutoff_9 / 3 - sigma_by_cutoff_3)
        
        # Multiply by number of particles
        U_lrc_per_particle *= num_particles
        U_lrc_total += U_lrc_per_particle

    return U_lrc_total

def compute_real_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19  # C
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23  # (1/(4 pi eps0)) / kB

    alpha = system_data["alpha"]  # 1/Å
    cutoff = system_data["cutoff"]  # Å
    L = system_data["box_length"]  # Å

    def min_image(dx, box_length):
        return dx - round(dx / box_length) * box_length

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    mol_ids = configuration["Molecule"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types])

    n_atoms = len(positions)
    real_energy = 0.0

    for j in range(n_atoms - 1):
        for l in range(j + 1, n_atoms):
            # Skip intramolecular pairs
            if mol_ids[j] == mol_ids[l]:
                continue

            dx = min_image(positions[l, 0] - positions[j, 0], L)
            dy = min_image(positions[l, 1] - positions[j, 1], L)
            dz = min_image(positions[l, 2] - positions[j, 2], L)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < cutoff and r > 1e-14:
                q_j = charges[j] * e_charge
                q_l = charges[l] * e_charge
                r_m = r * 1e-10
                factor_erfc = erfc(alpha * r)
                real_energy += coulomb_factor * (q_j * q_l / r_m) * factor_erfc

    return real_energy

def compute_fourier_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23  # 1/(4 pi eps0) in K·m / C^2

    alpha = system_data["alpha"]
    kmax = system_data["kmax"]
    L = system_data["box_length"]
    V_m = (L * 1e-10) ** 3

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types])
    charges_c = charges * e_charge

    positions_m = positions * 1e-10
    # 1/(4 pi eps0)*1/(2V) => coulomb_factor*(1/(2 V_m))
    prefactor = coulomb_factor / (2.0 * V_m)

    alpha_m = alpha * 1e10
    fourier_energy = 0.0

    max_sq = kmax * kmax + 2

    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            for kz in range(-kmax, kmax + 1):
                if (kx == 0 and ky == 0 and kz == 0):
                    continue

                k_int_sq = kx * kx + ky * ky + kz * kz
                if k_int_sq >= max_sq:
                    continue

                kx_m = (2.0 * math.pi / (L * 1e-10)) * kx
                ky_m = (2.0 * math.pi / (L * 1e-10)) * ky
                kz_m = (2.0 * math.pi / (L * 1e-10)) * kz
                k_sq = kx_m * kx_m + ky_m * ky_m + kz_m * kz_m
                if k_sq < 1e-14:
                    continue

                real_part = 0.0
                imag_part = 0.0
                for j, (xj, yj, zj) in enumerate(positions_m):
                    kr = kx_m*xj + ky_m*yj + kz_m*zj
                    real_part += charges_c[j]*math.cos(kr)
                    imag_part += charges_c[j]*math.sin(kr)

                sk_sq = real_part*real_part + imag_part*imag_part
                exponent = math.exp(- (k_sq)/(4.0*(alpha_m**2)))

                # 4 pi / k^2
                term = prefactor * (4.0 * math.pi / k_sq) * exponent * sk_sq

                fourier_energy += term

    return fourier_energy


def compute_self_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data["alpha"]
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t,"charge"] for t in atom_types])
    charges_c = charges * e_charge

    alpha_m = alpha*1e10

    sum_q2 = np.sum(charges_c**2)

    self_energy = - coulomb_factor * (alpha_m / math.sqrt(math.pi)) * sum_q2
    return self_energy


def compute_intra_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data["alpha"]
    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t,"charge"] for t in atom_types])
    mol_ids = configuration["Molecule"].values

    alpha_dimless = alpha

    intra_energy = 0.0
    unique_mols = np.unique(mol_ids)

    for m_id in unique_mols:
        idxs = np.where(mol_ids == m_id)[0]
        n_mol_atoms = len(idxs)
        for i in range(n_mol_atoms - 1):
            for j in range(i+1, n_mol_atoms):
                idx_i = idxs[i]
                idx_j = idxs[j]
                dx = positions[idx_j,0] - positions[idx_i,0]
                dy = positions[idx_j,1] - positions[idx_i,1]
                dz = positions[idx_j,2] - positions[idx_i,2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                if r > 1e-14:
                    q_i = charges[idx_i]*e_charge
                    q_j = charges[idx_j]*e_charge
                    r_m = r*1e-10
                    erf_val = erf(alpha_dimless*r)
                    val = coulomb_factor*(q_i*q_j / r_m)*erf_val
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
    # Merge df1 and df2 based on the number of particles
    df_merged = df1.merge(df2, left_on='Number of Particles', right_on='M (number of SPC/E molecules)', how='left')

    # Initialize counters
    matched_real = matched_fourier = matched_self = matched_intra = 0
    matched_dispersion = matched_lrc = 0
    not_matched_real = not_matched_fourier = not_matched_self = not_matched_intra = 0
    not_matched_dispersion = not_matched_lrc = 0
    l2_dispersion = l2_lrc = l2_real = 0.0
    l2_fourier = l2_self = l2_intra = 0.0

    # Initialize output lists
    real_energy_output, fourier_energy_output = [], []
    self_energy_output, intra_energy_output = [], []
    lrc_energy_output, dispersion_energy_output = [], []

    # Iterate over merged DataFrame
    for idx, row in df_merged.iterrows():
        # Extract computed values from df1
        real_energy = row['real_energies']
        fourier_energy = row['fourier_energies']
        self_energy = row['self_energies']
        intra_energy = row['intra_energies']
        num_molecules = row['Number of Particles']
        # LJ Components
        lrc_energy = row['lrc_Energies']
        dispersion_energy = row['dispersion_energies']

        # Extract reference values from df2
        if pd.isna(row['Ereal/kB (K)']):
            continue  # Skip if no match is found in df2

        nist_real_energy = float(row['Ereal/kB (K)'])
        nist_fourier_energy = float(row['Efourier/kB (K)'])
        nist_self_energy = float(row['Eself/kB (K)'])
        nist_intra_energy = float(row['Eintra/kB (K)'])
        nist_lrc_energy = float(row['ELRC/kB (K)'])
        nist_dispersion_energy = float(row['Edisp/kB (K)'])

        # Perform numeric comparisons with a tolerance
        match_real = np.isclose(real_energy, nist_real_energy, atol=tolerance)
        match_fourier = np.isclose(fourier_energy, nist_fourier_energy, atol=tolerance)
        match_self = np.isclose(self_energy, nist_self_energy, atol=tolerance)
        match_intra = np.isclose(intra_energy, nist_intra_energy, atol=tolerance)

        # Perform numeric comparisons with a tolerance
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

        # Store formatted outputs
        dispersion_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {dispersion_energy:.4E}, NIST: {nist_dispersion_energy:.4E}, Match: {match_dispersion}")
        lrc_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {lrc_energy:.4E}, NIST: {nist_lrc_energy:.4E}, Match: {match_lrc}")
        real_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {real_energy:.4E}, NIST: {nist_real_energy:.4E}, Match: {match_real}")
        fourier_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {fourier_energy:.4E}, NIST: {nist_fourier_energy:.4E}, Match: {match_fourier}")
        self_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {self_energy:.4E}, NIST: {nist_self_energy:.4E}, Match: {match_self}")
        intra_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {intra_energy:.4E}, NIST: {nist_intra_energy:.4E}, Match: {match_intra}")

        # Calculate L2 (Squared Euclidean) values for each energy type using the computed values
        l2_dispersion = np.square(dispersion_energy - nist_dispersion_energy)
        l2_lrc = np.square(lrc_energy - nist_lrc_energy)
        l2_real = np.square(real_energy - nist_real_energy)
        l2_fourier = np.square(fourier_energy - nist_fourier_energy)
        l2_self = np.square(self_energy - nist_self_energy)
        l2_intra = np.square(intra_energy - nist_intra_energy)

    # Print final results
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

    """
    # Print out the L2 values
    print(f"L2 Value Comparison of Squared Differences Between Computed and NIST Energy Values with tolerance: {tolerance}")
    print(f"L2 value for Dispersion: {np.sqrt(np.sum(l2_dispersion))}")
    print(f"L2 value for LRC: {np.sqrt(np.sum(l2_lrc))}")
    print(f"L2 value for Real Energy: {np.sqrt(np.sum(l2_real))}")
    print(f"L2 value for Fourier Energy: {np.sqrt(np.sum(l2_fourier))}")
    print(f"L2 value for Self Energy: {np.sqrt(np.sum(l2_self))}")
    print(f"L2 value for Intra Energy: {np.sqrt(np.sum(l2_intra))}")
    """

    # Compute L2 norms
    l2_dispersion_val = np.sqrt(np.sum(l2_dispersion))
    l2_lrc_val = np.sqrt(np.sum(l2_lrc))
    l2_real_val = np.sqrt(np.sum(l2_real))
    l2_fourier_val = np.sqrt(np.sum(l2_fourier))
    l2_self_val = np.sqrt(np.sum(l2_self))
    l2_intra_val = np.sqrt(np.sum(l2_intra))

    # Compute relative errors
    rel_dispersion = l2_dispersion_val / (np.abs(nist_dispersion_energy) + 1e-12)
    rel_lrc = l2_lrc_val / (np.abs(nist_lrc_energy) + 1e-12)
    rel_real = l2_real_val / (np.abs(nist_real_energy) + 1e-12)
    rel_fourier = l2_fourier_val / (np.abs(nist_fourier_energy) + 1e-12)
    rel_self = l2_self_val / (np.abs(nist_self_energy) + 1e-12)
    rel_intra = l2_intra_val / (np.abs(nist_intra_energy) + 1e-12)

    # Print results
    print(f"L2 Value Comparison with Tolerance = {tolerance}")
    print(f"Dispersion: L2 = {l2_dispersion_val:.6g}, Relative Error = {rel_dispersion:.2e}, Match: {rel_dispersion < tolerance}")
    print(f"LRC:        L2 = {l2_lrc_val:.6g}, Relative Error = {rel_lrc:.2e}, Match: {rel_lrc < tolerance}")
    print(f"Real:       L2 = {l2_real_val:.6g}, Relative Error = {rel_real:.2e}, Match: {rel_real < tolerance}")
    print(f"Fourier:    L2 = {l2_fourier_val:.6g}, Relative Error = {rel_fourier:.2e}, Match: {rel_fourier < tolerance}")
    print(f"Self:       L2 = {l2_self_val:.6g}, Relative Error = {rel_self:.2e}, Match: {rel_self < tolerance}")
    print(f"Intra:      L2 = {l2_intra_val:.6g}, Relative Error = {rel_intra:.2e}, Match: {rel_intra < tolerance}")
    
    total_correct = matched_real + matched_fourier + matched_self + matched_intra + matched_dispersion + matched_lrc
    total_incorrect = not_matched_real + not_matched_fourier + not_matched_self + not_matched_intra + not_matched_dispersion + not_matched_lrc

    print(f"Total correct answers: {total_correct}")
    print(f"Total incorrect answers: {total_incorrect}")


# calling compare_coulomb_energy function
compare_LJ_coulomb_energy(results, NIST_TRICLINIC_SPC_E_Water)