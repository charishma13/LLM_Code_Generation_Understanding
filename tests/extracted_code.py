import numpy as np
import pandas as pd
from scipy.special import erfc, erf
import math

###############################################
# New or Modified Functions for Triclinic/Monoclinic
###############################################
def build_triclinic_box_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    '''
    Constructs a 3x3 box matrix and its inverse for a triclinic (or monoclinic) cell.
    a, b, c  : side lengths (float)
    alpha, beta, gamma : angles in degrees (float)
    Returns (box_matrix, inv_box_matrix) as NumPy arrays.
    '''
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    Ax = a
    Ay = 0.0
    Az = 0.0

    Bx = b * math.cos(gamma)
    By = b * math.sin(gamma)
    Bz = 0.0

    Cx = c * math.cos(beta)
    # Safeguard for the formula inside the sqrt:
    # Also note the standard expression for the third vector
    Cy = c * ( ( math.cos(alpha) - math.cos(beta)*math.cos(gamma) ) / ( math.sin(gamma) + 1e-30 ) )
    tmp = 1.0 + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma) \
          - ( math.cos(alpha)**2 + math.cos(beta)**2 + math.cos(gamma)**2 )
    if tmp < 0.0:
        tmp = 0.0  # numerical safeguard
    Cz = c * math.sqrt(tmp) / (math.sin(gamma) + 1e-30)

    box_matrix = np.array([[Ax, Bx, Cx],
                           [Ay, By, Cy],
                           [Az, Bz, Cz]], dtype=float)
    inv_box_matrix = np.linalg.inv(box_matrix)
    return box_matrix, inv_box_matrix

def minimum_image_distance_triclinic(r_ij, box_matrix, inv_box_matrix):
    '''
    Applies the minimum image convention by converting a real-space displacement (r_ij)
    to fractional coordinates, wrapping into [-0.5, 0.5) by rounding, and converting back.
    '''
    frac = inv_box_matrix.dot(r_ij)
    frac -= np.round(frac)
    r_mic = box_matrix.dot(frac)
    return r_mic

def apply_minimum_image(r_ij, configuration, system_data):
    '''
    Wrapper that decides whether to apply a standard cubic minimum-image
    or a triclinic/monoclinic approach based on stored box matrix.
    '''
    cutoff = system_data['cutoff']
    # If box_matrix is present in configuration.attrs, we do the triclinic version
    if 'box_matrix' in configuration.attrs and 'inv_box_matrix' in configuration.attrs:
        box_matrix = configuration.attrs['box_matrix']
        inv_box_matrix = configuration.attrs['inv_box_matrix']
        r_ij_mic = minimum_image_distance_triclinic(r_ij, box_matrix, inv_box_matrix)
        return r_ij_mic
    else:
        # Fallback to cubic
        box_length = system_data['box length']
        return r_ij - box_length * np.round(r_ij / box_length)

###############################################
# Updated extracting_positions for reading cell vectors
###############################################
def extracting_positions(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Parse cell side lengths from the first line
    line0 = lines[0].split()
    a = float(line0[0])
    b = float(line0[1])
    c = float(line0[2])

    # Parse angles from the second line
    line1 = lines[1].split()
    alpha_deg = float(line1[0])
    beta_deg = float(line1[1])
    gamma_deg = float(line1[2])

    # Parse number of molecules from the third line
    line2 = lines[2].split()
    n_mol = int(line2[0])  # not strictly required, but keep for sanity check

    # Build the box matrix and the inverse box matrix
    box_matrix, inv_box_matrix = build_triclinic_box_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)

    # Remaining lines contain atomic data
    data_lines = lines[3:]

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

    columns = ['X', 'Y', 'Z', 'Atom Type']
    configuration = pd.DataFrame(data_list, columns=columns)

    # The standard SPC/E water has 3 atoms per molecule.
    configuration.index = range(1, len(configuration) + 1)
    configuration['Molecule'] = ((configuration.index - 1) // 3) + 1

    # Store cell info in DataFrame attributes
    configuration.attrs['a'] = a
    configuration.attrs['b'] = b
    configuration.attrs['c'] = c
    configuration.attrs['alpha'] = alpha_deg
    configuration.attrs['beta'] = beta_deg
    configuration.attrs['gamma'] = gamma_deg
    configuration.attrs['box_matrix'] = box_matrix
    configuration.attrs['inv_box_matrix'] = inv_box_matrix

    return configuration

###############################################
# Original dictionaries, data, and creation of DataFrames
###############################################
atom_properties = {
    'O': {'type': 'O', 'sigma': 3.165558, 'epsilon': 78.197431, 'charge': -0.8476, 'num_particles': 1},
    'H': {'type': 'H', 'sigma': 0.000000, 'epsilon': 0.000000, 'charge': 0.4238, 'num_particles': 2},
}

file_paths = [
    '../data/spce-sample-non-cuboid-configurations/spce_triclinic_sample_periodic3.txt'
]

NIST_TRICLINIC_SPC_E_Water = {
    'Configuration': [1, 2, 3, 4],
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

def creating_dataframes(file_paths, atom_properties, NIST_SPC_E_Water):
    NIST_SPC_E_Water = pd.DataFrame(NIST_SPC_E_Water)
    NIST_SPC_E_Water['Sum of energies'] = (
        NIST_SPC_E_Water['Edisp/kB (K)'] + NIST_SPC_E_Water['ELRC/kB (K)'] +
        NIST_SPC_E_Water['Ereal/kB (K)'] + NIST_SPC_E_Water['Efourier/kB (K)'] +
        NIST_SPC_E_Water['Eself/kB (K)'] + NIST_SPC_E_Water['Eintra/kB (K)']
    )

    force_field = pd.DataFrame(atom_properties).from_dict(atom_properties, orient='index')

    system = pd.DataFrame(file_paths, columns=['file_paths'])
    system['configuration #'] = (
        system['file_paths'].str.extract(r'(\\d+)', expand=False).fillna('0').astype(int)
    )

    # (Note: if needed, we could parse the side lengths & angles from NIST_SPC_E_Water.
    #  For now, we keep a single 'box length' for demonstration. You could store a, b, c, alpha, beta, gamma similarly.)

    system[["number of particles", "box length"]] = system["configuration #"].apply(
        lambda x: pd.Series({
            "number of particles": float(
                NIST_SPC_E_Water.loc[
                    NIST_SPC_E_Water["Configuration"] == x,
                    "M (number of SPC/E molecules)"
                ].values[0]
            ) if x in NIST_SPC_E_Water["Configuration"].values else 0.0,
            "box length": 30.0  # For demonstration, override with 30.0.
        })
    )

    system['cutoff'] = 10
    system['alpha'] = system.apply(
        lambda row: 5.6 / row['box length'] if row['box length'] != 0 else 0.28,
        axis=1
    )
    system['kmax'] = 5
    system['ε0'] = float(8.854187817E-12)
    system['kB'] = float(1.3806488E-23)

    return system, force_field, NIST_SPC_E_Water

###############################################
# Pairwise Dispersion Function (unchanged except using apply_minimum_image)
###############################################
def pair_dispersion_energy(system_data, configuration, force_field):
    positions = configuration[['X', 'Y', 'Z']].values
    atom_types = configuration['Atom Type'].values
    cutoff = system_data['cutoff']
    num_atoms = len(positions)

    total_dispersion_energy = 0.0

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            r_ij = positions[i] - positions[j]
            # Use the new minimum-image approach
            r_ij = apply_minimum_image(r_ij, configuration, system_data)
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
                potential_energy = 4 * epsilon_ij * (s_over_r**12 - s_over_r**6)
                total_dispersion_energy += potential_energy

    return total_dispersion_energy

###############################################
# Compute LRC energy for LJ (unchanged)
###############################################
def compute_lrc_energy(system_row, force_field):
    U_lrc_total = 0.0
    for atom_type, atom_data in force_field.iterrows():
        num_particles = system_row['number of particles'] * atom_data['num_particles']
        volume = system_row['box length']**3
        rho = num_particles / volume
        sigma = atom_data['sigma']
        epsilon = atom_data['epsilon']
        cutoff = system_row['cutoff']
        sigma_by_cutoff_3 = (sigma / cutoff)**3
        sigma_by_cutoff_9 = sigma_by_cutoff_3**3
        U_lrc_per_particle = (8 / 3) * np.pi * rho * epsilon * sigma**3 * (sigma_by_cutoff_9 / 3 - sigma_by_cutoff_3)
        U_lrc_per_particle *= num_particles
        U_lrc_total += U_lrc_per_particle
    return U_lrc_total

###############################################
# Real-space Coulomb Energy (replacing min_image with apply_minimum_image)
###############################################
def compute_real_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data['alpha']
    cutoff = system_data['cutoff']

    positions = configuration[['X', 'Y', 'Z']].values
    atom_types = configuration['Atom Type'].values
    mol_ids = configuration['Molecule'].values
    charges = np.array([force_field.loc[t, 'charge'] for t in atom_types])

    n_atoms = len(positions)
    real_energy = 0.0

    for j in range(n_atoms - 1):
        for l in range(j + 1, n_atoms):
            if mol_ids[j] == mol_ids[l]:
                continue
            r_ij = positions[l] - positions[j]
            r_ij = apply_minimum_image(r_ij, configuration, system_data)
            r = np.linalg.norm(r_ij)
            if r < cutoff and r > 1e-14:
                q_j = charges[j] * e_charge
                q_l = charges[l] * e_charge
                r_m = r * 1e-10
                real_energy += coulomb_factor * (q_j * q_l / r_m) * erfc(alpha * r)

    return real_energy

###############################################
# Fourier-space Coulomb Energy (expanded for triclinic by building reciprocal)
###############################################
def compute_fourier_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data['alpha']
    kmax = system_data['kmax']
    positions = configuration[['X', 'Y', 'Z']].values
    atom_types = configuration['Atom Type'].values
    charges = np.array([force_field.loc[t, 'charge'] for t in atom_types])
    charges_c = charges * e_charge
    positions_m = positions * 1e-10

    # Build reciprocal box if available
    if 'inv_box_matrix' in configuration.attrs:
        box_matrix = configuration.attrs['box_matrix']
        inv_box_matrix = configuration.attrs['inv_box_matrix']
        # Reciprocal box is 2*pi * (inv_box_matrix^T)
        recip_box = 2.0 * np.pi * inv_box_matrix.T
        # Volume in m^3
        V_m = abs(np.linalg.det(box_matrix * 1e-10))
    else:
        # fallback to cubic
        L = system_data['box length'] * 1e-10
        recip_box = (2.0 * np.pi / L) * np.eye(3)
        V_m = (system_data['box length'] * 1e-10)**3

    prefactor = coulomb_factor / (2.0 * V_m)
    alpha_m = alpha * 1e10

    fourier_energy = 0.0

    # We'll iterate over i, j, k from -kmax..kmax, build the wave vector from the reciprocal box.
    # Then we skip if it's near zero or outside the cutoff in k-space.

    for i in range(-kmax, kmax+1):
        for j in range(-kmax, kmax+1):
            for k in range(-kmax, kmax+1):
                if i == 0 and j == 0 and k == 0:
                    continue
                # k-vector in reciprocal space
                k_vec = i * recip_box[:,0] + j * recip_box[:,1] + k * recip_box[:,2]
                k_sq = np.dot(k_vec, k_vec)
                if k_sq < 1e-14:
                    continue
                # optional restriction if we want to approximate a spherical cutoff
                # or replicate old logic: if i^2 + j^2 + k^2 >= kmax^2 + 2: skip
                if (i*i + j*j + k*k) >= (kmax*kmax + 2):
                    continue

                real_part = 0.0
                imag_part = 0.0
                for idx, (xj, yj, zj) in enumerate(positions_m):
                    kr = k_vec[0]*xj + k_vec[1]*yj + k_vec[2]*zj
                    real_part += charges_c[idx]*math.cos(kr)
                    imag_part += charges_c[idx]*math.sin(kr)

                sk_sq = real_part*real_part + imag_part*imag_part
                exponent = math.exp(-k_sq/(4.0*alpha_m**2))
                term = prefactor * (4.0*math.pi / k_sq) * exponent * sk_sq
                fourier_energy += term

    return fourier_energy

###############################################
# Self-Energy and Intra-molecular (unchanged except references)
###############################################
def compute_self_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data['alpha']
    atom_types = configuration['Atom Type'].values
    charges = np.array([force_field.loc[t,'charge'] for t in atom_types])
    charges_c = charges * e_charge
    alpha_m = alpha*1e10
    sum_q2 = np.sum(charges_c**2)
    self_energy = - coulomb_factor * (alpha_m / math.sqrt(math.pi)) * sum_q2
    return self_energy

def compute_intra_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23

    alpha = system_data['alpha']
    positions = configuration[['X', 'Y', 'Z']].values
    atom_types = configuration['Atom Type'].values
    charges = np.array([force_field.loc[t,"charge"] for t in atom_types])
    mol_ids = configuration['Molecule'].values

    intra_energy = 0.0
    unique_mols = np.unique(mol_ids)

    for m_id in unique_mols:
        idxs = np.where(mol_ids == m_id)[0]
        n_mol_atoms = len(idxs)
        for i in range(n_mol_atoms - 1):
            for j in range(i+1, n_mol_atoms):
                idx_i = idxs[i]
                idx_j = idxs[j]
                r_ij = positions[idx_j] - positions[idx_i]
                # For intramolecular, we might still want periodic images if the molecule crosses a boundary
                r_ij = apply_minimum_image(r_ij, configuration, system_data)
                r = np.linalg.norm(r_ij)
                if r > 1e-14:
                    q_i = charges[idx_i]*e_charge
                    q_j = charges[idx_j]*e_charge
                    r_m = r*1e-10
                    erf_val = erf(alpha*r)
                    val = coulomb_factor*(q_i*q_j / r_m)*erf_val
                    intra_energy -= val

    return intra_energy

###############################################
# Create DataFrames and Run Calculations
###############################################
system, force_field, NIST_SPC_E_Water = creating_dataframes(file_paths, atom_properties, NIST_TRICLINIC_SPC_E_Water)

results = pd.DataFrame()
results['Number of Particles'] = system['number of particles'].astype(int)

results['lrc_Energies'] = system.apply(
    lambda row: compute_lrc_energy(row, force_field), axis=1
)

results['dispersion_energies'] = system['file_paths'].apply(
    lambda file_path: pair_dispersion_energy(
        system[system['file_paths'] == file_path].iloc[0],
        extracting_positions(file_path),
        force_field
    )
)

results['real_energies'] = system['file_paths'].apply(
    lambda file_path: compute_real_energies(
        system[system['file_paths'] == file_path].iloc[0],
        extracting_positions(file_path),
        force_field
    )
)

results['fourier_energies'] = system['file_paths'].apply(
    lambda file_path: compute_fourier_energies(
        system[system['file_paths'] == file_path].iloc[0],
        extracting_positions(file_path),
        force_field
    )
)

results['self_energies'] = system['file_paths'].apply(
    lambda file_path: compute_self_energies(
        system[system['file_paths'] == file_path].iloc[0],
        extracting_positions(file_path),
        force_field
    )
)

results['intra_energies'] = system['file_paths'].apply(
    lambda file_path: compute_intra_energies(
        system[system['file_paths'] == file_path].iloc[0],
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

    real_energy_output, fourier_energy_output = [], []
    self_energy_output, intra_energy_output = [], []
    lrc_energy_output, dispersion_energy_output = [], []

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

    print()
    print("Lennard-Jones Pair Dispersion Energy Comparison:")
    print(*dispersion_energy_output, sep=chr(10))
    print("Lennard-Jones long-range corrections Energy Comparison:")
    print(*lrc_energy_output)
    print()
    print("Real Energy Comparison:")
    print(*real_energy_output)
    print()
    print("Fourier Energy Comparison:")
    print(*fourier_energy_output)
    print()
    print("Self Energy Comparison:")
    print(*self_energy_output)
    print()
    print("Intra Energy Comparison:")
    print(*intra_energy_output,  sep=chr(10))
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

    total_correct = (matched_real + matched_fourier + matched_self +
                     matched_intra + matched_dispersion + matched_lrc)
    total_incorrect = (not_matched_real + not_matched_fourier + not_matched_self +
                       not_matched_intra + not_matched_dispersion + not_matched_lrc)
    print(f"Total correct answers: {total_correct}")
    print(f"Total incorrect answers: {total_incorrect}")

# Example usage:
compare_LJ_coulomb_energy(results, NIST_SPC_E_Water)
