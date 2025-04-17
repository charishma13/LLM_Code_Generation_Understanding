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
#        'spce_sample_config_periodic4.txt',
#        'spce_sample_config_periodic2.txt',
#        '../data/spce_sample_config_periodic3.txt',
        '../data/spce_sample_config_periodic1_modified.txt'
    ]

NIST_SPC_E_Water = {
        'Configuration': [1, 2, 3, 4],
        'M (number of SPC/E molecules)': [100, 200, 300, 750],
        'Lx=Ly=Lz (Å)': [20.0, 20.0, 20.0, 30.0],
        'Edisp/kB (K)': [9.95387E+04, 1.93712E+05, 3.54344E+05, 4.48593E+05],
        'ELRC/kB (K)': [-8.23715E+02, -3.29486E+03, -7.41343E+03, -1.37286E+04],
        'Ereal/kB (K)': [-5.58889E+05, -1.19295E+06, -1.96297E+06, -3.57226E+06],
        'Efourier/kB (K)': [6.27009E+03, 6.03495E+03, 5.24461E+03, 7.58785E+03],
        'Eself/kB (K)': [-2.84469E+06, -5.68938E+06, -8.53407E+06, -1.42235E+07],
        'Eintra/kB (K)': [2.80999E+06, 5.61998E+06, 8.42998E+06, 1.41483E+07],
        'Etotal/kB (K)': [-4.88604E+05, -1.06590E+06, -1.71488E+06, -3.20501E+06]
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

# create the target dataframes
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

    system[["number of particles", "box length"]] = system["configuration #"].apply(
        lambda x: pd.Series({
            "number of particles": float(
                NIST_SPC_E_Water.loc[
                    NIST_SPC_E_Water["Configuration"] == x,
                    "M (number of SPC/E molecules)"
                ].values[0]
            ) if x in NIST_SPC_E_Water["Configuration"].values else 0.0,
            "box length": float(
                NIST_SPC_E_Water.loc[
                    NIST_SPC_E_Water["Configuration"] == x,
                    "Lx=Ly=Lz (Å)"
                ].values[0]
            ) if x in NIST_SPC_E_Water["Configuration"].values else 20.0
        })
    )

    system['cutoff'] = 10

    # Fix alpha so that alpha = 5.6 / L
    system['alpha'] = system.apply(
        lambda row: 5.6 / row['box length'] if row['box length'] != 0 else 0.28,
        axis=1
    )

    system['kmax'] = 5
    system['ε0'] = float(8.854187817E-12)
    system['kB'] = float(1.3806488E-23)

    return system, force_field, NIST_SPC_E_Water

def compute_real_energies(system_data, configuration, force_field):
    e_charge = 1.602176634e-19  # C
    coulomb_factor = 8.9875517923e9 / 1.3806488e-23  # (1/(4 pi eps0)) / kB

    alpha = system_data["alpha"]  # 1/Å
    cutoff = system_data["cutoff"]  # Å
    L = system_data["box length"]  # Å

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
    L = system_data["box length"]
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

# 1. NIST_SPC_E_Water DataFrame:
#    - Contains thermodynamic properties of SPC/E water configurations in monoclinic cells.
#    - Columns:
#        - 'Configuration' (int): Configuration ID (1-4).
#        - 'M (number of SPC/E molecules)' (int): Number of SPC/E molecules in the system.
#        - 'Lx=Ly=Lz (Å)' (float): Box dimensions in Ångströms (single dimension for cubic cell).
#        - 'Edisp/kB (K)' (float), 'ELRC/kB (K)' (float), 'Ereal/kB (K)' (float), 'Efourier/kB (K)' (float),
#          'Eself/kB (K)' (float), 'Eintra/kB (K)' (float), 'Etotal/kB (K)' (float): Various energy components in Kelvin.
#        - 'Sum of energies' (float): Sum of all energy components.

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


system, force_field, NIST_SPC_E_Water = creating_dataframes(file_paths, atom_properties,NIST_SPC_E_Water)

# Computing energies storing in results
results = pd.DataFrame()

results['Number of Particles'] = system['number of particles'].astype(int)

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

def compare_coulomb_energy(df1, df2, tolerance=1e-4):
    # Merge df1 and df2 based on the number of particles
    df_merged = df1.merge(df2, left_on='Number of Particles', right_on='M (number of SPC/E molecules)', how='left')

    # Initialize counters
    matched_real = matched_fourier = matched_self = matched_intra = 0
    not_matched_real = not_matched_fourier = not_matched_self = not_matched_intra = 0

    # Initialize output lists
    real_energy_output, fourier_energy_output = [], []
    self_energy_output, intra_energy_output = [], []

    # Iterate over merged DataFrame
    for idx, row in df_merged.iterrows():
        # Extract computed values from df1
        real_energy = row['real_energies']
        fourier_energy = row['fourier_energies']
        self_energy = row['self_energies']
        intra_energy = row['intra_energies']
        num_molecules = row['Number of Particles']

        # Extract reference values from df2
        if pd.isna(row['Ereal/kB (K)']):
            continue  # Skip if no match is found in df2

        nist_real_energy = float(row['Ereal/kB (K)'])
        nist_fourier_energy = float(row['Efourier/kB (K)'])
        nist_self_energy = float(row['Eself/kB (K)'])
        nist_intra_energy = float(row['Eintra/kB (K)'])

        # Perform numeric comparisons with a tolerance
        match_real = np.isclose(real_energy, nist_real_energy, atol=tolerance)
        match_fourier = np.isclose(fourier_energy, nist_fourier_energy, atol=tolerance)
        match_self = np.isclose(self_energy, nist_self_energy, atol=tolerance)
        match_intra = np.isclose(intra_energy, nist_intra_energy, atol=tolerance)

        matched_real += int(match_real)
        not_matched_real += int(not match_real)

        matched_fourier += int(match_fourier)
        not_matched_fourier += int(not match_fourier)

        matched_self += int(match_self)
        not_matched_self += int(not match_self)

        matched_intra += int(match_intra)
        not_matched_intra += int(not match_intra)

        # Store formatted outputs
        real_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {real_energy:.4E}, NIST: {nist_real_energy:.4E}, Match: {match_real}")
        fourier_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {fourier_energy:.4E}, NIST: {nist_fourier_energy:.4E}, Match: {match_fourier}")
        self_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {self_energy:.4E}, NIST: {nist_self_energy:.4E}, Match: {match_self}")
        intra_energy_output.append(f"Test {idx+1} ({num_molecules} molecules): Computed: {intra_energy:.4E}, NIST: {nist_intra_energy:.4E}, Match: {match_intra}")

    # Print final results
    print("Real Energy Comparison:")
    print(*real_energy_output, sep=chr(10))
    print("Fourier Energy Comparison:")
    print(*fourier_energy_output, sep=chr(10))
    print("Self Energy Comparison:")
    print(*self_energy_output, sep=chr(10))
    print("Intra Energy Comparison:")
    print(*intra_energy_output, sep=chr(10))
    print()
    print(f"Count of correct Real Energy answers: {matched_real}")
    print(f"Count of incorrect Real Energy answers: {not_matched_real}")
    print(f"Count of correct Fourier Energy answers: {matched_fourier}")
    print(f"Count of incorrect Fourier Energy answers: {not_matched_fourier}")
    print(f"Count of correct Self Energy answers: {matched_self}")
    print(f"Count of incorrect Self Energy answers: {not_matched_self}")
    print(f"Count of correct Intra Energy answers: {matched_intra}")
    print(f"Count of incorrect Intra Energy answers: {not_matched_intra}")
    print()

    total_correct = matched_real + matched_fourier + matched_self + matched_intra
    total_incorrect = not_matched_real + not_matched_fourier + not_matched_self + not_matched_intra

    print(f"Total correct answers: {total_correct}")
    print(f"Total incorrect answers: {total_incorrect}")


# calling compare_coulomb_energy function
compare_coulomb_energy(results, NIST_SPC_E_Water)