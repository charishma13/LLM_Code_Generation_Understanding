import pandas as pd
import numpy as np
import re
from scipy.special import erfc,erf
import math 

atom_properties = {
    'HW': {'type': 'HW', 'sigma': '3.165558', 'epsilon': '78.197431', 'charge': '-0.8476', 'num_particles': '2'},
    'OW': {'type': 'OW', 'sigma': '0.0', 'epsilon': '0.0', 'charge': '0.4238', 'num_particles': '1'},
    'SiZ': {'type': 'SiZ', 'sigma': '2.3', 'epsilon': '22.0', 'charge': '-2 + OZ', 'num_particles': '191'},
    'OZ': {'type': 'OZ', 'sigma': '3.3', 'epsilon': '53.0', 'charge': '-0.75', 'num_particles': '384'},
    'AlZ': {'type': 'AlZ', 'sigma': '2.3', 'epsilon': '22.0', 'charge': 'SiZ - 1', 'num_particles': '1'},
    'HZ': {'type': 'HZ', 'sigma': '1.0', 'epsilon': '100.0', 'charge': '1', 'num_particles': '1'}
}


# Trying for single configurations first
file_paths = [
    #   'Zeolites_acid_base/configuration_1.xyz'
        'Zeolites_acid_base/configuration_2.xyz'
    ]

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
                atom_type = parts[0]
                data_list.append([x, y, z, atom_type])
            except ValueError:
                continue  
    
    # Create a DataFrame
    columns = ["X", "Y", "Z", "Atom Type"]
    configuration = pd.DataFrame(data_list, columns=columns)

    # Rename atom types
    rename_map = {
        "O": "OZ",
        "Si": "SiZ",
        "H": "HZ",
        "Al": "AlZ"
    }
    configuration.loc[configuration.index < len(configuration) - 3, 'Atom Type'] = configuration.loc[
        configuration.index < len(configuration) - 3, 'Atom Type'].replace(rename_map)
    configuration.iloc[-3:, configuration.columns.get_loc("Atom Type")] = ["HW", "OW", "HW"]

    # Initialize Molecule column
    configuration["Molecule"] = 0

    # Assign molecule ID for HZ
    configuration.loc[configuration["Atom Type"] == "HZ", "Molecule"] = 1

    # Assign molecule ID for H2O (OW + 2 HW) — last 3 atoms
    configuration.iloc[-3:, configuration.columns.get_loc("Molecule")] = 2

    # Assign molecule IDs to SiO2 units: 1 SiZ + 2 OZ atoms per unit
    framework = configuration.iloc[:576].copy()
    si_o_pairs = []
    mol_id = 3
    used_indices = set()

    for si_index in framework[framework["Atom Type"] == "SiZ"].index:
        # Find 2 closest unused OZ atoms
        si_coords = framework.loc[si_index, ["X", "Y", "Z"]].values
        oz_candidates = framework[(framework["Atom Type"] == "OZ") & (~framework.index.isin(used_indices))]
        oz_coords = oz_candidates[["X", "Y", "Z"]].values

        if len(oz_coords) < 2:
            continue

        distances = ((oz_coords - si_coords) ** 2).sum(axis=1)
        closest_oz = oz_candidates.iloc[distances.argsort()[:2]]

        # Assign molecule ID to SiZ and two closest OZ
        configuration.loc[[si_index], "Molecule"] = mol_id
        configuration.loc[closest_oz.index, "Molecule"] = mol_id
        used_indices.update(closest_oz.index)
        mol_id += 1

    # Reorder columns
    configuration = configuration[["Molecule", "X", "Y", "Z", "Atom Type"]]

    return configuration



# Create the target dataframes
def creating_dataframes(file_paths, atom_properties):
    # Creating the force_field dataframe
    force_field = pd.DataFrame.from_dict(atom_properties, orient='index')

    # Create the system dataframe with initialized columns
    system_data = []

    for path in file_paths:
        with open(path, "r") as file:
            lines = file.readlines()

        prefix_lines = lines[:2]

        # Extract energy
        energy_match = re.search(r'energy=([-+]?\d*\.\d+|\d+)', prefix_lines[1])
        energy = float(energy_match.group(1)) if energy_match else None

        # Extract lattice constants
        lattice_match = re.search(r'Lattice="([^"]+)"', prefix_lines[1])
        if lattice_match:
            lattice_values = lattice_match.group(1).split()
            lattice_floats = list(map(float, lattice_values))
            box_length = lattice_floats[0]  # Assuming cubic box
        else:
            lattice_floats = []
            box_length = None

        # Number of particles can be parsed from the first line
        try:
            num_particles = int(prefix_lines[0].strip())
        except ValueError:
            num_particles = None

        # Extracting the pbc value using regex
        pbc_match = re.search(r'pbc="([^"]+)"', prefix_lines[1])

        # Check if the match was found
        if pbc_match:
            pbc_value = pbc_match.group(1)
        else:
            print("pbc value not found")

        # Append a row of data
        system_data.append({
            "file_paths": path,
            "energy": energy,
            "number of particles": num_particles,
            "box length": box_length,
            "lattice floats": lattice_floats,
            "pbc value": pbc_value,
            "cutoff": 10,
            "alpha": 5.6 / box_length if box_length and box_length != 0 else 0.28,
            "kmax": 5,
            "ε0": 8.854187817E-12,
            "kB": 1.3806488E-23
        })

    system = pd.DataFrame(system_data)
    return system, force_field

system, force_field = creating_dataframes(file_paths, atom_properties)

results = pd.DataFrame()
results['Number of Particles'] = system['number of particles'].astype(int)

def minimum_image_distance(r_ij, cell_length):
    # Apply the minimum image convention to distances.
    return r_ij - cell_length * np.round(r_ij / cell_length)

import numpy as np

def evaluate_charge(charge_str, force_field):
    """Evaluate charge expression safely."""
    # Replace atom types with corresponding charges in the expression
    for atom_type in force_field.index:
        charge_value = force_field.loc[atom_type, 'charge']
        charge_str = charge_str.replace(atom_type, str(charge_value))
    
    # Evaluate the final expression for charge
    try:
        charge_value = eval(charge_str)
    except Exception as e:
        print(f"Error evaluating charge: {charge_str} -> {e}")
        charge_value = 0.0  # Default value in case of error
    
    return charge_value

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
    cell_length = system_data['box length']
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

                epsilon_i = float(force_field.loc[type_i, 'epsilon'])
                epsilon_j = float(force_field.loc[type_j, 'epsilon'])
                sigma_i = float(force_field.loc[type_i, 'sigma'])
                sigma_j = float(force_field.loc[type_j, 'sigma'])

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
        num_particles = system_row['number of particles'] * float(atom_data['num_particles'])
        
        # Calculate the system's volume
        volume = system_row['box length'] ** 3
        rho = num_particles / volume

        # Compute the LRC energy for each particle
        total_lrc_energy = 0.0

        # Get epsilon and sigma for each particle
        sigma = float(atom_data['sigma'])
        epsilon = float(atom_data['epsilon'])

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
    L = system_data["box length"]  # Å

    def min_image(dx, box_length):
        return dx - round(dx / box_length) * box_length

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    mol_ids = configuration["Molecule"].values
    charges = np.array([evaluate_charge(force_field.loc[t, "charge"], force_field) for t in atom_types])

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
    charges = np.array([evaluate_charge(force_field.loc[t, "charge"], force_field) for t in atom_types])
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
    charges = np.array([evaluate_charge(force_field.loc[t, "charge"], force_field) for t in atom_types])
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
    charges = np.array([evaluate_charge(force_field.loc[t, "charge"], force_field) for t in atom_types])
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

def acid_base(system_data, configuration, force_field):
    """
    Calculates the acid/base interaction term for a system using force field parameters.

    This function computes the contribution of acid/base interactions by adjusting the force field 
    parameters for specific atoms (e.g., AlZ and HZ), while keeping others (e.g., SiZ, OZ, HW, OW) fixed. 
    It uses force field models for water (SPC/E), zeolite (TraPPE-Zeo), and other molecules, and incorporates 
    Lennard-Jones and Coulombic interaction energies.

    **Parameters:**
    
    1. **system_data (pd.DataFrame)**: 
       - Contains metadata for the system such as number of particles, box length, cutoff distance, etc.
    
    2. **configuration (pd.DataFrame)**:
       - Contains atomic positions and molecule indices for the system.

    3. **force_field (pd.DataFrame)**:
       - Contains force field parameters (e.g., sigma, epsilon, charge) for various atoms, including water (SPC/E), zeolite (TraPPE-Zeo), and other components like AlZ and HZ.

    **Returns:**
    - **acid_base_part (float)**: The computed acid/base interaction energy contribution.

    **Notes:**
    - Fixed parameters: SiZ, OZ, HW, OW.
    - Optimized parameters: AlZ, HZ (with initial guesses for AlZ and HZ).
    - The function uses standard literature force fields for water, zeolite, and acid/base interactions.
    """
    acid_base_part = 0
    return acid_base_part

# Calculate pairwise energy for all system configurations
results['dispersion_energies'] = system['file_paths'].apply(
    lambda file_path: pair_dispersion_energy(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path), 
        force_field
    )
)

# Calculate LRC energy for all system configurations
results['lrc_Energies'] = system.apply(
    lambda row: compute_lrc_energy(row, force_field), axis=1
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

# Calculate pairwise energy for all system configurations
results['acid_base'] = system['file_paths'].apply(
    lambda file_path: acid_base(
        system[system['file_paths'] == file_path].iloc[0],  # Ensure single row selection
        extracting_positions(file_path),
        force_field
    )
)

def compare_LJ_coulomb_energy(results, energies, tolerance=1e-4):
    # Iterate over merged DataFrame
    df_merged = results.merge(energies, left_on='Number of Particles', right_on='number of particles', how='left')

    energy_output = []
    matched_energy, not_matched_energy = 0,0
    energy_val, rel_error = 0,0
    energy_dict = {}

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
        acid_base = row['acid_base']

        total_energy = row['energy']
        computed_energy = fourier_energy + self_energy + lrc_energy + dispersion_energy + real_energy + intra_energy + acid_base

        match_energy = np.isclose(total_energy, computed_energy, atol=tolerance)

        matched_energy += int(match_energy)
        not_matched_energy += int(not match_energy)

        energy_output.append(f"TraPPE-Zeo ({num_molecules} molecules): Computed: {computed_energy:.4E}, DFT Energy: {total_energy:.4E}, Match: {match_energy}")

        energy_sqr = np.square(total_energy - computed_energy)
        energy_val = np.sqrt(np.sum(energy_sqr))
        rel_error = energy_val / (np.abs(total_energy) + 1e-12)

        energy_dict = {
            "fourier_energy": fourier_energy,
            "self_energy": self_energy,
            "lrc_energy": lrc_energy,
            "dispersion_energy": dispersion_energy,
            "real_energy": real_energy,
            "intra_energy": intra_energy,
            "acid_base": acid_base
        }

    # Print final results
    print()
    print("Terms of Energy: ")
    for label, val in energy_dict.items():
        print(f"{label}: {int(val):.4E}")
    print()
    print("Energy Comparison:")
    print(*energy_output)
    print()
    print(f"Count of correct pairwise answers: {matched_energy}")
    print(f"Count of incorrect pairwise answers: {not_matched_energy}")
    print()
    print(f"Dispersion: L2 = {energy_val:.6g}, Relative Error = {rel_error:.2e}, Match: {rel_error < tolerance}")


compare_LJ_coulomb_energy(results, system[['number of particles','energy']])