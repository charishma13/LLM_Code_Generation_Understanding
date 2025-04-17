import numpy as np
import pandas as pd

# defining all variables
atom_properties = {
    'O': {'type': 'O', 'sigma': 3.165558, 'epsilon': 78.197431, 'charge': -0.8476, 'num_particles': 1},
    'H': {'type': 'H', 'sigma': 0.000000, 'epsilon': 0.000000, 'charge': 0.4238, 'num_particles': 2},
}

file_paths = [
        '../data/spce_sample_config_periodic4.txt',
        '../data/spce_sample_config_periodic2.txt',
        '../data/spce_sample_config_periodic3.txt',
        '../data/spce_sample_config_periodic1.txt'
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
def creating_dataframes(file_paths, atom_properties,NIST_SPC_E_Water):
    
    # Create the NIST_SPC_E_Water dataframe
    NIST_SPC_E_Water = pd.DataFrame(NIST_SPC_E_Water)
    
    NIST_SPC_E_Water['Sum of energies'] = (NIST_SPC_E_Water['Edisp/kB (K)'] + NIST_SPC_E_Water['ELRC/kB (K)'] +
                             NIST_SPC_E_Water['Ereal/kB (K)'] + NIST_SPC_E_Water['Efourier/kB (K)'] +
                             NIST_SPC_E_Water['Eself/kB (K)'] + NIST_SPC_E_Water['Eintra/kB (K)'])

    #for col in NIST_SPC_E_Water.columns[3:]:  # Skip the first column (Configuration)
     #   NIST_SPC_E_Water[col] = NIST_SPC_E_Water[col].apply(lambda x: f"{x:.4E}")
        
    # Creating the force_field dataframe
    force_field = pd.DataFrame(atom_properties).from_dict(atom_properties, orient='index')

    # Create the system dataframe contaning some variables
    system = pd.DataFrame(file_paths, columns=["file_paths"])

    system['configuration #'] = system['file_paths'].str.extract(r'(\d+)').astype(int)

    system[["number of particles", "box length"]] = system["configuration #"].apply(
    lambda x: pd.Series({
        "number of particles": float(NIST_SPC_E_Water.loc[NIST_SPC_E_Water["Configuration"] == x, 
                                                          "M (number of SPC/E molecules)"].values[0]),
        "box length": float(NIST_SPC_E_Water.loc[NIST_SPC_E_Water["Configuration"] == x, 
                                                 "Lx=Ly=Lz (Å)"].values[0])}))

    system['cutoff'] = 10
    system['alpha'] = 5.6
    system['kmax'] = 3
        
    return system, force_field, NIST_SPC_E_Water

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
        volume = system_row['box length'] ** 3
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


# pairwise dispersion energy functions

# Minimum Image Distance function and Pair Dispersion Energy calculation (Code 3)
def minimum_image_distance(r_ij, cell_length):
    # Apply the minimum image convention to distances.
    return r_ij - cell_length * np.round(r_ij / cell_length)

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



# dataframes
system, force_field, NIST_SPC_E_Water = creating_dataframes(file_paths, atom_properties,NIST_SPC_E_Water)

# Computing energies storing in results
results = pd.DataFrame()

results['Number of Particles'] = system['number of particles'].astype(int)

# Calculate LRC energy for all system configurations
results['LRC Energy'] = system.apply(
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

# Comparision function
def compare_LJ_energy(df1, df2, tolerance=1e-4):
    # Initialize output variables
    pairwise_output = []
    lrc_output = []
    matched_pairwise = 0
    matched_lrc = 0
    not_matched_pairwise = 0
    not_matched_lrc = 0

    # Ensure both dataframes have the same length
    if len(df1) != len(df2):
        raise ValueError("DataFrames must have the same length")

    # Iterate over rows in the dataframes
    for idx, row in df1.iterrows():
        # Extract values from each row
        dispersion_energy = row['dispersion_energies']
        lrc_energy = row['LRC Energy']
        nist_dispersion_energy = df2.loc[idx, 'Edisp/kB (K)']
        nist_lrc_energy = df2.loc[idx, 'ELRC/kB (K)']
        num_molecules = row['Number of Particles']  
        
        nist_lrc_energy = float(NIST_SPC_E_Water.loc[NIST_SPC_E_Water['M (number of SPC/E molecules)'] == num_molecules, 'ELRC/kB (K)'].values[0])
        
        nist_dispersion_energy = float(NIST_SPC_E_Water.loc[NIST_SPC_E_Water['M (number of SPC/E molecules)'] == num_molecules, 'Edisp/kB (K)'].values[0])

        # Perform numeric comparisons with a tolerance
        match_dispersion = np.isclose(dispersion_energy, nist_dispersion_energy, atol=tolerance)
        match_lrc = np.isclose(lrc_energy, nist_lrc_energy, atol=tolerance)

        matched_pairwise += int(match_dispersion)
        not_matched_pairwise += int(not match_dispersion)

        matched_lrc += int(match_lrc)
        not_matched_lrc += int(not match_lrc)

        # Format output strings
        pairwise_str = (
            f"Test {idx+1} ({num_molecules} molecules): "
            f"LLM answer: {dispersion_energy:.4E}, NIST answer: {nist_dispersion_energy:.4E}, "
            f"Match: {match_dispersion}"
        )

        lrc_str = (
            f"Test {idx+1} ({num_molecules} molecules): "
            f"LLM answer: {lrc_energy:.4E}, NIST answer: {nist_lrc_energy:.4E}, "
            f"Match: {match_lrc}"
        )

        pairwise_output.append(pairwise_str)
        lrc_output.append(lrc_str)

    # Print final results
    print("Lennard-Jones Pair Dispersion Energy:")
    print(*pairwise_output, sep='\n')
    print("\nLennard-Jones long-range corrections:")
    print(*lrc_output, sep='\n')
    print()
    print(f"Count of correct pairwise answers: {matched_pairwise}")
    print(f"Count of incorrect pairwise answers: {not_matched_pairwise}")
    print(f"Count of correct LRC answers: {matched_lrc}")
    print(f"Count of incorrect LRC answers: {not_matched_lrc}")

# calling compare_LJ_energy function 
compare_LJ_energy(results, NIST_SPC_E_Water)