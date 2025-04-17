import numpy as np
import pandas as pd
from scipy.special import erfc

# defining all variables
atom_properties = {
    'O': {'type': 'O', 'sigma': 3.165558, 'epsilon': 78.197431, 'charge': -0.8476, 'num_particles': 1},
    'H': {'type': 'H', 'sigma': 0.000000, 'epsilon': 0.000000, 'charge': 0.4238, 'num_particles': 2},
}

# Trying for small configurations first
file_paths = [
#        'spce_sample_config_periodic4.txt',
#        'spce_sample_config_periodic2.txt',
#        'spce_sample_config_periodic3.txt',
        'spce_sample_config_periodic1_modified.txt'
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
    system['kmax'] = 5
    system['ε0'] = float(8.854187817E-12)
    system['kB'] = float(1.3806488E-23)
        
    return system, force_field, NIST_SPC_E_Water

def min_image(dx, box):
        return dx - np.rint(dx/box)*box
def min_image(dx, box):
    return dx - np.rint(dx/box)*box
def compute_real_energies(system_data, configuration, force_field):
    import numpy as np

    # --- Complete this code --- #
    alpha_dimless = system_data['alpha']  
    box_length = system_data['box length']
    alpha_local = alpha_dimless / box_length  # Å⁻¹

    cutoff = system_data['cutoff']
    epsilon0 = system_data['ε0']
    kB = system_data['kB']

    e_si = 1.602176634e-19
    coul_SI = 1.0/(4.0*np.pi*epsilon0) * e_si*e_si  # units: J·m
    coul_prefactor = coul_SI/(1e-10)/kB            # => K·Å / (e^2)

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types], dtype=float)
    molecules = configuration["Molecule"].values   # Identify each atom’s molecule

    real_energy = 0.0
    N = len(positions)

    for i in range(N - 1):
        xi, yi, zi = positions[i]
        qi = charges[i]
        for j in range(i + 1, N):
            # Skip if in the same molecule so as not to double-count with Eintra
            if molecules[i] == molecules[j]:
                continue

            xj, yj, zj = positions[j]
            qj = charges[j]
            dx = min_image(xi - xj, box_length)
            dy = min_image(yi - yj, box_length)
            dz = min_image(zi - zj, box_length)
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            if r < cutoff and r > 1e-12:
                real_energy += qi * qj * erfc(alpha_local * r) / r

    real_energy *= coul_prefactor
    return real_energy






def compute_fourier_energies(system_data, configuration, force_field):
    import numpy as np

    # --- Complete this code --- #
    alpha_dimless = system_data['alpha']  
    box_length = system_data['box length']
    alpha_local = alpha_dimless / box_length  # α in Å⁻¹

    kmax = system_data['kmax']
    epsilon0 = system_data['ε0']
    kB = system_data['kB']

    e_si = 1.602176634e-19
    coul_SI = 1.0/(4.0*np.pi*epsilon0) * e_si*e_si
    coul_prefactor = coul_SI/(1e-10)/kB  # => K·Å/e²

    # Volume
    V = box_length**3
    # Adjust prefactor to match the standard Ewald formula
    prefactor = coul_prefactor * (2.0 * np.pi / V)

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types], dtype=float)
    N = len(positions)

    E_fourier = 0.0

    for nx in range(-kmax, kmax+1):
        for ny in range(-kmax, kmax+1):
            for nz in range(-kmax, kmax+1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue

                # kactual^2 = (2π/L)^2 * (nx^2 + ny^2 + nz^2)
                k_sq_integer = nx*nx + ny*ny + nz*nz
                kactual_sq = (2.0*np.pi / box_length)**2 * k_sq_integer

                if kactual_sq < 1e-12:
                    continue

                # Ewald damping: exp( - k^2 / (4 α^2) )
                exponent = - (kactual_sq / (4.0 * alpha_local**2))
                exp_factor = np.exp(exponent)

                # Structure factor
                sf_real, sf_imag = 0.0, 0.0
                for j in range(N):
                    kr = (2.0*np.pi / box_length)*(nx*positions[j,0] + ny*positions[j,1] + nz*positions[j,2])
                    c = np.cos(kr)
                    s = np.sin(kr)
                    sf_real += charges[j]*c
                    sf_imag += charges[j]*s

                S_k_sq = sf_real*sf_real + sf_imag*sf_imag

                E_fourier += exp_factor / kactual_sq * S_k_sq

    E_fourier *= prefactor
    return E_fourier








def compute_self_energies(system_data, configuration, force_field):
    import numpy as np

    # --- Complete this code --- #
    alpha_dimless = system_data['alpha']  
    box_length = system_data['box length']
    alpha_local = alpha_dimless / box_length  # Å⁻¹

    epsilon0 = system_data['ε0']
    kB = system_data['kB']

    e_si = 1.602176634e-19
    coul_SI = 1.0/(4.0*np.pi*epsilon0) * e_si*e_si
    coul_prefactor = coul_SI/(1e-10)/kB

    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types], dtype=float)
    sum_q2 = np.sum(charges**2)

    # Use alpha_local
    self_energy = - alpha_local/np.sqrt(np.pi) * sum_q2 * coul_prefactor
    return self_energy





def compute_intra_energies(system_data, configuration, force_field):
    import numpy as np

    alpha_dimless = system_data['alpha']
    box_length = system_data['box length']
    alpha_local = alpha_dimless / box_length  # Å⁻¹

    epsilon0 = system_data['ε0']
    kB = system_data['kB']

    e_si = 1.602176634e-19
    coul_SI = 1.0/(4.0*np.pi*epsilon0) * e_si*e_si
    coul_prefactor = coul_SI/(1e-10)/kB

    positions = configuration[["X", "Y", "Z"]].values
    atom_types = configuration["Atom Type"].values
    charges = np.array([force_field.loc[t, "charge"] for t in atom_types], dtype=float)
    molecules = configuration["Molecule"].values

    intra_energy = 0.0
    unique_mols = np.unique(molecules)

    # Pairwise sum i<j within each molecule
    for mol in unique_mols:
        indices = np.where(molecules == mol)[0]
        if len(indices) < 2:
            continue

        for i in range(len(indices) - 1):
            iidx = indices[i]
            qi = charges[iidx]
            xi, yi, zi = positions[iidx]
            for j in range(i+1, len(indices)):
                jidx = indices[j]
                qj = charges[jidx]
                xj, yj, zj = positions[jidx]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r > 1e-12:
                    val = qi * qj * (np.math.erf(alpha_local * r) / r)
                    intra_energy += val

    # Remove “×0.5” and instead flip the sign
    intra_energy *= -1.0 * coul_prefactor
    return intra_energy





    

# DataFrame Descriptions:

# 1. NIST_SPC_E_Water DataFrame:
#    - Contains thermodynamic properties of SPC/E water configurations.
#    - Columns:
#        - 'Configuration': Configuration ID (1-4).
#        - 'M (number of SPC/E molecules)': Number of SPC/E molecules.
#        - 'Lx=Ly=Lz (Å)': Box length (Ångströms).
#        - 'Edisp/kB (K)', 'ELRC/kB (K)', 'Ereal/kB (K)', 'Efourier/kB (K)', 'Eself/kB (K)', 'Eintra/kB (K)', 'Etotal/kB (K)': Various energy components in Kelvin.
#        - 'Sum of energies': Computed sum of all energy components.

# 2. force_field DataFrame:
#    - Contains force field parameters for oxygen ('O') and hydrogen ('H').
#    - Columns:
#        - 'type': Atom type.
#        - 'sigma': Lennard-Jones parameter (Å).
#        - 'epsilon': Lennard-Jones well depth (K).
#        - 'charge': Partial charge (e).
#        - 'num_particles': Number of particles per molecule.

# 3. system DataFrame:
#    - Contains metadata about each system configuration.
#    - Columns:
#        - 'file_paths': File names containing atomic configurations.
#        - 'configuration #': Extracted configuration number (1-4).
#        - 'number of particles': Number of molecules (from 'NIST_SPC_E_Water').
#        - 'box length': Box length (from 'NIST_SPC_E_Water').
#        - 'cutoff': Fixed cutoff distance for interactions (10 Å).
#        - 'alpha': Ewald summation parameter (5.6 / min(Lx,Ly,Lz)).
#        - 'kmax': Maximum wave vector index (5).
#        - 'ε0': Permittivity of Vacuum (8.854187817E-12 C2/(J m)).
#        - 'kB': Boltzmann Constant (1.3806488E-23 J/K).

# 4. configuration DataFrame (from 'extracting_positions'):
#    - Created per file, containing atomic positions.
#    - Columns:
#        - 'X', 'Y', 'Z': Atom coordinates.
#        - 'Atom Type': Type of atom ('O' or 'H').
#        - 'Molecule': Molecule index assigned based on position.

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


# calling compare_LJ_energy function 
compare_coulomb_energy(results, NIST_SPC_E_Water)
