allow_n_bonds = {
    1: [1],
    5: [2, 3, 4],
    6: [2, 3, 4],
    7: [1,2,3,4],
    8: [1,2],
    9: [1],
    14: [2,3,4,6],
    15: [2,3,4,5,6],
    16: [1,2,3,4,6],
    17: [1],
    33: [2,3,4,5], 
    34: [1,2,3,4,6], 
    35: [1],
    51: [3],
    52: [1,2],
    53: [1],
    80: [2],
    82: [2],
    83: [2],
}

valid_valencies = {
    "H": [1],
    "B": [3],
    "C": [4],
    "N": [3],
    "O": [2],
    "F": [1],
    "Si": [4],
    "P": [3, 5, 7],
    "S": [2, 4, 6],
    "Cl": [1],
    "Ge": [4],
    "As": [3, 5],
    "Se": [2, 4, 6],
    "Br": [1],
    "Sn": [4],
    "Sb": [3, 5],
    "Te": [2, 4, 6],
    "I": [1],
    "Hg": [1, 2],
    "Bi": [3, 5],
}


vertices_labels = {
    2: ["L-2", "vT-2", "vOC-2"],
    3: ["TP-3", "vT-3","fvOC-3", "mvOC-3"],
    4: ["SP-4", "T-4", "SS-4", "vTBPY-4"],
    5: ["PP-5", "vOC-5", "TBPY-5", "SPY-5", "JTBPY-5"],
    6: ["HP-6", "PPY-6", "OC-6", "TPR-6"],
}

allowed_shape = {
    1: [],  # Hydrogen typically does not take these shapes
    5: ["T-4",
        "TP-3"],  # Boron
    6: ["T-4", # 4 bonds
        "TP-3", "vT-3",  # 3 bonds
        "L-2", "vT-2" # 2 bonds for C#C and carbene
        ],  # Carbon
    7: ["T-4", # 4 bonds for charged N
        "TP-3", "vT-3", "mvOC-3", # 3 bonds
        "L-2", "vT-2", "vOC-2"],  # Nitrogen
    8: ["vT-3", # 3 bonds for charged O
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Oxygen
    9: [],
    14: ["OC-6", #6 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3",
        "L-2"],  # Silicon
    15: [
        "TBPY-5", "SPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "vT-2" # 2 bonds
            ],  # Phosphorus
    16: [
        "OC-6", # 6 bonds
        "TBPY-5", "SPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Sulfur
    17: [],  # Chlorine typically does not take these shapes

    32: [
        "OC-6", # 6 bonds
        "TBPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "L-2"], # Germanium
    33: [
        "OC-6", # 6 bonds
        "TBPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Arsenic
    34: [
        "OC-6", # 6 bonds
        "TBPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", # 3 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Selenium
    35: [],  # Bromine typically does not take these shapes
    50: [
        "OC-6", # 6 bonds
        "TBPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Tin
    51: [
        "OC-6", # 6 bonds
        "TBPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Antimony
    52: [
        "OC-6", # 6 bonds
        "TBPY-5", # 5 bonds
        "T-4", # 4 bonds
        "TP-3", "vT-3", # 3 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Tellurium
    53: [],  # Iodine typically does not take these shapes
    80: [
        "T-4", # 4 bonds
        "L-2", "vT-2", "vOC-2" # 2 bonds
        ],  # Mercury
}


degree_angles_ref = {
    # Hydrogen
    1: {
        1: [180],
    },
    # Boron
    5: {
        4: [109.5],
        3: [120],
        2: [180],
    },
    # Carbon
    6: {
        4: [109.5, 107, 102],  # ** Added 102 for small ring strain (e.g., cyclopropane)
        3: [120, 118, 109],    # ** Added 109 for distorted environments in strained systems
        2: [180], # does not count carbene for now
    },
    # Nitrogen
    # NOTE too many for 2: potentialky false positives
    7: {
        4: {109.5},
        3: [107, 110, 120, 105],  # ** Added 105 for small ring systems
        2: [90, 110, 120, 104, 180, 70, 140],  # ** Added 104 for lone pair and ring strain effects, Add 180 for N=N=N, Added 90,140 for electronic effects
        1: [180],
    },
    # Oxygen
    # NOTE too many for 2: potentialky false positives
    8: {
        1: [180],
        2: [90, 110, 120, 104, 180, 140, 70],  # ** Added 100 for distortions in medium-sized rings
    },
    # Fluorine
    9: {
        1: [180],
    },
    # Silicon
    14: {
        6: [90, 120],           # ** Octahedral and trigonal bipyramidal distortions
        4: [109.5, 108, 100],  # ** Added 100 for strain in small silacycles
        3: [120, 110],          # ** Added 110 for distorted trigonal planar systems
        2: [180],
    },
    # Phosphorus
    15: {
        6: [90, 120],           # ** Octahedral and trigonal bipyramidal distortions
        4: [109.5, 100],        # ** Added 100 for small ring effects
        3: [102, 105, 95],      # ** Added 95 for significant lone pair effects
        2: [120, 80],          # ** Slight distortion
    },
    # Sulfur
    16: {
        1: [180],
        2: [70, 95, 100, 109.5, 120, 85, 140],  # ** Added 85 for strained ring systems
        3: [108, 115, 120, 105],           # ** Added 105 for ring systems
        4: [109.5, 107, 102],              # ** Added 102 for strained tetrahedral geometries
        6: [90, 85, 80],                   # ** Added 80 for extreme strain
    },
    # Chlorine
    17: {
        1: [180],
    },
    # Arsenic
    33: {
        4: [109.5, 100],
        3: [120, 90],
        2: [180],
    },
    # Selenium
    34: {
        1: [180],
        2: [95, 100, 85],               # ** Added 85 for ring strain
        3: [115, 118, 105],             # ** Added 105 for distorted planar geometries
        4: [109.5, 108, 100],          # ** Added 100 for strain
        6: [90, 85, 80],               # ** Added 80 for severe distortion
    },
    # Bromine
    35: {
        1: [180],
    },
    # Antimony
    51: {
        3: [115, 120, 105],            # ** Added 105 for lone pair effects
    },
    # Tellurium
    52: {
        2: [105, 100, 90],             # ** Added 90 for small ring distortions
    },
    # Iodine
    53: {
        1: [180],
    },
    # Mercury
    80: {
        2: [180],               # ** Linear geometry
    },
    # Lead
    82: {
        2: [180],               # ** Linear geometry
    },
    # Bismuth
    83: {
        2: [180],               # ** Linear geometry
    },
}
