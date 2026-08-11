---
kernelspec:
  name: python3
  display_name: Python 3
---

# Tutorial 1: Chemical Property Estimation (Enthalpy, Entropy, and Gibbs Free Energy for Gases)

***

In this tutorial you will learn how to use ASE and MACE-OFF to get Chemical Properties of Atoms and Molecules using a Machine Learning Potential (MACE-OFF)

***

### Code Overview: 


This code will compute Chemical Properties of Propanol.  First, the molecule is built using the SMILES identificiation.  The code then runs a simulations for the molecule which allows the chemical properties to be calculated.  Then the code will display the results.  


<img src="/Reference_Files/Workflows/Tutorial_1_Workflow.svg"/>


The libraries / packages listed in cell 2 will have a brief explanation of their function in the code, but for more information, please use the links below

| Library / Package | Link to Documentations |
| :--: | :--: |
| numpy | [NumPy Documentation](https://numpy.org/doc/stable/) |
| pandas | [Pandas Documentation](https://pandas.pydata.org/docs/) |
| chem | [rdkit.chem Documentation](https://www.rdkit.org/docs/source/rdkit.html) |
| AllChem | Same Link as above |
| mace_off | [MACE Calculator Documentation](https://mace-web-interface.readthedocs.io/en/latest/guide/mace-calculator-parameters/#mace_off-organic-force-field-mace-off23) |
| mace_off | [MACE Descriptors Documentation](https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html) |
| atoms | [Atoms Object Documentation](https://ase.gitlab.io/ase/ase/atoms.html#ase.Atoms) |
| molecule | [Molecules Documentation](https://docs.ase-lib.org/ase/build/build.html#ase.build.molecule) |
| QuasiNewton | [Structure Optimization Documentation](https://docs.ase-lib.org/ase/optimize.html) |
| Vibrations | [Vibrational Modes Documentation](https://ase.gitlab.io/ase/ase/vibrations/modes.html#module-ase.vibrations) |
| IdealGasThermo | [Ideal-gas limit Documentation](https://ase.gitlab.io/ase/ase/thermochemistry/thermochemistry.html#ase.thermochemistry.IdealGasThermo) |
| units | [Units Documentation](https://ase.gitlab.io/ase/ase/units.html#module-ase.units) |

Click the button below to open this code in Google Colab


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cacherowan/CACHE-Rowan/blob/main/Reference_Files/Google_Colab_Files/Tutorial_1.ipynb)


### Outputs Should Appear Like This: 




```{code-cell} python
# Cell 1: Install Required Packages

!pip install ASE
!pip install mace-torch ase rdkit weas-widget
```


```{code-cell} python
# Cell 2: Import Required Libraries

import numpy as np                              # Computational library
import pandas as pd                             # Excel of python

from rdkit import Chem                          # Used to build molecules (basic package)
from rdkit.Chem import AllChem                  # Used to build molecules (advanced package)

from mace.calculators import mace_off           # MACE-OFF (Machine Learning Potential)

# Atomic Simulation Environment Libraries
from ase import Atoms                           # Represents a molecule object with information
from ase.build import molecule                  # Creates an atomic structure from the database
from ase.optimize import QuasiNewton            # Optimization / energy minimization
from ase.vibrations import Vibrations           # Used to calculate vibrational modes of the Atom object
from ase.thermochemistry import IdealGasThermo  # Allows you to calculate entropy, enthalpy, and gibbs free energy
from ase.units import kJ, mol                   # Conversion for units
```


```{code-cell} python
# Cell 3: Load MACE-OFF

print("Loading MACE-OFF (medium model)...")
calc_mol = mace_off(model="small", default_dtype="float64")
print("MACE-OFF loaded.")
```


```{code-cell} python
# Cell 4: Calculate Propanol (CCCO) Chemical Properties
# Build the Molecule using SMILES

smiles = 'CCCO' # SMILES for Propanol (CCCO)
seed = 42

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol) # Adds explicit hydrogens to the molecule
AllChem.EmbedMolecule(mol, randomSeed=seed) # Randomly places atoms at correct distances from each other
AllChem.MMFFOptimizeMolecule(mol) # Optimizes the geometry using the MMFF94 classical force field (Starting Position for MACE-OFF optimization)
conf = mol.GetConformer()
symbols = [a.GetSymbol() for a in mol.GetAtoms()]
positions = conf.GetPositions()

atoms_CCCO = Atoms(symbols=symbols, positions=positions)

atoms_CCCO.calc = calc_mol # Calls to MACE-OFF to be used
dyn = QuasiNewton(atoms_CCCO, logfile=None)
dyn.run(fmax=0.01) # Finding energy minimum
potentialenergy_CCCO = atoms_CCCO.get_potential_energy() # Get potential energy

vib = Vibrations(atoms_CCCO, name='ccco_vib')
vib.clean()
vib.run()
vib_energies = vib.get_energies() # Gets vibrational energy based on molecule geometry
vib_energies = np.array([e.real for e in vib_energies if e.real > 0.01]) # Filters out imaginary numbers and very low frequencies

# Takes inputs of vibrational energies, potential energy, and geometry to compute chemical properties
thermo = IdealGasThermo(
    vib_energies=vib_energies,
    potentialenergy=potentialenergy_CCCO,
    atoms=atoms_CCCO,
    geometry='nonlinear', # Linear (Straight Line) or Nonlinear (Bent in any way)
    symmetrynumber=1, # How many times you can rotate the molecule and get the same configuration
    spin=0, # 0.5 for each unpaired electrons
)

Conv_eV_kJ = 96.485 # eV -> kJ

# Records Chemical Property Data at 6 Temperatures at 1 atmosphere and Displays it
temps = [298.15, 400, 500, 600, 700, 800]
P = 101325.

records = []
for T in temps:
    H = thermo.get_enthalpy(T, verbose=False) * Conv_eV_kJ
    S = thermo.get_entropy(T, P, verbose=False) * Conv_eV_kJ
    G = thermo.get_gibbs_energy(T, P, verbose=False) * Conv_eV_kJ
    records.append({"T (K)": T, "H (kJ)": H, "S (kJ/K)": S, "G (kJ)": G})

df1 = pd.DataFrame(records)
display(df1)
```


<a href="../5_Molecule_Viewer/5_Phase_1_Molecule_Viewer.ipynb#tutorial-1" 
   target="_blank"
   style="display:inline-block; padding:10px 20px; background-color:#f9ab00; color:white; 
          text-decoration:none; border-radius:6px; font-family:sans-serif; font-weight:bold;">
  See Propanol Molecule
</a>

### Discussion / Analysis

In this code, you were able to calculate the Enthalpy, Entropy, and Gibbs Free Energy of Propanol by explicitly defining the geometry and constructing the molecule.   

::::{grid} 2
:gutter: 3

:::{grid-item-card} Molecular Dynamics Simulation Background
:link: 0_index_P1.md#tutorials

Background information of Molecular Dynamics Simulations
:::

:::{grid-item-card} Tutorial 2: Heat Capacity for Gases
:link: 2_Tutorial_2.md

Perform multiple molecular simulations using MACE-OFF, which will be used to calculate the heat capacity of multiple molecules.  Also some of the molecules will be constructed using RDKit.  
:::

::::