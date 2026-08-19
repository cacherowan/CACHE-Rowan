# Citations

## [1] MACE-OFF
MACE-OFF was used throughout the thermodynamic tutorials as the machine learning potential for estimating molecular energies and forces.

Kovács, D. P., Moore, J. H., Browning, N. J., Batatia, I., Horton, J. T., Pu, Y., Kapil, V., Witt, W. C., Magdău, I.-B., Cole, D. J., & Csányi, G.  "MACE-OFF: Transferable Short Range Machine Learning Force Fields for Organic Molecules." 
[Paper](https://arxiv.org/abs/2312.15211) |
[GitHub](https://github.com/ACEsuit/mace-off)

***

<a id="ref-2"></a>

## [2] LAMMPS

LAMMPS is a molecular simulation software package referenced in the Molecular Dynamics background material.

Thompson, A. P., Aktulga, H. M., Berger, R., Bolintineanu, D. S., Brown, W. M., Crozier, P. S., in 't Veld, P. J., Kohlmeyer, A., Moore, S. G., Nguyen, T. D., 
Shan, R., Stevens, M. J., Tranchida, J., Trott, C., & Plimpton, S. J. (2022).   "LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, 
meso, and continuum scales."  *Computer Physics Communications*, 271, 108171. DOI: 10.1016/j.cpc.2021.108171

[Website](https://www.lammps.org/) |
[Documentation](https://docs.lammps.org/)
***

<a id="ref-3"></a>

## [3] GROMACS

GROMACS is a molecular dynamics software package referenced in the Molecular Dynamics background material.

Abraham, M. J., Murtola, T., Schulz, R., Páll, S., Smith, J. C., Hess, B., & Lindahl, E. (2015).  "GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers."  *SoftwareX*, 1-2, 19-25.  DOI: 10.1016/j.softx.2015.06.001

[Website](https://www.gromacs.org/) |
[Documentation](https://manual.gromacs.org/)

***

<a id="ref-4"></a>

## [4] Anvil High-Performance Computing System

Anvil is a U.S. academic high-performance computing resource referenced in the Molecular Dynamics background material.

Song, X. C., Smith, P., Kalyanam, R., Zhu, X., Adams, E., Colby, K., Finnegan, P., Gough, E., Hillery, E., Irvine, R., Maji, A., & St. John, J. (2022).  "Anvil - System Architecture and Experiences from Deployment and Early User Operations." *Practice and Experience in Advanced Research Computing (PEARC '22)*, Article 23, 1-9. DOI: 10.1145/3491418.3530766

[Anvil Website](https://www.rcac.purdue.edu/anvil)

***

<a id="ref-5"></a>

## [5] Chemical Property Database and Environmental Impact Forecasting Framework

The chemical-property database used in the tutorials was developed as part of research in  the Sustainable Design and Systems Medicine Lab and supports early-stage environmental impact assessment of chemicals and processes.

Appiah, H. D., Conway, M., Patel, J., McMahon, M., Hesketh, R., & Yenkie, K. M. (2026). "Early-stage environmental impact forecasting of chemicals and processes with machine 
learning and data analytics tools." *Clean Technologies and Environmental Policy*, 28(5), 120. DOI: 10.1007/s10098-026-03479-8

[Paper](https://link.springer.com/article/10.1007/s10098-026-03479-8)

***

<a id="ref-6"></a>

## [6] Atomic Simulation Environment (ASE)

Atomic Simulation Environment (ASE) was used extensively throughout the thermodynamic tutorials for constructing atomic systems, geometry optimization, vibrational calculations, and thermochemical calculations.

Larsen, A. H., Mortensen, J. J., Blomqvist, J., Castelli, I. E., Christensen, R., Dułak, M., Friis, J., Groves, M. N., Hammer, B., Hargus, C., Hermes, E. D., Jennings, P. C., 
Jensen, P. B., Kermode, J., Kitchin, J. R., Kolsbjerg, E. L., Kubal, J., Kaasbjerg, K.,  Lysgaard, S., Maronsson, J. B., Maxson, T., Olsen, T., Pastewka, L., Peterson, A., 
Rostgaard, C., Schiøtz, J., Schütt, O., Strange, M., Thygesen, K. S., Vegge, T.,  Vilhelmsen, L., Walter, M., Zeng, Z., & Jacobsen, K. W. (2017).  
"The atomic simulation environment—a Python library for working with atoms."  *Journal of Physics: Condensed Matter*, 29, 273002.  DOI: 10.1088/1361-648X/aa680e

[ASE Website](https://ase-lib.org/) |
[Repository](https://gitlab.com/ase/ase/)

***

<a id="ref-7"></a>

## [7] NumPy

NumPy was used for numerical calculations and array operations throughout the tutorials.
Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).  "Array programming with NumPy." *Nature*, 585, 357-362. DOI: 10.1038/s41586-020-2649-2

[Documentation](https://numpy.org/doc/stable/)

***

<a id="ref-8"></a>

## [8] pandas

pandas was used for reading, organizing, manipulating, and displaying tabular chemical-property data.

McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 56-61.
[Documentation](https://pandas.pydata.org/docs/)

***

<a id="ref-9"></a>

## [9] RDKit

RDKit was used to interpret SMILES strings, construct molecular structures, calculate molecular descriptors, and generate initial three-dimensional molecular geometries.

Landrum, G., et al. (2026). *RDKit: rdkit/rdkit: 2026_03_5 (Q1 2026) Release* (Version Release_2026_03_5) [Software]. Zenodo. DOI: 10.5281/zenodo.21741729

[GitHub](https://github.com/rdkit/rdkit/tree/Release_2026_03_5) |
[Documentation](https://www.rdkit.org/docs/)

***

<a id="ref-10"></a>

## [10] MACE Calculator Documentation

The MACE calculator interface was used to assign MACE-OFF machine learning potentials to molecular systems.

[MACE Calculator Documentation](https://mace-web-interface.readthedocs.io/en/latest/guide/mace-calculator-parameters/#mace_off-organic-force-field-mace-off23)

***

<a id="ref-11"></a>

## [11] MACE Descriptors Documentation

MACE descriptor functionality is referenced for obtaining learned atomic and molecular representations from MACE models.

[MACE Descriptors Documentation](https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html)

***

<a id="ref-12"></a>

## [12] ASE Atoms Object

The ASE `Atoms` object is used to represent molecular systems, including their chemical elements and Cartesian coordinates.

[ASE Atoms Documentation](https://ase.gitlab.io/ase/ase/atoms.html#ase.Atoms)

***

<a id="ref-13"></a>

## [13] ASE Molecule Builder

The ASE molecule-building functionality is used to construct predefined molecular structures in several tutorial examples.

[ASE Molecules Documentation](https://docs.ase-lib.org/ase/build/build.html#ase.build.molecule)

***

<a id="ref-14"></a>

## [14] ASE Structure Optimization

ASE optimization algorithms, including QuasiNewton and LBFGS, are used to minimize molecular geometries before vibrational and thermochemical calculations.

[ASE Structure Optimization Documentation](https://docs.ase-lib.org/ase/optimize.html)

***

<a id="ref-15"></a>

## [15] ASE Vibrational Analysis

ASE's `Vibrations` functionality is used to calculate molecular vibrational modes from finite atomic displacements and force evaluations.

[ASE Vibrations Documentation](https://ase.gitlab.io/ase/ase/vibrations/modes.html#module-ase.vibrations)

***

<a id="ref-16"></a>

## [16] ASE Ideal-Gas Thermochemistry

ASE's `IdealGasThermo` implementation is used to calculate ideal-gas thermochemical quantities from molecular energies, vibrational modes, geometry, symmetry, spin,  temperature, and pressure.

[IdealGasThermo Documentation](https://ase.gitlab.io/ase/ase/thermochemistry/thermochemistry.html#ase.thermochemistry.IdealGasThermo)

***

<a id="ref-17"></a>

## [17] ASE Units

ASE unit definitions and conversion factors are used throughout the tutorials to convert between atomistic and molar energy units.

[ASE Units Documentation](https://ase.gitlab.io/ase/ase/units.html#module-ase.units)

***

<a id="ref-18"></a>

## [18] Matplotlib

Matplotlib was used to visualize calculated thermodynamic properties and comparisons with reference data.

Hunter, J. D. (2007).  
"Matplotlib: A 2D Graphics Environment." *Computing in Science & Engineering*, 9(3), 90-95.  DOI: 10.1109/MCSE.2007.55

[Documentation](https://matplotlib.org/stable/)

***

<a id="ref-19"></a>

## [19] PyTorch

PyTorch is the underlying deep learning framework used by MACE and MACE-OFF.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., 
Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library."  *Advances in Neural Information Processing Systems*, 32.

[Paper](https://arxiv.org/abs/1912.01703) |
[PyTorch](https://pytorch.org/)

***

<a id="ref-20"></a>

## [20] NIST Chemistry WebBook

Reference thermochemical data used for comparison and validation in the tutorials were obtained from the National Institute of Standards and Technology (NIST) Chemistry WebBook.

Linstrom, P. J., & Mallard, W. G. (Eds.). *NIST Chemistry WebBook, NIST Standard Reference Database Number 69*. National Institute of Standards and Technology, Gaithersburg, MD. DOI: 10.18434/T4D303

[NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/)