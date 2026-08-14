---
kernelspec:
  name: python3
  display_name: Python 3
---

# Model Validation Metrics

***

The Standard Enthalpy of Formation is the energy required to turn 1 mole of base elements in their natural state into 1 mole of the molecule you are computing this for.  When using MACE-OFF to get these values, you need to calculate the enthalpy of the desired molecule and subtract the enthalpy of the same amount of atoms in their base elements in their natural state.  For example to compute the Standard Enthalpy of Formation for water (H{sub}`2`O), you would compute the enthalpy of a water molecule and subtract the base elements in their natural state (O{sub}`2` and H{sub}`2`).  The equation ends up being: 

Standard Enthalpy of Formation of H{sub}`2`O = H(H{sub}`2`O) - H(H{sub}`2`) - $ \frac{1}{2} $ H(O{sub}`2`) 

***

### Code Overview: 

The Code below will go through using Atomic Simulation Environment (ASE) and Multi Atomic Cluster Expansion - Organic Force Field (MACE-OFF) to calculate the Standard Enthalpy of Formation of Molecules including, Ammonia, Propanol, Methanol, Propane, and Thiophene.  Then you will see how to use the Database of Chemical Properties to obtain the Standard Enthalpy of Formation of the same materials, and then finally compare the values from both methods against the National Institute of Standards and Technologies (NIST) Database.  


<img src="/Reference_Files/Workflows/Error_4_Workflow.svg"/>


The libraries / packages listed in cell 2 will have a brief explanation of their function in the code, but for more information, please use the links below

| Library / Package | Link to Documentations |
| :--: | :--: |
| numpy | [NumPy Documentation](https://numpy.org/doc/stable/) |
| matplotlib.pyplot | [Matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_summary.html) |
| pandas | [Pandas Documentation](https://pandas.pydata.org/docs/) |
| chem | [rdkit.chem Documentation](https://www.rdkit.org/docs/source/rdkit.html) |
| AllChem | Same link as above |
| mace_off | [MACE Calculator Documentation](https://mace-web-interface.readthedocs.io/en/latest/guide/mace-calculator-parameters/#mace_off-organic-force-field-mace-off23) |
| mace_off | [MACE Descriptors Documentation](https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html) |
| atoms | [Atoms Object Documentation](https://ase.gitlab.io/ase/ase/atoms.html#ase.Atoms) |
| molecule | [Molecules Documentation](https://docs.ase-lib.org/ase/build/build.html#ase.build.molecule) |
| QuasiNewton | [Structure Optimization Documentation](https://docs.ase-lib.org/ase/optimize.html) |
| Vibrations | [Vibrational Modes Documentation](https://ase.gitlab.io/ase/ase/vibrations/modes.html#module-ase.vibrations) |
| IdealGasThermo | [Ideal-gas limit Documentation](https://ase.gitlab.io/ase/ase/thermochemistry/thermochemistry.html#ase.thermochemistry.IdealGasThermo) |
| kJ, mol | [Units Documentation](https://ase.gitlab.io/ase/ase/units.html#module-ase.units) |


Click the link below to open the Colab notebook:


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cacherowan/CACHE-Rowan/blob/main/Reference_Files/Google_Colab_Files/model_validation_metrics.ipynb)


### Outputs Should Appear Like This:


```
# Cell 1: Install Required Packages

!pip install ASE
!pip install mace-torch ase rdkit weas-widget
```

<details>
<summary>Expected output</summary>

```text
Collecting ASE
  Downloading ase-3.29.0-py3-none-any.whl.metadata (4.4 kB)
Requirement already satisfied: numpy>=1.21.6 in /usr/local/lib/python3.12/dist-packages (from ASE) (2.0.2)
Requirement already satisfied: scipy>=1.8.1 in /usr/local/lib/python3.12/dist-packages (from ASE) (1.16.3)
Requirement already satisfied: matplotlib>=3.5.2 in /usr/local/lib/python3.12/dist-packages (from ASE) (3.10.0)
Requirement already satisfied: typing_extensions>=4.13.1 in /usr/local/lib/python3.12/dist-packages (from ASE) (4.16.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (4.63.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (26.2)
Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (11.3.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.5.2->ASE) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib>=3.5.2->ASE) (1.17.0)
Downloading ase-3.29.0-py3-none-any.whl (3.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 17.6 MB/s eta 0:00:00
Installing collected packages: ASE
Successfully installed ASE-3.29.0
Collecting mace-torch
  Downloading mace_torch-0.3.16-py3-none-any.whl.metadata (27 kB)
Requirement already satisfied: ase in /usr/local/lib/python3.12/dist-packages (3.29.0)
Collecting rdkit
  Downloading rdkit-2026.3.4-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (3.8 kB)
Collecting weas-widget
  Downloading weas_widget-0.2.6-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: torch>=1.12 in /usr/local/lib/python3.12/dist-packages (from mace-torch) (2.11.0+cpu)
Collecting e3nn==0.4.4 (from mace-torch)
  Downloading e3nn-0.4.4-py3-none-any.whl.metadata (5.1 kB)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from mace-torch) (2.0.2)
Requirement already satisfied: opt_einsum in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.4.0)
Collecting torch-ema (from mace-torch)
  Downloading torch_ema-0.3-py3-none-any.whl.metadata (415 bytes)
Requirement already satisfied: prettytable in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.18.0)
Collecting matscipy (from mace-torch)
  Downloading matscipy-1.2.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
Requirement already satisfied: h5py in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.16.0)
Collecting torchmetrics (from mace-torch)
  Downloading torchmetrics-1.9.0-py3-none-any.whl.metadata (23 kB)
Collecting python-hostlist (from mace-torch)
  Downloading python_hostlist-2.3.0.tar.gz (37 kB)
  Preparing metadata (setup.py) ... done
Collecting configargparse (from mace-torch)
  Downloading configargparse-1.7.5-py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: GitPython in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.1.51)
Requirement already satisfied: pyYAML in /usr/local/lib/python3.12/dist-packages (from mace-torch) (6.0.3)
Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from mace-torch) (4.67.3)
Collecting lmdb (from mace-torch)
  Downloading lmdb-2.3.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (1.2 kB)
Requirement already satisfied: orjson in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.11.9)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.10.0)
Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (from mace-torch) (2.2.2)
Requirement already satisfied: sympy in /usr/local/lib/python3.12/dist-packages (from e3nn==0.4.4->mace-torch) (1.14.0)
Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from e3nn==0.4.4->mace-torch) (1.16.3)
Collecting opt-einsum-fx>=0.1.4 (from e3nn==0.4.4->mace-torch)
  Downloading opt_einsum_fx-0.1.4-py3-none-any.whl.metadata (3.3 kB)
Requirement already satisfied: typing_extensions>=4.13.1 in /usr/local/lib/python3.12/dist-packages (from ase) (4.16.0)
Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from rdkit) (11.3.0)
Requirement already satisfied: anywidget>=0.9.11 in /usr/local/lib/python3.12/dist-packages (from weas-widget) (0.9.21)
Collecting appdirs>=1.4.4 (from weas-widget)
  Downloading appdirs-1.4.4-py2.py3-none-any.whl.metadata (9.0 kB)
Requirement already satisfied: click>=8.1.7 in /usr/local/lib/python3.12/dist-packages (from weas-widget) (8.4.2)
Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from weas-widget) (2.32.4)
Requirement already satisfied: scikit-image in /usr/local/lib/python3.12/dist-packages (from weas-widget) (0.25.2)
Requirement already satisfied: ipywidgets>=7.6.0 in /usr/local/lib/python3.12/dist-packages (from anywidget>=0.9.11->weas-widget) (7.7.1)
Requirement already satisfied: psygnal>=0.8.1 in /usr/local/lib/python3.12/dist-packages (from anywidget>=0.9.11->weas-widget) (0.15.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (4.63.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (26.2)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mace-torch) (2.9.0.post0)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (3.29.7)
Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (75.2.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (2025.3.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from GitPython->mace-torch) (4.0.12)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas->mace-torch) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas->mace-torch) (2026.3)
Requirement already satisfied: wcwidth>=0.3.5 in /usr/local/lib/python3.12/dist-packages (from prettytable->mace-torch) (0.8.2)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (3.4.9)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (3.18)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (2026.6.17)
Requirement already satisfied: imageio!=2.35.0,>=2.33 in /usr/local/lib/python3.12/dist-packages (from scikit-image->weas-widget) (2.37.3)
Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.12/dist-packages (from scikit-image->weas-widget) (2026.4.11)
Requirement already satisfied: lazy-loader>=0.4 in /usr/local/lib/python3.12/dist-packages (from scikit-image->weas-widget) (0.5)
Collecting lightning-utilities>=0.15.3 (from torchmetrics->mace-torch)
  Downloading lightning_utilities-0.15.3-py3-none-any.whl.metadata (5.5 kB)
Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->GitPython->mace-torch) (5.0.3)
Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.12/dist-packages (from ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (6.17.1)
Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.12/dist-packages (from ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.2.0)
Requirement already satisfied: traitlets>=4.3.1 in /usr/local/lib/python3.12/dist-packages (from ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (5.7.1)
Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.12/dist-packages (from ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.6.10)
Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.12/dist-packages (from ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (7.34.0)
Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.12/dist-packages (from ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.0.16)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib->mace-torch) (1.17.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy->e3nn==0.4.4->mace-torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.12->mace-torch) (3.0.3)
Requirement already satisfied: debugpy>=1.0 in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.8.15)
Requirement already satisfied: jupyter-client>=6.1.12 in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (7.4.9)
Requirement already satisfied: matplotlib-inline>=0.1 in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.2.2)
Requirement already satisfied: nest-asyncio in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.6.0)
Requirement already satisfied: psutil in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (5.9.5)
Requirement already satisfied: pyzmq>=17 in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (26.2.1)
Requirement already satisfied: tornado>=6.1 in /usr/local/lib/python3.12/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (6.5.7)
Collecting jedi>=0.16 (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget)
  Downloading jedi-0.20.0-py2.py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: decorator in /usr/local/lib/python3.12/dist-packages (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.4.2)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.12/dist-packages (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.7.5)
Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.0.52)
Requirement already satisfied: pygments in /usr/local/lib/python3.12/dist-packages (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.20.0)
Requirement already satisfied: backcall in /usr/local/lib/python3.12/dist-packages (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.2.0)
Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.12/dist-packages (from ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.9.0)
Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.12/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (6.5.7)
Requirement already satisfied: parso<0.9.0,>=0.8.6 in /usr/local/lib/python3.12/dist-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.8.7)
Requirement already satisfied: entrypoints in /usr/local/lib/python3.12/dist-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.4)
Requirement already satisfied: jupyter-core>=4.9.2 in /usr/local/lib/python3.12/dist-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (5.9.1)
Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (25.1.0)
Requirement already satisfied: nbformat in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (5.10.4)
Requirement already satisfied: nbconvert>=5 in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (7.17.1)
Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.1.0)
Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.18.1)
Requirement already satisfied: prometheus-client in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.25.0)
Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.12/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.3.3)
Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.12/dist-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.7.0)
Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.12/dist-packages (from jupyter-core>=4.9.2->jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.10.0)
Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.12/dist-packages (from nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.2.4)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.13.5)
Requirement already satisfied: bleach!=5.0.0 in /usr/local/lib/python3.12/dist-packages (from bleach[css]!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (6.4.0)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.7.1)
Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.3.0)
Requirement already satisfied: mistune<4,>=2.0.3 in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.3.3)
Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.10.4)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.5.1)
Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.12/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.21.2)
Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.12/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.26.0)
Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.12/dist-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (25.1.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.12/dist-packages (from bleach!=5.0.0->bleach[css]!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.5.1)
Requirement already satisfied: tinycss2>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from bleach[css]!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.5.1)
Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (26.1.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2026.6.3)
Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.12/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.20.0)
Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.1.0)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.12/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.8.4)
Requirement already satisfied: pycparser in /usr/local/lib/python3.12/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.0)
Requirement already satisfied: anyio>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.14.2)
Requirement already satisfied: jupyter-events>=0.11.0 in /usr/local/lib/python3.12/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.12.1)
Requirement already satisfied: jupyter-server-terminals>=0.4.4 in /usr/local/lib/python3.12/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.5.4)
Requirement already satisfied: websocket-client>=1.7 in /usr/local/lib/python3.12/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.9.0)
Requirement already satisfied: python-json-logger>=2.0.4 in /usr/local/lib/python3.12/dist-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.1.0)
Requirement already satisfied: rfc3339-validator in /usr/local/lib/python3.12/dist-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.1.4)
Requirement already satisfied: rfc3986-validator>=0.1.1 in /usr/local/lib/python3.12/dist-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (0.1.1)
Requirement already satisfied: fqdn in /usr/local/lib/python3.12/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.5.1)
Requirement already satisfied: isoduration in /usr/local/lib/python3.12/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (20.11.0)
Requirement already satisfied: jsonpointer>1.13 in /usr/local/lib/python3.12/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.1.1)
Requirement already satisfied: rfc3987-syntax>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.1.0)
Requirement already satisfied: uri-template in /usr/local/lib/python3.12/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.3.0)
Requirement already satisfied: webcolors>=24.6.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (25.10.0)
Requirement already satisfied: lark>=1.2.2 in /usr/local/lib/python3.12/dist-packages (from rfc3987-syntax>=1.1.0->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.3.1)
Requirement already satisfied: arrow>=0.15.0 in /usr/local/lib/python3.12/dist-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (1.4.0)
Downloading mace_torch-0.3.16-py3-none-any.whl (316 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 316.0/316.0 kB 6.5 MB/s eta 0:00:00
Downloading e3nn-0.4.4-py3-none-any.whl (387 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 387.7/387.7 kB 25.4 MB/s eta 0:00:00
Downloading rdkit-2026.3.4-cp312-cp312-manylinux_2_28_x86_64.whl (37.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.4/37.4 MB 46.2 MB/s eta 0:00:00
Downloading weas_widget-0.2.6-py3-none-any.whl (345 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 345.3/345.3 kB 25.3 MB/s eta 0:00:00
Downloading appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
Downloading configargparse-1.7.5-py3-none-any.whl (27 kB)
Downloading lmdb-2.3.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (344 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 344.7/344.7 kB 23.6 MB/s eta 0:00:00
Downloading matscipy-1.2.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (453 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 453.1/453.1 kB 32.4 MB/s eta 0:00:00
Downloading torch_ema-0.3-py3-none-any.whl (5.5 kB)
Downloading torchmetrics-1.9.0-py3-none-any.whl (983 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 983.4/983.4 kB 54.9 MB/s eta 0:00:00
Downloading lightning_utilities-0.15.3-py3-none-any.whl (31 kB)
Downloading opt_einsum_fx-0.1.4-py3-none-any.whl (13 kB)
Downloading jedi-0.20.0-py2.py3-none-any.whl (4.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 81.9 MB/s eta 0:00:00
Building wheels for collected packages: python-hostlist
  Building wheel for python-hostlist (setup.py) ... done
  Created wheel for python-hostlist: filename=python_hostlist-2.3.0-py3-none-any.whl size=39449 sha256=e3886f4c170cf7f45a1e04a9b30e0ad9159fbba3e60a3fc9c0b9766710156a20
  Stored in directory: /root/.cache/pip/wheels/02/e4/34/75fc0cd5b7889d8cc4ce6fb2f74c9fd17b3c6138cb03832481
Successfully built python-hostlist
Installing collected packages: python-hostlist, appdirs, rdkit, lmdb, lightning-utilities, jedi, configargparse, torchmetrics, torch-ema, opt-einsum-fx, matscipy, e3nn, mace-torch, weas-widget
Successfully installed appdirs-1.4.4 configargparse-1.7.5 e3nn-0.4.4 jedi-0.20.0 lightning-utilities-0.15.3 lmdb-2.3.0 mace-torch-0.3.16 matscipy-1.2.0 opt-einsum-fx-0.1.4 python-hostlist-2.3.0 rdkit-2026.3.4 torch-ema-0.3 torchmetrics-1.9.0 weas-widget-0.2.6
```

</details>




```
# Cell 2: Import Required Libraries

# General Libraries
import numpy as np                              # Stores numbers in arrays and runs fast calculations on them
import matplotlib.pyplot as plt                 # Plot Graphs
import pandas as pd                             # Opens the data file as a table and find rows by their SMILES string

# Building Geometry of Molecules / Atoms
from rdkit import Chem                          # Used to build molecules (basic package)
from rdkit.Chem import AllChem                  # Used to build molecules (advanced package)

# MACE Library
from mace.calculators import mace_off           # MACE-OFF (Machine Learning Potential)

# Atomic Simulation Environment Libraries
from ase import Atoms                           # Represents a molecule object with information
from ase.build import molecule                  # Creates an atomic structure from the database
from ase.optimize import QuasiNewton            # Optimization / energy minimization
from ase.vibrations import Vibrations           # Used to calculate vibrational modes of the Atom object
from ase.thermochemistry import IdealGasThermo  # Allows you to calculate entropy, enthalpy, and gibbs free energy
from ase.units import kJ, mol                   # Conversion for units
```
<details>
<summary>Expected output</summary>

```text
/usr/local/lib/python3.12/dist-packages/e3nn/o3/_wigner.py:10: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.
  _Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))
cuequivariance or cuequivariance_torch is not available. Cuequivariance acceleration will be disabled.
```

</details>




```
# Cell 3: Load MACE-OFF

print("Loading MACE-OFF (medium model)...")
calc_mol = mace_off(model="small", default_dtype="float64")
print("MACE-OFF loaded.")
```
<details>
<summary>Expected output</summary>

```text
Loading MACE-OFF (medium model)...
Downloading MACE model from 'https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model'
The model is distributed under the Academic Software License (ASL) license, see https://github.com/gabor1/ASL 
 To use the model you accept the terms of the license.
ASL is based on the Gnu Public License, but does not permit commercial use
Downloading: 100.0% (7.0 MB / 7.0 MB)
Cached MACE model to /root/.cache/mace/MACE-OFF23_small.model
Using MACE-OFF23 MODEL for MACECalculator with /root/.cache/mace/MACE-OFF23_small.model
Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization.
/usr/local/lib/python3.12/dist-packages/mace/calculators/mace.py:226: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.
  torch.load(f=model_path, map_location=device)
MACE-OFF loaded.
```

</details>




```
# Cell 4: Define Function to Compute Chemical Properties

# Geometry, Symmetry, and Spin are used to get accurate properties
def compute_thermo(
    atoms,                      # Attach atoms / molecule
    calc,                       # Attach calculator (MACE-OFF)
    geometry,                   # (Linear or nonlinear)
    symmetrynumber,             # Symmetry of molecule
    spin,                       # 0.5 for each unpaired electron
    temperature=298.15,
    vib_name='molecule_vib',
    fmax=0.01,                  # Force threshold
    vib_energy_threshold=0.01,
    verbose=False
):
    """
    Compute potential energy and enthalpy for a molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecule to compute thermodynamics for.
    calc : ASE calculator
        The calculator to use (e.g. EMT, GPAW, etc.).
    geometry : str
        'linear' or 'nonlinear' — molecular geometry type.
    symmetrynumber : int
        Rotational symmetry number (e.g. 3 for NH3, 2 for H2).
    spin : float
        Total spin (0 for closed-shell, 0.5 per unpaired electron).
    temperature : float
        Temperature in Kelvin. Default: 298.15.
    vib_name : str
        Prefix for vibrational frequency cache files.
    fmax : float
        Force convergence criterion for geometry optimization (eV/Å).
    vib_energy_threshold : float
        Minimum vibrational energy (eV) to include; filters near-zero/imaginary modes.
    verbose : bool
        Whether to print thermochemistry details.

    Returns
    -------
    dict with keys:
        'potential_energy'  : float, eV
        'enthalpy_eV'       : float, eV
        'enthalpy_kJ_mol'   : float, kJ/mol
    """
    # Attach Calculator and Optimize Geometry
    atoms.calc = calc
    dyn = QuasiNewton(atoms, logfile=None)
    dyn.run(fmax=fmax)

    potential_energy = atoms.get_potential_energy()

    # Compute Vibrational Frequencies
    vib = Vibrations(atoms, name=vib_name)
    vib.clean()
    vib.run()
    vib_energies = vib.get_energies()

    # Filter Out Imaginary and Near-zero Modes
    vib_energies = np.array([
        e.real for e in vib_energies
        if e.real > vib_energy_threshold
    ])

    # Compute Chemical Properties
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=potential_energy,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetrynumber,
        spin=spin,
    )

    enthalpy_eV = thermo.get_enthalpy(temperature=temperature, verbose=verbose)
    enthalpy_kJ_mol = enthalpy_eV * (1 / (kJ / mol))

    return {
        'potential_energy': potential_energy,
        'enthalpy_eV': enthalpy_eV,
        'enthalpy_kJ_mol': enthalpy_kJ_mol,
    }
```





```
# Cell 5: Compute Chemical Properties for Ammonia (NH3)

results = compute_thermo(
    atoms=molecule('NH3'),
    calc=calc_mol,
    geometry='nonlinear',
    symmetrynumber=3,
    spin=0,
    temperature=298.15,
    vib_name='nh3_vib',
)

potentialenergy_NH3 = results['potential_energy']
enthalpy_NH3_eV = results['enthalpy_eV']
enthalpy_NH3_kJ_mol = results['enthalpy_kJ_mol']

print(f"NH3 Potential Energy: {potentialenergy_NH3:.4f} eV = {potentialenergy_NH3 * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of NH3: {enthalpy_NH3_eV:.4f} eV = ")
print(f"Enthalpy of NH3 at 298 K: {enthalpy_NH3_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
NH3 Potential Energy: -1540.0721 eV = -148594.3708 kJ
Enthalpy of NH3: -1539.0324 eV = 
Enthalpy of NH3 at 298 K: -148494.0510 kJ/mol
```

</details>


### <span style="color:Green">**Ammonia**</span>

SMILES: N

Chemical Formula: NH{sub}`3`

Click the power icon and then run the cell below to view an interactive model of an Ammonia Molecule

```{code-cell} python
import micropip
await micropip.install("py3Dmol")

from pyodide.http import pyfetch
import py3Dmol


response = await pyfetch("https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Molecules_To_View/Ammonia(NH3).xyz")
xyz = await response.string()

view = py3Dmol.view(width=800, height=400)
view.addModel(xyz, "xyz")
view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
view.zoom(2)
view.show()
```


```
# Cell 6: Compute Chemical Properties for CH3CHOHCH3 (Propanol)
# Create Molecule from SMILES, optimize geometry, and compute properties

# Propanol is not in the G2 database, so we need to create the geometry from the SMILES string and optimize it before computing thermochemistry.

def smiles_to_atoms(smiles, seed=64):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol) # Adds explicit hydrogens to the molecule
    AllChem.EmbedMolecule(mol, randomSeed=64) # Randomly places atoms at correct distances from each other
    AllChem.MMFFOptimizeMolecule(mol) # Optimizes the geometry using the MMFF94 classical force field (Starting Position for MACE-OFF optimization))
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = conf.GetPositions()
    return Atoms(symbols=symbols, positions=positions)

atoms_iPrOH = smiles_to_atoms('CCCO')

results = compute_thermo(
    atoms=atoms_iPrOH,
    calc=calc_mol,
    geometry='nonlinear',
    symmetrynumber=1,
    spin=0,
    temperature=298.15,
    vib_name='1propanol_vib',
)

potentialenergy_CH3CHOHCH3 = results['potential_energy']
enthalpy_CH3CHOHCH3_eV = results['enthalpy_eV']
enthalpy_CH3CHOHCH3_kJ_mol = results['enthalpy_kJ_mol']

print(f"1-Propanol Potential Energy: {potentialenergy_CH3CHOHCH3:.4f} eV = {potentialenergy_CH3CHOHCH3 * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of CH3CHOHCH3: {enthalpy_CH3CHOHCH3_eV:.4f} eV")
print(f"Enthalpy of CH3CHOHCH3 at 298 K: {enthalpy_CH3CHOHCH3_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
1-Propanol Potential Energy: -5292.0758 eV = -510607.6927 kJ
Enthalpy of CH3CHOHCH3: -5288.9310 eV
Enthalpy of CH3CHOHCH3 at 298 K: -510304.2678 kJ/mol
```

</details>


### <span style="color:Green">**Propanol**</span>

SMILES: CCCO

Chemical Formula: C{sub}`3`H{sub}`8`O

Click the power icon and then run the cell below to view an interactive model of a Propanol Molecule

```{code-cell} python
import micropip
await micropip.install("py3Dmol")

from pyodide.http import pyfetch
import py3Dmol


response = await pyfetch("https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Molecules_To_View/Propanol(C3H8O).xyz")
xyz = await response.string()

view = py3Dmol.view(width=800, height=400)
view.addModel(xyz, "xyz")
view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
view.zoom(2)
view.show()
```


```
# Cell 7: Compute Chemical Properties for Methanol (CH3OH)

results = compute_thermo(
    atoms=molecule('CH3OH'),
    calc=calc_mol,
    geometry='nonlinear',
    symmetrynumber=1,
    spin=0,
    temperature=298.15,
    vib_name='ch3oh_vib',
)

potentialenergy_CH3OH = results['potential_energy']
enthalpy_CH3OH_eV = results['enthalpy_eV']
enthalpy_CH3OH_kJ_mol = results['enthalpy_kJ_mol']

print(f"CH3OH Potential Energy: {potentialenergy_CH3OH:.4f} eV = {potentialenergy_CH3OH * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of CH3OH: {enthalpy_CH3OH_eV:.4f} eV")
print(f"Enthalpy of CH3OH at 298 K: {enthalpy_CH3OH_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
CH3OH Potential Energy: -3151.0012 eV = -304025.3976 kJ
Enthalpy of CH3OH: -3149.4851 eV
Enthalpy of CH3OH at 298 K: -303879.1149 kJ/mol
```

</details>


### <span style="color:Green">**Methanol**</span>

SMILES: CO

Chemical Formula: CH{sub}`4`O

Click the power icon and then run the cell below to view an interactive model of a Methanol Molecule

```{code-cell} python
import micropip
await micropip.install("py3Dmol")

from pyodide.http import pyfetch
import py3Dmol


response = await pyfetch("https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Molecules_To_View/Methanol(CH4O).xyz")
xyz = await response.string()

view = py3Dmol.view(width=800, height=400)
view.addModel(xyz, "xyz")
view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
view.zoom(2)
view.show()
```


```
# Cell 8: Compute Chemical Properties for C3H8 (Propane)

results = compute_thermo(
    atoms=molecule('C3H8'),
    calc=calc_mol,
    geometry='nonlinear',
    symmetrynumber=2,
    spin=0,
    temperature=298.15,
    vib_name='c3h8_vib',
)

potentialenergy_C3H8 = results['potential_energy']
enthalpy_C3H8_eV = results['enthalpy_eV']
enthalpy_C3H8_kJ_mol = results['enthalpy_kJ_mol']

print(f"C3H8 Potential Energy: {potentialenergy_C3H8:.4f} eV = {potentialenergy_C3H8 * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of C3H8: {enthalpy_C3H8_eV:.4f} eV")
print(f"Enthalpy of C3H8 at 298 K: {enthalpy_C3H8_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
C3H8 Potential Energy: -3243.9350 eV = -312992.1486 kJ
Enthalpy of C3H8: -3240.9635 eV
Enthalpy of C3H8 at 298 K: -312705.4433 kJ/mol
```

</details>


### <span style="color:Green">**Propane**</span>

SMILES: CCC

Chemical Formula: C{sub}`3`H{sub}`8`

Click the power icon and then run the cell below to view an interactive model of a Propane Molecule

```{code-cell} python
import micropip
await micropip.install("py3Dmol")

from pyodide.http import pyfetch
import py3Dmol


response = await pyfetch("https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Molecules_To_View/Propane(C3H8).xyz")
xyz = await response.string()

view = py3Dmol.view(width=800, height=400)
view.addModel(xyz, "xyz")
view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
view.zoom(2)
view.show()
```


```
# Cell 9: Compute Chemical Properties for C4H4S (Thiophene)

results = compute_thermo(
    atoms=molecule('C4H4S'),
    calc=calc_mol,
    geometry='nonlinear',
    symmetrynumber=2,
    spin=0,
    temperature=298.15,
    vib_name='c4h4s_vib',
)

potentialenergy_C4H4S = results['potential_energy']
enthalpy_C4H4S_eV = results['enthalpy_eV']
enthalpy_C4H4S_kJ_mol = results['enthalpy_kJ_mol']

print(f"C4H4S Potential Energy: {potentialenergy_C4H4S:.4f} eV = {potentialenergy_C4H4S * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of C4H4S: {enthalpy_C4H4S_eV:.4f} eV")
print(f"Enthalpy of C4H4S at 298 K: {enthalpy_C4H4S_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
C4H4S Potential Energy: -15052.7811 eV = -1452372.5993 kJ
Enthalpy of C4H4S: -15050.8213 eV
Enthalpy of C4H4S at 298 K: -1452183.5026 kJ/mol
```

</details>


### <span style="color:Green">**Thiophene**</span>

SMILES: c1ccsc1

Chemical Formula: C{sub}`4`H{sub}`4`S

Click the power icon and then run the cell below to view an interactive model of a Thiophene Molecule

```{code-cell} python
import micropip
await micropip.install("py3Dmol")

from pyodide.http import pyfetch
import py3Dmol


response = await pyfetch("https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Molecules_To_View/Thiophene(C4H4S).xyz")
xyz = await response.string()

view = py3Dmol.view(width=800, height=400)
view.addModel(xyz, "xyz")
view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
view.zoom(2)
view.show()
```


### Compute Molecular Properties for Standard Forms


```
# Cell 10: Carbon Properties (Assumed to be Graphite, anchored via experimental sublimation enthalpy using NIST)

H_sub_C_kJ = 716.7  # Experimental enthalpy of sublimation: C(graphite) -> C(g), NIST

c_atom = Atoms('C', positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=False)
c_atom.calc = calc_mol # Attach Calculator (MACE-OFF)
E_C = c_atom.get_potential_energy()

# Convert to kJ/mol and add translational thermal contribution (5/2 RT) for consistency
R = 8.314462618e-3  # kJ/(mol K)
T = 298.15
E_C_kJ = E_C * (1 / (kJ / mol))
H_C_atom_kJ = E_C_kJ + (5/2) * R * T # Thermal correction for C atom (Typically IdealGasThermo handles this)

# Anchor to graphite using the sublimation enthalpy cycle:
# E_C(atom) = H(C, graphite) + H_sub  =>  H(C, graphite) = H(C, atom) - H_sub
H_graphite_kJ = H_C_atom_kJ - H_sub_C_kJ

print(f"C atom potential energy: {E_C:.4f} eV = {E_C * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of C atom at 298 K: {H_C_atom_kJ:.4f} kJ/mol")
print(f"Enthalpy of C (graphite) at 298 K: {H_graphite_kJ:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
C atom potential energy: -1030.5672 eV = -99434.6160 kJ
Enthalpy of C atom at 298 K: -99428.4186 kJ/mol
Enthalpy of C (graphite) at 298 K: -100145.1186 kJ/mol
```

</details>




```
# Cell 11: Compute Chemical Properties for H2

results = compute_thermo(
    atoms=molecule('H2'),
    calc=calc_mol,
    geometry='linear',
    symmetrynumber=2,
    spin=0,
    temperature=298.15,
    vib_name='h2_vib',
)

potentialenergy_H2 = results['potential_energy']
enthalpy_H2_eV = results['enthalpy_eV']
enthalpy_H2_kJ_mol = results['enthalpy_kJ_mol']

print(f"H2 Potential Energy: {potentialenergy_H2:.4f} eV = {potentialenergy_H2 * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of H2: {enthalpy_H2_eV:.4f} eV")
print(f"Enthalpy of H2 at 298 K: {enthalpy_H2_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
H2 Potential Energy: -31.8512 eV = -3073.1751 kJ
Enthalpy of H2: -31.5653 eV
Enthalpy of H2 at 298 K: -3045.5872 kJ/mol
```

</details>




```
# Cell 12: Compute Chemical Properties for N2

results = compute_thermo(
    atoms=molecule('N2'),
    calc=calc_mol,
    geometry='linear',
    symmetrynumber=2,
    spin=0,
    temperature=298.15,
    vib_name='n2_vib',
)

potentialenergy_N2 = results['potential_energy']
enthalpy_N2_eV = results['enthalpy_eV']
enthalpy_N2_kJ_mol = results['enthalpy_kJ_mol']

print(f"N2 Potential Energy: {potentialenergy_N2:.4f} eV = {potentialenergy_N2 * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of N2: {enthalpy_N2_eV:.4f} eV")
print(f"Enthalpy of N2 at 298 K: {enthalpy_N2_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
N2 Potential Energy: -2980.9789 eV = -287620.7400 kJ
Enthalpy of N2: -2980.7486 eV
Enthalpy of N2 at 298 K: -287598.5243 kJ/mol
```

</details>




```
# Cell 13: Compute Chemical Properties for O2

results = compute_thermo(
    atoms=molecule('O2'),
    calc=calc_mol,
    geometry='linear',
    symmetrynumber=2,
    spin=1,
    temperature=298.15,
    vib_name='o2_vib',
)

potentialenergy_O2 = results['potential_energy']
enthalpy_O2_eV = results['enthalpy_eV']
enthalpy_O2_kJ_mol = results['enthalpy_kJ_mol']

print(f"O2 Potential Energy: {potentialenergy_O2:.4f} eV = {potentialenergy_O2 * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of O2: {enthalpy_O2_eV:.4f} eV")
print(f"Enthalpy of O2 at 298 K: {enthalpy_O2_kJ_mol:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
O2 Potential Energy: -4092.3851 eV = -394855.1390 kJ
Enthalpy of O2: -4092.2217 eV
Enthalpy of O2 at 298 K: -394839.3741 kJ/mol
```

</details>




```
# Cell 14: Compute Chemical Properties for S (rhombic)

H_sub_S_kJ = 277.0 / 8  # Experimental enthalpy of sublimation: S(rhombic) -> S(g), NIST

s_atom = Atoms('S', positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=False)
s_atom.calc = calc_mol
E_S = s_atom.get_potential_energy()

E_S_kJ = E_S * (1 / (kJ / mol))
H_S_atom_kJ = E_S_kJ + (5/2) * R * T # Thermal Correction for S atom (Typically IdealGasThermo handles this)
H_rhombic_S_kJ = H_S_atom_kJ - H_sub_S_kJ

print(f"S atom potential energy: {E_S:.4f} eV = {E_S * (1 / (kJ / mol)):.4f} kJ")
print(f"Enthalpy of S atom at 298 K: {H_S_atom_kJ:.4f} kJ/mol")
print(f"Enthalpy of S (rhombic) at 298 K: {H_rhombic_S_kJ:.4f} kJ/mol")
```
<details>
<summary>Expected output</summary>

```text
S atom potential energy: -10834.4845 eV = -1045368.8408 kJ
Enthalpy of S atom at 298 K: -1045362.6434 kJ/mol
Enthalpy of S (rhombic) at 298 K: -1045397.2684 kJ/mol
```

</details>




```
# Cell 15: Compute Error in MACE-OFF vs Experimental Data

# Ammonia
dH_NH3 = enthalpy_NH3_kJ_mol - (0.5*enthalpy_N2_kJ_mol + 1.5 * enthalpy_H2_kJ_mol)
dH_Act_NH3 = -45.9 # Experimental, kJ/mol

print(f"Enthalpy change for NH3 formation at 298 K: {(dH_NH3):.4f} kJ/mol")
print(f"Experimental: {dH_Act_NH3} kJ/mol")

Percent_Error_NH3_MACE = abs((dH_NH3 - dH_Act_NH3) / dH_Act_NH3) * 100
print(f"Percent Error: {Percent_Error_NH3_MACE:.2f}%")
print("")



# Propanol
dH_Pro = enthalpy_CH3CHOHCH3_kJ_mol - (3*H_graphite_kJ + 4*enthalpy_H2_kJ_mol + 0.5*enthalpy_O2_kJ_mol)
dH_Act_Pro = -256.0 # Experimental, kJ/mol

print(f"Enthalpy change for ISP formation at 298 K: {(dH_Pro):.4f} kJ/mol")
print(f"Experimental: {dH_Act_Pro} kJ/mol")

Percent_Error_Pro_MACE = abs((dH_Pro - dH_Act_Pro) / dH_Act_Pro) * 100
print(f"Percent Error: {Percent_Error_Pro_MACE:.2f}%")
print("")



# Methanol
dH_CH3OH = enthalpy_CH3OH_kJ_mol - (H_graphite_kJ + 2*enthalpy_H2_kJ_mol + 0.5*enthalpy_O2_kJ_mol)

dH_Act_CH3OH = -205.0  # Experimental, kJ/mol

print(f"Enthalpy change for CH3OH formation at 298 K: {dH_CH3OH:.4f} kJ/mol")
print(f"Experimental: {dH_Act_CH3OH} kJ/mol")

Percent_Error_CH3OH_MACE = abs((dH_CH3OH - dH_Act_CH3OH) / dH_Act_CH3OH) * 100
print(f"Percent Error: {Percent_Error_CH3OH_MACE:.2f}%")
print("")



# Propane
dH_C3H8 = enthalpy_C3H8_kJ_mol - (3*H_graphite_kJ + 4*enthalpy_H2_kJ_mol)

dH_Act_C3H8 = -104.7  # Experimental, kJ/mol

print(f"Enthalpy change for C3H8 formation at 298 K: {dH_C3H8:.4f} kJ/mol")
print(f"Experimental: {dH_Act_C3H8} kJ/mol")

Percent_Error_C3H8_MACE = abs((dH_C3H8 - dH_Act_C3H8) / dH_Act_C3H8) * 100
print(f"Percent Error: {Percent_Error_C3H8_MACE:.2f}%")
print("")



# Thiophene
dH_C4H4S = enthalpy_C4H4S_kJ_mol - (4*H_graphite_kJ + 2*enthalpy_H2_kJ_mol + H_rhombic_S_kJ)

dH_Act_C4H4S = 116.4  # Experimental, kJ/mol

print(f"Enthalpy change for C4H4S formation at 298 K: {dH_C4H4S:.4f} kJ/mol")
print(f"Experimental: {dH_Act_C4H4S} kJ/mol")

Percent_Error_C4H4S_MACE = abs((dH_C4H4S - dH_Act_C4H4S) / dH_Act_C4H4S) * 100
print(f"Percent Error: {Percent_Error_C4H4S_MACE:.2f}%")
```
<details>
<summary>Expected output</summary>

```text
Enthalpy change for NH3 formation at 298 K: -126.4080 kJ/mol
Experimental: -45.9 kJ/mol
Percent Error: 175.40%

Enthalpy change for ISP formation at 298 K: -266.8762 kJ/mol
Experimental: -256.0 kJ/mol
Percent Error: 4.25%

Enthalpy change for CH3OH formation at 298 K: -223.1348 kJ/mol
Experimental: -205.0 kJ/mol
Percent Error: 8.85%

Enthalpy change for C3H8 formation at 298 K: -87.7388 kJ/mol
Experimental: -104.7 kJ/mol
Percent Error: 16.20%

Enthalpy change for C4H4S formation at 298 K: -114.5856 kJ/mol
Experimental: 116.4 kJ/mol
Percent Error: 198.44%
```

</details>


### Obtain Values from Database Method


```
# Cell 16: Define Location of Database (Located in Github Repository)

DATABASE_PATH = "https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Chemical_Property_Database/Processed_Solvent_DF_v6_TEST.xlsx"
df = pd.read_excel(DATABASE_PATH)
```





```
# Cell 17: Define Function to Obtain Values from Database Method

def obtain_Enthalpy_Of_Formation(MOLECULE, SMILES):
    """
    Find Properties of input Molecule using the ANN model and database.

    Parameters:
        MOLECULE (str): Name of the molecule (for display purposes).
        SMILES (str): SMILES string of the molecule to look up in the database.

    Returns:
        float: Predicted standard formation enthalpy in kJ/mol.
    """

    # 2. Read the DB:
    db = pd.read_excel(DATABASE_PATH)
    db = db.set_index('SMILES')
    descriptors = db.loc[SMILES]

    Enthalpy_Of_Formation = ['Standard Formation Enthalpy (Gas) [J/mol]']

    Property = descriptors.loc[Enthalpy_Of_Formation]

    return descriptors['Standard Formation Enthalpy (Gas) [J/mol]']
```
<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 18: Use Function to Obtain Values of Standard Enthalpy of Formation in (J/mol)

Ammonia_Enthalpy_Of_Formation = obtain_Enthalpy_Of_Formation("Ammonia", "N")
print("Ammonia Enthalpy of Formation", Ammonia_Enthalpy_Of_Formation, "J/mol")

Propanol_Enthalpy_Of_Formation = obtain_Enthalpy_Of_Formation("Propanol", "CCCO")
print("Propanol Enthalpy of Formation", Propanol_Enthalpy_Of_Formation, "J/mol")

Methanol_Enthalpy_Of_Formation = obtain_Enthalpy_Of_Formation("Methanol", "CO")
print("Methanol Enthalpy of Formation", Methanol_Enthalpy_Of_Formation, "J/mol")

Propane_Enthalpy_Of_Formation = obtain_Enthalpy_Of_Formation("Propane", "CCC")
print("Propane Enthalpy of Formation", Propane_Enthalpy_Of_Formation, "J/mol")

Thiophene_Enthalpy_Of_Formation = obtain_Enthalpy_Of_Formation("Thiophene", "s1cccc1")
print("Thiophene Enthalpy of Formation", Thiophene_Enthalpy_Of_Formation, "J/mol")
```
<details>
<summary>Expected output</summary>

```text
Ammonia Enthalpy of Formation -45940.0 J/mol
Propanol Enthalpy of Formation -256000.0 J/mol
Methanol Enthalpy of Formation -205000.0 J/mol
Propane Enthalpy of Formation -104700.0 J/mol
Thiophene Enthalpy of Formation 114900.0 J/mol
```

</details>




```
# Cell 19: Calculate Percent Error in ANN Model vs Experimental Data (NIST)

# Convert from J/mol to kJ/mol
Ammonia_Enthalpy_Of_Formation = Ammonia_Enthalpy_Of_Formation / 1000
Propanol_Enthalpy_Of_Formation = Propanol_Enthalpy_Of_Formation / 1000
Methanol_Enthalpy_Of_Formation = Methanol_Enthalpy_Of_Formation / 1000
Propane_Enthalpy_Of_Formation = Propane_Enthalpy_Of_Formation / 1000
Thiophene_Enthalpy_Of_Formation = Thiophene_Enthalpy_Of_Formation / 1000

Percent_Error_NH3_ANN = abs((Ammonia_Enthalpy_Of_Formation - -45.9) / -45.9) * 100

Percent_Error_CH3CHOHCH3_ANN = abs((Propanol_Enthalpy_Of_Formation - -256.0) / -256.0) * 100

Percent_Error_CH3OH_ANN = abs((Methanol_Enthalpy_Of_Formation - -205.0) / -205.0) * 100

Percent_Error_C3H8_ANN = abs((Propane_Enthalpy_Of_Formation - -104.7) / -104.7) * 100

Percent_Error_C4H4S_ANN = abs((Thiophene_Enthalpy_Of_Formation - 116.4) / 116.4) * 100
```





```
# Cell 20: Show Table of All Percent Errors

import pandas as pd

data = {
    "Chemical Name": ["Ammonia", "Propanol", "Methanol", "Propane", "Thiophene"],
    "MACE-OFF Standard Formation Enthalpy [kJ/mol]": [dH_NH3, dH_Pro, dH_CH3OH, dH_C3H8, dH_C4H4S],
    "Database Standard Formation Enthalpy [kJ/mol]": [Ammonia_Enthalpy_Of_Formation, Propanol_Enthalpy_Of_Formation, Methanol_Enthalpy_Of_Formation, Propane_Enthalpy_Of_Formation, Thiophene_Enthalpy_Of_Formation],
    "Experimental Standard Formation Enthalpy [kJ/mol]": [-45.9, -256.0, -205, -104.7, 116.4], # Values from NIST in kJ/mol
    "MACE-OFF Percent Error": [Percent_Error_NH3_MACE, Percent_Error_Pro_MACE, Percent_Error_CH3OH_MACE, Percent_Error_C3H8_MACE, Percent_Error_C4H4S_MACE],
    "Database Percent Error": [Percent_Error_NH3_ANN, Percent_Error_CH3CHOHCH3_ANN, Percent_Error_CH3OH_ANN, Percent_Error_C3H8_ANN, Percent_Error_C4H4S_ANN]
}
pd.set_option("display.width", 1000)
df = pd.DataFrame(data)

df['MACE-OFF Percent Error'] = df['MACE-OFF Percent Error'].map(lambda x: f'{x:.3g}%')
df['Database Percent Error'] = df['Database Percent Error'].map(lambda x: f'{x:.3g}%')

display(df)
```
<details>
<summary>Expected output</summary>

```text
	Chemical Name	MACE-OFF Standard Formation Enthalpy [kJ/mol]	Database Standard Formation Enthalpy [kJ/mol]	Experimental Standard Formation Enthalpy [kJ/mol]	MACE-OFF Percent Error	Database Percent Error
0	Ammonia	-126.408027	-45940.0	-45.9	175%	0.0871%
1	Propanol	-266.876195	-256000.0	-256.0	4.25%	0%
2	Methanol	-223.134847	-205000.0	-205.0	8.85%	0%
3	Propane	-87.738800	-104700.0	-104.7	16.2%	0%
4	Thiophene	-114.585552	114900.0	116.4	198%	1.29%
```

</details>

### Discussion / Analysis

In this code, you were able to use ASE and MACE-OFF to compute Chemical Properties, and a few edge cases were shown to help if you need to compute the Standard Enthalpy of Formation of a molecule with a product that has a irregular natural form.  You also saw how to use the Database to compute any property of any chemical included in the excel sheet.  After that, the error in each method was shown.  The error in the Database compared to NIST was extremely small, but the error in MACE-OFF was small in some cases, but in others, it was extremely large.  That was due to trying to analyze elements that the model wasn't trained on specifically.  The Database is extensive and has many properties for known chemicals currently, but if you need a property for a chemical that isn't included, you will need to use another method.  MACE-OFF can be accurate on data it was trained on, but could significantly off when simulating atoms / molecules it has insufficient data on.  

::::{grid} 2
:gutter: 3

:::{grid-item-card} Molecular Dynamics Simulation Background
:link: 0_index_P1.md#tutorials

Background information of Molecular Dynamics Simulations
:::

:::{grid-item-card} Machine Learning Background
:link: ../2_Production Phase Impact Prediction/0_index_P2.md

Background information on Machine Learning and Artificial Neural Networks
:::

::::