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
Requirement already satisfied: typing_extensions>=4.13.1 in /usr/local/lib/python3.12/dist-packages (from ASE) (4.15.0)
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
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 32.3 MB/s eta 0:00:00
Installing collected packages: ASE
Successfully installed ASE-3.29.0
Collecting mace-torch
  Downloading mace_torch-0.3.16-py3-none-any.whl.metadata (27 kB)
Requirement already satisfied: ase in /usr/local/lib/python3.12/dist-packages (3.29.0)
Collecting rdkit
  Downloading rdkit-2026.3.3-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (3.8 kB)
Collecting weas-widget
  Downloading weas_widget-0.2.6-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: torch>=1.12 in /usr/local/lib/python3.12/dist-packages (from mace-torch) (2.11.0+cpu)
Collecting e3nn==0.4.4 (from mace-torch)
  Downloading e3nn-0.4.4-py3-none-any.whl.metadata (5.1 kB)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from mace-torch) (2.0.2)
Requirement already satisfied: opt_einsum in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.4.0)
Collecting torch-ema (from mace-torch)
  Downloading torch_ema-0.3-py3-none-any.whl.metadata (415 bytes)
Requirement already satisfied: prettytable in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.17.0)
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
Requirement already satisfied: GitPython in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.1.50)
Requirement already satisfied: pyYAML in /usr/local/lib/python3.12/dist-packages (from mace-torch) (6.0.3)
Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from mace-torch) (4.67.3)
Collecting lmdb (from mace-torch)
  Downloading lmdb-2.2.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (1.2 kB)
Requirement already satisfied: orjson in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.11.9)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (from mace-torch) (3.10.0)
Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (from mace-torch) (2.2.2)
Requirement already satisfied: sympy in /usr/local/lib/python3.12/dist-packages (from e3nn==0.4.4->mace-torch) (1.14.0)
Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from e3nn==0.4.4->mace-torch) (1.16.3)
Collecting opt-einsum-fx>=0.1.4 (from e3nn==0.4.4->mace-torch)
  Downloading opt_einsum_fx-0.1.4-py3-none-any.whl.metadata (3.3 kB)
Requirement already satisfied: typing_extensions>=4.13.1 in /usr/local/lib/python3.12/dist-packages (from ase) (4.15.0)
Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from rdkit) (11.3.0)
Requirement already satisfied: anywidget>=0.9.11 in /usr/local/lib/python3.12/dist-packages (from weas-widget) (0.9.21)
Collecting appdirs>=1.4.4 (from weas-widget)
  Downloading appdirs-1.4.4-py2.py3-none-any.whl.metadata (9.0 kB)
Requirement already satisfied: click>=8.1.7 in /usr/local/lib/python3.12/dist-packages (from weas-widget) (8.4.1)
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
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (3.29.3)
Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (75.2.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.12->mace-torch) (2025.3.0)
Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from GitPython->mace-torch) (4.0.12)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas->mace-torch) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas->mace-torch) (2026.2)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.12/dist-packages (from prettytable->mace-torch) (0.8.1)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (3.4.7)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (3.18)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->weas-widget) (2026.5.20)
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
Requirement already satisfied: mistune<4,>=2.0.3 in /usr/local/lib/python3.12/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.2.1)
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
Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2026.5.1)
Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.12/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.18.2)
Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.0.0)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.12/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (2.8.4)
Requirement already satisfied: pycparser in /usr/local/lib/python3.12/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (3.0)
Requirement already satisfied: anyio>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.0->anywidget>=0.9.11->weas-widget) (4.13.0)
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
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 316.0/316.0 kB 11.1 MB/s eta 0:00:00
Downloading e3nn-0.4.4-py3-none-any.whl (387 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 387.7/387.7 kB 23.1 MB/s eta 0:00:00
Downloading rdkit-2026.3.3-cp312-cp312-manylinux_2_28_x86_64.whl (37.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.2/37.2 MB 50.7 MB/s eta 0:00:00
Downloading weas_widget-0.2.6-py3-none-any.whl (345 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 345.3/345.3 kB 20.7 MB/s eta 0:00:00
Downloading appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
Downloading configargparse-1.7.5-py3-none-any.whl (27 kB)
Downloading lmdb-2.2.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (338 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 338.3/338.3 kB 20.0 MB/s eta 0:00:00
Downloading matscipy-1.2.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (453 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 453.1/453.1 kB 26.3 MB/s eta 0:00:00
Downloading torch_ema-0.3-py3-none-any.whl (5.5 kB)
Downloading torchmetrics-1.9.0-py3-none-any.whl (983 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 983.4/983.4 kB 46.7 MB/s eta 0:00:00
Downloading lightning_utilities-0.15.3-py3-none-any.whl (31 kB)
Downloading opt_einsum_fx-0.1.4-py3-none-any.whl (13 kB)
Downloading jedi-0.20.0-py2.py3-none-any.whl (4.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 100.2 MB/s eta 0:00:00
Building wheels for collected packages: python-hostlist
  Building wheel for python-hostlist (setup.py) ... done
  Created wheel for python-hostlist: filename=python_hostlist-2.3.0-py3-none-any.whl size=39449 sha256=a8e1aa6e2f17f28c66f2bc7e157da7f5046cd29a5b7f375a8b400747e54fcbe5
  Stored in directory: /root/.cache/pip/wheels/02/e4/34/75fc0cd5b7889d8cc4ce6fb2f74c9fd17b3c6138cb03832481
Successfully built python-hostlist
Installing collected packages: python-hostlist, appdirs, rdkit, lmdb, lightning-utilities, jedi, configargparse, torchmetrics, torch-ema, opt-einsum-fx, matscipy, e3nn, mace-torch, weas-widget
Successfully installed appdirs-1.4.4 configargparse-1.7.5 e3nn-0.4.4 jedi-0.20.0 lightning-utilities-0.15.3 lmdb-2.2.1 mace-torch-0.3.16 matscipy-1.2.0 opt-einsum-fx-0.1.4 python-hostlist-2.3.0 rdkit-2026.3.3 torch-ema-0.3 torchmetrics-1.9.0 weas-widget-0.2.6
```

</details>




```
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

# Same as last code except atoms_CCCO is defined instead of being called from g2 list
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


# Records Chemical Property Data at 6 Temperatures at 1 atmosphere and Displays it
temps = [298.15, 400, 500, 600, 700, 800]
P = 101325.

records = []
for T in temps:
    H = thermo.get_enthalpy(T, verbose=False)
    S = thermo.get_entropy(T, P, verbose=False)
    G = thermo.get_gibbs_energy(T, P, verbose=False)
    records.append({"T (K)": T, "H (eV)": H, "S (eV/K)": S, "G (eV)": G})

df1 = pd.DataFrame(records)
display(df1)
```
<details>
<summary>Expected output</summary>

```text
	T (K)	H (eV)	S (eV/K)	G (eV)
0	298.15	-5288.936388	0.003120	-5289.866516
1	400.00	-5288.834825	0.003411	-5290.199227
2	500.00	-5288.712134	0.003684	-5290.554071
3	600.00	-5288.568944	0.003944	-5290.935592
4	700.00	-5288.408152	0.004192	-5291.342519
5	800.00	-5288.232302	0.004427	-5291.773550
```

</details>


### Discussion / Analysis

In this code, you were able to calculate the Enthalpy, Entropy, and Gibbs Free Energy of Propanol by explicitly defining the geometry and constructing the molecule.   

::::{grid} 2
:gutter: 3

:::{grid-item-card} Molecular Dynamics Simulation Background
:link: index_P1.md#tutorials

Background information of Molecular Dynamics Simulations
:::

:::{grid-item-card} Tutorial 2: Heat Capacity for Gases
:link: 2_Tutorial_2.md

Perform multiple molecular simulations using MACE-OFF, which will be used to calculate the heat capacity of multiple molecules.  Also some of the molecules will be constructed using RDKit.  
:::

::::