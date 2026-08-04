# Tutorial 2: Human Health Impact Prediction Using ANN Model

***

In this code you will be able to estimate the impact to Human Health using the ANN Model.  Then you will be able to see how changing the amount of that chemical will affect the predicted impact.  

***

### Code Overview: 

This code will download the human health impact weight of the model locally and give this file to the code, which then will take the input of a chemical using the SMILES identification to tell you the impact in kg to human health.  Then you will be able to see the impact scaled according to how much mass in kg you have of that chemical.  


<img src="/Reference_Files/Workflows/Human_Health_Impact_Workflow.svg"/>

The libraries / packages listed in cell 1 will have a brief explanation of their function in the code, but for more information, please use the links below

| Library / Package | Link to Documentation |
| :--: | :--: |
| numpy | [NumPy Documentation](https://numpy.org/doc/stable/) |
| pandas | [Pandas Documentation](https://pandas.pydata.org/docs/) |
| matplotlib.pyplot | [Matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_summary.html) |
| ipywidgets | [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/stable/) |
| IPython.display | [Display Module](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html) |
| tensorflow | [Tensorflow](https://www.tensorflow.org/api_docs) |
| os | [OS](https://docs.python.org/3/library/os.html) |
| requests | [Requests](https://requests.readthedocs.io/en/latest/) |

Click the button below to open this code in Google Colab


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cacherowan/CACHE-Rowan/blob/main/Reference_Files/Google_Colab_Files/Human_Health_Impact_Prediction.ipynb)


### Outputs Should Appear Like This: 


```
# Cell 1: Import Required Libraries / Packages

import numpy as np                    # Stores numbers in arrays and runs fast calculations on them
import pandas as pd                   # Opens the data file as a table and finds rows by their SMILES string
import matplotlib.pyplot as plt       # Plot Graphs
import ipywidgets as widgets          # Creates UI elements such as the slider below
from IPython.display import display   # Displays UI elements

import tensorflow as tf               # Loads the trained neural network (ANN) and runs it to make predictions

import os                             # Allows interactions with the file system
import requests                       # Makes requests to internet files
```
<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 2: Define Function to Download Weight (.h5 file) of ANN Model

def download_if_needed(url, local_path):
    """Download a file from GitHub if it doesn't already exist."""
    if os.path.exists(local_path):
        print(f"✓ Using existing {local_path}")
        return

    print(f"Downloading {local_path}...")
    response = requests.get(url)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    print("Download complete.")
```
<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 3: Create Path to Dowloaded Files and Database

# URL / path to download (.h5 file)
HH_MODEL_URL = ("https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Chemical_Property_Database/model_v5_HH_okayvaltest.h5")

# Name of weight file
HH_MODEL_PATH = "model_v5_HH_okayvaltest.h5"

# URL / path to database (the spreadsheet holding the molecular properties)
DATABASE_PATH = "https://raw.githubusercontent.com/cacherowan/CACHE-Rowan/main/Reference_Files/Chemical_Property_Database/Processed_Solvent_DF_v6_TEST.xlsx"

# Call to Function to Download Weight (.h5 file)
download_if_needed(HH_MODEL_URL, HH_MODEL_PATH)

# The trained model file (.h5) that predicts Human Health Impact
HH_MODEL = HH_MODEL_PATH
```

<details>
<summary>Expected output</summary>

```text
Downloading model_v5_HH_okayvaltest.h5...
Download complete.
```

</details>




```
# Cell 4: Load the trained model
# Load the .h5 file, which contains the trained network: its structure and its learned weights.  
# This rebuilds the model as "model_CC," ready to make predictions.  
# compile=false in the code below helps skip the training setup, since we are only using the model to predict (not train it).  

import warnings
warnings.filterwarnings("ignore")

model_HH = tf.keras.models.load_model(
    HH_MODEL_PATH,
    custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    compile=False
)
```

</details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 5: Load the molecule database
# Opens the spreadshet into a table, and uses each molecule's SMILES code as its row name so molecules are easy to look up

db = pd.read_excel(DATABASE_PATH)
db = db.set_index('SMILES')
```

<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 6: Select the thermodynamic properties and the molecular descriptors used as inputs for the Human Health model
# The Human Health model was trained on a specific set of 10 features.
# We must feed the model the exact same features, in the exact same order, every time we ask it to make a prediction.

# Thermodynamic properties:
thermo_feat_HH = [
    'Heat of Vaporization(J/mol)',    # energy needed to turn the liquid into vapour
    'Heat Capacity (kJ/kgC)',         # heat needed to raise the molecule's temperature
    'XLogP',                          # a measure of how "fat-loving" vs "water-loving" it is
    'Pitzer’s Acentric Factor [-]',   # a measure of how non-spherical the molecule is
    'Critical Temperature [K]'        # temperature above which it can't be liquefied
]
# Molecular descriptors:
mol_desc_feat_HH = [
    'Chi0n',                # a connectivity index describing molecular structure
    'HallKierAlpha',        # a shape-related descriptor
    'SMR_VSA7',             # surface-area descriptor related to molar refractivity
    'VSA_EState6',          # surface-area descriptor related to electronic state
    'NumValenceElectrons'   # total number of valence electrons in the molecule
]
```

<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 7: Predict Human Health Impact for one molecule
# This function takes a molecule, looks up its properties in the database, feeds them to the trained model, and returns the predicted Human Health Impact.  
# Run this cell once.  It won't show any output on its own, it just sets yp the function so the cells below can use it.  

def predict_human_health_impact(molecule_name, smiles):
    """
    Predict the Human Health Impact of one molecule.

    molecule_name : name for printing only (e.g. "Methanol") — doesn't affect the result.
    smiles        : SMILES string; must exactly match a row in the database.
    Returns the predicted Human Health Impact in DALY per kg of chemical.
    """
    descriptors = db.loc[smiles]
    descriptors_hh = descriptors.loc[thermo_feat_HH + mol_desc_feat_HH]

    print(f"\n{molecule_name}  ({smiles})")
    print("-" * 40)
    for feature_name, value in descriptors_hh.items():
        print(f"  {feature_name:<32} {value:>8.4f}")
    print("-" * 40)

    model_input = np.array([list(descriptors_hh)])
    prediction = model_HH.predict(model_input, verbose=0)
    impact_value = prediction.item()

    print(f"  Human Health Impact: {round(impact_value, 4)} DALY/kg\n")
    return impact_value
```

<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 8: Calculate the Human Health Impact of a chemical
# Give the function below two things: the chemical's name (for display) and its SMILES code, which the model uses to look it up.  Run the cell to see its features and its predicted human health impact impact.  
# To calculate the impact for another chemical, copy the line below into a new cell and change the name and SMILES.  

predict_human_health_impact("Methanol", "CO");
```

<details>
<summary>Expected output</summary>

```text
Methanol  (CO)
----------------------------------------
  Heat of Vaporization(J/mol)      37460.1385
  Heat Capacity (kJ/kgC)             2.5318
  XLogP                             -0.5000
  Pitzer’s Acentric Factor [-]       0.5560
  Critical Temperature [K]         512.5000
  Chi0n                              1.4472
  HallKierAlpha                     -0.0400
  SMR_VSA7                           0.0000
  VSA_EState6                        0.0000
  NumValenceElectrons               14.0000
----------------------------------------
  Human Health Impact: 0.9432 DALY/kg
```

</details>




```
# Cell 9: Example with Ethanol

predict_human_health_impact("Ethanol", "CCO");
```

<details>
<summary>Expected output</summary>

```text
Ethanol  (CCO)
----------------------------------------
  Heat of Vaporization(J/mol)      42319.9941
  Heat Capacity (kJ/kgC)             2.5700
  XLogP                             -0.1000
  Pitzer’s Acentric Factor [-]       0.6440
  Critical Temperature [K]         514.0000
  Chi0n                              2.1543
  HallKierAlpha                     -0.0400
  SMR_VSA7                           0.0000
  VSA_EState6                        0.0000
  NumValenceElectrons               20.0000
----------------------------------------
  Human Health Impact: 0.9486 DALY/kg
```

</details>




```
# Cell 10: Example with Benzene

predict_human_health_impact("Benzene", "c1ccccc1");
```

<details>
<summary>Expected output</summary>

```text
Benzene  (c1ccccc1)
----------------------------------------
  Heat of Vaporization(J/mol)      33879.4484
  Heat Capacity (kJ/kgC)             1.7469
  XLogP                              2.1000
  Pitzer’s Acentric Factor [-]       0.2120
  Critical Temperature [K]         562.0500
  Chi0n                              3.4641
  HallKierAlpha                     -0.7800
  SMR_VSA7                          36.3982
  VSA_EState6                       12.0000
  NumValenceElectrons               30.0000
----------------------------------------
  Human Health Impact: 0.9432 DALY/kg
```

</details>




```
# Cell 11: Example with Toluene

predict_human_health_impact("Toluene", "Cc1ccccc1");
```

<details>
<summary>Expected output</summary>

```text
Toluene  (Cc1ccccc1)
----------------------------------------
  Heat of Vaporization(J/mol)      38009.5928
  Heat Capacity (kJ/kgC)             1.7190
  XLogP                              2.7000
  Pitzer’s Acentric Factor [-]       0.2630
  Critical Temperature [K]         591.7500
  Chi0n                              4.3868
  HallKierAlpha                     -0.7800
  SMR_VSA7                          35.8953
  VSA_EState6                       10.2616
  NumValenceElectrons               36.0000
----------------------------------------
  Human Health Impact: 0.9432 DALY/kg
```

</details>




```
# Cell 12: Example using Phenol (Which will be displayed later on a slider and graph)

# 1. Example: Phenol
MOLECULE = "PHENOL"
SMILES = "Oc1ccccc1"

# 2. Read the DB:
descriptors = db.loc[SMILES]

# Selected properties for Human Health Impact:
thermo_feat_HH   = ['Heat of Vaporization(J/mol)', 'Heat Capacity (kJ/kgC)', 'XLogP','Pitzer’s Acentric Factor [-]', 'Critical Temperature [K]']
mol_desc_feat_HH = ['Chi0n', 'HallKierAlpha', 'SMR_VSA7', 'VSA_EState6','NumValenceElectrons']

# 3. Obtain the relevant properties:
descriptors_hh = descriptors.loc[thermo_feat_HH + mol_desc_feat_HH]

molecule_human_health_impact = model_HH.predict(np.array([list(descriptors_hh)]), verbose=0) # DALY/kg chemical
```

<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 13: Phenol Continued, Creating Slider to Visualize Impact

kg_slider = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description='kg amount:')
output_slider = widgets.Output()

def update_slider(change):
    with output_slider:
        output_slider.clear_output()
        kg_slider_kg = kg_slider.value
        total_hh_slider = molecule_human_health_impact[0][0] * kg_slider_kg
        print(f"Health impact:  {total_hh_slider:.6f} Disability-Adjusted Life Years (DALY)")

kg_slider.observe(update_slider, names='value')
display(kg_slider, output_slider)
update_slider(None)
```

<details>
<summary>Expected output</summary>

<img src="/Reference_Files/Human_Health_Impact_Files/Cell_8.png"/>

</details>




```
# Cell 14: Phenol Continued, Creating Slider and Graph to Visualize Impact

kg_slider_graph = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description='kg amount:')
output_graph = widgets.Output()

hh_per_kg = molecule_human_health_impact[0][0]
kg_range = np.linspace(0, 100, 200)

def update_graph(change):
    with output_graph:
        output_graph.clear_output(wait=True)
        kg_graph_kg = kg_slider_graph.value
        total_hh_graph = hh_per_kg * kg_graph_kg

        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.plot(kg_range, hh_per_kg * kg_range, color='#e0685c')
        ax.scatter([kg_graph_kg], [total_hh_graph], color='#e0685c', zorder=5, s=80)
        ax.set_title(f"Health: {total_hh_graph:.6f} Disability-Adjusted Life Years (DALY)")
        ax.set_xlabel(f"kg {MOLECULE}")

        plt.tight_layout()
        plt.show()

kg_slider_graph.observe(update_graph, names='value')
display(kg_slider_graph, output_graph)
update_graph(None)
```
<details>
<summary>Expected output</summary>

<img src="/Reference_Files/Human_Health_Impact_Files/Cell_9.png"/>

</details>

### Discussion / Analysis

In this code, you were able to see how a chemical impacts human health using the ANN Model.  Then you could see how changing the amount of that chemical affects the human health impact prediction.   

::::{grid} 2
:gutter: 3

:::{grid-item-card} Machine Learning Background
:link: index.md#Tutorials

Background information on Machine Learning and Artificial Neural Networks
:::

:::{grid-item-card} Climate Change Impact Prediction Using a Power Law Correlation
:link: ../3_Use and End of Life Phase Impact Prediction/Climate_Change_Impact_Prediction_Using_a_Power_Law_Correlation.md

Description
:::

::::