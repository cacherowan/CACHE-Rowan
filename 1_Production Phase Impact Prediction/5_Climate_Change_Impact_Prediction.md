# Climate Change Impact Prediction Using ANN Model

***

In this code you will be able to estimate the impact to Climate Change using the ANN Model.  Then you will be able to see how changing the amount of that chemical will affect the predicted impact.  

***

### Code Overview: 

This code will download the climate change impact weight of the model locally and give this file to the code, which then will take the input of a chemical using the SMILES identification to tell you the impact in kg to climate change.  Then you will be able to see the impact scaled according to how much mass in kg you have of that chemical.  



Click the button below to open this code in Google Colab


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Earlyrizer64/MyST_site/blob/main/Reference_Files/Google_Colab_Files/Climate_Change_Impact_Prediction.ipynb)


### Outputs Should Appear Like This: 


```
# Cell 1: Import Required Libraries

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.impute import KNNImputer
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold

# Downloading Weights
import os
import requests

# For Giving Weights
from tensorflow.keras.models import load_model

# For Widgets and Graphs at end
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
```
<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 2: Define Function to Download Weights of ANN

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

CC_MODEL_URL = ("https://raw.githubusercontent.com/Earlyrizer64/MyST_site/main/Reference_Files/Chemical_Property_Database/model_v5_CC_okayvaltest.h5")

CC_MODEL_PATH = "model_v5_CC_okayvaltest.h5"

DATABASE_PATH = "https://raw.githubusercontent.com/Earlyrizer64/MyST_site/main/Reference_Files/Chemical_Property_Database/Processed_Solvent_DF_v6_TEST.xlsx"

# Call to Function
download_if_needed(CC_MODEL_URL, CC_MODEL_PATH)

CC_MODEL = CC_MODEL_PATH
```

</details>
<summary>Expected output</summary>

```text
Downloading model_v5_CC_okayvaltest.h5...
Download complete.
```

</details>




```
# Cell 4: Load Weights

model_CC = tf.keras.models.load_model(
    CC_MODEL_PATH,
    custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    compile=False
)
```
<details>
<summary>Expected output</summary>

```text
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
/usr/local/lib/python3.12/dist-packages/keras/src/layers/activations/leaky_relu.py:41: UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
  warnings.warn(
```

</details>




```
# Cell 5: Example of Phenol

# 1. Example: Phenol
MOLECULE = "PHENOL"
SMILES = "Oc1ccccc1"

# 2. Read the DB:
db = pd.read_excel(DATABASE_PATH)
db = db.set_index('SMILES')
descriptors = db.loc[SMILES]

# Selected properties for Climate Change:
thermo_feat_CC   = ['Heat Capacity (kJ/kgC)', 'Boiling Point(K)', 'XLogP', 'Critical Temperature [K]', 'Critical Molar Volume [m3/mol]']
mol_desc_feat_CC = ['BertzCT', 'ExactMolWt', 'HallKierAlpha', 'PEOE_VSA6', 'NOCount']

# 3. Obtain the relevant properties:
descriptors_cc = descriptors.loc[thermo_feat_CC + mol_desc_feat_CC]
```
<details>
<summary>Expected output</summary>

```text
No Visible Output
```

</details>




```
# Cell 6: Print Descriptors

descriptors_cc
```
<details>
<summary>Expected output</summary>

```text
	Oc1ccccc1
Heat Capacity (kJ/kgC)	2.1408
Boiling Point(K)	454.99
XLogP	1.5
Critical Temperature [K]	694.25
Critical Molar Volume [m3/mol]	0.000229
BertzCT	134.10737
ExactMolWt	94.041865
HallKierAlpha	-0.98
PEOE_VSA6	18.199101
NOCount	1
dtype: object
```

</details>




```
# Cell 7: Print Climate Change Impact and Human Health Impact

molecule_climate_change_impact = model_CC.predict(np.array([list(descriptors_cc)]), verbose=0)  # kgCO2-eq/kg chemcal

print("Climate Change Impact: ", round(molecule_climate_change_impact.item(), 4), f"kgCO2-eq/kg {MOLECULE}")
```
<details>
<summary>Expected output</summary>

```text
Climate Change Impact:  107.0432 kgCO2-eq/kg PHENOL
```

</details>




```
# Cell 8: Creating Slider to Visualize Impact

kg_slider = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description='kg amount:')
output_slider = widgets.Output()

def update_slider(change):
    with output_slider:
        output_slider.clear_output()
        kg_slider_kg = kg_slider.value
        total_cc_slider = molecule_climate_change_impact[0][0] * kg_slider_kg
        print(f"Climate Impact: {total_cc_slider:.4f} kgCO2-eq")

kg_slider.observe(update_slider, names='value')
display(kg_slider, output_slider)
update_slider(None)
```
<details>
<summary>Expected output</summary>

<img src="/Reference_Files/Climate_Change_Impact_Files/Cell_8.png"/>

</details>




```
# Cell 9: Creating Slider and Graph to Visualize Impact

kg_slider_graph = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description='kg amount:')
output_graph = widgets.Output()

cc_per_kg = molecule_climate_change_impact[0][0]
kg_range = np.linspace(0, 100, 200)

def update_graph(change):
    with output_graph:
        output_graph.clear_output(wait=True)
        kg_graph_kg = kg_slider_graph.value
        total_cc_graph = cc_per_kg * kg_graph_kg

        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.plot(kg_range, cc_per_kg * kg_range, color='#e0a458')
        ax.scatter([kg_graph_kg], [total_cc_graph], color='#e0a458', zorder=5, s=80)
        ax.set_title(f"Climate: {total_cc_graph:.3f} kgCO2-eq")
        ax.set_xlabel(f"kg {MOLECULE}")

        plt.tight_layout()
        plt.show()

kg_slider_graph.observe(update_graph, names='value')
display(kg_slider_graph, output_graph)
update_graph(None)
```
<details>
<summary>Expected output</summary>

<img src="/Reference_Files/Climate_Change_Impact_Files/Cell_9.png"/>

</details>

### Discussion / Analysis

In this code, you were able to see how a chemical impacts climate change using the ANN Model.  Then you could see how changing the amount of that chemical affects the climate change impact prediction.   