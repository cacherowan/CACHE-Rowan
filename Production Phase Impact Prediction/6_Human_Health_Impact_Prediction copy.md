# Human Health Impact Prediction Using ANN Model

***

In this code you will be able to estimate the impact to Human Health using the ANN Model.  Then you will be able to see how changing the amount of that chemical will affect the predicted impact.  

***

### Code Overview: 

This code will download the human health impact weight of the model locally and give this file to the code, which then will take the input of a chemical using the SMILES identification to tell you the impact in kg to human health.  Then you will be able to see the impact scaled according to how much mass in kg you have of that chemical.  



Click the button below to open this code in Google Colab


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Earlyrizer64/MyST_site/blob/main/Reference_Files/Google_Colab_Files/Human_Health_Impact_Prediction.ipynb)


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

HH_MODEL_URL = ("https://raw.githubusercontent.com/Earlyrizer64/MyST_site/main/Reference_Files/Chemical_Property_Database/model_v5_HH_okayvaltest.h5")

HH_MODEL_PATH = "model_v5_HH_okayvaltest.h5"

DATABASE_PATH = "https://raw.githubusercontent.com/Earlyrizer64/MyST_site/main/Reference_Files/Chemical_Property_Database/Processed_Solvent_DF_v6_TEST.xlsx"

# Call to Function
download_if_needed(HH_MODEL_URL, HH_MODEL_PATH)

HH_MODEL = HH_MODEL_PATH
```

</details>
<summary>Expected output</summary>

```text
Downloading model_v5_HH_okayvaltest.h5...
Download complete.
```

</details>




```
# Cell 4: Load Weights

model_HH = tf.keras.models.load_model(
    HH_MODEL_PATH,
    custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    compile=False
)
```
<details>
<summary>Expected output</summary>

```text

# Cell 4: Load Weights

model_HH = tf.keras.models.load_model(
    HH_MODEL_PATH,
    custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    compile=False
)
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
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

# Selected properties for Human Health Impact:
thermo_feat_HH   = ['Heat of Vaporization(J/mol)', 'Heat Capacity (kJ/kgC)', 'XLogP','Pitzer’s Acentric Factor [-]', 'Critical Temperature [K]']
mol_desc_feat_HH = ['Chi0n', 'HallKierAlpha', 'SMR_VSA7', 'VSA_EState6','NumValenceElectrons']

# 3. Obtain the relevant properties:
descriptors_hh = descriptors.loc[thermo_feat_HH + mol_desc_feat_HH]
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
Heat of Vaporization(J/mol)	57735.363515
Heat Capacity (kJ/kgC)	2.1408
XLogP	1.5
Pitzer’s Acentric Factor [-]	0.438
Critical Temperature [K]	694.25
Chi0n	3.833965
HallKierAlpha	-0.98
SMR_VSA7	30.331835
VSA_EState6	8.712685
NumValenceElectrons	36
dtype: object
```

</details>




```
# Cell 7: Print Climate Change Impact and Human Health Impact

molecule_human_health_impact = model_HH.predict(np.array([list(descriptors_hh)]), verbose=0) # DALY/kg chemical

print("Human Health Impact: ", round(molecule_human_health_impact.item(), 4), f"DALY/kg {MOLECULE}")
```
<details>
<summary>Expected output</summary>

```text
Human Health Impact:  0.9596 DALY/kg PHENOL
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
# Cell 9: Creating Slider and Graph to Visualize Impact

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