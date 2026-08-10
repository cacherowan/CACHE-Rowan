# Isopropanol (IPA) Recovery from a Pharmaceutical Waste Stream

## Description: 

Celecoxib is the active ingredient in Celebrex, a medication used to treat arthritis. Its production involves downstream purification steps such as centrifugation and drying, which generate isopropanol (IPA) laden waste streams. This case study focuses on the dryer distillate, which is modeled as a binary mixture containing 51 wt.% IPA and 49 wt.% water at a flow rate of 1,000 kg/hr. The recovery of IPA from this stream is limited by the IPA-water azeotrope, which occurs at approximately 87.7 wt.% IPA and 80.37 °C. As a result, distillation alone cannot achieve the required IPA purity. To overcome this limitation, distillation is combined with pervaporation, a membrane-based separation process that enables separation beyond the azeotropic composition and recovery of high-purity IPA.

<img src="../Reference_Files/Case_Study_For_Validation_Files/Image_1.png"/>

| Step 1: Distillation | Step 2: Pervaporation |
| :--: | :--: |
| The distillation column, equipped with a kettle reboiler and an overhead condenser, performs the initial separation by removing most of the water and concentrating IPA to approximately 88 wt.%, near its azeotropic limit. It handles the bulk separation; however, because of the IPA-water azeotrope, it cannot further increase the IPA purity on its own | The concentrated IPA-water mixture is sent to the pervaporation membrane, which selectively removes water under vacuum. This allows the separation to move beyond the azeotropic limit, producing high-purity IPA as the retentate. The water-rich permeate passes through the membrane, where it is subsequently condensed and removed |

***

**IPA Recovery Flowsheet — Distillation + Pervaporation (Animated) Code**

```{code-cell} python
:tags: [thebe-active-cell]
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from IPython.display import HTML

plt.ioff()  # don't auto-display the figure after every cell; we only want
            # the final inline animation to show, once everything is drawn

# ============================================================ SETTINGS ===
OUTPUT_FILE = "ipa_flowsheet_animated.gif"
FPS = 24
DURATION_SECONDS = 3.5           # length of one loop
FRAMES = int(FPS * DURATION_SECONDS)

# ---------------------------------------------------------------- colors --
INK = "#111111"
GRAY_COL = "#BEBEBE"
MEMBRANE = "#A9C88A"
DASH_PATTERN = (6, 5)             # (on, off) in points
DASH_CYCLE = sum(DASH_PATTERN)
```





```{code-cell} python
:tags: [thebe-active-cell]
x = 5
y = 10
print(f"The sum is {x + y}")
```



# Example Page

This is regular descriptive text. You can write as much as you want here —
explain the molecule, the method, whatever context is needed. Regular
markdown formatting like **bold**, *italics*, and [links](https://example.com)
all work normally.

![Example image](../images/example.png)

## Try it yourself

Edit the values below and click Run to see the result update.

```{code-cell} python
:tags: [thebe-active-cell]
x = 5
y = 10
print(f"The sum is {x + y}")
```

More text can follow after the cell, continuing the page normally.