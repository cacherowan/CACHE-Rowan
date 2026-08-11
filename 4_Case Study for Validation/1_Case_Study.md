# Isopropanol (IPA) Recovery from a Pharmaceutical Waste Stream

## Description: 

Celecoxib is the active ingredient in Celebrex, a medication used to treat arthritis. Its production involves downstream purification steps such as centrifugation and drying, which generate isopropanol (IPA) laden waste streams. This case study focuses on the dryer distillate, which is modeled as a binary mixture containing 51 wt.% IPA and 49 wt.% water at a flow rate of 1,000 kg/hr. The recovery of IPA from this stream is limited by the IPA-water azeotrope, which occurs at approximately 87.7 wt.% IPA and 80.37 °C. As a result, distillation alone cannot achieve the required IPA purity. To overcome this limitation, distillation is combined with pervaporation, a membrane-based separation process that enables separation beyond the azeotropic composition and recovery of high-purity IPA.

<img src="../Reference_Files/Case_Study_For_Validation_Files/ipa_flowsheet_3d_front.gif"/>

| Step 1: Distillation | Step 2: Pervaporation |
| :--: | :--: |
| The distillation column, equipped with a kettle reboiler and an overhead condenser, performs the initial separation by removing most of the water and concentrating IPA to approximately 88 wt.%, near its azeotropic limit. It handles the bulk separation; however, because of the IPA-water azeotrope, it cannot further increase the IPA purity on its own | The concentrated IPA-water mixture is sent to the pervaporation membrane, which selectively removes water under vacuum. This allows the separation to move beyond the azeotropic limit, producing high-purity IPA as the retentate. The water-rich permeate passes through the membrane, where it is subsequently condensed and removed |

## Process results:

The table below summarizes the key stream flows and energy demands for the distillation and pervaporation stages

| Parameter | Value |
| :--: | :--: |
| Waste Stream Feed (Distillation) [kg/h] | 1000 |
| Distillate (Distillation) [kg/h] | 585 |
| Bottom (Distillation) [kg/h] | 415 |
| Cooling Energy (Distillation) [kW] | 576.22 |
| Steam Energy (Distillation) [kW] | 608.06 |
| Retentate (Pervaporation) [kg/h] | 511.89 |
| Permeate (Pervaporation) [kg/h] | 73.31 |
| Steam Energy (Pervaporation) [kW] | 3.3371 |
| Flux (Pervaporation) [L/m{sup}`2`-h] | 555.00 |



<table>
<tr style="background-color:#552D20; color:white;">
  <th style="text-align:center;">Parameter</th>
  <th style="text-align:center;">Result</th>
</tr>
<tr style="background-color:#FFFFFF; color:#1a1a1a;">
  <td style="text-align:center;">Waste Stream Feed (Distillation) [kg/h]</td>
  <td style="text-align:center;">1000</td>
</tr>
<tr style="background-color:#ECEBE8; color:#1a1a1a;">
  <td style="text-align:center;">Distillate (Distillation) [kg/h]</td>
  <td style="text-align:center;">585</td>
</tr>
<tr style="background-color:#FFFFFF; color:#1a1a1a;">
  <td style="text-align:center;">Bottom (Distillation) [kg/h]</td>
  <td style="text-align:center;">415</td>
</tr>
<tr style="background-color:#ECEBE8; color:#1a1a1a;">
  <td style="text-align:center;">Cooling Energy (Distillation) [kW]</td>
  <td style="text-align:center;">576.22</td>
</tr>
<tr style="background-color:#FFFFFF; color:#1a1a1a;">
  <td style="text-align:center;">Steam Energy (Distillation) [kW]</td>
  <td style="text-align:center;">608.06</td>
</tr>
<tr style="background-color:#ECEBE8; color:#1a1a1a;">
  <td style="text-align:center;">Retentate (Pervaporation) [kg/h]</td>
  <td style="text-align:center;">511.89</td>
</tr>
<tr style="background-color:#FFFFFF; color:#1a1a1a;">
  <td style="text-align:center;">Permeate (Pervaporation) [kg/h]</td>
  <td style="text-align:center;">73.31</td>
</tr>
<tr style="background-color:#ECEBE8; color:#1a1a1a;">
  <td style="text-align:center;">Steam Energy (Pervaporation) [kW]</td>
  <td style="text-align:center;">3.3371</td>
</tr>
<tr style="background-color:#FFFFFF; color:#1a1a1a;">
  <td style="text-align:center;">Flux (Pervaporation) [L/m<sup>2</sup>-h]</td>
  <td style="text-align:center;">550.00</td>
</tr>
</table>