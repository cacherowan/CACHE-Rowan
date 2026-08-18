# Global Warming Potential (GWP) Estimation of IPA-Water Case Study

## 1. General Framework

For any environmental impact metric i, the total life cycle assessment (LCA) is computed separately across three lifecycle phases: production, use-phase, and end-of-life, using the following general equations:

$$
\color{red}{LCA_{i,production} = \sum_{j} LCI_{i,j,production} * Flows_j}
$$

$$
\color{orange}{LCA_{i,use-phase} = \sum_{k_{all}} LCA_{i,k,use-phase} + \sum_{j} LCA_{i,j,use-phase}}
$$

$$
\color{green}{LCA_{i,End-of-Life(EoL)-phase} = \sum_{k_{all}} LCA_{i,k,EoL-phase} + \sum_{j} LCA_{i,j,EoL-phase}}
$$

## 2. Application To The IPA-Water Recovery Case Study

For this case study:

$$ i = GWP $$

The assessment includes the production phase (cradle-to-gate) and the IPA recovery process (gate-to-gate). Because IPA is recovered and reused rather than discarded, the end-of-life phase term does not apply here.

Therefore, the general framework narrows to:

$$
GWP_{Total} = GWP_{Production-phase} + GWP_{Use-phase}
$$

## 2.1 Production Phase GWP

The ANN predicts IPA's production GWP directly from its molecular and thermodynamic descriptors  as 1.902 kg CO₂-eq/kg

| Chemical      | Impact (Production Phase, kg CO₂-eq/kg) |
|---------------|:-----------------------------------------:|
| Isopropanol   |                                             |
| Water         |                                             |
| **Total**     |                                             |

:::{note}
To calculate this yourself, go to Production Phase Impact Prediction → Tutorial 1: Climate Change Impact Prediction Using ANN Model, select IPA as the chemical, and run the prediction to obtain its climate change (GWP) impact directly from the ANN model.
::: 

## 2.2 Use Phase GWP

The recovery-phase GWP sums over k ∈ {distillation, pervaporation, each scaled from its own reference case through the power law}:

$$
\frac{GWP_{new}}{GWP_{ref}} = (\frac{F_{new}}{F_{ref}})^\alpha = (\frac{E_{new}}{E_{ref}})^\beta
$$