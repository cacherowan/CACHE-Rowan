# Environmental Impact Forecasting of Chemicals and Processes with Molecular Simulation, Machine Learning and Data Analytics Tools

**The modules contained on this site are made to educate undergraduate level students in courses related to sustainability.  These modules will teach students how to use molecular dynamics simulations to find chemical properties of novel chemicals as well as show them how to use a machine learning artificial neural network algorithm to determine the impact of chemicals on both climate change and human health.**


<div style="display: flex; gap: 20px; justify-content: center;">

  <!-- Left Column -->
  <div style="flex: 1; text-align: left;">
    <img src="/Reference_Files/Main_Page_1.png" alt="Image 1" style="width:100%; max-width:300px;">
    <div style="padding:10px; margin-top:10px; min-height:100px;">
      Life Cycle Assessment (LCA) is the foundation of our environmental forecasting framework. It measures the environmental impacts of a chemical throughout its lifecycle by accounting for the energy consumed, materials used, and emissions generated from production through end-of-life. Depending on the objective of the study, different lifecycle boundaries can be selected to focus on specific stages of the product's journey. 
    </div>
  </div>

  <!-- Right Column -->
  <div style="flex: 1; text-align: left;">
    <img src="/Reference_Files/water_box.gif" alt="Image 2" style="width:100%; max-width:300px;">
    <div style="padding:10px; margin-top:10px; min-height:100px;">
      A molecular dynamics (MD) simulation is a computer experiment that tracks how atoms move over time. At every step, the computer looks at where the atoms are, calculates the forces between them, and nudges each atom forward by a tiny amount. That tiny amount is called the timestep, and it is usually about one femtosecond (10-15 seconds). String millions of these steps together and you get a movie of atomic motion. The heart of any MD simulation is the potential. This is the mathematical function that tells the computer how strongly atoms push or pull on each other based on their positions. Without a potential, there are no forces, and nothing moves. 
    </div>
  </div>

</div>


## Common LCA System Boundaries

**Cradle-to-Gate (Production Phase):** Evaluates the environmental impacts from raw material extraction through manufacturing, ending when the product leaves the production facility.

**Gate-to-Gate (Process Phase):** Focuses on the environmental impacts of a single manufacturing or processing step, making it useful for comparing different production technologies.

**Gate-to-Grave (End-of-Life Phase):** Assesses the impacts from the factory gate through transportation, product use, recycling, and final disposal.

**Cradle-to-Cradle (Full Life Cycle):** Evaluates the entire lifecycle, including the recovery and reuse of materials to create new products instead of treating them as waste.

## The Challenge: Missing Life Cycle Inventory (LCI) Data

LCA relies on Life Cycle Inventory (LCI) data that describes the resources consumed and emissions generated during chemical production. While these data are available for established chemicals, they are unavailable for newly developed compounds. Our framework enables the prediction of environmental impact categories, including climate change and human health, even in the absence of complete LCI data.

<img src="/Reference_Files/MainPage_2.png">

## Module 1A: Thermodynamic and Molecular Descriptor Generation through Molecular Simulation

- SMILES fingerprints serve as the input to the framework. 

- Molecular simulation and direct property prediction are used to generate thermodynamic and molecular descriptors. 

- The predicted descriptors provide the input features required by the Artificial Neural Network (ANN) model.

## Module 2B: Environmental Impact Prediction Using Artificial Neural Networks (ANN)

- The predicted thermodynamic and molecular descriptors are provided as inputs to the ANN.

- The ANN predicts human health and climate change impact categories.

## Module 2C: Climate Change Impact Prediction Using Data Analytics

- Collect process throughput, energy consumption, and climate change impact data for representative separation technologies. 

- Develop regression-based scaling models relating climate change impacts to process throughput and energy consumption. 

- Predict gate-to-gate climate change impacts for industrial-scale chemical processes.