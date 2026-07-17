# Environmental Impact Forecasting of Chemicals and Processes with Molecular Simulation, Machine Learning and Data Analytics Tools

### Life Cycle Assessment (LCA) is the foundation of our environmental forecasting framework. It measures the environmental impacts of a chemical throughout its lifecycle by accounting for the energy consumed, materials used, and emissions generated from production through end-of-life. Depending on the objective of the study, different lifecycle boundaries can be selected to focus on specific stages of the product's journey.

<img src="/Reference_Files/Main_Page_1.png">

## **Common LCA System Boundaries**

### **Cradle-to-Gate (Production Phase):** Evaluates the environmental impacts from raw material extraction through manufacturing, ending when the product leaves the production facility.

### **Gate-to-Gate (Process Phase):** Focuses on the environmental impacts of a single manufacturing or processing step, making it useful for comparing different production technologies.

### **Gate-to-Grave (End-of-Life Phase):** Assesses the impacts from the factory gate through transportation, product use, recycling, and final disposal.

### **Cradle-to-Cradle (Full Life Cycle):** Evaluates the entire lifecycle, including the recovery and reuse of materials to create new products instead of treating them as waste.

## The Challenge: Missing Life Cycle Inventory (LCI) Data

### LCA relies on Life Cycle Inventory (LCI) data that describes the resources consumed and emissions generated during chemical production. While these data are available for established chemicals, they are unavailable for newly developed compounds. Our framework enables the prediction of environmental impact categories, including climate change and human health, even in the absence of complete LCI data.

<img src="/Reference_Files/Main_Page_2.png">

## Module 1A: Thermodynamic and Molecular Descriptor Generation through Molecular Simulation

- SMILES fingerprints serve as the input to the framework. 

- Molecular simulation and direct property prediction are used to generate thermodynamic and molecular descriptors. 

- The predicted descriptors provide the input features required by the Artificial Neural Network (ANN) model.

## **Module 2B: Environmental Impact Prediction Using Artificial Neural Networks (ANN)**

- The predicted thermodynamic and molecular descriptors are provided as inputs to the ANN.

- The ANN predicts Human health and climate change impact categories.

## Module 2C: Climate Change Impact Prediction Using Data Analytics

- Collect process throughput, energy consumption, and climate change impact data for representative separation technologies. 

- Develop regression-based scaling models relating climate change impacts to process throughput and energy consumption. 

- Predict gate-to-gate climate change impacts for industrial-scale chemical processes.


***


***




This site provides intuitive tutorials on obtaining chemical properties using two main methods: simulating molecules and extracting measured property data from a database. The database currently contains a few hundred chemicals and numerous chemical properties. The site includes three tutorials and one comparison of the different methods.

Tutorial 1 walks you through simulating a gas molecule and calculating its enthalpy, entropy, and Gibbs free energy. Tutorial 2 shows how to calculate the heat capacity of gases, from creating the molecular geometry using the Simplified Molecular Input Line Entry System (SMILES) to simulating molecular behavior using the MACE-OFF machine learning potential. Tutorial 3 demonstrates the database search approach, where SMILES is used to define a chemical and query the database for any available property. The final code example shows how to use both MACE-OFF and the Atomic Simulation Environment (ASE), as well as the database, to calculate the standard enthalpy of formation of gases. It then compares the error of each method against reference values from the National Institute of Standards and Technology (NIST) and discusses the advantages and limitations of each approach.

Together, these tutorials can be used as provided or modified to meet the user's needs. They are designed to help users obtain chemical properties for different substances to support the early-stage life cycle assessment of chemicals.