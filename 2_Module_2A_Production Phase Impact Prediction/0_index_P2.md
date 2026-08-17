# Machine Learning Background

## What is Machine Learning

**Machine learning**  is a branch of artificial intelligence that enables computers to learn from data rather than relying on explicitly programmed rules. During training, a model is shown many examples so it can identify the underlying patterns and relationships in the data; once trained, it applies what it has learned to new, similar data to make predictions or solve problems. Because a model's performance depends on what it was trained on, it generally performs best on data that resembles its training examples.
Machine learning approaches are typically grouped by how a model learns from data. In supervised learning, each training example is paired with a known, correct answer that the model learns to reproduce. In unsupervised learning, the model instead identifies patterns or groupings within unlabeled data. In reinforcement learning, the model learns through trial and error, receiving rewards for favorable decisions.

This work uses a supervised learning approach, which covers two main types of problems: regression, which predicts a continuous numerical value such as a quantity or rate, and classification, which assigns an input to one of several discrete categories. Figure 1 illustrates the simplest case of a regression model: a straight line, y = β0 + β1x, fit through a scatter of data points so that it captures the overall upward trend between the two variables. The line itself is the model's prediction, for any value of the independent variable, the corresponding point on the line is the model's best estimate of the dependent variable.
<img src="/Reference_Files/Production_Phase_Impact_Prediction/regression.png" style="max-width: 100%; width: 600px; height: auto; display: block; margin: 0 auto;">
**Figure 1.** A linear regression model fit to a set of data points. The line represents the predicted relationship between the independent and dependent variables.

The specific model used in this phase is an artificial neural network, a machine learning architecture inspired by the structure of the human brain and well suited to capturing complex, nonlinear relationships that a single straight line cannot. As shown in Figure 2, a neural network is built from layers of simple, interconnected units, or neurons: an input layer, one or more hidden layers, and an output layer, with every connection between neurons carrying a weight that adjusts as the network learns. Here, the network is applied to a regression problem: predicting a continuous value, the climate-change impact of a chemical, from that chemical's input properties
<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <div style="flex: 1; text-align: center;">
    <img src="/Reference_Files/Production_Phase_Impact_Prediction/ANN.png" style="max-width: 100%; height: auto;">
    <p><em>Figure 2. Architecture of a fully connected artificial neural network, showing the input layer, hidden layers, and output layer.</em></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/Reference_Files/Production_Phase_Impact_Prediction/image.png" style="max-width: 100%; height: auto;">
    <p><em>Figure 3. Structure of a single neuron, showing how inputs, weights, and a bias combine through an activation function to produce an output.</em></p>
  </div>
</div>
Figure 3 details how a single neuron performs this computation. Each input xᵢ is multiplied by its corresponding weight wᵢ, and the weighted inputs are summed together with a bias term b that shifts the result up or down. This value then passes through an activation function f, which introduces the nonlinearity that lets the network learn curved relationships rather than being limited to straight lines. Learning consists of adjusting these weights and biases: during training, the network compares its predictions to the correct answers and updates each one to reduce the resulting error, gradually tuning itself to fit the data.
# Tutorials

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorial 1: Climate Change Impact
:link: 1_Climate_Change_Impact_Prediction.md

Learn how to load a model weight and have the model take an input of a specific chemical and deliver a predicted value for Climate Change Impact
:::

:::{grid-item-card} Tutorial 2: Human Health Impact
:link: 2_Human_Health_Impact_Prediction.md

Learn how to load a model weight and have the model take an input of a specific chemical and deliver a predicted value for Human Health Impact
:::

::::

Both tutorial 1 and 2 have similar setups but tutorial 1 predicts the climate change impact and tutorial 2 predicts the human health impact of different chemicals.  