![Passing Status](https://github.com/bluegardier/contamination_model/actions/workflows/github-ci.yml/badge.svg)

# Contamination Rate Prediction
Disclaimer: This project is in a **WIP** state.

### Project Description
This project is a regression type that aims to estimate the contamination rate of a population, 
using best practices from traditional software.



### Data Dependencies
We are using the following Data Source:

| Source | Description | Year |
|--------|-------------|------|
|Sintetic data for Data Science/Machine Learning practices.|Data containing info about fictional individuals like age, IMC and how they connect with each other. |XXXX|

### Usage
To run this project locally:
```
git clone https://github.com/bluegardier/contamination_model.git
cd airbnb_prediction
pip install .
``` 

### Repository Structure
- `contamination_model`: Project modules.
- `workspace`: Project's data containing both raw and clean data for modelling and the model's binary.
- `requirements.txt`: contains python dependencies to reproduce the experiments.

### Running the Project
- `python main.py --help`: Shows usage information.
- `python main.py features`: Generate features
- `python main.py deploy_model`: Deploy model
- `python main.py predict_model`: Predicts model
- `python main.py run`: Run all model pipeline steps sequentially
