![Passing Status](https://github.com/bluegardier/contamination_model/actions/workflows/github-ci.yml/badge.svg)

# Contamination Rate Prediction
Disclaimer: This project is in a **WIP** state.

### Project Description
This project is a regression type that aims to estimate the contamination rate of a population, 
using best practices from traditional software.



### Data Dependencies
We are using the following Data Source:

| Source | Description |
|--------|-------------|
|Sintetic data for Data Science/Machine Learning practices.|Data containing info about fictional individuals like age, IMC and how they connect with each other. |XXXX|


Columns Input - individuos_espec:

| Column | Type | Description |
|--------|------|-------------|
|name|int|Person's ID|
|idade|float|Person's Age|
|estado_civil|str|Person's marital status|
|qt_filhos|float|Person's number of childrens|
|estuda|float|Person's study status. Receives 1 if they study, 0 otherwise|
|trabalha|float|Person's work status. Receives 1  if they work, 0 otherwise|
|pratica_esportes|float|Person's sports status. Receives 1  if they practice any sport, 0 otherwise|
|transporte_mais_utilizado|str|Person's most frequent means of transport|
|IMC|float|Person's IMC|

Columns Input - conexoes_espec:

| Column | Type | Description |
|--------|------|-------------|
|V1|int|The contamined person ID|
|V2|int|The new infected person ID by V1|
|grau|str|How V1 and V2 are connected|
|proximidade|str|How close V1 and V2 are|
|prob_V1_V2|float64|The probability V1 infects V2|


### Usage
To run this project locally:
```
git clone https://github.com/bluegardier/contamination_model.git
cd contamination_model
pip install -r requirements.txt && pip install .
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
