# A multi-agent model of urban microgrids: Assessing the effects of energy-market shocks using real-world data
based on work done during the FIM individual research phase in fall 2022 at Research Center Finance & Information Management, Branch Business & Information Systems Engineering of the Fraunhofer FIT.

This code implements an agent-based model to simulate residential smart home behavior with peer-to-peer trading. The resulting paper is published in the Journal of Applied Energy ([Link](https://www.sciencedirect.com/science/article/pii/S0306261923005445?via%3Dihub)).

## Data
The underlying housing data was provided by tetraeder.solar and the city of Ingolstadt. Therefore, it is proprietary and not uploaded. However, data for an anonymized community can be found at `data_03/custom_communities`. This is the community composition used in the paper.

The rest of the data (load profiles, electricity price, etc.) comes from literature and is therefore made available. It can also be found in the folder `03_data`.

## Code
The agent-based simulation leverages the mesa library, and the smart home energy management system (HEMS) optimization is implemented in Python Gurobi interface gurobipy. All of the code can be found in the folder `02_implementation/energy_model`.

The relevant code files are:
- `Energy_Community_Model.py`: Initialization, P2P trading mechanism, and logging
- `CommunityMember`: Implementation of the HEMS optimization and other agent behavior
- `data_setup.py`: Read in data that was preprocessed to initialize the model
- `model_eval.py`: Functions to compute metrics and visualize results
- `model_run.ipynb`: Execution of simulation
- `model_eval_benchmarks.ipynb`: Analysis with important metrics used in the paper and some plots
- `model_eval_graphs.ipynb`: Analysis with lots of plots that are used in the paper

Other less relevant files are:
- `DataBase`: Storage of data for model and agents
- `network_setup.py`: Preprocess data to create a custom community from the proprietary input
- `model_test.ipynb`: Testing and debugging of model

Nonrelevant, because not used files are:
- `server.py`
- `model_vis.py`
- `main.py`

## Final remarks
In my opinion, although the code works, is not super clean. I am currently working on a follow-up project that builds on this model, but is larger in scope (master thesis). Therefore, you can consider this repository to be archived, i.e. to not change in the future. In the new project that is not open-source yet, I hope to learn from this project and produce higher-quality code. I will probably open-source the new repository around November 2023.

If you have any questions regarding this or the new project, feel free to reach me at [jochen.madler@tum.de](mailto:jochen.madler@tum.de). Cheers! :)





