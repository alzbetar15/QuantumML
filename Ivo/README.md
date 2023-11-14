# qml_sin

## Installation

Begin by cloning the repository to your local machine:

`git clone https://github.com/QML-GW22/qml_sin.git`

Create a virtual environment for dependencies:

Linux:

`virtualenv <path-to-venv>`

Windows:

`python -m venv <path-to-venv>`

Activate the virtual environment using the commands for your OS and CLI of choice:

| Platform | Shell | Command to activate virtual environment |
| :-: | :-: | :-: |
| POSIX | bash/zsh | source \<venv-path>/bin/activate |
| | fish | source \<venv-path>/bin/activate.fish |
| | csh/tsch | source \<venv-path>/bin/activate.csh |
| | Powershell | \<venv-path>/bin/Activate.ps1 |
| Windows | cmd.exe | \<venv-path>\Scripts\activate.bat |
| | Powershell | \<venv-path>\Scripts\Activate.ps1 |

Dependencies are then installed using `pip`:

`pip install -r requirements.txt`

Dependencies are now installed and this will be the environment this code was ran with.

# Quantum Machine Learning for detecting sin waves

This repo contains code for categorising sin waves

This is an implementation of Quantum convolutional neural network for classical data classification from this (paper)[https://arxiv.org/abs/2108.00661]. It uses (Pennylane softwarT)(https://pennylane.ai), to classify sin waves with noise added. This repo is forked from this (source)[https://github.com/takh04/QCNN].


## Sin_Generator.py

This produces a dataset containing both data and labels for 10,000 examples, half of which are sin waves with gaussian noise and half are just gaussian noise *(can be extended in future for other types of noise)*. For a given SNR a noisy sin wave or just noise with 256 datapoints is produced. The sin waves have a random frequency and phase. 

## QCNN_Circuit.py

Different parameterised unitary ansatz are defined and used to create quantum and pooling layers. These layers are then stacked together depending on which type of architecture is used.

ADD CIRCUIT DIAGRAMS
## Training.py

The training set is trained with cross entropy loss in random batches.

## Benchmarking.py

The dataset is loaded and split into 80% train / 20% test, normalised with l2-normalisation and embedded with amplitude embedding into 8 qubits. The above training file is then ran and the training loss history is recorded. Lastly predictions are made on the test set and its accuracy is recorded. 
