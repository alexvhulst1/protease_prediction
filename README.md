# protease_prediction

## Description
Interaction Finder is a Streamlit application designed to predict protein interactions with kinases, E3 ligases, and proteases based on a UniProt identifier or a protein sequence. The application also supports batch processing via Excel file uploads.

## Features
- Prediction of interactions with kinases, E3 ligases, and proteases.
- Support for individual predictions through UniProt ID or protein sequence input.
- Train the different models to predict interactions with kinases, E3 ligases, and proteases
- Batch processing by uploading an Excel file.
- Visualization of results in graphical format.

## Prerequisites
Ensure you have the following dependencies installed:

- Python 3.6 or higher
- torch
- scikit-learn
- pandas
- numpy
- ijson
- h5py
- json
- streamlit
- transformers
- matplotlib
- seaborn

You can install them using pip:

```bash
pip install torch scikit-learn pandas numpy ijson h5py json streamlit transformers matplotlib seaborn
```

## Project Structure
- `uniprotkb_AND_reviewed_true_2024_03_26.json`: UniProt database of all SwissProt proteins.
- `per-protein.h5`: Embeddings with ProtTrans.
- `protease` directory:
  - `cleavage.csv`: List of cleavages downloaded on the MEROPS site publicly available
  - `problem_1_final.ipynb`: Notebook for training models with and without GO terms based on cleavage lists from `cleavage.csv`.
  - `dic_enzyme.json`: List of the unique integer corresponding to a particular protease (one hot encoding of proteases)
  - `dic_GO_problem_1.json`: List of the unique integer corresponding to a particular GO (one hot encoding of GO)
  - `model_embedding_pep_site_problem_1_final.pt`: Protease Model trained without GO
  - `model_embedding_pep_go_site_problem_1_final.pt`: Protease Model trained with GO
- `E3` directory:
  - `literature.E3.txt`: List of E3 ligases interactions downloaded on the UbiBrowser site publicly available
  - `dic_E3.json`: List of the unique integer corresponding to a particular ligase (one hot encoding of ligases)
  - `dic_GO_problem_2.json`: List of the unique integer corresponding to a particular GO (one hot encoding of GO)
  - `model_embedding_pep_site_problem_2_final.pt`: E3 ligase Model trained without GO
  - `model_embedding_pep_go_site_problem_2_final.pt`: E3 ligase Model trained with GO
  - `problem_2_final.ipynb`: Notebook for training E3 ligase models based on respective data.
- `kinase` directory:
  - `Kinase_Substrate_Dataset`: List of Kinase interactions downloaded on the PhosphoSitePlus site publicly available
  - `dic_kinase.json`: List of the unique integer corresponding to a particular kinase (one hot encoding of kinases)
  - `dic_GO_problem_3.json`: List of the unique integer corresponding to a particular GO (one hot encoding of GO)
  - `model_embedding_pep_site_problem_3_final.pt`: Kinase Model trained without GO
  - `model_embedding_pep_go_site_problem_3_final.pt`: Kinase Model trained with GO
  - `problem_3_final.ipynb`: Notebook for training kinase models based on respective data.
- `final_interface.py`: Script for creating the Streamlit application.

## Usage
1. Clone the repository:

```bash
git clone https://github.com/alexandre.ver-hulst/protease_prediction.git
```

2. Navigate to the project directory:

```bash
cd protease_prediction
```

3. Run the Streamlit application:

```bash
streamlit run final_interface.py
```

4. Open your browser and go to the URL provided by Streamlit (default is `http://localhost:8501`).

## Training Models
### Protease Models
Navigate to the `protease` directory and open `problem_1_final.ipynb` to train protease models using `cleavage.csv` and GO terms data.

### E3 Ligase Models
Navigate to the `E3` directory and open `problem_2_final.ipynb` to train E3 ligase models based on their respective data.

### Kinase Models
Navigate to the `kinase` directory and open `problem_3_final.ipynb` to train kinase models based on their respective data.

## Contribution
Contributions are welcome! Please submit a pull request for any improvements or bug fixes.

## Authors
- [Alexandre ver Hulst](https://github.com/alexandre.ver-hulst)
