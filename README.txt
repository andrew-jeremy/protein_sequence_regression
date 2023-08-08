There are 3 Jupyter Notebooks in this repo demonstrating large language models for protein modeling:

1) "protein_regressor_EDA.ipynb": This covers the exploratory data analysis (EDA) including saving a pandas dataframe, stored
in the sub directory "data" as:

'data/20230331_pembro_SSMplus_AlphaSeq_data_unique_embedding.csv'

It contains a new  tokenized protein sequences column for the subsequent models in the following corresponding notebooks.
2) "quantile_regression.ipynb": This is a an XGBoost Regressor implementation that achieves a Pearson correlation coefficient  of 0.82 on the held out test data.
A trained model is saved as a pickle file as "data/Trained_Model.pkl" and can be loaded and subsequent inference done for all the cell starting with the one
labelled as: "Saved Trained Models"

3) "protein_regressor_GPT.ipynb": contain a mini GPT with a regression head. It is implemented and is training but my Colab Pro account has limited
compute and I have not been able to run it to the end. The idea with this implementation is that the tokens in the protein sequence need to 'talk' to detach
other (self-attention) for a much deeper connection to the affinity values as the ordering of the tokens is important in determining the resulting affinity values.
Additional files needed for the GPT model: "train.py", trainer.py", "transformer.py", "dataset.py", "utils.py"
