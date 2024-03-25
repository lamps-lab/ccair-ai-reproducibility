# Citation Contexts for AI Reproducibility [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10871052.svg)](https://doi.org/10.5281/zenodo.10871052)
___
This repository is for the proof of concept project for identifying the correlation between citation context of citing papers and the reproducibility of cited paper (original paper) within the field  of Artificial Intelligence

This repository contains the code and the data to reproduce results for the work titled: <b>"Can citations tell us about a paper’s reproducibility? A
case study of machine learning papers</b>" 

## Folder structure 
```
    .
    ├── data              # datasets
    ├── documents         # Documentation files (alternatively `doc`)
    ├── notebooks         # .ipynb notebook files
    ├── plots             # Visualizations stored location
    ├── scripts           # Executable files
    └── README.md
```


## Dependencies ##
> <font face="consolas" color='#000080'><b>1. python:</b><br> 
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;3.8.17&emsp;&emsp;|&emsp;&emsp;3.9.13<br><br>
<b>2. Jupyter Notebook:</b><br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;notebook server: 6.5.4<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Kernel Information: Python 3.8.17 (default, Jul  5 2023, 20:44:21) [MSC v.1916 64 bit (AMD64)]<br><br>
<b>3. pip installations:</b><br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;TensorFlow<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pip install tensorflow==2.12.0<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pip install keras-core --upgrade<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pip install -q keras-nlp --upgrade<br>
      &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;PyTorch<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pip install torch==1.13.0<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;transformers<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pip install -q transformers<br>
<b>4. conda installations:</b><br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pandas==2.0.3<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Beatifulsoup4==4.11.1<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Selenium==4.11.2<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;conda install -c conda-forge selenium<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;webdriver-manager=4.0.0<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;conda install -c conda-forge webdriver-manager<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;bibtexparser-1.4.0<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;conda install -c conda-forge bibtexparser!<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;pdfminer.six-20221105<br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;conda install -c conda-forge pdfminer.six<br>

</font>

## Dataset ##
Avaialable in the `data` directory

![alt text](documents/citaion_contexts_relationships.png "Citation Contexts for AI Reproducibility - Dataset")


## Steps ##
1. Create a new Anaconda environment and install the dependencies
2. Clone the repository [lamps-lab/ai-reproducibility](https://github.com/lamps-lab/ai-reproducibility)
3. Use either the available data in `data` directory or create the datasets from scratch using `notebooks/R_001_Creating_the_RS_superset.ipynb` and `notebooks/R_001_Extract_Citing_Paper_Details_from_S2GA.ipynb` 
4. For model training use the `notebooks/R_001_multiclass_text_classification_from_colab_for_5_classes.ipynb` and `notebooks/R_001_binary_text_classification_related_not_related.ipynb`
5. Use `notebooks/R_001_Visualizations.ipynb` for vizualizations

<!-- ## Citation ## -->

```BibTeX

```

Rochana R. Obadage<br>
03/25/2024