# Clustering-Financial-Institutions-with-Soft-Information

Code repository for the paper **_Clustering Financial Institutions with Soft Information: A Computational Linguistics Approach_**, submitted in partial fulfilment of the requirements for the degree of MSc Financial Technology at Imperial College London

## Code Definition
- The `pdf_preprocess_and_parse` directory contains code for document extraction, text cleaning, and normalisation.
- The `w2v` directory contains code for creating Negative Sampling Word2Vec training datasets and Word2Vec model using the Skip-Gram architecture.
- The `sbert` directory contains code for encoding sentences using SBERT.
- The `empirical_result` directory contains code for clustering and testing the significance of soft information against hard information.

## Data Definition
- Soft information consists of annual report PDF files of 117 financial institutions, spanning the years 2013 to 2022, downloaded from the official websites of these institutions.
- Hard information includes Global Systemically Important Banks (G-SIB) Indicators and the G-SIB Score, downloaded from [(Bank for International Settlements, 2022)](https://www.bis.org/bcbs/gsib/gsib_assessment_samples.htm).

## Requirements and Environment
- **Environment:** Python 3.11.8
- **Requirements:** Please refer to `requirements.txt`.
- **Original Results Produced On:**
    - **Operating System:** Windows 11 Pro Version 23H2
    - **CPU:** 13th Gen Intel(R) Core(TM) i5-13400F 2.50 GHz
    - **GPU:** GeForce RTX 4060 Ti