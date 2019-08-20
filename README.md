# Patient Embedding 
## Introduction:

Recently, the healthcare industry has been focused on distributing personalized care management and treatments to patients. This requires that individualized patient phenotyping drive the clinical decision making process. The electronic health record (EHR) provides each patient with a digital representation of the medical information collected over the course of their encounters. The EHR, therefore, provides a highly personalized representation of a patient. However, the abundance of data and longitudinal of the EHR makes it difficult to concisely summarize and compare patients for the development of personalized clinical decision support.

Advances in machine and deep learning techniques, especially with the field of natural language processing, has enabled concise representations of sequentially ordered information. Such methods offer the potential to capitulate patterns in the data within personalized representations that account for both the heterogeneity and similarities among patient phenotypes across time.

In this proposal, we intend to develop patient embeddings by employing longitudinal unsupervised methods. Unsupervised learning allows for the development of representations that are inherently task independent and, therefore, generalizable across prediction tasks. Fundamentally, we aim to ensure that our embeddings both provide an accurate representation of the longitudinal information provided as well as a contextual understanding of future clinical events.  

## Prior Work and Background:

Prior research on developing concise patient representations has utilized both supervised and unsupervised learning, with only some studies accounting for the temporality of the patient events. Unsupervised approaches follow an encoding and decoding paradigm. Miotto et. al. utilize stacked denoising autoencoders to develop a robust and concise patient representation, though does not consider the longitudinality of the EHR. Lyu et. al. meanwhile utilized an LSTM with attention to encode sequential patient records, where separate embeddings were constructed that either reconstruct a current or future sequence of medical events. Supervised approaches, instead, develop task specific patient representations. Zhang et. al. train a LSTM with attention to predict future hospitalization accounting for longitudinally of the EHR and the explicit contextualization of individual medical concepts. However, the patient representation learned is unlikely to generalize across heterogenous tasks.

Our work instead attempts to address not only the longitudinal nature of the EHR in an unsupervised fashion, but also ensure that our representation is both reconstructive of the patient data provided and predictive of future clinical events. We propose to do so by utilizing advances in language modeling, namely the transformer model which allows for parallelized attention with sequential data. We also intend to borrow inutions from the BERT, Bidirectional Encoder Representations from Transformers, model by incorporating both masked language/sequence modeling (MLM/MSM) and future sequence construction (FSC). It is important to note that BERT’s handling of Next Sequence Prediction (NSP) differs from our formulation of future sequence construction (FSC) in that NSP simply predicts whether or not two sequences occur sequentially, while with FSC we will predict the next unseen sequence.

## Proposed Architecture:

![](https://github.com/neiljethani/EHR/blob/master/patient_embedding/PTE_FINAL.png)

## Proposed Formulation:

We will use the eICU or MIMIC III dataset. Clinical events occurring during a visit will be aggregated hourly. The transformer model suggested will intake masked 24hr sequences. From these sequences the masked event will be predicted as well as the following 24hr sequence. A joint cross entropy loss will be used for learning.

## Evaluation:

### Visualization with Domain Analysis:

Patient representations will be evaluated by visualization of patient representations via t-SNE. Future visualization of task specific separability can be inspected. Additionally, nearest neighbors can be inspected by domain experts to qualitatively evaluate embeddings.

### Performance on Prediction Tasks:

We will evaluate the AUROC, AUPRC, and F1 Scores of both the unsupervised embeddings directly and via transfer learning of the initialed model. We will aim to predict 24hr Mortality, Discharge, and Sepsis prediction.

=======

## Structure
- Generating the Datasets
- Generating/Training the Embedding Models
- Generating/Training the Classification Models
- Evaluating and Visualizing Models

### Generating the Datasets

#### Benchmarking Data Processing
##### Running the Scripts: Please Reference https://github.com/YerevaNN/mimic3-benchmarks/blob/master/README.md
- `python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} $data_root`
- `python -m mimic3benchmark.scripts.validate_events $data_root`
- `python -m mimic3benchmark.scripts.extract_episodes_from_subjects $data_root`
- `python -m mimic3benchmark.scripts.split_train_and_test $data_root`
    - Uses benchmarking TEST set but subdivided remains data into TRAIN, VAL, and VAL_TEST sets
- Under the Hood
    - mimic3benchmark/mimic3csv.py
        - Contains functions to read and merge raw MIMIC III data.
    - mimic3benchmark/subject.py
        - Contains functions to read extracted data and process information into a timeseries structure.
    - mimic3benchmark/preprocessing.py
        - Contains functions to extract and maps data to variables. Cleans variables and removes outliers.
        - Added ‘clean_events_remove_outliers’ to remove outliers
    - mimic3benchmark/readers.py
        - Contains reader class to reader the data from the various datasets
        - Class is latter wrapped in pytorch Dataset Class
        
#### Task Specific Dataset Generation
- Running the Scripts
    - Patient Embedding Dataset
        - python -m mimic3benchmark.scripts.create_patient_embedding $data_root $data_pe
    - In Hospital Mortality Dataset
        - python -m mimic3benchmark.scripts.create_in_hospital_mortality $data_root $data_ihm
    - Extended Length of Stay Dataset
        - python -m mimic3benchmark.scripts.create_extended_length_of_stay $data_root $data_elos
    - Decompensation Dataset
        - python -m mimic3benchmark.scripts.create_decompensation $data_root $data_dec
    - Discharge Dates
        - python -m mimic3benchmark.scripts.create_discharge $data_root $data_dis
- Under the Hood
    - Patient Embedding Dataset
        - Requirements
            - Removes data without a length of stay or events 
            - Removes data without icu events in the first 24hrs or last 48hrs
            - Creates 48hrs Windows and Streams along each patient Hourly
                - Only Stores up to 200 windows per patient 
        - Output
            - Folder for Each Data Split
                - Each folder has a ‘listfile.csv’,  a csv file for each window, and a total file.csv
                    - Each row in listfile.csv has the file name for each icu episode, the total number of windows for the patient, and the current window number.
                    - The total file.csv stores the total number of visits in the dataset
    - In Hospital Mortality/ Extended Length of Stay Dataset
        - Requirements
            - Only uses first 24hrs of patient data
            - Removes data without a length of stay or events 
        - Output
            - Folder for Each Data Split
                - Each folder has a ‘listfile.csv’ and a csv file for each icu visit
                    - Each row in listfile.csv has the file name for each icu episode and the corresponding label 
    - Discharge/ Decompensation Dataset
        - Requirements
            - Removes data without a length of stay or events 
        - Output
            - Folder for Each Data Split
                - Each folder has a ‘listfile.csv’ and a csv file for each icu visit
                    - Each row in listfile.csv has the file name for each icu episode, the current time window, and the label of the corresponding time window
                    

### Generating/Training the Embedding Models
- Running the Scripts
    - python -m mimic3models.patient_embedding.main {ARGUMENTS} 
        - {ARGUMENTS} 
            - --data
                - Where input data Is located
            - --output_dir
                - Where to output models
            - --embed_method
                - Embedding method to use (RAW, DAE, DFE, TRANS, PCA)
            - -e 
                - Epochs to train for
            - --lr 
                - Learning rate 
            - -b 
                - Batch size 
            - -l 
                - Layers to use 
                - Used for TRANS, DAE, DFE
            - --cuda_devices 
                - Cuda Device to use
            - --num_workers
                - Number of workers for data loading
            - -mp  (Only used in TRANS)
                - Proportion of reconstruction loss in total loss
            - --nc
                - Option indicates not to use cuda and use cpu instead
                - USED FOR PCA
        - Example:
            - python -m mimic3models.patient_embedding.main --data $data_pe --output_dir $models_pe --embed_method TRANS -l 3 -a 3 -b 1024 -w 30 -e 100 -mp 0.5 --cuda_devices 1 --lr 0.001
- Under the Hood
    - Main File = mimic3models/patient_embedding/main
        - File to run training from command line
    - mimic3models/pytorch_models/embedding = Directory with All Code for Patient Embedding Training
        - File Structure + Function
            - model = contains scripts with pytorch model architectures and function to initialize models
                - model.py
                    - Function to compile and initialize model based on provided arguments 
                - utils
                    - DAE.py, PCA.py, TRANS.py = scripts encoding model architectures 
            - train
                - train.py
                    - Has trainer class with initializes model based on input by from main.py 
                    - Has functions to iterate epochs, save model, and store training statistics
                - utils.py
                    - Has functions to compute loss and take optimizer steps
            - dataset
                - utils.py
                    - Contains pytorch Dataset class for patient embedding
        - Output
            - models/patient_embedding/{embed_method}/{timestamp}
                - models are stored as model.ep#
                - Lossfile.csv has loss statistics
                - best directory contains the best model

### Generating/Training the Classification Models
- Running the Scripts
    - python -m mimic3models.{TASK}.main {ARGUMENTS} 
        - {TASK} 
            - in_hospital_mortality, discharge, extended_length_of_stay, decompensation
        - {ARGUMENTS} 
            - --data
                - Where input data Is located
            - --output_dir 
                - Where to output models
            - --embed_method 
                - Embedding method to use (RAW, DAE, DFE, TRANS, PCA)
            - --embed_model
                - Where to 
            - -e 
                - Epochs to train for
            - --lr 
                - Learning rate 
            - -b 
                - Batch size 
            - --cuda_devices  
                - Cuda Device to use
                - *Note: For TRANS use same device used to train embedding model
            - --num_workers 
                - Number of workers for data loading
            - --nc
                - Option indicates not to use cuda and use cpu instead
                - USED FOR PCA
            - -th
                - AUROC threshold above which to save models
            - --percent_data
                - Percent of data to use for training
        - Examples:
            - python -m mimic3models.discharge.main --data $data_dis --output_dir $models_dis --embed_method TRANS  --embed_model $models_pe/TRANS/2019-08-05_20-24-07/best/TRANS.ep90 --cuda_devices 1 --num_workers 30 -e 100 -b 1024 --lr 0.001 -th 0.83
- Under the Hood
    - Main File = mimic3models/patient_embedding/main
        - File to run training from command line
    - mimic3models/pytorch_models/embedding = Directory with All Code for Patient Embedding Training
        - File Structure + Function
            - model = contains scripts with pytorch model architectures and function to initialize models
                - model.py
                    - Function to compile and initialize model based on provided arguments 
                - utils
                    - DAE.py, PCA.py, TRANS.py = scripts encoding model architectures 
            - train
                - train.py
                    - Has trainer class with initializes model based on input by from main.py 
                    - Has functions to iterate epochs, save model, and store training statistics
                - utils.py
                    - Has functions to compute loss and take optimizer steps
            - dataset
                - utils.py
                    - Contains pytorch Dataset class for patient embedding
        - Output
            - models/{task}/{embed_method}/{timestamp}
                - models are stored as model.ep#
                - Lossfile.csv has performance statistics
                - best directory contains the best model

### Evaluating and Visualizing Models
- Running the Scripts 
    - Generating Bootstrap Performance Metrics as well as ROC and PRC
        - python -m mimic3models.visualization.metrics {ARGUMENTS}
            - {ARGUMENTS}
                - --embed_method
                    - Embedding method to evaluate
                - --num_workers
                    - Number of workers for data loading
                - -b
                    - batch size to use
                - --cuda_devices
                    - Cuda device to use
                - -nc
                    - Instruct not to use cuda (PCA)
            - Example
                - python -m mimic3models.visualization.metrics --embed_method TRANS --num_workers 30  -b 1024 --cuda_devices 1
    - Generating t-SNE Plots
        - python -m mimic3models.visualization.tsne {ARGUMENTS}
            - {ARGUMENTS}
                - --embed_method
                    - Embedding method to evaluate
                - --embed_model
                    - Embedding model to use
                - --num_workers
                    - Number of workers for data loading
                - -b
                    - batch size to use
                - --cuda_devices
                    - Cuda device to use
                - -nc
                    - Instruct not to use cuda (PCA)
            - Example
                - python -m mimic3models.visualization.tsne --embed_method TRANS --num_workers 30 --embed_model $models_pe/TRANS/2019-08-05_20-24-07/best/TRANS.ep90 -b 1024 --cuda_devices 1
- Under the Hood
    - mimic3models/visualization/metrics.py
        - Only requires Embedding Method as inputs —> Finds best model and dataset for EACH TASK
        - mimic3models/visualization/utilis.py
            - EmbeddingVisualizer Class
                - Set data —> set task —> set model —> predict 
                - Computes bootstrapped metrics: AUPRC, AUROC, F1, and Accuracy
                - Plots PRC and ROC 
        - Output
            - vis/{embed_method}/{task}/{timestamp}/{task}.png
    - mimic3models/visualization/tsne.py
        - Requires embedding method and model —> generates tsne with each label visualized for EACH TASK
        - mimic3models/visualization/utilis.py
            - ClassificationVisualizer Class
                - Set data —> Set Task —> fit tsne (sklearn) —> Plots tsne
        - Output
            - vis/{embed_method}/{task}/{timestamp}/
                - BM_{task}.csv
                - PRC_{task}.png
                - ROC_{task}.png



