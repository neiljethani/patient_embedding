=======
# Patient Embedding 
## Introduction:

Recently, the healthcare industry has been focused on distributing personalized care management and treatments to patients. This requires that individualized patient phenotyping drive the clinical decision making process. The electronic health record (EHR) provides each patient with a digital representation of the medical information collected over the course of their encounters. The EHR, therefore, provides a highly personalized representation of a patient. However, the abundance of data and longitudinal of the EHR makes it difficult to concisely summarize and compare patients for the development of personalized clinical decision support.

Advances in machine and deep learning techniques, especially with the field of natural language processing, has enabled concise representations of sequentially ordered information. Such methods offer the potential to capitulate patterns in the data within personalized representations that account for both the heterogeneity and similarities among patient phenotypes across time.

In this proposal, we intend to develop patient embeddings by employing longitudinal unsupervised methods. Unsupervised learning allows for the development of representations that are inherently task independent and, therefore, generalizable across prediction tasks. Fundamentally, we aim to ensure that our embeddings both provide an accurate representation of the longitudinal information provided as well as a contextual understanding of future clinical events.  

## Prior Work and Background:

Prior research on developing concise patient representations has utilized both supervised and unsupervised learning, with only some studies accounting for the temporality of the patient events. Unsupervised approaches follow an encoding and decoding paradigm. Miotto et. al. utilize stacked denoising autoencoders to develop a robust and concise patient representation, though does not consider the longitudinality of the EHR. Lyu et. al. meanwhile utilized an LSTM with attention to encode sequential patient records, where separate embeddings were constructed that either reconstruct a current or future sequence of medical events. Supervised approaches, instead, develop task specific patient representations. Zhang et. al. train a LSTM with attention to predict future hospitalization accounting for longitudinally of the EHR and the explicit contextualization of individual medical concepts. However, the patient representation learned is unlikely to generalize across heterogenous tasks.

Our work instead attempts to address not only the longitudinal nature of the EHR in an unsupervised fashion, but also ensure that our representation is both reconstructive of the patient data provided and predictive of future clinical events. We propose to do so by utilizing advances in language modeling, namely the transformer model which allows for parallelized attention with sequential data. We also intend to borrow inutions from the BERT, Bidirectional Encoder Representations from Transformers, model by incorporating both masked language/sequence modeling (MLM/MSM) and future sequence construction (FSC). It is important to note that BERT’s handling of Next Sequence Prediction (NSP) differs from our formulation of future sequence construction (FSC) in that NSP simply predicts whether or not two sequences occur sequentially, while with FSC we will predict the next unseen sequence.

## Proposed Architecture:

![](https://github.com/neiljethani/EHR/blob/master/patient_embedding/PTE_architecture.png)

## Proposed Formulation:

We will use the eICU or MIMIC III dataset. Clinical events occurring during a visit will be aggregated hourly. The transformer model suggested will intake masked 24hr sequences. From these sequences the masked event will be predicted as well as the following 24hr sequence. A joint cross entropy loss will be used for learning.

## Evaluation:

### Visualization with Domain Analysis:

Patient representations will be evaluated by visualization of patient representations via t-SNE. Future visualization of task specific separability can be inspected. Additionally, nearest neighbors can be inspected by domain experts to qualitatively evaluate embeddings.

### Performance on Prediction Tasks:

We will evaluate the AUROC, AUPRC, and F1 Scores of both the unsupervised embeddings directly and via transfer learning of the initialed model. We will aim to predict 24hr Mortality, Discharge, and Sepsis prediction.

