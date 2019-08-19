#!/bin/bash

cd $src
python -m mimic3models.patient_embedding.main --data $data_pe --output_dir $models_pe --embed_method TRANS -b 1024 -e 50 --log_freq 100
python -m mimic3models.patient_embedding.main --data $data_pe --output_dir $models_pe --embed_method DAE -b 1024 -e 50 --log_freq 100
python -m mimic3models.patient_embedding.main --data $data_pe --output_dir $models_pe --embed_method PCA -b 4096 -e 1 --log_freq 50
