#!/bin/bash

#echo starting in_hospital_mortality
#for percent in 1 5 10 25 50; do
#	python -m mimic3models.in_hospital_mortality.main --data $data_ihm --output_dir $models_ihm --embed_method RAW --cuda_devices 0 --num_workers 15 -e 500 -b 1024 --lr 0.001 -th 0.86 --percent_data $percent
#        python -m mimic3models.in_hospital_mortality.main --data $data_ihm --output_dir $models_ihm --embed_method DAE  --embed_model $models_pe/DAE/2019-07-31/best/DAE.ep46 --cuda_devices 0 --num_workers 15 -e 500 -b 1024 --lr 0.001 -th 0.85 --percent_data $percent
#        python -m mimic3models.in_hospital_mortality.main --data $data_ihm --output_dir $models_ihm --embed_method DFE  --embed_model $models_pe/DFE/2019-08-05/best/DAE.ep45 --cuda_devices 0 --num_workers 15 -e 500 -b 1024 --lr 0.001 -th 0.86 --percent_data $percent#
#	python -m mimic3models.in_hospital_mortality.main --data $data_ihm --output_dir $models_ihm --embed_method PCA  --embed_model $models_pe/PCA/2019-08-02/best/PCA.ep0 -nc --num_workers 15 -e 500 -b 1024 --lr 0.001 -th 0.845 --percent_data $percent
#done

#echo starting discharge
#for percent in 1 5 10 25 50; do
#	python -m mimic3models.discharge.main --data $data_dis --output_dir $models_dis --embed_method RAW --cuda_devices 0 --num_workers 15 -e 50 -b 1024 --lr 0.001 -th 0.834 --percent_data $percent
#	python -m mimic3models.discharge.main --data $data_dis --output_dir $models_dis --embed_method DAE  --embed_model $models_pe/DAE/2019-07-31/best/DAE.ep46 --cuda_devices 0 --num_workers 15 -e 50 -b 1024 --lr 0.001 -th 0.832 --percent_data $percent
#	python -m mimic3models.discharge.main --data $data_dis --output_dir $models_dis  --embed_method PCA  --embed_model $models_pe/PCA/2019-08-02/best/PCA.ep0 -nc --num_workers 15 -e 50 -b 1024 --lr 0.001 -th 0.832 --percent_data $percent
#	python -m mimic3models.discharge.main --data $data_dis --output_dir $models_dis --embed_method DFE  --embed_model $models_pe/DFE/2019-08-05/best/DAE.ep45 --cuda_devices 0 --num_workers 15 -e 50 -b 1024 --lr 0.001 -th 0.79 --percent_data $percent
#done

echo starting extended length of stay
for percent in 1 5 10 25 50; do
	python -m mimic3models.extended_length_of_stay.main --data $data_elos --output_dir $models_elos --embed_method RAW --cuda_devices 0 --num_workers 2 -e 500 -b 1024 --lr 0.001 -th 0.799 --percent_data $percent
	python -m mimic3models.extended_length_of_stay.main --data $data_elos --output_dir $models_elos --embed_method DAE  --embed_model $models_pe/DAE/2019-07-31/best/DAE.ep46 --cuda_devices 0 --num_workers 2 -e 500 -b 1024 --lr 0.001 -th 0.79 --percent_data $percent
	python -m mimic3models.extended_length_of_stay.main --data $data_elos --output_dir $models_elos  --embed_method PCA  --embed_model $models_pe/PCA/2019-08-02/best/PCA.ep0 -nc --num_workers 2 -e 500 -b 1024 --lr 0.001 -th 0.798 --percent_data $percent
	python -m mimic3models.extended_length_of_stay.main --data $data_elos --output_dir $models_elos --embed_method DFE  --embed_model $models_pe/DFE/2019-08-05/best/DAE.ep45 --cuda_devices 0 --num_workers 2 -e 500 -b 1024 --lr 0.001 -th 0.802 --percent_data $percent
done

echo starting decompensation
for percent in 1 5 10 25 50; do
	python -m mimic3models.decompensation.main --data $data_dec --output_dir $models_dec --embed_method RAW --cuda_devices 0 --num_workers 15 -e 50 -b 1024 --lr 0.001 -th 0.88 --percent_data $percent
	python -m mimic3models.decompensation.main --data $data_dec --output_dir $models_dec  --embed_method PCA  --embed_model $models_pe/PCA/2019-08-02/best/PCA.ep0 -nc --num_workers 30 -e 50 -b 1024 --lr 0.001 -th 0.87 --percent_data $percent
	python -m mimic3models.decompensation.main --data $data_dec --output_dir $models_dec --embed_method DAE  --embed_model $models_pe/DAE/2019-07-31/best/DAE.ep46 --cuda_devices 0 --num_workers 30 -e 50 -b 1024 --lr 0.001 -th 0.87 --percent_data $percent
	python -m mimic3models.decompensation.main --data $data_dec --output_dir $models_dec --embed_method DFE  --embed_model $models_pe/DFE/2019-08-05/best/DAE.ep45 --cuda_devices 0 --num_workers 30 -e 50 -b 1024 --lr 0.001 -th 0.88 --percent_data $percent
done
