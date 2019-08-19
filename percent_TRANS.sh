#!/bin/bash

#echo starting in_hospital_mortality
#for percent in 1 5 10 25 50; do
#	python -m mimic3models.in_hospital_mortality.main --data $data_ihm --output_dir $models_ihm --embed_method TRANS  --embed_model $models_pe/TRANS/2019-08-05_20-24-07/best/TRANS.ep90 --cuda_devices 1 --num_workers 15 -e 500 -b 1024 --lr 0.01 -th 0.875 --percent_data $percent
#done

#echo starting discharge
#for percent in 1 5 10 25 50; do
#	python -m mimic3models.discharge.main --data $data_dis --output_dir $models_dis --embed_method TRANS  --embed_model $models_pe/TRANS/2019-08-05_20-24-07/best/TRANS.ep90 --cuda_devices 1 --num_workers 15 -e 100 -b 1024 --lr 0.001 -th 0.83 --percent_data $percent
#done

#echo All done

echo starting decompensation
for percent in 1 5 10 25 50; do
	python -m mimic3models.decompensation.main --data $data_dec --output_dir $models_dec --embed_method TRANS  --embed_model $models_pe/TRANS/2019-08-05_20-24-07/best/TRANS.ep90 --cuda_devices 1 --num_workers 30 -e 50 -b 1024 --lr 0.001 -th 0.88 --percent_data $percent
done

echo starting extended length of stay
for percent in 1 5 10 25 50; do
	python -m mimic3models.extended_length_of_stay.main --data $data_elos --output_dir $models_elos --embed_method TRANS  --embed_model $models_pe/TRANS/2019-08-05_20-24-07/best/TRANS.ep90 --cuda_devices 1 --num_workers 15 -e 500 -b 1024 --lr 0.01 -th 0.802 --percent_data $percent
done

echo ALL DONE
