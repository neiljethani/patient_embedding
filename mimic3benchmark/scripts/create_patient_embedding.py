from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
import math
import itertools

random.seed(49297)


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data_list = []
    data_list_visit = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    total = 0
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            data_list_ts = []
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours

                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]
                
                # no measurements in ICU
                if len(ts_lines) == 0: #len(ts_lines_x) == 0 or len(ts_lines_y) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue
                
                #Remove Patients With Nothing in the First 24 Hours
                if len([line for (line, t) in zip(ts_lines, event_times) 
                        if -eps < t < 24]) == 0:
                    print("\n\t(no events in first 24hrs of ICU) ", patient, ts_filename)
                    continue
                    
                #Remove Patients With Nothing in the Last 48 Hours
                if len([line for (line, t) in zip(ts_lines, event_times) 
                        if los-48-eps < t < los]) == 0:
                    print("\n\t(no events in last 48hrs of ICU) ", patient, ts_filename)
                    continue
                
                ts_lines_hourly = [[] for _ in range(math.ceil(los))]
                for hr in range(math.ceil(los)):
                    ts_lines_hourly[hr] = [line for (line, t) in zip(ts_lines, event_times) 
                                           if -eps + hr < t < hr + 1]
                
                #Create File for Each Window Up to 200 Windows per Patient (200 windows randomly selected if over)
                n_windows = math.ceil(los) - n_hours
                if n_windows <= 200:
                    for hr in range(n_windows):
                        end_time = hr + n_hours
                        if hr == 0:
                            ts_lines_sel = list(itertools.chain.from_iterable(ts_lines_hourly[hr:end_time]))
                        else:
                            ts_lines_sel.extend(ts_lines_hourly[end_time])
                            
                        output_ts_filename = "{}_{}_{}.csv".format(patient, ts_filename.split('.')[0], hr)
                        with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                            outfile.write(header)
                            for line in ts_lines_sel:
                                outfile.write(line)

                        data_list.append((output_ts_filename, n_windows, hr))
                        data_list_ts.append((output_ts_filename, n_windows, hr))
                else:
                    windows = list(range(n_windows))
                    random.shuffle(windows)
                    for hr in windows[:200]:
                        end_time = hr + n_hours
                        ts_lines_sel = [line for (line, t) in zip(ts_lines, event_times) 
                                        if -eps < t < end_time]
                        #Try this out next time if needed
                        #ts_lines_sel = list(itertools.chain.from_iterable(ts_lines_hourly[0:end_time]))

                        output_ts_filename = "{}_{}_{}.csv".format(patient, ts_filename.split('.')[0], hr)
                        with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                            outfile.write(header)
                            for line in ts_lines_sel:
                                outfile.write(line)

                        data_list.append((output_ts_filename, 200, hr))
                        data_list_ts.append((output_ts_filename, 200, hr))
                        
                #Randomly Select One Window for PCA
                if len(data_list_ts) >= 1:
                    data_list_visit.append(data_list_ts[random.randint(0, len(data_list_ts)-1)])
                
                #Collect Number of Visits (To be used to Average Loss)
                total += 1

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')

    print("\n", len(data_list))
    if partition == "train":
        data_list = sorted(data_list)
        data_list_visit = sorted(data_list_visit)
        #random.shuffle(data_list)
    if partition == "val":
        data_list = sorted(data_list)
        data_list_visit = sorted(data_list_visit)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('ts_file, total_windows, window\n')
        for (x, y, z) in data_list:
            listfile.write('{},{},{}\n'.format(x, y, z)) 
    
    with open(os.path.join(output_dir, "listfile_visit.csv"), "w") as listfile_visit:
        listfile_visit.write('ts_file, total_windows, window\n')
        for (x, y, z) in data_list_visit:
            listfile_visit.write('{},{},{}\n'.format(x, y, z))
            
    with open(os.path.join(output_dir, "totalfile.csv"), "w") as totalfile:
        totalfile.write(str(total))


def main():
    parser = argparse.ArgumentParser(description="Create data for embedding task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #process_partition(args, "train")
    process_partition(args, "val")


if __name__ == '__main__':
    main()
