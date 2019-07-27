import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
import copy

INPUT_FILE = 'data/' + 'M_anfaelle_mit_patientendaten_kurz.csv'
OUTPUT_FILE = 'data/processed/' + 'processed_scaled.csv'
NORMALIZE_DATA = False
SAVE = False


# This file preprocesses all important data from /data/M_anfaelle_mit_patientendaten.csv and turns it into nurmerical values.
# NeuralNetworks and the SVM are only able to use numerical values. The new data will be turned into a csv in the folder /data.

# Included data
# 0. time_start     OR 21. group_time
# 1. age            OR 20. group_age
# 2. gender
# 3. duration
# 4. intensity
# 5. pain_location
# 6. pain_type
# 7. pain_origin
# 8. uebelkeit
# 9. erbrechen
# 10. lichtempfindlichkeit
# 11. laermempfindlichkeit
# 12. geruchsempfindlichkeit
# 13. ruhebeduerfnis
# 14. bewegungsdrang
# 15. schwindel
# 16. sonstiges
# 17. week_day
# 18. med_1
# 19. med_effect
# 20. group_age
# 21. group_time
# 22. k_type


def process_data():
    # Turns the data into a panda datafile and ignore first row
    df = pd.read_csv(INPUT_FILE, dtype={'Medname6': str, 'Medname7': str})
    dataset = df.values

    # Turns groups of strings into integers (e.g.: female=0, male=1)
    label_encoder = preprocessing.LabelEncoder()

    # Extract all important columns
    gender = label_encoding(label_encoder, dataset[:, 3])
    age = dataset[:, 4]
    begin_time = copy.deepcopy(dataset[:, 5])
    end_time = dataset[:, 6]
    duration = []
    time_start = []

    i = 0
    for date in begin_time:
        begin_time[i] = datetime.strptime(date[:16], '%Y-%m-%d %H:%M')
        time_start.append(datetime.strptime(date[11:16], '%H:%M'))
        i += 1

    i = 0
    for date in end_time:
        end_time[i] = datetime.strptime(date[:16], '%Y-%m-%d %H:%M')
        i += 1

    for i in range(len(begin_time)):
        dur = end_time[i] - begin_time[i]
        duration.append(int(dur.total_seconds() / 60))

        dur2 = time_start[i] - datetime(1900, 1, 1, 0, 0, 0)
        time_start[i] = int(dur2.total_seconds() / 60)

    duration = np.array(duration)
    time_start = np.array(time_start)
    intensity = dataset[:, 7]
    pain_location = label_encoding(label_encoder, dataset[:, 8])
    pain_type = label_encoding(label_encoder, dataset[:, 9])
    pain_origin = label_encoding(label_encoder, dataset[:, 10])

    symptoms = dataset[:, 11]
    uebelkeit = [0] * len(dataset[:, 11])
    erbrechen = [0] * len(dataset[:, 11])
    lichtempfindlichkeit = [0] * len(dataset[:, 11])
    laermempfindlichkeit = [0] * len(dataset[:, 11])
    geruchsempfindlichkeit = [0] * len(dataset[:, 11])
    ruhebeduerfnis = [0] * len(dataset[:, 11])
    bewegungsdrang = [0] * len(dataset[:, 11])
    schwindel = [0] * len(dataset[:, 11])
    sonstiges = [0] * len(dataset[:, 11])

    # 10010100 -> 00101001 -> 101001 -> 101001 000
    idx = -1
    for ele in symptoms:
        idx += 1
        ele = str(int(ele))
        ele = ele[::-1]

        if ele == '0':
            continue

        ele = ele[2:len(ele)]
        col = -1
        for char in ele:
            col += 1
            if char == '1':
                if col == 0:
                    uebelkeit[idx] = 1
                if col == 1:
                    erbrechen[idx] = 1
                if col == 2:
                    lichtempfindlichkeit[idx] = 1
                if col == 3:
                    laermempfindlichkeit[idx] = 1
                if col == 4:
                    geruchsempfindlichkeit[idx] = 1
                if col == 5:
                    ruhebeduerfnis[idx] = 1
                if col == 6:
                    bewegungsdrang[idx] = 1
                if col == 7:
                    schwindel[idx] = 1
                if col == 8:
                    sonstiges[idx] = 1

    week_day = label_encoding(label_encoder, dataset[:, 17])
    med_1 = dataset[:, 23]
    med_effect = dataset[:, 32]

    i = 0
    for ele in med_1:
        if type(ele) != str:
            med_1[i] = 'keins'
        i += 1

    i = 0
    for ele in med_effect:
        if type(ele) != str:
            med_effect[i] = 'keins'
        i += 1

    med_1 = label_encoding(label_encoder, med_1)
    med_effect = label_encoding(label_encoder, med_effect)

    group_age = []
    i = 0
    for a in dataset[:, 4]:
        if a < 21:
            group_age.append(0)
        elif a < 39:
            group_age.append(1)
        elif a < 60:
            group_age.append(2)
        else:
            group_age.append(3)
        i += 1
    group_age = np.array(group_age)

    group_time = []
    i = 0
    for date in dataset[:,5]:
        hours_minutes = date[11:16]
        hours_minutes = int(hours_minutes.replace(':', ''))

        if hours_minutes < 1000 and hours_minutes >= 500:
            group_time.append(0)
        elif hours_minutes < 1700:
            group_time.append(1)
        elif hours_minutes < 2200:
            group_time.append(2)
        else:
            group_time.append(3)
        i += 1
    group_time = np.array(group_time)

    k_type = label_encoding(label_encoder, dataset[:, 34])

    processed_data = np.column_stack((time_start, age))
    processed_data = np.column_stack((processed_data, gender))
    processed_data = np.column_stack((processed_data, duration))
    processed_data = np.column_stack((processed_data, intensity))
    processed_data = np.column_stack((processed_data, pain_location))
    processed_data = np.column_stack((processed_data, pain_type))
    processed_data = np.column_stack((processed_data, pain_origin))

    processed_data = np.column_stack((processed_data, uebelkeit))
    processed_data = np.column_stack((processed_data, erbrechen))
    processed_data = np.column_stack((processed_data, lichtempfindlichkeit))
    processed_data = np.column_stack((processed_data, laermempfindlichkeit))
    processed_data = np.column_stack((processed_data, geruchsempfindlichkeit))
    processed_data = np.column_stack((processed_data, ruhebeduerfnis))
    processed_data = np.column_stack((processed_data, bewegungsdrang))
    processed_data = np.column_stack((processed_data, schwindel))
    processed_data = np.column_stack((processed_data, sonstiges))

    processed_data = np.column_stack((processed_data, week_day))
    processed_data = np.column_stack((processed_data, med_1))
    processed_data = np.column_stack((processed_data, med_effect))
    processed_data = np.column_stack((processed_data, group_age))
    processed_data = np.column_stack((processed_data, group_time))
    processed_data = np.column_stack((processed_data, k_type))

    # Normalize data
    if NORMALIZE_DATA:
        min_max_scaler = preprocessing.MinMaxScaler()
        processed_data = min_max_scaler.fit_transform(processed_data)

    if SAVE:
        np.savetxt(OUTPUT_FILE, processed_data, fmt='%f', delimiter=',')

    show_num_patients_per_age_group_has_aura(group_age, k_type)
    show_num_patients_per_age_group(group_age)
    show_num_patients_with_aura(k_type)


def label_encoding(label_encoder, data):
    label_encoder.fit(data)
    return label_encoder.transform(data)


def show_num_patients_with_aura(k_type):
    has_aura = 0
    has_no_aura = 0

    for data in k_type:
        if data == 0:
            has_aura += 1
        else:
            has_no_aura += 1

    print('Patients that have an aura: ', has_aura)
    print('Patients that have no aura: ', has_no_aura)


def show_num_patients_per_age_group(age_group):
    g0 = 0
    g1 = 0
    g2 = 0
    g3 = 0

    for data in age_group:
        if data == 0:
            g0 += 1
        if data == 1:
            g1 += 1
        if data == 2:
            g2 += 1
        if data == 3:
            g3 += 1

    print('Patients in group zero: ', g0)
    print('Patients in group one: ', g1)
    print('Patients in group two: ', g2)
    print('Patients in group three: ', g3)


def show_num_patients_per_age_group_has_aura(age_group, k_type):
    g0 = 0
    g1 = 0
    g2 = 0
    g3 = 0

    print(k_type)

    i = 0
    for data in age_group:
        if data == 0 and (k_type[i] == 0):
            g0 += 1
        if data == 1 and (k_type[i] == 0):
            g1 += 1
        if data == 2 and (k_type[i] == 0):
            g2 += 1
        if data == 3 and (k_type[i] == 0):
            g3 += 1
        i += 1

    print('Patients in group zero that have aura: ', g0)
    print('Patients in group one that have aura: ', g1)
    print('Patients in group two that have aura: ', g2)
    print('Patients in group three that have aura: ', g3)


if __name__ == '__main__':
    process_data()
