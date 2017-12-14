import csv
import numpy as np

def compare_subs(file1, file2, threshold=0.5):
    with open(file1, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',') 
        next(reader)
        all_rows1 = np.array(list(reader))

        preds1 = ([float(r[1]) for r in all_rows1])
        
    with open(file2, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',') 
        next(reader)
        all_rows2 = np.array(list(reader))

        preds2 = np.asarray([float(r[1]) for r in all_rows2])

    diff = abs(preds1 - preds2)
    
    return np.hstack((all_rows1[diff > threshold], all_rows2[diff > threshold]))