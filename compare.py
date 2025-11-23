import pandas as pd
from pathlib import Path

txt_file = Path("catboost.txt")
#csv_file = Path("./Classifier Testing/Evaluate Labeled.csv")
csv_file = Path("xgb.txt")

# 1) Läs txt (en etikett per rad)
txt = pd.read_csv(txt_file, header=None, names=["label"], dtype=str)

# 2) Läs csv, utan header, ta bara första kolumnen
csv = list(pd.read_csv(csv_file, names=["label"])["label"])

counter = 0

for i in range(len(txt.label)):
    #print(txt.label[i].strip(), csv.label[i].strip())
    if txt.label[i].strip() == csv[i].strip():
        
        
        counter+=1
    else:
        #print(txt.label[i].strip(), csv[i].strip())
        pass
accuracy = counter / len(txt.label)

print(f"Antal matchande rader: {counter}")
print(f"Accuracy: {accuracy:.4f}")