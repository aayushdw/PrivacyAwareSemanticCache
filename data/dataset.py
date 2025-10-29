import pandas as pd

csv_file_path = 'questions.csv'
df = pd.read_csv(csv_file_path, escapechar='\\', sep=',')
print(len(df))