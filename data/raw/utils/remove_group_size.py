import csv

input_file = 'unique_questions_filtered.csv'
output_file = 'unique_questions_filtered_temp.csv'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['qid', 'question'])

    writer.writeheader()
    for row in reader:
        writer.writerow({'qid': row['qid'], 'question': row['question']})

import os
os.replace(output_file, input_file)
print(f"Successfully removed 'group_size' column from {input_file}")
