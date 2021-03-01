import csv

def write_to_file(filename, data):
    with open(filename, "w") as f:
        for text in data:
            f.write("%s\n" % text)

def write_to_csv(filename, data, authors, works):
    with open(filename, 'a') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for text, author, work in zip(data, authors, works):
            csv_writer.writerow([text, author, work])

def write_spurious_to_csv(filename, data, works):
    with open(filename, 'a') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for text, work in zip(data, works):
            csv_writer.writerow([text, work])