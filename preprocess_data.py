import csv


average_reading_speed = 225



def read_txt(txt_file):
    total_words = 0

    with open(txt_file) as txt_file:
        reader = csv.reader(txt_file, delimiter='\n')
        for row in reader:
            if row in [[]]:
                continue

            # print(row)

            num_words_in_row = len(row[0].split())
            total_words += num_words_in_row


    print()
    print('# words:', total_words)
    print('Time to read agreement: {:.2f} minutes'.format(total_words/average_reading_speed))








if __name__ == '__main__':
    # read_txt('data/facebook_terms.txt')
    read_txt('data/tiktok_terms.txt')
    read_txt('data/microsoft_terms.txt')

