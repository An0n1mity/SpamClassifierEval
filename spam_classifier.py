import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import numpy as np
import re

def train_model(training_percent=0.8, SPAM_FOLDER='HAMS', HAM_FOLDER='SPAMS'):
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    # Make a dictionary : word | count in SPAM | count in HAM | P(word|SPAM) | P(word|HAM) | P(SPAM|word)
    word_count_dict = {}

    files = os.listdir(SPAM_FOLDER)
    # Take 99% of the files for training randomly
    training_files_spams = np.random.choice(files, int(len(files) * training_percent), replace=False)
    # Take the rest for testing
    testing_files_spams = [SPAM_FOLDER + '/' + file for file in files if file not in training_files_spams]

    files = os.listdir(HAM_FOLDER)
    # Take 90% of the files for training randomly
    training_files_hams = np.random.choice(files, int(len(files) * training_percent), replace=False)
    # Take the rest for testing
    testing_files_hams = [HAM_FOLDER + '/' + file for file in files if file not in training_files_hams]


    # Store them in a new file
    with open('training_files_spams.txt', 'w') as f:
        for file in training_files_spams:
            f.write(file + '\n')

    with open('testing_files_spams.txt', 'w') as f:
        for file in testing_files_spams:
            f.write(file + '\n')

    files = os.listdir(HAM_FOLDER)
    # Take 90% of the files for training randomly
    training_files_hams = np.random.choice(files, int(len(files) * training_percent), replace=False)
    # Take the rest for testing
    testing_files_hams = [HAM_FOLDER + '/' + file for file in files if file not in training_files_hams]

    # Store them in a new file
    with open('training_files_hams.txt', 'w') as f:
        for file in training_files_hams:
            f.write(file + '\n')

    with open('testing_files_hams.txt', 'w') as f:
        for file in testing_files_hams:
            f.write(file + '\n')

    pattern = r'<[^<]+?>|\d+|[^\w\s]|[^\x00-\x7F]+|http\S+'

    for filename in training_files_spams:
        with open(os.path.join(SPAM_FOLDER, filename), 'r', errors="ignore") as f:
            file_words = f.read()
            # Remove pattern
            file_words = re.sub(pattern, '', file_words)
            file_words = file_words.lower().split()

            for word in file_words:
                if word not in stop_words:
                    # Lemmatize the word
                    word = lemmatizer.lemmatize(word)
                    if word not in word_count_dict:
                        word_count_dict[word] = {'SPAM': 1, 'HAM': 0}
                    else:
                        word_count_dict[word]['SPAM'] += 1

    for filename in training_files_hams:
        with open(os.path.join(HAM_FOLDER, filename), 'r', errors="ignore") as f:
            file_words = f.read()
            # # Remove pattern
            file_words = re.sub(pattern, '', file_words)
            file_words = file_words.lower().split()
            for word in file_words:
                if word not in stop_words:
                    # Lemmatize the word
                    word = lemmatizer.lemmatize(word)
                    if word not in word_count_dict:
                        word_count_dict[word] = {'SPAM': 0, 'HAM': 1}
                    else:
                        word_count_dict[word]['HAM'] += 1

    # normalize the counts
    for word in word_count_dict:
        word_count_dict[word]['SPAM'] /= len(training_files_spams)
        word_count_dict[word]['HAM'] /= len(training_files_hams)

    # Calculate the probabilities
    for word in word_count_dict:
        word_count_dict[word]['P(word|SPAM)'] = word_count_dict[word]['SPAM'] / (word_count_dict[word]['SPAM'] + word_count_dict[word]['HAM'])
        word_count_dict[word]['P(word|HAM)'] = word_count_dict[word]['HAM'] / (word_count_dict[word]['SPAM'] + word_count_dict[word]['HAM'])
        word_count_dict[word]['P(SPAM|word)'] = word_count_dict[word]['P(word|SPAM)'] / (word_count_dict[word]['P(word|SPAM)'] + word_count_dict[word]['P(word|HAM)'])
        if word_count_dict[word]['P(SPAM|word)'] == 0:
            word_count_dict[word]['P(SPAM|word)'] = 0.01
        elif word_count_dict[word]['P(SPAM|word)'] == 1:
            word_count_dict[word]['P(SPAM|word)'] = 0.99

    # Store the dictionary in a file not binary
    with open('word_count_dict.txt', 'w') as f:
        # Write the header
        f.write('word SPAM HAM P(word|SPAM) P(word|HAM) P(SPAM|word)\n')
        for word in word_count_dict:
            f.write(word + ' ' + str(word_count_dict[word]['SPAM']) + ' ' + str(word_count_dict[word]['HAM']) + ' ' + str(word_count_dict[word]['P(word|SPAM)']) + ' ' + str(word_count_dict[word]['P(word|HAM)']) + ' ' + str(word_count_dict[word]['P(SPAM|word)']) + '\n')

# New file for classification should be split in words and spamicity of new word is set to 0.4
# If word is not in dictionary, spamicity is set to 0.4
# Open file test.txt 

def get_file_spamicity(filename, n=8, plot=False):
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    # Make a dictionary : word | count in SPAM | count in HAM | P(word|SPAM) | P(word|HAM) | P(SPAM|word)
    word_count_dict = {}
    # Read word count dictionary from file
    word_count_dict = {}
    with open('word_count_dict.txt', 'r') as f:
        for line in f:
            if not line.startswith('word'):
                line = line.split()
                word_count_dict[line[0]] = {'SPAM': float(line[1]), 'HAM': float(line[2]), 'P(word|SPAM)': float(line[3]), 'P(word|HAM)': float(line[4]), 'P(SPAM|word)': float(line[5])}

    new_file_spamicities = {}
    pattern = r'<[^<]+?>|\d+|[^\w\s]|[^\x00-\x7F]+|http\S+'
    with open(os.path.join(filename), 'r', errors="ignore") as f:
        file_words = f.read()
        # Remove patterns
        file_words = re.sub(pattern, '', file_words)
        # Converting to Lowercase and split to get the words
        file_words = file_words.lower().split()

        for word in file_words:
            if word not in stop_words:
                # Lemmatize the word
                word = lemmatizer.lemmatize(word)
                if word in word_count_dict:
                    spamicity = word_count_dict[word]['P(SPAM|word)']
                    new_file_spamicities[word] = {'P(SPAM|word)': spamicity}
                else:
                    new_file_spamicities[word] = {'P(SPAM|word)': 0.4}
                
    # calculate mean of spamicities
    mean_spamicity = sum(new_file_spamicities[word]['P(SPAM|word)'] for word in new_file_spamicities) / len(new_file_spamicities)

    # Take the n/2 words with highest spamicity above mean and the n/2 words with lowest spamicity below mean
    word_with_lowest_spamicity = [word for word in new_file_spamicities if new_file_spamicities[word]['P(SPAM|word)'] < mean_spamicity]
    word_with_highest_spamicity = [word for word in new_file_spamicities if new_file_spamicities[word]['P(SPAM|word)'] > mean_spamicity]

    # order words with highest spamicity above mean
    word_with_highest_spamicity = sorted(word_with_highest_spamicity, key=lambda x: new_file_spamicities[x]['P(SPAM|word)'], reverse=True)
    # order words with lowest spamicity below mean
    word_with_lowest_spamicity = sorted(word_with_lowest_spamicity, key=lambda x: new_file_spamicities[x]['P(SPAM|word)'])

    # Take the n/2 words with highest spamicity above mean and the n/2 words with lowest spamicity below mean
    word_with_lowest_spamicity = word_with_lowest_spamicity[:int(n/2)]
    word_with_highest_spamicity = word_with_highest_spamicity[:int(n/2)]

    words_with_high_deviation = word_with_highest_spamicity + word_with_lowest_spamicity
    # sort in ascending order
    words_with_high_deviation = sorted(words_with_high_deviation, key=lambda x: new_file_spamicities[x]['P(SPAM|word)'])

    # calulate the spamicity of the file based on the words with strong deviation from mean
    # rule p_1 * p_2 * ... * p_n / (p_1 * p_2 * ... * p_n + (1 - p_1) * (1 - p_2) * ... * (1 - p_n))
    # where p_i is the spamicity of the word i

    # calculate the numerator
    numerator = 1
    denominator = 1
    l = 1

    for word in words_with_high_deviation:
        spamicity = new_file_spamicities[word]['P(SPAM|word)']
        numerator *= spamicity
        denominator *= spamicity
        l *= (1 - spamicity)

    denominator += l

    if plot:
        fig, ax = plt.subplots()
        #Plot a hline where xaxis is the spamicity
        ax.hlines(y=0, xmin=0, xmax=1, color='b', linestyle='solid', linewidth=2)
        for word in new_file_spamicities:
            ax.axvline(x=new_file_spamicities[word]['P(SPAM|word)'], ymin=0.45, ymax=0.55, color='b', linestyle='solid', linewidth=2)
        for word in words_with_high_deviation:
            spamicity_ = new_file_spamicities[word]['P(SPAM|word)']
            # Change color of the segment
            color = 'r' if word in word_with_highest_spamicity else 'g'
            ax.axvline(spamicity_, ymin=0.45, ymax=0.55, color=color, linestyle='solid', linewidth=2, label=word + ' ' + '{:.3f}'.format(spamicity_))
        
        ax.axvline(x=mean_spamicity, ymin=0.45, ymax=0.55, color='y', linestyle='solid', linewidth=2)
        ax.set_title('Selecting a threshold for spamicity for n = ' + str(n))
        # Add green legend for words with lowest spamicity below mean

        # remove ticks from y axis
        ax.set_xlabel('Spamicity')

        ax.set_yticks([])
        plt.text(0.5, 0.01, "File spamicity = " + str(spamicity), ha='center', va='bottom', transform=ax.transAxes)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()

    return spamicity

# test misclassification for a given n
def test_misclassification(testing_files_spams, testing_files_hams, n = (8, 16, 32), threshold=0.6, unseen_spamicity = 0.4, plot = False, verbose = False):
    nb_false_positives = [0] * len(n)
    nb_true_negatives = [0] * len(n)

    pattern = r'<[^<]+?>|\d+|[^\w\s]|[^\x00-\x7F]+|http\S+'
    testing_files = testing_files_spams + testing_files_hams
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    
    # Read word count dictionary from file
    word_count_dict = {}
    with open('word_count_dict.txt', 'r') as f:
        for line in f:
            if not line.startswith('word'):
                line = line.split()
                word_count_dict[line[0]] = {'SPAM': float(line[1]), 'HAM': float(line[2]), 'P(word|SPAM)': float(line[3]), 'P(word|HAM)': float(line[4]), 'P(SPAM|word)': float(line[5])}

    for file in testing_files:
        new_file_spamicities = {}
        with open(file, 'r', errors='ignore') as f:
            file_words = f.read()
            # Remove patterns
            file_words = re.sub(pattern, '', file_words)
            # Converting to Lowercase and split to get the words
            file_words = file_words.lower().split()

            for word in file_words:
                if word not in stop_words:
                    # Lemmatize the word
                    word = lemmatizer.lemmatize(word)
                    if word in word_count_dict:
                        spamicity = word_count_dict[word]['P(SPAM|word)']
                        new_file_spamicities[word] = {'P(SPAM|word)': spamicity}
                    else:
                        new_file_spamicities[word] = {'P(SPAM|word)': unseen_spamicity}

        # calculate mean of spamicities
        mean_spamicity = sum(new_file_spamicities[word]['P(SPAM|word)'] for word in new_file_spamicities) / len(new_file_spamicities)
        words_with_lowest_spamicity = tuple([] for _ in range(len(n)))
        words_with_highest_spamicity = tuple([] for _ in range(len(n)))
        # Take the n/2 words with highest spamicity above mean and the n/2 words with lowest spamicity below mean
        word_with_lowest_spamicity = [word for word in new_file_spamicities if new_file_spamicities[word]['P(SPAM|word)'] < mean_spamicity]
        word_with_highest_spamicity = [word for word in new_file_spamicities if new_file_spamicities[word]['P(SPAM|word)'] > mean_spamicity]

        # order words with highest spamicity above mean
        word_with_highest_spamicity = sorted(word_with_highest_spamicity, key=lambda x: new_file_spamicities[x]['P(SPAM|word)'], reverse=True)
        # order words with lowest spamicity below mean
        word_with_lowest_spamicity = sorted(word_with_lowest_spamicity, key=lambda x: new_file_spamicities[x]['P(SPAM|word)'])

        # Take the n/2 words with highest spamicity above mean and the n/2 words with lowest spamicity below mean
        words_with_lowest_spamicity = tuple(word_with_lowest_spamicity[:int(n[i]/2)] for i in range(len(n)))        
        words_with_highest_spamicity = tuple(word_with_highest_spamicity[:int(n[i]/2)] for i in range(len(n))) 

        words_with_high_deviation = tuple(words_with_lowest_spamicity[i] + words_with_highest_spamicity[i] for i in range(len(n)))
        # sort in ascending order
        #words_with_high_deviation = sorted(words_with_high_deviation, key=lambda x: new_file_spamicities[x]['P(SPAM|word)'])

        # calulate the spamicity of the file based on the words with strong deviation from mean
        # rule p_1 * p_2 * ... * p_n / (p_1 * p_2 * ... * p_n + (1 - p_1) * (1 - p_2) * ... * (1 - p_n))
        # where p_i is the spamicity of the word i

        # calculate the numerator
        numerators = [1] * len(n)
        denominators = [1] * len(n)
        ls = [1] * len(n)
        spamicities = [0] * len(n)

        for i in range(len(n)):
            for word in words_with_high_deviation[i]:
                spamicity = new_file_spamicities[word]['P(SPAM|word)']
                numerators[i] *= spamicity
                denominators[i] *= spamicity
                ls[i] *= (1 - spamicity)

            denominators[i] += ls[i]
            spamicities[i] = numerators[i]/denominators[i]

            if file in testing_files_spams and spamicities[i] < threshold:
                nb_false_positives[i] += 1
            elif file in testing_files_hams and spamicities[i] > threshold:
                nb_true_negatives[i] += 1

    if plot:
        x = np.arange(len(n))
        plt.bar(x, nb_false_positives, width=0.4, color='r', label='False Positives')
        plt.bar(x + 0.4, nb_true_negatives, width=0.4, color='g', label='True Negatives')
        plt.legend()
        plt.xticks(x + 0.2, ['n=' + str(v) for v in n])
        plt.xlabel('n')
        plt.ylabel('Number of misclassifications')
        plt.show()

        # plot with percent 
        plt.bar(x, [nb_false_positives[i]/len(testing_files_spams) for i in range(len(n))], width=0.4, color='r', label='False Positives')
        plt.bar(x + 0.4, [nb_true_negatives[i]/len(testing_files_hams) for i in range(len(n))], width=0.4, color='g', label='True Negatives')
        plt.legend()
        plt.xticks(x + 0.2, ['n=' + str(v) for v in n])
        plt.xlabel('n')
        plt.ylabel('Percent of misclassifications')
        plt.show()

    if verbose:
        print('Threshold:', threshold)
        print('n =', n)
        print('unseen spamicity'  , unseen_spamicity)
        for i in range(len(n)):
            print('n =', n[i])
            print('Number of false positives:', nb_false_positives[i])
            print('Number of true negatives:', nb_true_negatives[i])
            print('Percent of false positives:', nb_false_positives[i]/len(testing_files_spams))
            print('Percent of true negatives:', nb_true_negatives[i]/len(testing_files_hams))
            print('')

if __name__ == '__main__':
    # Train the model
    train_model()
    # read testing files
    with open('testing_files_hams.txt', 'r') as f:
        testing_files_hams = f.read().splitlines()

    with open('testing_files_spams.txt', 'r') as f:
        testing_files_spams = f.read().splitlines()

    test_file = np.random.choice(testing_files_hams)
    test_spamicity = get_file_spamicity(test_file, n=8, plot=True)
    test_spamicity = get_file_spamicity(test_file, n=16, plot=True)
    test_spamicity = get_file_spamicity(test_file, n=32, plot=True)

    test_file = np.random.choice(testing_files_spams)
    test_spamicity = get_file_spamicity(test_file, n=8, plot=True)
    test_spamicity = get_file_spamicity(test_file, n=16, plot=True)
    test_spamicity = get_file_spamicity(test_file, n=32, plot=True)

    test_misclassification(testing_files_spams, testing_files_hams, n=(8, 16, 32, 64, 128), threshold=0.4, unseen_spamicity=0.2, plot=True, verbose=True)

