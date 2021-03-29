# Create NLP pipelines to clean Reviews data
# Load input files and read reviews
# Tokenize
# Remove stopwords
# Perform stemming
# Write cleaned data to output file

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
import os

sample_text = """I loved this movie since I was 7 and I saw it on the opening day. It was so touching and beautiful. I strongly recommend seeing for all. It's a movie to watch with your family by far.<br /><br />My MPAA rating: PG-13 for thematic elements, prolonged scenes of disastor, nudity/sexuality and some language."""

# Initialising Objects
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# Creating a function for cleaning, tokenizing, removing stopwords and stemming the reviews
def getProcessedText(reviews):
    reviews = reviews.lower()

    # Cleaning reviews data that is removing the <br /><br /> from the text
    reviews = reviews.replace("<br /><br />", " ")

    # Tokenize
    tokens = tokenizer.tokenize(reviews)

    # Stopword removal
    new_tokens = [token for token in tokens if token not in en_stopwords]

    # Stemming
    stemmed_tokens = [ps.stem(token) for token in new_tokens]

    # Joining the above list of words
    cleaned_review = ' '.join(stemmed_tokens)

    return cleaned_review


# Creating a function that accepts input file to process data and stores the processed data in an output file
def processDocument(inputFile, outputFile):
    out = open(outputFile, 'w', encoding='utf8')

    with open(inputFile, encoding='utf8') as f:
        reviews = f.readlines()

    for review in reviews:
        cleaned_review = getProcessedText(review)
        print(cleaned_review, file=out)

    out.close()


# Read command line arguments as we will run this script from command line
inputFile = sys.argv[1]
outputFile = sys.argv[2]
cwd = os.getcwd()
inputFile = cwd+'/data/'+inputFile
outputFile = cwd+'/data/'+outputFile
processDocument(inputFile, outputFile)
