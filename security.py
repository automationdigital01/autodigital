import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import re
from dateutil.parser import parse
from dateutil import parser
from nltk.tokenize import sent_tokenize
import spacy
from word2number import w2n
import zipfile
import gdown


def remove_specific_dates(text):
    pattern = r'\w+,\s\w+\s\d{1,2},\s\d{4}\s\d{1,2}:\d{2}:\d{2}\s(?:AM|PM)'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()

def remove_day_month_date(text):
    pattern = r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()

def replace_words_with_numbers(input_string):
    words = input_string.split()
    converted_words = []

    for word in words:
        try:
            # Attempt to convert word to number
            number = w2n.word_to_num(word)
            converted_words.append(str(number))
        except ValueError:
            # If conversion fails, keep the original word
            converted_words.append(word)

    converted_string = ' '.join(converted_words)
    return converted_string

def getresult(sentence):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Process the text using spaCy
    doc = nlp(sentence)

    verbs_and_numbers = []

    # Extract verbs and numbers and store them in a list
    for token in doc:
        if token.pos_ == "VERB":
            verbs_and_numbers.append(token.text)
        elif re.match(r'\d+', token.text):  # Check if the token is a number using regular expression
            verbs_and_numbers.append(token.text)

    # Keywords lists
    injury_keywords = ["injury", "injuries", "injured", "injuring"]
    fatality_keywords = ["death", "deaths", "fatality", "fatalities", "dead", "died", "killing"]

    # Filter and print the items that match the keywords or are numbers
    matched_verbs_and_numbers = []
    for item in verbs_and_numbers:
        if item.lower() in injury_keywords + fatality_keywords or re.match(r'\d+', item):
            matched_verbs_and_numbers.append(item)

    # Lemmatize the verbs
    #lemmatized_list = [lemmatizer.lemmatize(word, pos='v') if word.lower() in injury_keywords + fatality_keywords else word for word in matched_verbs_and_numbers]
    return matched_verbs_and_numbers

def checklist(result):
    listresult=[]   
    if len(result) == 2:
        if ((result[0] == "killing" or result[0] == "injuring") and result[1].isdigit()):
            print("Condition21 met:", result[0], result[1])
            listresult.append(result[0]+" " + result[1])
        if (result[0].isdigit() and result[1] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition22 met:", result[0], result[1])
            listresult.append(result[0]+" " +result[1])
            print("text")
            print(result[0]+" " +result[1])
        
    if len(result) == 3:
        if ((result[0] == "killing" or result[0] == "injuring") and result[1].isdigit()):
            print("Condition31 met:", result[0], result[1])
            listresult.append(result[0]+" " +result[1])
        if ((result[1] == "killing" or result[1] == "injuring") and result[2].isdigit()):
            print("Condition311 met:", result[1], result[2])
            listresult.append(result[1]+" " +result[2])
        if (result[1].isdigit() and result[2] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition34 met:", result[1], result[2])
            listresult.append(result[1]+" " +result[2])
        if (result[0].isdigit() and result[1] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition35 met:", result[0], result[1])            
            listresult.append(result[0]+" " +result[1])
            print(listresult)
        if (result[1].isdigit() and result[2] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition36 met:", result[1], result[2])
            listresult.append(result[1]+" " +result[2])
        
    if len(result) == 4:
        if ((result[0] == "killing" or result[0] == "injuring") and result[1].isdigit()):
            print("Condition41 met:", result[0], result[1])
            listresult.append(result[0]+" " +result[1])
        if ((result[1] == "killing" or result[1] == "injuring") and result[2].isdigit()):
            print("Condition42 met:", result[1], result[2])
            listresult.append(result[1]+" " +result[2])
        if (result[1].isdigit() and result[2] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition43 met:", result[1], result[2])
            listresult.append(result[1]+" " +result[2])
        if (result[0].isdigit() and result[1] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition44 met:", result[0], result[1])
            listresult.append(result[0]+" " +result[1])
        if (result[2].isdigit() and result[3] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition423 met:", result[2], result[3])
            listresult.append(result[2]+" " +result[3])
        if (result[1].isdigit() and result[2] in ["injury", "injuries","injured","death", "deaths", "fatality", "fatalities","dead", "died"]):
            print("Condition2 met:", result[0], result[1])
            listresult.append(result[0]+" " +result[1])
        if (result[2] == "killing" or result[2] == "injuring") and result[3].isdigit():
            print("Condition423 met:", result[2], result[3]) 
            listresult.append(result[2]+" " +result[3])
    return listresult

# Step 3: Remove all dates using dateutil
def remove_dates(text):
    try:
        parsed_date = parser.parse(text)
        return "" if parsed_date else text
    except (ValueError, OverflowError):
        return text

#cleaned_text = ' '.join(remove_dates(word) for word in text_without_newlines_tabs.split())
def tokenize_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

def contains_keywords(sentence, keywords):
    for keyword in keywords:
        if keyword in sentence.lower():
            return True
    return False

# Function to replace keywords
def replace_keywords(text, keyword_list, replacement):
    for keyword in keyword_list:
        if keyword in text:
            return replacement
    return text


def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    additional_chars = set(string.punctuation) | set(['/'])
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token not in additional_chars]
    return filtered_tokens

def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower(), pos='n') for token in tokens]

def threat_actor(text):
    dfThreat= pd.read_excel('data/CatThreatActor.xlsx')# Read the Excel sheet into a DataFrame

    # Lemmatize the keywords in the 'Keywords' column of dfThreat
    dfThreat['Keywords'] = dfThreat['Keywords'].apply(lambda x: ' '.join(lemmatize_tokens(word_tokenize(x))))
    # Tokenize, remove stopwords, and specified characters
    filtered_tokens = remove_stopwords(text)
    
    # Lemmatize the tokens (convert to singular form)
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    
    # Get individual tokens, bigrams, trigrams, 4-grams, and 5-grams from the filtered tokens
    individual_tokens = lemmatized_tokens
    bigrams = get_ngrams(lemmatized_tokens, 2)
    trigrams = get_ngrams(lemmatized_tokens, 3)
    four_grams = get_ngrams(lemmatized_tokens, 4)
    five_grams = get_ngrams(lemmatized_tokens, 5)
    
    # Convert n-grams to lowercase strings for comparison
    all_ngrams = [gram if isinstance(gram, str) else ' '.join(gram) for gram in individual_tokens + bigrams + trigrams + four_grams + five_grams]
    
    # Find the matching keywords in the "Keywords" column
    matching_keywords = dfThreat[dfThreat['Keywords'].apply(lambda x: any(ngram == x.lower() for ngram in all_ngrams))]
    
    # Sort by the number of matched tokens in descending order
    matching_keywords['Matched Tokens'] = matching_keywords['Keywords'].apply(lambda x: sum(1 for ngram in all_ngrams if ngram == x.lower()))
    matching_keywords = matching_keywords.sort_values(by='Matched Tokens', ascending=False)
    
    # Extract the corresponding "Main risk category" up to top 3 matches
    result_Threat_risk = matching_keywords[['Threat actor', 'Keywords', 'Matched Tokens']].head(3)
    st.header("Threat Actor")
    st.dataframe(result_Threat_risk)
    csv = result_Threat_risk.to_csv().encode('utf-8')
    st.download_button(label="Download Threat Actor output",data=csv,file_name='threat_actor.csv',mime='text/csv',)

def mainsubrisk(text):
        df = pd.read_excel('data/CatMainRisk.xlsx')
        #st.header("Main Risk Category")
        #st.dataframe(df)
    
# Tokenize, remove stopwords, and specified characters
        filtered_tokens = remove_stopwords(text)



# Get individual tokens, bigrams, trigrams, 4-grams, and 5-grams from the filtered tokens
        individual_tokens = filtered_tokens
        bigrams = get_ngrams(filtered_tokens, 2)
        trigrams = get_ngrams(filtered_tokens, 3)
        four_grams = get_ngrams(filtered_tokens, 4)
        five_grams = get_ngrams(filtered_tokens, 5)

# Convert n-grams to lowercase strings for comparison
        all_ngrams = [gram.lower() if isinstance(gram, str) else ' '.join(gram).lower() for gram in individual_tokens + bigrams + trigrams + four_grams + five_grams]

# Find the matching keywords in the "Keywords" column
        matching_keywords = df[df['Keywords'].apply(lambda x: any(ngram == x.lower() for ngram in all_ngrams))]
        
# Sort by the number of matched tokens in descending order
        matching_keywords['Matched Tokens'] = matching_keywords['Keywords'].apply(lambda x: sum(1 for ngram in all_ngrams if ngram == x.lower()))
        matching_keywords = matching_keywords.sort_values(by='Matched Tokens', ascending=False)

# Extract the corresponding "Main risk category" up to top 3 matches
        result_main_risk = matching_keywords[['Main risk categories', 'Keywords', 'Matched Tokens']].head(3)
        st.header("Main Risk Category Matched Tokens")

        st.dataframe(result_main_risk)



# Read the Excel sheet into a DataFrame
        df = pd.read_excel('data/CatSubRisk.xlsx')

# Tokenize, remove stopwords, and specified characters
        filtered_tokens = remove_stopwords(text)

# Get individual tokens, bigrams, trigrams, 4-grams, and 5-grams from the filtered tokens
        individual_tokens = filtered_tokens
        bigrams = get_ngrams(filtered_tokens, 2)
        trigrams = get_ngrams(filtered_tokens, 3)
        four_grams = get_ngrams(filtered_tokens, 4)
        five_grams = get_ngrams(filtered_tokens, 5)

# Convert n-grams to lowercase strings for comparison
        all_ngrams = [gram.lower() if isinstance(gram, str) else ' '.join(gram).lower() for gram in individual_tokens + bigrams + trigrams + four_grams + five_grams]

# Find the matching keywords in the "Keywords" column
        matching_keywords = df[df['Keywords'].apply(lambda x: any(ngram == x.lower() for ngram in all_ngrams))]

# Sort by the number of matched tokens in descending order
        matching_keywords['Matched Tokens'] = matching_keywords['Keywords'].apply(lambda x: sum(1 for ngram in all_ngrams if ngram == x.lower()))
        matching_keywords = matching_keywords.sort_values(by='Matched Tokens', ascending=False)

# Extract the corresponding "Main risk category" and "Sub-risk categories" up to top 3 matches
        result_main_sub_risk = matching_keywords[['Main risk categories', ' Sub-Risk categories', 'Keywords', 'Matched Tokens']].head(3)
        st.header("Main Sub Risk")
        st.dataframe(result_main_sub_risk)

        if result_main_risk.empty:
        # Update result_main_risk with data from result_main_sub_risk
            result_main_risk['Main risk categories']= result_main_sub_risk[['Main risk categories']].copy()

    # Merge the data frames on the common column "Main risk categories"
        merged_df = pd.merge(result_main_risk,result_main_sub_risk,  on="Main risk categories")
        st.header("Merged")
        st.dataframe(merged_df)
        csv = merged_df.to_csv().encode('utf-8')
        st.download_button(label="Download Main/Sub risk output",data=csv,file_name='merged_risk.csv',mime='text/csv',)

def fatality(text):
    
    # Apply the functions sequentially
    text = remove_specific_dates(text)
    text = remove_day_month_date(text)
    text = replace_words_with_numbers(text)
    
    # Step 1: Remove links
    text=text.replace("at least ",' ')
    text =text.replace("is",' ')
    text_without_links = re.sub(r'http[s]?://\S+', '', text)

    # Step 2: Remove newlines and tabs
    text_without_newlines_tabs = text_without_links.replace('\n', ' ').replace('\t', ' ')
    injury_keywords = ["injury", "injuries","injured","injuring"]
    fatality_keywords = ["death", "deaths", "fatality", "fatalities","dead", "died" ,"killing"]

    listfinal = []
    sentences = tokenize_into_sentences(text_without_newlines_tabs)
    for i, sentence in enumerate(sentences, 1):
        if contains_keywords(sentence.lower(), injury_keywords + fatality_keywords):
            result = getresult(sentence)
            result = [word.lower() for word in result]
            listfinal +=checklist(result)
                    
        # Initialize lists to store the extracted information
        numbers_list = []
        text_list = []

    for item in listfinal:
        # Extract numbers from the item
        numbers = [int(s) for s in item.split() if s.isdigit()]
        
        if numbers:
            numbers_list.append(numbers[0])
            text_list.append(item.replace(str(numbers[0]), '').strip())
        else:
            numbers_list.append(0)
            text_list.append(item)

    # Create a DataFrame
    dffinalfatal = pd.DataFrame({ 'KilledInjured': text_list , 'Numbers': numbers_list,})
    # Apply keyword replacement to the column
    dffinalfatal['KilledInjured'] = dffinalfatal['KilledInjured'].apply(lambda x: replace_keywords(x, fatality_keywords, 'killed'))
    dffinalfatal['KilledInjured'] = dffinalfatal['KilledInjured'].apply(lambda x: replace_keywords(x, injury_keywords, 'injured'))
    st.header("Fatality Info")
    st.dataframe(dffinalfatal)
    csv = dffinalfatal.to_csv().encode('utf-8')
    st.download_button(label="Download Fatality output",data=csv,file_name='fatality.csv',mime='text/csv',)


def main():
    st.title("Security App")
    text = st.text_area('enter text',)
    if text:
        model_url = "en_core_web_sm"
        #model_file = "en_core_web_sm1.zip"
        #gdown.download(model_url, model_file, quiet=False)
        #with zipfile.ZipFile(model_file, "r") as zip_ref:
         #   zip_ref.extractall("en_core_web_sm")

        model_path = "./en_core_web_sm-3.5.0"
        nlp = spacy.load(model_path)
    
    if st.button("Submit"):
        mainsubrisk(text)
        threat_actor(text)
        fatality(text)


if __name__ == "__main__":
    nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
    nltk.download('stopwords')   # Download the stopwords (if not already downloaded)
    nltk.download('wordnet')
    
    
    main()
