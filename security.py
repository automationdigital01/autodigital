import pandas as pd
#import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
import streamlit as st





#requests.get(url="link") #without authentication

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    additional_chars = set(string.punctuation) | set(['/'])
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token not in additional_chars]
    return filtered_tokens

def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def main():
    text = st.text_area('enter text',)

#making dataframes.

# Read the Excel sheet into a DataFrame
    df = pd.read_excel('CatMainRisk.xlsx')

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

    st.dataframe(result_main_risk)



# Read the Excel sheet into a DataFrame
    df = pd.read_excel('CatSubRisk.xlsx')

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

    if result_main_risk.empty:
        # Update result_main_risk with data from result_main_sub_risk
        result_main_risk['Main risk categories']= result_main_sub_risk[['Main risk categories']].copy()

    # Merge the data frames on the common column "Main risk categories"
    merged_df = pd.merge(result_main_risk,result_main_sub_risk,  on="Main risk categories")
    st.dataframe(merged_df)

    #merged_df.to_excel('FinalMainCatSubCat.xlsx', index=False)

if __name__ == "__main__":
    download('words')
    nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
    nltk.download('stopwords')   # Download the stopwords (if not already downloaded)
    main()

