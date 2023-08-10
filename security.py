import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st



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
    dfThreat= pd.read_excel('data/CatMainRisk.xlsx')# Read the Excel sheet into a DataFrame

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
    
    st.dataframe(result_Threat_risk)
    #csv = result_Threat_risk.to_csv().encode('utf-8')
    #st.download_button(label="Download Threat Actor output",data=csv,file_name='merged_risk.csv',mime='text/csv',)

def main_subrisk(text):
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
        st.download_button(label="Download Main/Sub risk output",
                           data=csv,
                           file_name='merged_risk.csv',
                           mime='text/csv',)

        

def main():
    st.title("Security App")
    text = st.text_area('enter text',)
    main_subrisk=st.checkbox("Main/Sub Risk")
    threatactor=st.checkbox("Threat Actor")
    if st.button("Submit") and main_subrisk and threatactor:
        main_subrisk(text)
        threat_actor(text)
        
    elif st.button("Submit") and threatactor:
        threat_actor(text)
        
    elif st.button("Submit") and main_subrisk:
        main_subrisk(text)


    
          
    
        
    

if __name__ == "__main__":
    nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
    nltk.download('stopwords')   # Download the stopwords (if not already downloaded)
    main()
