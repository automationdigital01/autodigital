import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
from fuzzywuzzy import fuzz


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

def ngramprocessing(text,df):
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
    all_ngrams = [gram.lower() if isinstance(gram, str) else ' '.join(gram).lower() for gram in individual_tokens + bigrams + trigrams + four_grams + five_grams]

    # Find the matching keywords in the "Keywords" column main
    matching_keywords = df[df['Keywords'].apply(lambda x: any(ngram == x.lower() for ngram in all_ngrams))]
    print(matching_keywords)

    # Sort by the number of matched tokens in descending order
    matching_keywords['Matched Tokens'] = matching_keywords['Keywords'].apply(lambda x: sum(1 for ngram in all_ngrams if ngram == x.lower()))
    matching_keywords = matching_keywords.sort_values(by='Matched Tokens', ascending=False)

    # Extract the corresponding "Main risk category" up to top 3 matches
    
    
    return matching_keywords

def load_keywords_from_excel(file_path):
    df = pd.read_excel(file_path)
    keywords = df.set_index('Keywords').to_dict()['Threat actor']
    return keywords

def find_threat_actor(input_string, keywords):
    max_ratio = 0
    threat_actor = None

    for keyword, actor in keywords.items():
        ratio = fuzz.partial_ratio(input_string.lower(), keyword.lower())
        if ratio > max_ratio:
            max_ratio = ratio
            threat_actor = actor
    return threat_actor

def GetThreatActor(text):
    excel_file_path_master = 'data/CatThreatActor.xlsx'
    # excel_file_path_actor = 'CatThreatActor.xlsx'

    # Load the master category Excel into a DataFrame
    master_df = pd.read_excel(excel_file_path_master)

    # Extract keywords from the master DataFrame
    keywords = master_df.set_index('Keywords').to_dict()['Threat actor']

    # Identify the threat actor based on input text and keywords
    threat_actor = find_threat_actor(text, keywords)

    if threat_actor:
        #print(f"Threat Actor: {threat_actor}")
      # Create DataFrame with matched result
        result_df = pd.DataFrame({'Threat Actor': [threat_actor], 'Keyword': [next((k for k, v in keywords.items() if v == threat_actor), None)]})
        #result_df.to_excel('ThreatCategoryResult.xlsx', index=False)
        #print("Result saved to 'ThreatCategoryResult.xlsx'")

  
    
    st.header("Threat Actor")
    st.dataframe(result_df)
    csv = result_df.to_csv().encode('utf-8')
    st.download_button(label="Download Threat Actor output",data=csv,file_name='threat_actor.csv',mime='text/csv',)

def mainsubrisk(text):
    # Read the Excel sheet into a DataFrame
    dfMainRisk = pd.read_excel('data/CatMainRisk.xlsx')

    # Read the Excel sheet into a DataFrame
    dfSubRisk = pd.read_excel('data/CatSubRisk.xlsx')
        
    main_matching_keywords =ngramprocessing(text, dfMainRisk)
    #print(main_matching_keywords)
    sub_matching_keywords =ngramprocessing(text, dfSubRisk)
        
    result_main_risk = main_matching_keywords[['Main risk categories', 'Keywords', 'Matched Tokens']].head(3)
    st.header("Main Risk")
    st.dataframe(result_main_risk)
    csv_main = result_main_risk.to_csv().encode('utf-8')
    st.download_button(label="Download main risk output",data=csv_main,file_name='main_risk.csv',mime='text/csv',)

    result_main_sub_risk = sub_matching_keywords[['Main risk categories', ' Sub-Risk categories', 'Keywords', 'Matched Tokens']].head(3)
    st.header("Main Sub Risk")
    st.dataframe(result_main_sub_risk)
    csv_sub = result_main_sub_risk.to_csv().encode('utf-8')
    st.download_button(label="Download sub risk output",data=csv_sub,file_name='sub_risk.csv',mime='text/csv',)
    

    # Merge the data frames on the common column "Main risk categories"
    merged_df = pd.merge(result_main_risk,result_main_sub_risk,  on="Main risk categories")

    if(len(result_main_sub_risk)>0 and len(result_main_risk)>0):
        if len(merged_df) == 0:
            # Create a new row with the values you want
            new_row = {
                'Main risk categories': 'Unmatched',
                'Sub-Risk categories': 'Unmatched',
                }

            # Append the new row to the DataFrame
            merged_df = merged_df.append(new_row, ignore_index=True)
            
    if(len(result_main_sub_risk)>0 or len(result_main_risk)>0):
        if len(merged_df) == 0:
            # Create a new row with the values you want
            new_row = {
                'Main risk categories': 'KeywordIssue',
                'Sub-Risk categories': 'KeywordIssue',
                }

            # Append the new row to the DataFrame
            merged_df = merged_df.append(new_row, ignore_index=True)
    
    st.header("Merged")
    st.dataframe(merged_df)
    csv_merged = merged_df.to_csv().encode('utf-8')
    st.download_button(label="Download merged output",data=csv_merged,file_name='merged.csv',mime='text/csv',)

       

def main():
    st.title("Security App")
    text = st.text_area('enter text',)
    if st.button("Submit"):
        mainsubrisk(text)
        GetThreatActor(text)
        
        

      
if __name__ == "__main__":
    nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
    nltk.download('stopwords')   # Download the stopwords (if not already downloaded)
    nltk.download('wordnet')
    main()
