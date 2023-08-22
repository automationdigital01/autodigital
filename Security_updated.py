import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
nltk.download('stopwords')   # Download the stopwords (if not already downloaded)

#text = text_string
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


def main():
    st.title("Security App")
    text = st.text_area('enter text',)
    # Read the Excel sheet into a DataFrame
    dfMainRisk = pd.read_excel('data/CatMainRisk.xlsx')

    # Read the Excel sheet into a DataFrame
    dfSubRisk = pd.read_excel('data/CatSubRisk.xlsx')

    # Read the Excel sheet into a DataFrame
    dfThreat = pd.read_excel('data/CatThreatActor.xlsx')

    # Lemmatize the keywords in the 'Keywords' column of dfThreat
    dfThreat['Keywords'] = dfThreat['Keywords'].apply(lambda x: ' '.join(lemmatize_tokens(word_tokenize(x))))

    #GetMatching Keywords
    main_matching_keywords =ngramprocessing(text, dfMainRisk)
    sub_matching_keywords =ngramprocessing(text, dfSubRisk)
    Threat_matching_keywords =ngramprocessing(text, dfThreat)

    #GetMainRisk
    result_main_risk = main_matching_keywords[['Main risk categories', 'Keywords', 'Matched Tokens']].head(3)
    if result_main_risk.empty:
        # Update result_main_risk with data from result_main_sub_risk
        result_main_risk['Main risk categories']= result_main_sub_risk[['Main risk categories']].copy()
    st.header("Main Risk")
    st.dataframe(result_main_risk)
    csv_main = result_main_risk.to_csv().encode('utf-8')
    st.download_button(label="Download main risk output",data=csv_main,file_name='main_risk.csv',mime='text/csv',)


    #GetSubRisk
    result_main_sub_risk = sub_matching_keywords[['Main risk categories', 'Keywords', 'Matched Tokens']].head(3)
    st.header("Main Sub Risk")
    st.dataframe(result_main_sub_risk)
    csv_sub = result_main_sub_risk.to_csv().encode('utf-8')
    st.download_button(label="Download sub risk output",data=csv_sub,file_name='sub_risk.csv',mime='text/csv',)

    # Extract the corresponding "Main risk category" up to top 3 matches
    result_Threat_risk = Threat_matching_keywords[['Threat actor', 'Keywords', 'Matched Tokens']].head(3)
    st.header("Threat Actor")
    st.dataframe(result_Threat_risk)
    csv_threat = result_Threat_risk.to_csv().encode('utf-8')
    st.download_button(label="Download Threat Actor output",data=csv_threat,file_name='threat_actor.csv',mime='text/csv',)



    # Merge the data frames on the common column "Main risk categories"
    merged_df = pd.merge(result_main_risk,result_main_sub_risk,  on="Main risk categories")
    st.header("Merged")
    st.dataframe(merged_df)
    csv_merged = merged_df.to_csv().encode('utf-8')
    st.download_button(label="Download merged output",data=csv_merged,file_name='merged.csv',mime='text/csv',)


    #Convert to Excel
    #merged_df.to_excel('FinalMainCatSubCat.xlsx', index=False)


    if(len(result_Threat_risk)!=0 and len(merged_df)!=0):
        # Create an empty list to store combined rows
        combined_rows = []

        # Iterate through rows of dfThreat
        for idx_threat, row_threat in result_Threat_risk.iterrows():
            num_repeats = len(merged_df)  # Repeat for each row in merged_df

            # Repeat rows from merged_df for the current threat row
            repeated_merged = pd.concat([merged_df] * num_repeats, ignore_index=True)

            # Add the threat actor and keywords from dfThreat to the repeated_merged
            repeated_merged["Threat actor"] = [row_threat["Threat actor"]] * len(repeated_merged)
            repeated_merged["Keywords"] = [row_threat["Keywords"]] * len(repeated_merged)
            repeated_merged["Matched Tokens"] = [row_threat["Matched Tokens"]] * len(repeated_merged)

            # Append the combined rows to the list
            combined_rows.append(repeated_merged)

        # Concatenate all the combined rows into a final dataframe
        final_df = pd.concat(combined_rows, ignore_index=True)

    if(len(result_Threat_risk)==0):
        final_df =merged_df

    if(len(merged_df)==0):
        final_df =result_Threat_risk

    st.header("Final")    
    st.dataframe(final_df)

    csv_final = final_df.to_csv().encode('utf-8')
    st.download_button(label="Download final output",data=csv_final,file_name='final.csv',mime='text/csv',)
    #print(final_df)

if __name__ == "__main__":
    nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
    nltk.download('stopwords')   # Download the stopwords (if not already downloaded)
    nltk.download('wordnet')
    main()    