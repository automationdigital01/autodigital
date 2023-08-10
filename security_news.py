import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import geotext
import spacy
import pandas as pd
import streamlit as st
import nltk

# GetLocationForMain

def extract_entities(text):
    # Using geotext to extract cities
    places = geotext.GeoText(text)
   
    cities = list(set(places.cities))
    # Initialize sets to store recognized countries and states
    countries = set()
    states = set()

    # Load the spaCy NER model (en_core_web_lg)
    nlp = spacy.load("en_core_web_lg")

    # Process the text with spaCy NER
    doc = nlp(text)

    # Iterate through each named entity in the text
    for ent in doc.ents:
        # Check if the entity is a country
        if ent.label_ == "GPE" and ent.text not in cities:
            countries.add(ent.text)
        # Check if the entity is a state or province
        elif ent.label_ == "LOC" and ent.text not in cities:
            states.add(ent.text)
            
       
            
     # Extract exception country names
    country_exceptions = ['US','UK','USSR','Turkiye','UAE','USA','ENG']
    for country_name in country_exceptions:
        if country_name in text:
            countries.add(country_name)
    
            
    iraq_governates = ['Anbar','Qadisiyyah', 'Babil', 'Baghdad', 'Basra', 'Dhi Qar', 'Diyala', 'Dohuk', 'Erbil', 'Halabja','Karbala', 'Kirkuk', 'Maysan', 'Muthanna', 'Najaf', 'Nineveh', 'Saladin', 'Sulaymaniyah', 'Wasit']
    egypt_governates = ['Cairo', 'Alexandria', 'Port Said', 'Suez', 'North Sinai', 'South Sinai', 'Ismailia', 'Beheira', 'Giza', 'Faiyum', 'Beni Suef', 'Minya', 'Assiut', 'Sohag','Qena','Aswan','Luxor', 'Red Sea', 'New Valley', 'Matruh', 'Qalyubia', 'Dakahlia', 'Sharqia', 'Monufia', 'Gharbia', 'Kafr El Sheikh', 'Damietta']
    mozambique_governates = ['Zambezia','Tete','Sofala','Niassa','Nampula','Maputo','Maputo City','Manica','Inhambane','Gaza','Cabo Delgado']

    # Add governorates for specific countries
    if "Iraq" in countries:
        for gov in iraq_governates:
            if gov in text:
                states.add(gov)
                
    if "Egypt" in countries:
        for gov in egypt_governates:
            if gov in text:
                states.add(gov)
                
    if "Mozambique" in countries:
        for gov in mozambique_governates:
            if gov in text:
                states.add(gov)
    
    return list(countries),cities , list(states)



def Get_SubjectCountry(text):
    # Using geotext to extract cities
    places = geotext.GeoText(text)
   
    cities = list(set(places.cities))
    # Initialize sets to store recognized countries and states
    countries = set()
    states = set()

    # Load the spaCy NER model (en_core_web_lg)
    nlp = spacy.load("en_core_web_lg")

    # Process the text with spaCy NER
    doc = nlp(text)

    # Iterate through each named entity in the text
    for ent in doc.ents:
        # Check if the entity is a country
        if ent.label_ == "GPE" and ent.text not in cities:
            countries.add(ent.text)           
    # Extract exception country names
    country_exceptions = ['US','UK','USSR','Turkiye','UAE','USA','ENG']
    for country_name in country_exceptions:
        if country_name in text:
            countries.add(country_name)

    return list(countries)
# calling Text Summarization

def text_summarization(text, sentences_count=1):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_text

def main():
    subject_string = st.text_area("enter subject",) #text_subject
    text_string = st.text_area("text",) #text_Input
    total_string =text_string + subject_string
    summary = text_summarization(text_string)
    subject_countries = Get_SubjectCountry(subject_string)
    countries_main, cities_main, states_main = extract_entities(total_string)

    # converting to dataframe
    dfOutput = pd.DataFrame(
        columns=['Country', 'State', 'City', 'Summary', 'SubjectCountry'])
    data = [[countries_main, states_main, cities_main, summary, subject_countries]]
    dfOutput = pd.DataFrame(
        data, columns=['Country', 'State', 'City', 'Summary', 'SubjectCountry'])
    st.dataframe(dfOutput)
    csv = dfOutput.to_csv().encode('utf-8')
    st.download_button(label="Download data as CSV",
                        data=csv,
                        file_name='country_securityanalysis.csv',
                        mime='text/csv',)

    #dfOutput.to_excel("SecurityAnalysis.xlsx")

if __name__ == "__main__":
    nltk.download('punkt')       # Download the punkt tokenizer models (if not already downloaded)
    nltk.download('stopwords')   # Download the stopwords (if not already downloaded)
    main()
