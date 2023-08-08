import requests
import streamlit as st
txt = st.text_area('enter text',)
st.write(txt)
#df=requests.get(url="https://technip.sharepoint.com/:x:/r/sites/SecurityNewsDataConsolidation/_layouts/15/Doc.aspx?sourcedoc=%7B6ED3943E-E085-4D2C-B0F8-E76DFC8BCC65%7D&file=CatMainRisk.xlsx&action=default&mobileredirect=true") #without authentication
#st.dataframe(df)
