import requests
import streamlit as st
df=requests.get(url="https://technip.sharepoint.com/:x:/r/sites/SecurityNewsDataConsolidation/Shared%20Documents/General/Project-1%20Config/CatMainRisk.xlsx?d=w6ed3943ee0854d2cb0f8e76dfc8bcc65&csf=1&web=1&e=UVz17t") #without authentication
st.dataframe(df)
