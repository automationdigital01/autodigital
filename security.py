import requests
import streamlit as st
import pandas as pd

def read_csv(ctx, relative_url, pandas=False):
    response = File.open_binary(ctx, relative_url)
    bytes_data = response.content
    try:
        s = str(bytes_data, 'utf8')
    except Exception as e:
        st.write('utf8 encoding error')
        st.write(relative_url, e)
        try:
            s = str(bytes_data, 'cp1252')
        except Exception as e:
            st.write('CRITIAL ERROR cp1252 encoding error')
            st.write(relative_url, e)
    if pandas == False:
        return s
    else:
        data = StringIO(s)
        return data
      
FILE_URL= "https://technip.sharepoint.com/:x:/r/sites/SecurityNewsDataConsolidation/Shared%20Documents/General/Project-1%20Config/CatMainRisk.xlsx?d=w6ed3943ee0854d2cb0f8e76dfc8bcc65&csf=1&web=1&e=XUxlUY" 
df= pd.read_csv(read_csv(ctx=ctx, relative_url=FILE_URL, pandas=True), dtype=str, keep_default_na=False) # read master qrd db
st.dataframe(df)
#txt = st.text_area('enter text',)
#st.write(txt)
#df=requests.get(url="https://technip.sharepoint.com/:x:/r/sites/SecurityNewsDataConsolidation/_layouts/15/Doc.aspx?sourcedoc=%7B6ED3943E-E085-4D2C-B0F8-E76DFC8BCC65%7D&file=CatMainRisk.xlsx&action=default&mobileredirect=true") #without authentication
#st.dataframe(df)
