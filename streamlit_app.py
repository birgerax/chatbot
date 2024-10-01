import numpy as np
import streamlit as st
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from groq import Groq
groq_client = Groq(
    api_key="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",
)
llama_70B = "llama-3.1-70b-versatile"

st.set_page_config(layout='wide')
st.title("Statistik 📊")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Datum", "BP", "Konkurser", "Byggkostnadsindex", "KI", "Investeringar", "Ny- och ombyggnad", "Internationella jämförelser"])

with tab1:
    #### Få datum för nästa SCB-publicering: byggande
  import requests
  from bs4 import BeautifulSoup

  url = "https://www.scb.se/hitta-statistik/statistik-efter-amne/boende-byggande-och-bebyggelse/bostadsbyggande-och-ombyggnad/bygglov-nybyggnad-och-ombyggnad/"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, "html.parser")
  next_publication = soup.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  #### Få datum för nästa SCB-publicering: finansmarknadsstatistik
  import requests
  from bs4 import BeautifulSoup

  url_finans = "https://www.scb.se/hitta-statistik/statistik-efter-amne/finansmarknad/finansmarknadsstatistik/finansmarknadsstatistik/"
  response_fin = requests.get(url_finans)
  soup_fin = BeautifulSoup(response_fin.text, "html.parser")
  datum_finans = soup_fin.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  #### Få datum för nästa SCB-publicering: byggkostnadsindex
  import requests
  from bs4 import BeautifulSoup

  url_bki = "https://www.scb.se/hitta-statistik/statistik-efter-amne/priser-och-konsumtion/byggnadsprisindex-samt-faktorprisindex-for-byggnader/byggkostnadsindex-bki/"
  response_bki = requests.get(url_bki)
  soup_bki = BeautifulSoup(response_bki.text, "html.parser")
  datum_bki = soup_bki.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  #### Få datum för nästa SCB-publicering: konkurser
  import requests
  from bs4 import BeautifulSoup

  url_k = "https://www.scb.se/hitta-statistik/statistik-efter-amne/naringsverksamhet/konkurser-och-offentliga-ackord/konkurser-och-offentliga-ackord/"
  response_k = requests.get(url_k)
  soup_k = BeautifulSoup(response_k.text, "html.parser")
  datum_k = soup_k.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  #### Få datum för nästa SCB-publicering: nationalräkenskaper (bl.a. fasta bruttoinvesteringar, bostadsinvesteringar...)
  import requests
  from bs4 import BeautifulSoup

  url_nr = "https://www.scb.se/hitta-statistik/statistik-efter-amne/nationalrakenskaper/nationalrakenskaper/nationalrakenskaper-kvartals-och-arsberakningar/"
  response_nr = requests.get(url_nr)
  soup_nr = BeautifulSoup(response_nr.text, "html.parser")
  datum_nr = soup_nr.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  #### Få datum för nästa SCB-publicering: Prisindex i producent- och importled (bl.a. tjänsteprisindex)
  import requests
  from bs4 import BeautifulSoup

  url_tj = "https://www.scb.se/hitta-statistik/statistik-efter-amne/priser-och-konsumtion/prisindex-i-producent-och-importled/prisindex-i-producent-och-importled-ppi/"
  response_tj = requests.get(url_tj)
  soup_tj = BeautifulSoup(response_tj.text, "html.parser")
  datum_tj = soup_tj.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  datum_bygg = next_publication.split(": ")[1].split("-")
  datum_bygg = "-".join([datum_bygg[0], datum_bygg[1], datum_bygg[2]])
  datum_fin = datum_finans.split(": ")[1].split("-")
  datum_fin = "-".join([datum_fin[0], datum_fin[1], datum_fin[2]])
  datum_b = datum_bki.split(": ")[1].split("-")
  datum_b = "-".join([datum_b[0], datum_b[1], datum_b[2]])
  datum_ko = datum_k.split(": ")[1].split("-")
  datum_ko = "-".join([datum_ko[0], datum_ko[1], datum_ko[2]])
  datum_n = datum_nr.split(": ")[1].split("-")
  datum_n = "-".join([datum_n[0], datum_n[1], datum_n[2]])
  datum_t = datum_tj.split(": ")[1].split("-")
  datum_t = "-".join([datum_t[0], datum_t[1], datum_t[2]])

  # Printa kommande statistikuppdateringar i ordning efter närmast datum
  from datetime import datetime

  alla_datum = [(datetime.strptime(datum_bygg, "%Y-%m-%d"), "Bygglov, nybyggnad och ombyggnad", next_publication),
                (datetime.strptime(datum_fin, "%Y-%m-%d"), "Finansmarknadsstatistik", datum_finans),
                (datetime.strptime(datum_b, "%Y-%m-%d"), "Byggkostnadsindex", datum_bki),
                (datetime.strptime(datum_ko, "%Y-%m-%d"), "Konkurser", datum_k),
                (datetime.strptime(datum_n, "%Y-%m-%d"), "Nationalräkenskaper (bl.a. bostadsinvesteringar)", datum_nr),
                (datetime.strptime(datum_t, "%Y-%m-%d"), "Prisindex i producent- och importled (bl.a. tjänsteprisindex)", datum_tj)]

  sorterade_datum = sorted(alla_datum, key=lambda x: x[0])
  for date, category, variable in sorterade_datum:
    print(category + ":\n" + variable + "\n")

  @st.cache_data
  def convert_df(df):
      # IMPORTANT: Cache the conversion to prevent computation on every rerun
      return df.to_csv().encode("utf-8")

  def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

  # Tabell: statistikuppdateringar efter närmast datum
  from datetime import datetime
  from prettytable import PrettyTable

  def display_data():
    alla_datum = [(datetime.strptime(datum_bygg, "%Y-%m-%d"), "Bygglov, nybyggnad och ombyggnad", next_publication),
                  (datetime.strptime(datum_fin, "%Y-%m-%d"), "Finansmarknadsstatistik", datum_finans),
                  (datetime.strptime(datum_b, "%Y-%m-%d"), "Byggkostnadsindex", datum_bki),
                  (datetime.strptime(datum_ko, "%Y-%m-%d"), "Konkurser", datum_k),
                  (datetime.strptime(datum_n, "%Y-%m-%d"), "Nationalräkenskaper (bl.a. bostadsinvesteringar)", datum_nr),
                  (datetime.strptime(datum_t, "%Y-%m-%d"), "Prisindex i producent- och importled (bl.a. tjänsteprisindex)", datum_tj)]

    sorterade_datum = sorted(alla_datum, key=lambda x: x[0])

    table = PrettyTable()
    table.field_names = ["Kategori", "Nästa publicering"]
    table.align["Kategori"] = "1"
    table.align["Nästa publicering"] = "1"

    table.title = "Statistikuppdateringar sorterat efter datum 📆"

    for _, category, variable in sorterade_datum:
      table.add_row([category, _.strftime("%Y-%m-%d")])
    st.write(table)
  display_data()

with tab2:
  # Data till diagram 3.3
  import requests
  import json

  # Configure the Plotly figure to improve download quality
  config = {
    'toImageButtonOptions': {
        'format': 'png',  # Export format
        'width': None,
        'height': None,
        'filename': 'high_quality_plot',  # Filename for download
        'scale': 2  # Increase scale for higher resolution (scale=2 means 2x the default resolution)
    },
    'displaylogo': False  # Optionally remove the Plotly logo from the toolbar
  }

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "item",
          "values": [
            "00"
          ]
        }
      },
      {
        "code": "BruttoNetto",
        "selection": {
          "filter": "item",
          "values": [
            "2"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "000003I4",
            "000003I2",
            "000003I6"
          ]
        }
      },
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "2000",
            "2001",
            "2002",
            "2003",
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0201/BO0201A/PrisPerAreorFH02"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  #response_json

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import pandas as pd
  import math

  keys = [entry['key'][2] for entry in response_json['data']]
  values_mark = [float(entry['values'][0]) for entry in response_json['data']]
  values_byggnad = [float(entry['values'][1]) for entry in response_json['data']]
  values_totalt = [float(entry['values'][2]) for entry in response_json['data']]

  df = pd.DataFrame({
      'År': keys,
      'Markpris/lägenhetsarea, kr': values_mark,
      'Byggnadspris/lägenhetsarea, kr': values_byggnad,
      'Totalt produktionspris/lägenhetsarea, kr': values_totalt
  })

  # Convert all columns to numeric (if not already)
  df = df.apply(pd.to_numeric)

  # Separate the time column from the rest
  time_column = df.iloc[:, 0]
  data_to_normalize = df.iloc[:, 1:]

  # Normalize the rest of the DataFrame (excluding the time column)
  normalized_data = (data_to_normalize / data_to_normalize.iloc[0]) * 100

  # Combine the time column with the normalized data
  df_normalized = pd.concat([time_column, normalized_data], axis=1)

  #print(df_normalized)

  # Konsumentprisindex (KPI) fastställda årsmedeltal, totalt, 1980=100 efter år

  # Data till diagram 3.3
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "2000",
            "2001",
            "2002",
            "2003",
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "2023"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/PR/PR0101/PR0101L/KPIFastAmed"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  #response_json

  keys_kpi = [entry['key'][0] for entry in response_json['data']]
  values_kpi = [float(entry['values'][0]) for entry in response_json['data']]

  df_kpi = pd.DataFrame({
      'År': keys_kpi,
      'KPI': values_kpi,
  })

  # Convert all columns to numeric (if not already)
  df_kpi = df_kpi.apply(pd.to_numeric)

  # Separate the time column from the rest
  time_column = df_kpi.iloc[:, 0]
  data_to_normalize = df_kpi.iloc[:, 1:]

  # Normalize the rest of the DataFrame (excluding the time column)
  normalized_data = (data_to_normalize / data_to_normalize.iloc[0]) * 100

  # Combine the time column with the normalized data
  df_normalized_kpi = pd.concat([time_column, normalized_data], axis=1)

  #print(df_normalized_kpi)

  # Remove the last row from df_kpi to match the length of df_normalized
  df_kpi_trimmed = df_normalized_kpi.iloc[:-1].reset_index(drop=True)

  # Merge the two DataFrames on the 'Time' column from df_normalized and 'År' from df_kpi_trimmed
  df_combined = pd.merge(df_normalized, df_kpi_trimmed, left_on='År', right_on='År', how='left')
  df_combined_kpi = df_combined

  import matplotlib.pyplot as plt

  # Set the 'Series1' column as the index (assuming it's time)
  df_combined.set_index('År', inplace=True)

  # Plot the DataFrame
  df_combined.plot(kind='line', marker='o')

  # Add titles and labels
  plt.title('Index för produktionspris per lägenhetsarea i riket för nybyggda flerbostadshus')
  plt.xlabel('År')
  plt.ylabel('Index')

  # Data till tabell 3.3
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "vs:RegionRiket99",
          "values": [
            "00"
          ]
        }
      },
      {
        "code": "Hustyp",
        "selection": {
          "filter": "item",
          "values": [
            "FLERBO",
            "SMÅHUS"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "BO0101A4"
          ]
        }
      },
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "2000K1",
            "2000K2",
            "2000K3",
            "2000K4",
            "2001K1",
            "2001K2",
            "2001K3",
            "2001K4",
            "2002K1",
            "2002K2",
            "2002K3",
            "2002K4",
            "2003K1",
            "2003K2",
            "2003K3",
            "2003K4",
            "2004K1",
            "2004K2",
            "2004K3",
            "2004K4",
            "2005K1",
            "2005K2",
            "2005K3",
            "2005K4",
            "2006K1",
            "2006K2",
            "2006K3",
            "2006K4",
            "2007K1",
            "2007K2",
            "2007K3",
            "2007K4",
            "2008K1",
            "2008K2",
            "2008K3",
            "2008K4",
            "2009K1",
            "2009K2",
            "2009K3",
            "2009K4",
            "2010K1",
            "2010K2",
            "2010K3",
            "2010K4",
            "2011K1",
            "2011K2",
            "2011K3",
            "2011K4",
            "2012K1",
            "2012K2",
            "2012K3",
            "2012K4",
            "2013K1",
            "2013K2",
            "2013K3",
            "2013K4",
            "2014K1",
            "2014K2",
            "2014K3",
            "2014K4",
            "2015K1",
            "2015K2",
            "2015K3",
            "2015K4",
            "2016K1",
            "2016K2",
            "2016K3",
            "2016K4",
            "2017K1",
            "2017K2",
            "2017K3",
            "2017K4",
            "2018K1",
            "2018K2",
            "2018K3",
            "2018K4",
            "2019K1",
            "2019K2",
            "2019K3",
            "2019K4",
            "2020K1",
            "2020K2",
            "2020K3",
            "2020K4",
            "2021K1",
            "2021K2",
            "2021K3",
            "2021K4",
            "2022K1",
            "2022K2",
            "2022K3",
            "2022K4",
            "2023K1",
            "2023K2",
            "2023K3",
            "2023K4",
            "2024K1",
            "2024K2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101C/LagenhetNyKv16"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  #response_json

  keys_pkv = [entry['key'][2] for entry in response_json['data'] if entry['key'][1] == 'FLERBO']
  values_pfle = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'FLERBO']
  values_psma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'SMÅHUS']

  df_p = pd.DataFrame({
      'År': keys_pkv,
      'Flerbostadshus': values_pfle,
      'Småhus': values_psma,
  })

  #print(df_p)

  # Extract the year from the 'Quarter' column
  df_p['Year'] = df_p['År'].str[:4]

  # Sum the values by year
  df_yearly = df_p.groupby('Year').agg({
      'Flerbostadshus': 'sum',
      'Småhus': 'sum'
  }).reset_index()

  #print(df_yearly)

  # Transpose the DataFrame to switch rows and columns
  df_transposed = df_yearly.T

  # Rename the index to the new row names
  df_transposed.index = ['År', 'Flerbostadshus', 'Småhus']

  #print(df_transposed)

  # Separate the first row
  first_row = df_transposed.iloc[0:1]

  # Select and round the remaining part
  df_selected = df_transposed.iloc[1:, :].apply(pd.to_numeric, errors='coerce').round(-2)

  # Combine the first row with the rounded DataFrame
  df_combined = pd.concat([first_row, df_selected])

  # Compute the sum of the second and third rows
  # Note: Adjust index if your DataFrame starts from a different row.
  row_to_sum = df_combined.iloc[1:3].sum()

  # Add the sum row to df_combined
  df_combined.loc['Total nybyggnad'] = row_to_sum

  #print(df_combined.iloc[:,15:])

  # Data till tabell 3.3
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "vs:RegionRiket99",
          "values": [
            "00"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "000001O2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }
  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101B/LagenhetOmbNKv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  #response_json

  keys = [entry['key'][1] for entry in response_json['data']]
  values = [float(entry['values'][0]) for entry in response_json['data']]

  df_ombyggnad = pd.DataFrame({
      'År': keys,
      'Ombyggnad': values,
  })

  #print(df_ombyggnad)

  # Extract the year from the 'Quarter' column
  df_ombyggnad['Year'] = df_ombyggnad['År'].str[:4]

  # Sum the values by year
  df_yearly = df_ombyggnad.groupby('Year').agg({
      'Ombyggnad': 'sum',
  }).reset_index()

  #print(df_yearly.iloc[11:,:])

  # Transpose the DataFrame to switch rows and columns
  df_transposed = df_yearly.T

  # Rename the index to the new row names
  df_transposed.index = ['År', 'Ombyggnad']
  #print(df_transposed.iloc[:,11:])

  # Separate the first row
  first_row = df_transposed.iloc[0:1]

  # Select and round the remaining part
  df_selected = df_transposed.iloc[1:, :].apply(pd.to_numeric, errors='coerce').round(-2)

  # Combine the first row with the rounded DataFrame
  df_combined_o = pd.concat([first_row.iloc[:,11:], df_selected.iloc[:,11:]])
  df_combined_o.columns = range(df_combined_o.shape[1])

  #print(df_combined_o)

  df_combined_tot = pd.concat([df_combined, df_combined_o.iloc[1:, :]])
  #print(df_combined_tot)

  # Totalt påbörjade bostäder

  total_paborjad = df_combined_tot.iloc[3] + df_combined_tot.iloc[4]
  df_combined_tot.loc['Totalt påbörjade bostäder'] = total_paborjad
  #print(df_combined_tot)

  # Befolkningsutveckling
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Kon",
        "selection": {
          "filter": "item",
          "values": [
            "1+2"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "000000LV"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BE/BE0101/BE0101G/BefUtvKon1749"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  #response_json

  keys = [entry['key'][1] for entry in response_json['data']]
  values = [float(entry['values'][0]) for entry in response_json['data']]

  df_befolkning = pd.DataFrame({
      'År': keys,
      'Folkmängd': values,
  })

  #print(df_befolkning)

  # Transpose the DataFrame to switch rows and columns
  df_transposed = df_befolkning.T

  # Rename the index to the new row names
  df_transposed.index = ['År', 'Folkmängd']

  # Separate the first row
  first_row = df_transposed.iloc[0:1]

  # Select the remaining part (Folkmängd) and calculate first differences
  df_selected = df_transposed.iloc[1:, :].apply(pd.to_numeric, errors='coerce')
  first_differences = df_selected.diff(axis=1).round(-2)

  # Rename the first differences row as "Befolkningsutveckling"
  first_differences.index = ['Befolkningsutveckling']

  # Combine the first row, original row, and new row with first differences
  df_combined = pd.concat([first_row.iloc[:, 251:], first_differences.iloc[:, 251:]], axis=0)

  # Reset column indices
  df_combined.columns = range(df_combined.shape[1])

  # Print the final DataFrame
  #print(df_combined)
  df_combined_tot = pd.concat([df_combined_tot, df_combined.iloc[1:, :]])

  st.header('BP-underlag! :sunglasses:')
  st.pyplot(plt)
  st.write(df_combined_kpi.round(1))
  st.subheader("** Tabell 3.3 Påbörjade bostäder och befolkningsökning **")
  st.write(df_combined_tot.round(0).fillna(0).astype(int))

with tab4:
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Kostnadsslag",
        "selection": {
          "filter": "item",
          "values": [
            "TOTAL",
            "MATERIAL",
            "ARBL",
            "MASK",
            "ETTILLFEM",
            "BYGGKOST"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/PR/PR0502/PR0502A/FPIBOM2015"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import pandas as pd
  import math

  keys_kv = [entry['key'][2] for entry in response_json['data']]
  values_fle = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TOTAL']
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'TOTAL']

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_sma, values_fle])

  # Trim keys_barometer to match the minimum length
  keys_kv_trimmed = keys_kv[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_kv_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Flerbostadshus'] = values_fle[:min_length]
  df['Gruppbyggda småhus'] = values_sma[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']  # Colors from the previous code
  data_bki_tot = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6 if column != 'Total' else 1.5,  # Adjust line width for 'Total' line
              dash='dash' if column == 'Total' else 'solid',  # Set line style to dashed for 'Total' line
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_bki_tot.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.18,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}, 2015=100',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.2,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_bki_tot = go.Layout(
      title='Byggkostnadsindex för bostäder exkl. löneglidning och moms',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=0,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
          x=1.05,  # Position the legend to the right of the chart
          y=1,
          traceorder='normal',
          font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )
  layout_bki_tot['title']['y'] = 0.89
  df_tot = df

  import plotly.graph_objs as go
  import plotly.offline as pyo

  keys_kv = [entry['key'][2] for entry in response_json['data']]
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'TOTAL']
  values_arbl = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'ARBL']
  values_mask = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'MASK']
  values_mate = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'MATERIAL']
  values_kost = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'BYGGKOST']
  values_etti = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'ETTILLFEM']

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_sma, values_arbl, values_mask, values_mate, values_kost, values_etti])

  # Trim keys_barometer to match the minimum length
  keys_kv_trimmed = keys_kv[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_kv_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Total'] = values_sma[:min_length]
  df['Löner'] = values_arbl[:min_length]
  df['Maskiner'] = values_mask[:min_length]
  df['Byggherrekostnad'] = values_kost[:min_length]
  df['Entreprenörernas kostnad'] = values_etti[:min_length]
  df['Material'] = values_mate[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]
  df_sma = df

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']  # Colors from the previous code
  data_bki_sma = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6 if column != 'Total' else 1.5,  # Adjust line width for 'Total' line
              dash='dash' if column == 'Total' else 'solid',  # Set line style to dashed for 'Total' line
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_bki_sma.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.18,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}, 2015=100',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.2,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_bki_sma = go.Layout(
      title='Byggkostnadsindex: Gruppbyggda småhus',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=0,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )

  layout_bki_sma['title']['y'] = 0.89

  bki_sma = go.Figure(data=data_bki_sma, layout=layout_bki_sma, layout_width=700)
  #pyo.iplot(bki_sma, filename='line-mode')

  import plotly.graph_objs as go
  import plotly.offline as pyo

  keys_kv = [entry['key'][2] for entry in response_json['data']]
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TOTAL']
  values_arbl = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'ARBL']
  values_mask = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'MASK']
  values_mate = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'MATERIAL']
  values_kost = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'BYGGKOST']
  values_etti = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'ETTILLFEM']

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_sma, values_arbl, values_mask, values_mate, values_kost, values_etti])

  # Trim keys_barometer to match the minimum length
  keys_kv_trimmed = keys_kv[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_kv_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Total'] = values_sma[:min_length]
  df['Löner'] = values_arbl[:min_length]
  df['Maskiner'] = values_mask[:min_length]
  df['Byggherrekostnad'] = values_kost[:min_length]
  df['Entreprenörernas kostnad'] = values_etti[:min_length]
  df['Material'] = values_mate[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']  # Colors from the previous code
  data_bki_fler = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6 if column != 'Total' else 1.5,  # Adjust line width for 'Total' line
              dash='dash' if column == 'Total' else 'solid',  # Set line style to dashed for 'Total' line
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_bki_fler.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.18,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}, 2015=100',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.2,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_bki_fler = go.Layout(
      title='Byggkostnadsindex: Flerbostadshus',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=0,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )

  layout_bki_fler['title']['y'] = 0.89
  bki_fler = go.Figure(data=data_bki_fler, layout=layout_bki_fler, layout_width=700)
  #pyo.iplot(bki_fler, filename='line-mode')

  bki_tot = go.Figure(data=data_bki_tot, layout=layout_bki_tot, layout_width=700)
  st.plotly_chart(bki_tot, config=config)

  st.header("Senaste utfall", divider=True)
  col1, col2, col3 = st.columns(3)
  latest_value = df_tot['Flerbostadshus'].iloc[-1]
  previous_value = df_tot['Flerbostadshus'].iloc[-2]

  # Calculate the percentage change
  percentage_change = (latest_value / previous_value - 1) * 100

  # Format the numeric value with spaces as thousands separators
  latest_formatted_value = f"{latest_value:,.1f}".replace(',', ' ')

  # Format the percentage change to one decimal place with a comma instead of a dot
  formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'

  latest_value = df_tot['Gruppbyggda småhus'].iloc[-1]
  previous_value = df_tot['Gruppbyggda småhus'].iloc[-2]
  percentage_change = (latest_value / previous_value - 1) * 100
  latest_formatted_value_sma = f"{latest_value:,.1f}".replace(',', ' ')
  formatted_change_sma = f"{percentage_change:.1f}".replace('.', ',') + '%'

  new_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.15,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_bki_tot['annotations'] = [new_annotation, data_source_annotation]
  layout_bki_tot['title']['x'] = 0.1
  bki_tot = go.Figure(data=data_bki_tot, layout=layout_bki_tot)

  @st.cache_data
  def save_figure_as_image(fig, format='png'):
      # Save the figure to a BytesIO object
      img_bytes = BytesIO()
      fig.write_image(img_bytes, format=format, scale=2)  # Increase scale for higher resolution
      img_bytes.seek(0)
      return img_bytes

  st.header("Hämta", divider=True)
  # Save the figure as a high-resolution image (PNG format by default)
  image_data = save_figure_as_image(bki_tot, format='png')

  # Add a download button for the figure image
  st.download_button(
      label="📈 Hämta figur",
      data=image_data,
      file_name="figure.png",
      mime="image/png",
      key="1"
  )

  col1.metric(f"Flerbostadshus" , latest_formatted_value, formatted_change)
  col2.metric(f"Gruppbyggda småhus" , latest_formatted_value_sma, formatted_change_sma)

  st.plotly_chart(bki_sma, config=config)
  layout_bki_sma['annotations'] = [new_annotation, data_source_annotation]
  layout_bki_sma['title']['x'] = 0.1
  bki_sma = go.Figure(data=data_bki_sma, layout=layout_bki_sma)

  # Save the figure as a high-resolution image (PNG format by default)
  image_data = save_figure_as_image(bki_sma, format='png')

  st.header("Hämta", divider=True)
  # Add a download button for the figure image
  st.download_button(
      label="📈 Hämta figur",
      data=image_data,
      file_name="figure.png",
      mime="image/png",
      key="2"
  )

  st.plotly_chart(bki_fler, config=config)
  layout_bki_fler['annotations'] = [new_annotation, data_source_annotation]
  layout_bki_fler['title']['x'] = 0.1
  bki_fler = go.Figure(data=data_bki_fler, layout=layout_bki_fler)

    # Save the figure as a high-resolution image (PNG format by default)
  image_data = save_figure_as_image(bki_fler, format='png')

  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  # Add a download button for the figure image
  col1.download_button(
      label="📈 Hämta figur",
      data=image_data,
      file_name="figure.png",
      mime="image/png",
      key="3"
  )

  # Extract columns from df_tot and df_sma, excluding the first column ('Time')
  df_tot_columns = df_tot.iloc[:, 1:]
  df_sma_columns = df_sma.iloc[:, 1:]
  # Create gap columns filled with NaN to place between DataFrames
  gap_column_1 = pd.DataFrame(np.nan, index=df.index, columns=[''])
  gap_column_2 = pd.DataFrame(np.nan, index=df.index, columns=[''])
  # Concatenate df, gap_column_1, df_tot_columns, gap_column_2, and df_sma_columns horizontally
  df_combined = pd.concat([df, gap_column_1, df_tot_columns, gap_column_2, df_sma_columns], axis=1)

  df_xlsx = to_excel(df_combined)
  col2.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="10")

#  chat_completion = groq_client.chat.completions.create(
#    messages=[
#        {
#            "role": "user",
#            "content": f"Firstly, based on the data in the dataframe {df_tot}, which displays index levels for the monthly (each month and year is given in the column 'Time') cost of housing construction for multi-family homes (Flerbostadshus) as well as single-family homes (Gruppbyggda småhus), describe the trend for both with: 1) five-year change  2) one year change (see {df[48:]}) 3) last quarter change {df[57:]} 4) one month. Comment it from a perspective that a normal rate would be usually around 2% inflation rate (however: do not explicitly write anything about that 2% is expected). Secondly, do the same analysis for the dataframe {df}, this dataframe shows the different parts of the costs and how they have developed for Flerbostadshus (see figure {bki_fler})",
#        }
#    ],
#    model=llama_70B,
#)
#
#  st.write((chat_completion.choices[0].message.content))

with tab5:
  ##### Konfidensindikator, kvartal

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Indikator",
        "selection": {
          "filter": "item",
          "values": [
            "BBYG",
            "BBOA",
            "B41000",
            "B42000",
            "B43000"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/indikatorer/Indikatorq.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import math

  keys_konfidens = []
  values_bbyg = []
  values_bboa = []
  values_b41 = []
  values_b42 = []
  values_b43 = []

  for entry in response_json['data']:
      if entry['key'][1] >= '2010Q1':
          if entry['key'][0] == 'BBYG':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bbyg.append(float(value))
          elif entry['key'][0] == 'BBOA':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bboa.append(float(value))
          elif entry['key'][0] == 'B41000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b41.append(float(value))
          elif entry['key'][0] == 'B42000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b42.append(float(value))
          elif entry['key'][0] == 'B43000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b43.append(float(value))
          keys_konfidens.append(entry['key'][1])

  data = [go.Scatter(
              x=keys_konfidens,
              y=values_bbyg,
              name="Byggindustri (SNI 41-43)",
              hovertext=[f"Tidpunkt: {key}<br>Byggindustri (SNI 41-43) {value}" for key, value in zip(keys_konfidens, values_bbyg)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#1f77b4',
                  width=1.5,
                  #dash='dash'
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_konfidens,
              y=values_bboa,
              name="Bygg & anläggning (SNI 41-42)",
              hovertext=[f"Tidpunkt: {key}<br>Bygg & anläggning (SNI 41-42) {value}" for key, value in zip(keys_konfidens, values_bboa)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#ff7f0e',
                  width=1.5
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_konfidens,
              y=values_b41,
              name="Husbyggande (SNI 41)",
              hovertext=[f"Tidpunkt: {key}<br>Husbyggande (SNI 41) {value}" for key, value in zip(keys_konfidens, values_b41)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#2ca02c',
                  width=1.5
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_konfidens,
              y=values_b42,
              name="Anläggningsverksamhet (SNI 42)",
              hovertext=[f"Tidpunkt: {key}<br>Anläggningsverksamhet (SNI 42) {value}" for key, value in zip(keys_konfidens, values_b42)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#d62728',
                  width=1.5
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_konfidens,
              y=values_b43,
              name="Specialiserad byggverksamhet (SNI 43)",
              hovertext=[f"Tidpunkt: {key}<br>Specialiserad byggverksamhet (SNI 43) {value}" for key, value in zip(keys_konfidens, values_b43)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#9467bd',
                  width=1.5
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          )]

  layout = go.Layout(
      title='Konfidensindikator',
      xaxis=dict(
          title='Tidpunkt'
      ),
      yaxis=dict(
          title='Index'
      )
  )

  layout = go.Layout(
      title='Konfidensindikator',
      xaxis=dict(
        # title='Tidpunkt'
      ),
      yaxis=dict(
          title='Index'
      ),
      plot_bgcolor='white',  # set the background color to white
      #xaxis_gridcolor='lightgray',  # set the horizontal grid color
      yaxis_gridcolor='lightgray'  # set the vertical grid color
  )

  konjunkturbarometern = go.Figure(data=data, layout=layout)
  #pyo.iplot(konjunkturbarometern, filename='line-mode')

  #st.plotly_chart(konjunkturbarometern)

  konf_bbyg = pd.DataFrame({'År': keys_konfidens[:len(values_bbyg)], 'Byggindustri (SNI 41-43)': values_bbyg})
  konf_bboa = pd.DataFrame({'År2': keys_konfidens[:len(values_bbyg)], 'Bygg & anläggning (SNI 41-42)': values_bboa})
  konf_b41 = pd.DataFrame({'År3': keys_konfidens[:len(values_bbyg)], 'Husbyggande (SNI 41)': values_b41})
  konf_b42 = pd.DataFrame({'År4': keys_konfidens[:len(values_bbyg)], 'Anläggningsverksamhet (SNI 42)': values_b42})
  konf_b43 = pd.DataFrame({'År5': keys_konfidens[:len(values_bbyg)], 'Specialiserad byggverksamhet (SNI 43)': values_b43})

  df_konf = pd.concat([konf_bbyg, konf_bboa, konf_b41, konf_b42, konf_b43], axis=1)

  # Remove the third column
  df_konf = df_konf.drop(['År2', 'År3', 'År4', 'År5'], axis=1)

  ##### Barometerindikatorn, månadsvis. Ingår frågorna: Orderstocken, nulägesomdöme + Antal anställda, förväntningar.

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Indikator",
        "selection": {
          "filter": "item",
          "values": [
            "BTOT",
            "BBYG",
            "BBOA",
            "B41000",
            "B42000",
            "B43000"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "http://statistik.konj.se/PxWeb/api/v1/sv/KonjBar/indikatorer/Indikatorm.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import math

  keys_barometer = []
  values_btot = []
  values_bbyg = []
  values_bboa = []
  values_b41 = []
  values_b42 = []
  values_b43 = []

  for entry in response_json['data']:
      if entry['key'][1] >= '2010M05':
          if entry['key'][0] == 'BBYG':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bbyg.append(float(value))
          elif entry['key'][0] == 'BBOA':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bboa.append(float(value))
          elif entry['key'][0] == 'B41000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b41.append(float(value))
          elif entry['key'][0] == 'B42000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b42.append(float(value))
          elif entry['key'][0] == 'B43000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b43.append(float(value))
          elif entry['key'][0] == 'BTOT':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_btot.append(float(value))
          keys_barometer.append(entry['key'][1])

  data_barometerindikatorn = [go.Scatter(
              x=keys_barometer,
              y=values_bbyg,
              name="Byggindustri (SNI 41-43)",
              hovertext=[f"Tidpunkt: {key}<br>Byggindustri (SNI 41-43) {value}" for key, value in zip(keys_barometer, values_bbyg)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#A2B1B3',
                  width=2,
                  #dash='dash'
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_barometer,
              y=values_bboa,
              name="Bygg & anläggning (SNI 41-42)",
              hovertext=[f"Tidpunkt: {key}<br>Bygg & anläggning (SNI 41-42) {value}" for key, value in zip(keys_barometer, values_bboa)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#D9CC00',
                  width=2
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_barometer,
              y=values_b41,
              name="Husbyggande (SNI 41)",
              hovertext=[f"Tidpunkt: {key}<br>Husbyggande (SNI 41) {value}" for key, value in zip(keys_barometer, values_b41)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#2BB2F7',
                  width=2
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_barometer,
              y=values_b42,
              name="Anläggningsverksamhet (SNI 42)",
              hovertext=[f"Tidpunkt: {key}<br>Anläggningsverksamhet (SNI 42) {value}" for key, value in zip(keys_barometer, values_b42)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#509D00',
                  width=2
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_barometer,
              y=values_b43,
              name="Specialiserad byggverksamhet (SNI 43)",
              hovertext=[f"Tidpunkt: {key}<br>Specialiserad byggverksamhet (SNI 43) {value}" for key, value in zip(keys_barometer, values_b43)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#004B84',
                  width=2
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          ),
          go.Scatter(
              x=keys_barometer,
              y=values_btot,
              name="Totala näringslivet",
              hovertext=[f"Tidpunkt: {key}<br>Totala näringslivet {value}" for key, value in zip(keys_barometer, values_btot)],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color='#E87502',
                  width=2
              ),
              opacity=0.6,
              selected=dict(
                  marker=dict(
                      color='red'
                  )
              ),
              unselected=dict(
                  marker=dict(
                      opacity=0.1
                  )
              )
          )]

  layout_barometerindikatorn = go.Layout(
      title='Barometerindikatorn',
      xaxis=dict(
          title='Tidpunkt'
      ),
      yaxis=dict(
          title='Index'
      )
  )

  layout_barometerindikatorn = go.Layout(
      title='Konfidensindikatorn för totala näringslivet och hela byggindustrin samt uppdelat på husbyggande, anläggning, specialiserad byggverksamhet', #Barometerindikatorn
      xaxis=dict(
        # title='Tidpunkt'
      ),
      yaxis=dict(
          title='Index'
      ),
      plot_bgcolor='white',  # set the background color to white
      #xaxis_gridcolor='lightgray',  # set the horizontal grid color
      yaxis_gridcolor='lightgray'  # set the vertical grid color
  )

  barometerindikatorn = go.Figure(data=data_barometerindikatorn, layout=layout_barometerindikatorn)
  #pyo.iplot(barometerindikatorn, filename='line-mode')

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import math
  import pandas as pd

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_bbyg, values_bboa, values_b41, values_b42, values_b43, values_btot])

  # Trim keys_barometer to match the minimum length
  keys_barometer_trimmed = keys_barometer[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_barometer_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Byggindustri (SNI 41-43)'] = values_bbyg[:min_length]
  df['Bygg & anläggning (SNI 41-42)'] = values_bboa[:min_length]
  df['Husbyggande (SNI 41)'] = values_b41[:min_length]
  df['Anläggningsverksamhet (SNI 42)'] = values_b42[:min_length]
  df['Specialiserad byggverksamhet (SNI 43)'] = values_b43[:min_length]
  df['Totala näringslivet'] = values_btot[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']  # Colors from the previous code
  data_barometerindikatorn = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=1))
      )
      data_barometerindikatorn.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.25,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}, index medelvärde=100',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.23,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__indikatorer/Indikatorm.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_barometerindikatorn = go.Layout(
      title=dict(
        text='Konfidensindikatorn för totala näringslivet och hela byggindustrin<br>samt uppdelat på husbyggande, anläggning, specialiserad byggverksamhet',
        #font=dict(size=18),  # Adjust font size to fit the title within the available space
        y=0.9,  # Adjust this value to move the title higher or lower
    ),
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=270,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )

  # Create the figure
  barometerindikatorn = go.Figure(data=data_barometerindikatorn, layout=layout_barometerindikatorn, layout_width=700)
  #pyo.iplot(barometerindikatorn, filename='line-mode')
  st.plotly_chart(barometerindikatorn, config=config)

  st.header("Senaste utfall", divider=True)
  col1, col2, col3 = st.columns(3)
  latest_value = df['Bygg & anläggning (SNI 41-42)'].iloc[-1]
  previous_value = df['Bygg & anläggning (SNI 41-42)'].iloc[-2]

  # Calculate the absolute change
  absolute_change = latest_value - previous_value

  # Format the numeric value with spaces as thousands separators
  latest_formatted_value = f"{latest_value:,.1f}".replace(',', ' ')

  # Format the absolute change to one decimal place with a comma instead of a dot
  formatted_change = f"{absolute_change:,.1f}".replace(',', ' ')  # No percentage sign, just the absolute value

  # Display the metric with formatted absolute change
  col1.metric(f"Bygg & anläggning (SNI 41-42)", latest_formatted_value, formatted_change)

  latest_value = df['Husbyggande (SNI 41)'].iloc[-1]
  previous_value = df['Husbyggande (SNI 41)'].iloc[-2]
  absolute_change = latest_value - previous_value
  latest_formatted_value = f"{latest_value:,.1f}".replace(',', ' ')
  formatted_change = f"{absolute_change:,.1f}".replace(',', ' ')
  col2.metric(f"Husbyggande (SNI 41)", latest_formatted_value, formatted_change)

  latest_value = df['Anläggningsverksamhet (SNI 42)'].iloc[-1]
  previous_value = df['Anläggningsverksamhet (SNI 42)'].iloc[-2]
  absolute_change = latest_value - previous_value
  latest_formatted_value = f"{latest_value:,.1f}".replace(',', ' ')
  formatted_change = f"{absolute_change:,.1f}".replace(',', ' ')
  col3.metric(f"Anläggningsverksamhet (SNI 42)", latest_formatted_value, formatted_change)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="9")

  barometer_bbyg = pd.DataFrame({'År': keys_barometer[:len(values_bbyg)], 'Byggindustri (SNI 41-43)': values_bbyg})
  barometer_bboa = pd.DataFrame({'År2': keys_barometer[:len(values_bbyg)], 'Bygg & anläggning (SNI 41-42)': values_bboa})
  barometer_b41 = pd.DataFrame({'År3': keys_barometer[:len(values_bbyg)], 'Husbyggande (SNI 41)': values_b41})
  barometer_b42 = pd.DataFrame({'År4': keys_barometer[:len(values_bbyg)], 'Anläggningsverksamhet (SNI 42)': values_b42})
  barometer_b43 = pd.DataFrame({'År5': keys_barometer[:len(values_bbyg)], 'Specialiserad byggverksamhet (SNI 43)': values_b43})
  barometer_btot = pd.DataFrame({'År6': keys_barometer[:len(values_bbyg)], 'Totala näringslivet': values_btot})

  df_barometer = pd.concat([barometer_bbyg, barometer_bboa, barometer_b41, barometer_b42, barometer_b43, barometer_btot], axis=1)

  # Remove the third column
  df_barometer = df_barometer.drop(['År2', 'År3', 'År4', 'År5', 'År6'], axis=1)

  ### Anbudspriser, utfall (säsongsrensat)

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Fråga",
        "selection": {
          "filter": "item",
          "values": [
            "102"
          ]
        }
      },
      {
        "code": "Serie",
        "selection": {
          "filter": "item",
          "values": [
            "S"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/ftgmanad/Barboam.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import math

  keys_anbud = []
  values_bbyg = []
  values_bboa = []
  values_b41 = []
  values_b42 = []
  values_b43 = []

  for entry in response_json['data']:
      if entry['key'][3] >= '2010M05':
          if entry['key'][0] == 'BBYG':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bbyg.append(float(value))
          elif entry['key'][0] == 'BBOA':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bboa.append(float(value))
          elif entry['key'][0] == 'B41000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b41.append(float(value))
          elif entry['key'][0] == 'B42000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b42.append(float(value))
          elif entry['key'][0] == 'B43000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b43.append(float(value))
          keys_anbud.append(entry['key'][3])

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_bbyg, values_bboa, values_b41, values_b42, values_b43, values_btot])

  # Trim keys_barometer to match the minimum length
  keys_anbud_trimmed = keys_anbud[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_anbud_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Byggindustri (SNI 41-43)'] = values_bbyg[:min_length]
  df['Bygg & anläggning (SNI 41-42)'] = values_bboa[:min_length]
  df['Husbyggande (SNI 41)'] = values_b41[:min_length]
  df['Anläggningsverksamhet (SNI 42)'] = values_b42[:min_length]
  df['Specialiserad byggverksamhet (SNI 43)'] = values_b43[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Colors from the previous code
  data = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data.append(trace)

  # Create traces using DataFrame columns
  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']  # Colors from the previous code
  data_anbudspriser = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_anbudspriser.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.23,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_anbudspriser = go.Layout(
      title='Anbudspriser, utfall (säsongsrensat)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=270,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )

  layout_anbudspriser['title']['y'] = 0.89

  anbudspriser = go.Figure(data=data_anbudspriser, layout=layout_anbudspriser, layout_width=700)
  #pyo.iplot(anbudspriser, filename='line-mode')
  st.plotly_chart(anbudspriser, config=config)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="8")

  anbudspriser_bbyg = pd.DataFrame({'År': keys_anbud[:len(values_bbyg)], 'Byggindustri (SNI 41-43)': values_bbyg})
  anbudspriser_bboa = pd.DataFrame({'År2': keys_anbud[:len(values_bbyg)], 'Bygg & anläggning (SNI 41-42)': values_bboa})
  anbudspriser_b41 = pd.DataFrame({'År3': keys_anbud[:len(values_bbyg)], 'Husbyggande (SNI 41)': values_b41})
  anbudspriser_b42 = pd.DataFrame({'År4': keys_anbud[:len(values_bbyg)], 'Anläggningsverksamhet (SNI 42)': values_b42})
  anbudspriser_b43 = pd.DataFrame({'År5': keys_anbud[:len(values_bbyg)], 'Specialiserad byggverksamhet (SNI 43)': values_b43})

  df_anbudspriser= pd.concat([anbudspriser_bbyg, anbudspriser_bboa, anbudspriser_b41, anbudspriser_b42, anbudspriser_b43], axis=1)

  # Remove the third column
  df_anbudspriser = df_anbudspriser.drop(['År2', 'År3', 'År4', 'År5'], axis=1)

  ### Orderstock, nulägesomdöme (säsongsrensat)
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Fråga",
        "selection": {
          "filter": "item",
          "values": [
            "104"
          ]
        }
      },
      {
        "code": "Serie",
        "selection": {
          "filter": "item",
          "values": [
            "S"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/ftgmanad/Barboam.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  keys_orderstock = []
  values_bbyg = []
  values_bboa = []
  values_b41 = []
  values_b42 = []
  values_b43 = []

  for entry in response_json['data']:
      if entry['key'][3] >= '2010M05':
          if entry['key'][0] == 'BBYG':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bbyg.append(float(value))
          elif entry['key'][0] == 'BBOA':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bboa.append(float(value))
          elif entry['key'][0] == 'B41000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b41.append(float(value))
          elif entry['key'][0] == 'B42000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b42.append(float(value))
          elif entry['key'][0] == 'B43000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b43.append(float(value))
          keys_orderstock.append(entry['key'][3])

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_bbyg, values_bboa, values_b41, values_b42, values_b43, values_btot])

  # Trim keys_barometer to match the minimum length
  keys_orderstock_trimmed = keys_orderstock[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_orderstock_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Byggindustri (SNI 41-43)'] = values_bbyg[:min_length]
  df['Bygg & anläggning (SNI 41-42)'] = values_bboa[:min_length]
  df['Husbyggande (SNI 41)'] = values_b41[:min_length]
  df['Anläggningsverksamhet (SNI 42)'] = values_b42[:min_length]
  df['Specialiserad byggverksamhet (SNI 43)'] = values_b43[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']  # Colors from the previous code
  data_orderstock = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_orderstock.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.23,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_orderstock = go.Layout(
      title='Orderstock, nulägesomdöme (säsongsrensat)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=270,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )

  layout_orderstock['title']['y'] = 0.89

  orderstock = go.Figure(data=data_orderstock, layout=layout_orderstock, layout_width=700)
  #pyo.iplot(orderstock, filename='line-mode')
  st.plotly_chart(orderstock, config=config)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="7")

  orderstock_bbyg = pd.DataFrame({'År': keys_orderstock[:len(values_bbyg)], 'Byggindustri (SNI 41-43)': values_bbyg})
  orderstock_bboa = pd.DataFrame({'År2': keys_orderstock[:len(values_bbyg)], 'Bygg & anläggning (SNI 41-42)': values_bboa})
  orderstock_b41 = pd.DataFrame({'År3': keys_orderstock[:len(values_bbyg)], 'Husbyggande (SNI 41)': values_b41})
  orderstock_b42 = pd.DataFrame({'År4': keys_orderstock[:len(values_bbyg)], 'Anläggningsverksamhet (SNI 42)': values_b42})
  orderstock_b43 = pd.DataFrame({'År5': keys_orderstock[:len(values_bbyg)], 'Specialiserad byggverksamhet (SNI 43)': values_b43})

  df_orderstock= pd.concat([orderstock_bbyg, orderstock_bboa, orderstock_b41, orderstock_b42, orderstock_b43], axis=1)

  # Remove the third column
  df_orderstock = df_orderstock.drop(['År2', 'År3', 'År4', 'År5'], axis=1)

  ### Anställningsplaner (säsongsrensat)
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Fråga",
        "selection": {
          "filter": "item",
          "values": [
            "204"
          ]
        }
      },
      {
        "code": "Serie",
        "selection": {
          "filter": "item",
          "values": [
            "S"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/ftgmanad/Barboam.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  keys_planer = []
  values_bbyg = []
  values_bboa = []
  values_b41 = []
  values_b42 = []
  values_b43 = []

  for entry in response_json['data']:
      if entry['key'][3] >= '2010M05':
          if entry['key'][0] == 'BBYG':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bbyg.append(float(value))
          elif entry['key'][0] == 'BBOA':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_bboa.append(float(value))
          elif entry['key'][0] == 'B41000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b41.append(float(value))
          elif entry['key'][0] == 'B42000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b42.append(float(value))
          elif entry['key'][0] == 'B43000':
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  values_b43.append(float(value))
          keys_planer.append(entry['key'][3])

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_bbyg, values_bboa, values_b41, values_b42, values_b43, values_btot])

  # Trim keys_barometer to match the minimum length
  keys_planer_trimmed = keys_planer[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_planer_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Byggindustri (SNI 41-43)'] = values_bbyg[:min_length]
  df['Bygg & anläggning (SNI 41-42)'] = values_bboa[:min_length]
  df['Husbyggande (SNI 41)'] = values_b41[:min_length]
  df['Anläggningsverksamhet (SNI 42)'] = values_b42[:min_length]
  df['Specialiserad byggverksamhet (SNI 43)'] = values_b43[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']  # Colors from the previous code
  data_anstallningsplaner = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_anstallningsplaner.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.23,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_anstallningsplaner = go.Layout(
      title='Anställningsplaner (säsongsrensat)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          tickvals=tick_positions,
          ticktext=tick_labels,  # Use the modified x-tick labels
          #title='Tidpunkt',
          tickangle=270,  # Keep x-axis tick labels horizontal
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  )

  layout_anstallningsplaner['title']['y'] = 0.89

  planer = go.Figure(data=data_anstallningsplaner, layout=layout_anstallningsplaner, layout_width=700)
 #pyo.iplot(planer, filename='line-mode')
  st.plotly_chart(planer, config=config)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="6")

  planer_bbyg = pd.DataFrame({'År': keys_planer[:len(values_bbyg)], 'Byggindustri (SNI 41-43)': values_bbyg})
  planer_bboa = pd.DataFrame({'År2': keys_planer[:len(values_bbyg)], 'Bygg & anläggning (SNI 41-42)': values_bboa})
  planer_b41 = pd.DataFrame({'År3': keys_planer[:len(values_bbyg)], 'Husbyggande (SNI 41)': values_b41})
  planer_b42 = pd.DataFrame({'År4': keys_planer[:len(values_bbyg)], 'Anläggningsverksamhet (SNI 42)': values_b42})
  planer_b43 = pd.DataFrame({'År5': keys_planer[:len(values_bbyg)], 'Specialiserad byggverksamhet (SNI 43)': values_b43})

  df_planer = pd.concat([planer_bbyg, planer_bboa, planer_b41, planer_b42, planer_b43], axis=1)

  # Remove the third column
  df_planer = df_planer.drop(['År2', 'År3', 'År4', 'År5'], axis=1)

  ### Främsta hindren
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Fråga",
        "selection": {
          "filter": "item",
          "values": [
            "1081",
            "1082",
            "1083",
            "1084",
            "1085",
            "1086",
            "1087"
          ]
        }
      },
      {
        "code": "Serie",
        "selection": {
          "filter": "item",
          "values": [
            "O"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/ftgmanad/Barboam.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  ####### Främsta hinder (hela byggindustrin)
  import math
  import plotly.graph_objs as go

  keys_hinder = []
  values_efterfragan = []
  values_material = []
  values_arbetskraft = []
  values_finans = []
  values_annat = []

  for entry in response_json['data']:
      if entry['key'][3] >= '2022M01':
          if entry['key'][0] == 'BBYG': # Här väljer jag hela byggindustrin
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  if entry['key'][1] == '1082':
                      values_efterfragan.append(float(value))
                  elif entry['key'][1] == '1083':
                      values_material.append(float(value))
                  elif entry['key'][1] == '1084':
                      values_arbetskraft.append(float(value))
                  elif entry['key'][1] == '1086':
                      values_finans.append(float(value))
                  elif entry['key'][1] == '1087':
                      values_annat.append(float(value))
          keys_hinder.append(entry['key'][3])

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_efterfragan, values_material, values_arbetskraft, values_finans, values_annat])

  # Trim keys_barometer to match the minimum length
  keys_hinder_trimmed = keys_hinder[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_hinder_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Efterfrågan'] = values_efterfragan[:min_length]
  df['Material och/eller utrustning'] = values_material[:min_length]
  df['Arbetskraft'] = values_arbetskraft[:min_length]
  df['Finansiella restriktioner'] = values_finans[:min_length]
  df['Annat'] = values_annat[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']  # Professional consulting colors
  data_hinder = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_hinder.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.25,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_hinder = go.Layout(
      title='Främsta hinder för byggande (hela byggindustrin)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  ,
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )


  layout_hinder['title']['y'] = 0.89

  hinder = go.Figure(data=data_hinder, layout=layout_hinder, layout_width=700)
 # pyo.iplot(hinder, filename='line-mode')
  st.plotly_chart(hinder, config=config)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="5")

  hinder_efterfragan = pd.DataFrame({'År': keys_hinder[:len(values_efterfragan)], 'Efterfrågan': values_efterfragan})
  hinder_material = pd.DataFrame({'År2': keys_hinder[:len(values_efterfragan)], 'Material och/eller utrustning': values_material})
  hinder_arbetskraft = pd.DataFrame({'År3': keys_hinder[:len(values_efterfragan)], 'Arbetskraft': values_arbetskraft})
  hinder_finans = pd.DataFrame({'År4': keys_hinder[:len(values_efterfragan)], 'Finansiella restriktioner': values_finans})
  hinder_annat = pd.DataFrame({'År5': keys_hinder[:len(values_efterfragan)], 'Annat': values_annat})

  df_hinder = pd.concat([hinder_efterfragan, hinder_material, hinder_arbetskraft, hinder_finans, hinder_annat], axis=1)

  # Remove the third column
  df_hinder = df_hinder.drop(['År2', 'År3', 'År4', 'År5'], axis=1)

  import math
  import plotly.graph_objs as go

  keys_hinder = []
  values_efterfragan = []
  values_material = []
  values_arbetskraft = []
  values_finans = []
  values_annat = []

  for entry in response_json['data']:
      if entry['key'][3] >= '2022M01':
          if entry['key'][0] == 'B41000': # Här bestämmer jag att det är husbyggande, går att ändra till tex anläggning
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  if entry['key'][1] == '1082':
                      values_efterfragan.append(float(value))
                  elif entry['key'][1] == '1083':
                      values_material.append(float(value))
                  elif entry['key'][1] == '1084':
                      values_arbetskraft.append(float(value))
                  elif entry['key'][1] == '1086':
                      values_finans.append(float(value))
                  elif entry['key'][1] == '1087':
                      values_annat.append(float(value))
          keys_hinder.append(entry['key'][3])

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_efterfragan, values_material, values_arbetskraft, values_finans, values_annat])

  # Trim keys_barometer to match the minimum length
  keys_hinder_trimmed = keys_hinder[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_hinder_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Efterfrågan'] = values_efterfragan[:min_length]
  df['Material och/eller utrustning'] = values_material[:min_length]
  df['Arbetskraft'] = values_arbetskraft[:min_length]
  df['Finansiella restriktioner'] = values_finans[:min_length]
  df['Annat'] = values_annat[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']  # Professional consulting colors
  data_hinder_hus = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_hinder_hus.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.25,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_hinder_hus = go.Layout(
      title='Främsta hinder för husbyggande (SNI 41)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  ,
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_hinder_hus['title']['y'] = 0.89

  hinder_hus = go.Figure(data=data_hinder_hus, layout=layout_hinder_hus, layout_width=700)
  #pyo.iplot(hinder_hus, filename='line-mode')
  st.plotly_chart(hinder_hus, config=config)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="4")

  hinder_hus_efterfragan = pd.DataFrame({'År': keys_hinder[:len(values_efterfragan)], 'Efterfrågan': values_efterfragan})
  hinder_hus_material = pd.DataFrame({'År2': keys_hinder[:len(values_efterfragan)], 'Material och/eller utrustning': values_material})
  hinder_hus_arbetskraft = pd.DataFrame({'År3': keys_hinder[:len(values_efterfragan)], 'Arbetskraft': values_arbetskraft})
  hinder_hus_finans = pd.DataFrame({'År4': keys_hinder[:len(values_efterfragan)], 'Finansiella restriktioner': values_finans})
  hinder_hus_annat = pd.DataFrame({'År5': keys_hinder[:len(values_efterfragan)], 'Annat': values_annat})

  df_hinder_hus = pd.concat([hinder_hus_efterfragan, hinder_hus_material, hinder_hus_arbetskraft, hinder_hus_finans, hinder_hus_annat], axis=1)

  # Remove the third column
  df_hinder_hus = df_hinder_hus.drop(['År2', 'År3', 'År4', 'År5'], axis=1)

  import math
  import plotly.graph_objs as go

  keys_hinder = []
  values_efterfragan = []
  values_material = []
  values_arbetskraft = []
  values_finans = []
  values_annat = []

  for entry in response_json['data']:
      if entry['key'][3] >= '2022M01':
          if entry['key'][0] == 'B42000': # Här bestämmer jag att det är anläggning
              value = entry['values'][0]
              if value != '..' and not math.isnan(float(value)):
                  if entry['key'][1] == '1082':
                      values_efterfragan.append(float(value))
                  elif entry['key'][1] == '1083':
                      values_material.append(float(value))
                  elif entry['key'][1] == '1084':
                      values_arbetskraft.append(float(value))
                  elif entry['key'][1] == '1086':
                      values_finans.append(float(value))
                  elif entry['key'][1] == '1087':
                      values_annat.append(float(value))
          keys_hinder.append(entry['key'][3])

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_efterfragan, values_material, values_arbetskraft, values_finans, values_annat])

  # Trim keys_barometer to match the minimum length
  keys_hinder_trimmed = keys_hinder[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_hinder_trimmed})

  # Replace the values in the 'Time' column with the desired format
  month_map = {
      '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
      '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }
  df['Time'] = df['Time'].str[5:].map(month_map) + '-' + df['Time'].str[2:4]

  # Add columns for each line plot, ensuring lengths match
  df['Efterfrågan'] = values_efterfragan[:min_length]
  df['Material och/eller utrustning'] = values_material[:min_length]
  df['Arbetskraft'] = values_arbetskraft[:min_length]
  df['Finansiella restriktioner'] = values_finans[:min_length]
  df['Annat'] = values_annat[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']  # Professional consulting colors
  data_hinder_anlaggning = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_hinder_anlaggning.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=-0.25,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/">Konjunkturinstitutet</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_hinder_anlaggning = go.Layout(
      title='Främsta hinder för anläggningsverksamhet (SNI 42)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          title='',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  ,
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_hinder_anlaggning['title']['y'] = 0.89

  hinder_anlaggning = go.Figure(data=data_hinder_anlaggning, layout=layout_hinder_anlaggning, layout_width=700)
  #pyo.iplot(hinder_anlaggning, filename='line-mode')
  st.plotly_chart(hinder_anlaggning, config=config, width=1100, height=900)

  st.header("Hämta", divider=True)
  df_xlsx = to_excel(df)
  st.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="15")

with tab6:
  ### Bostadsinvesteringar (work in progress..)

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Typ",
        "selection": {
          "filter": "item",
          "values": [
            "111"
          ]
        }
      },
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "1993K1",
            "1993K2",
            "1993K3",
            "1993K4",
            "1994K1",
            "1994K2",
            "1994K3",
            "1994K4",
            "1995K1",
            "1995K2",
            "1995K3",
            "1995K4",
            "1996K1",
            "1996K2",
            "1996K3",
            "1996K4",
            "1997K1",
            "1997K2",
            "1997K3",
            "1997K4",
            "1998K1",
            "1998K2",
            "1998K3",
            "1998K4",
            "1999K1",
            "1999K2",
            "1999K3",
            "1999K4",
            "2000K1",
            "2000K2",
            "2000K3",
            "2000K4",
            "2001K1",
            "2001K2",
            "2001K3",
            "2001K4",
            "2002K1",
            "2002K2",
            "2002K3",
            "2002K4",
            "2003K1",
            "2003K2",
            "2003K3",
            "2003K4",
            "2004K1",
            "2004K2",
            "2004K3",
            "2004K4",
            "2005K1",
            "2005K2",
            "2005K3",
            "2005K4",
            "2006K1",
            "2006K2",
            "2006K3",
            "2006K4",
            "2007K1",
            "2007K2",
            "2007K3",
            "2007K4",
            "2008K1",
            "2008K2",
            "2008K3",
            "2008K4",
            "2009K1",
            "2009K2",
            "2009K3",
            "2009K4",
            "2010K1",
            "2010K2",
            "2010K3",
            "2010K4",
            "2011K1",
            "2011K2",
            "2011K3",
            "2011K4",
            "2012K1",
            "2012K2",
            "2012K3",
            "2012K4",
            "2013K1",
            "2013K2",
            "2013K3",
            "2013K4",
            "2014K1",
            "2014K2",
            "2014K3",
            "2014K4",
            "2015K1",
            "2015K2",
            "2015K3",
            "2015K4",
            "2016K1",
            "2016K2",
            "2016K3",
            "2016K4",
            "2017K1",
            "2017K2",
            "2017K3",
            "2017K4",
            "2018K1",
            "2018K2",
            "2018K3",
            "2018K4",
            "2019K1",
            "2019K2",
            "2019K3",
            "2019K4",
            "2020K1",
            "2020K2",
            "2020K3",
            "2020K4",
            "2021K1",
            "2021K2",
            "2021K3",
            "2021K4",
            "2022K1",
            "2022K2",
            "2022K3",
            "2022K4",
            "2023K1",
            "2023K2",
            "2023K3",
            "2023K4",
            "2024K1",
            "2024K2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/NR/NR0103/NR0103B/NR0103ENS2010T18Kv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  bostadsinvesteringar = response_json

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import pandas as pd

  keys_inv = [entry['key'][1] for entry in bostadsinvesteringar['data']]
  values_inv = [float(entry['values'][0]) for entry in bostadsinvesteringar['data']]
  df_inv = pd.DataFrame({'Time': keys_inv, 'Total': values_inv})

  # Slice the DataFrame to select the last 60 rows
  df_inv = df_inv.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)']  # Professional consulting colors
  data_inv = []
  for i, column in enumerate(df_inv.columns[1:]):
      trace = go.Scatter(
          x=df_inv['Time'],
          y=df_inv['Total'],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df_inv['Time'], df_inv['Total'])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[0],  # Use the first color defined in colors
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_inv.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df_inv['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.35,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Mnkr, säsongsrensade löpande priser, senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.2,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T18Kv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_inv = go.Layout(
      title='Bostadsinvesteringar',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
          font=dict(size=12)  # Set font size for legend text
      ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_inv['title']['y'] = 0.89

  bostadsinvesteringar_lopande = go.Figure(data=data_inv, layout=layout_inv, layout_width=700)
  #pyo.iplot(bostadsinvesteringar_lopande, filename='line-mode')
  #st.plotly_chart(bostadsinvesteringar_lopande)

  ### Bostadsinvesteringar (work in progress..)

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Typ",
        "selection": {
          "filter": "item",
          "values": [
            "112"
          ]
        }
      },
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "1993K1",
            "1993K2",
            "1993K3",
            "1993K4",
            "1994K1",
            "1994K2",
            "1994K3",
            "1994K4",
            "1995K1",
            "1995K2",
            "1995K3",
            "1995K4",
            "1996K1",
            "1996K2",
            "1996K3",
            "1996K4",
            "1997K1",
            "1997K2",
            "1997K3",
            "1997K4",
            "1998K1",
            "1998K2",
            "1998K3",
            "1998K4",
            "1999K1",
            "1999K2",
            "1999K3",
            "1999K4",
            "2000K1",
            "2000K2",
            "2000K3",
            "2000K4",
            "2001K1",
            "2001K2",
            "2001K3",
            "2001K4",
            "2002K1",
            "2002K2",
            "2002K3",
            "2002K4",
            "2003K1",
            "2003K2",
            "2003K3",
            "2003K4",
            "2004K1",
            "2004K2",
            "2004K3",
            "2004K4",
            "2005K1",
            "2005K2",
            "2005K3",
            "2005K4",
            "2006K1",
            "2006K2",
            "2006K3",
            "2006K4",
            "2007K1",
            "2007K2",
            "2007K3",
            "2007K4",
            "2008K1",
            "2008K2",
            "2008K3",
            "2008K4",
            "2009K1",
            "2009K2",
            "2009K3",
            "2009K4",
            "2010K1",
            "2010K2",
            "2010K3",
            "2010K4",
            "2011K1",
            "2011K2",
            "2011K3",
            "2011K4",
            "2012K1",
            "2012K2",
            "2012K3",
            "2012K4",
            "2013K1",
            "2013K2",
            "2013K3",
            "2013K4",
            "2014K1",
            "2014K2",
            "2014K3",
            "2014K4",
            "2015K1",
            "2015K2",
            "2015K3",
            "2015K4",
            "2016K1",
            "2016K2",
            "2016K3",
            "2016K4",
            "2017K1",
            "2017K2",
            "2017K3",
            "2017K4",
            "2018K1",
            "2018K2",
            "2018K3",
            "2018K4",
            "2019K1",
            "2019K2",
            "2019K3",
            "2019K4",
            "2020K1",
            "2020K2",
            "2020K3",
            "2020K4",
            "2021K1",
            "2021K2",
            "2021K3",
            "2021K4",
            "2022K1",
            "2022K2",
            "2022K3",
            "2022K4",
            "2023K1",
            "2023K2",
            "2023K3",
            "2023K4",
            "2024K1",
            "2024K2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/NR/NR0103/NR0103B/NR0103ENS2010T18Kv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  ovrigt_lopande = response_json

  keys_inv_fast = [entry['key'][1] for entry in ovrigt_lopande['data']]
  values_inv_fast = [float(entry['values'][0]) for entry in ovrigt_lopande['data']]
  df_inv_fast = pd.DataFrame({'Time': keys_inv_fast, 'Total': values_inv_fast})

  # Slice the DataFrame to select the last 60 rows
  df_inv_fast = df_inv_fast.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)']
  data_ovrigt_lopande = []
  for i, column in enumerate(df_inv_fast.columns[1:]):
      trace = go.Scatter(
          x=df_inv_fast['Time'],
          y=df_inv_fast['Total'],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df_inv_fast['Time'], df_inv_fast['Total'])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[0],  # Use the first color defined in colors
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_ovrigt_lopande.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df_inv_fast['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.30,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Mnkr, säsongsrensade löpande priser, senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.2,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T17Kv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_ovrigt_lopande = go.Layout(
      title='Investeringar i övriga byggnader och anläggningar',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
          font=dict(size=12)  # Set font size for legend text
      ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_ovrigt_lopande['title']['y'] = 0.89

  inv_ovrigt_lopande = go.Figure(data=data_ovrigt_lopande, layout=layout_ovrigt_lopande, layout_width=700)
  #st.plotly_chart(inv_ovrigt_lopande)

  ### Bostadsinvesteringar, fasta priser

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Typ",
        "selection": {
          "filter": "item",
          "values": [
            "111"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "0000000H"
          ]
        }
      },
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "1993K1",
            "1993K2",
            "1993K3",
            "1993K4",
            "1994K1",
            "1994K2",
            "1994K3",
            "1994K4",
            "1995K1",
            "1995K2",
            "1995K3",
            "1995K4",
            "1996K1",
            "1996K2",
            "1996K3",
            "1996K4",
            "1997K1",
            "1997K2",
            "1997K3",
            "1997K4",
            "1998K1",
            "1998K2",
            "1998K3",
            "1998K4",
            "1999K1",
            "1999K2",
            "1999K3",
            "1999K4",
            "2000K1",
            "2000K2",
            "2000K3",
            "2000K4",
            "2001K1",
            "2001K2",
            "2001K3",
            "2001K4",
            "2002K1",
            "2002K2",
            "2002K3",
            "2002K4",
            "2003K1",
            "2003K2",
            "2003K3",
            "2003K4",
            "2004K1",
            "2004K2",
            "2004K3",
            "2004K4",
            "2005K1",
            "2005K2",
            "2005K3",
            "2005K4",
            "2006K1",
            "2006K2",
            "2006K3",
            "2006K4",
            "2007K1",
            "2007K2",
            "2007K3",
            "2007K4",
            "2008K1",
            "2008K2",
            "2008K3",
            "2008K4",
            "2009K1",
            "2009K2",
            "2009K3",
            "2009K4",
            "2010K1",
            "2010K2",
            "2010K3",
            "2010K4",
            "2011K1",
            "2011K2",
            "2011K3",
            "2011K4",
            "2012K1",
            "2012K2",
            "2012K3",
            "2012K4",
            "2013K1",
            "2013K2",
            "2013K3",
            "2013K4",
            "2014K1",
            "2014K2",
            "2014K3",
            "2014K4",
            "2015K1",
            "2015K2",
            "2015K3",
            "2015K4",
            "2016K1",
            "2016K2",
            "2016K3",
            "2016K4",
            "2017K1",
            "2017K2",
            "2017K3",
            "2017K4",
            "2018K1",
            "2018K2",
            "2018K3",
            "2018K4",
            "2019K1",
            "2019K2",
            "2019K3",
            "2019K4",
            "2020K1",
            "2020K2",
            "2020K3",
            "2020K4",
            "2021K1",
            "2021K2",
            "2021K3",
            "2021K4",
            "2022K1",
            "2022K2",
            "2022K3",
            "2022K4",
            "2023K1",
            "2023K2",
            "2023K3",
            "2023K4",
            "2024K1",
            "2024K2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/NR/NR0103/NR0103B/NR0103ENS2010T17Kv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  bostadsinvesteringar_fast = response_json

  keys_inv_fast = [entry['key'][1] for entry in bostadsinvesteringar_fast['data']]
  values_inv_fast = [float(entry['values'][0]) for entry in bostadsinvesteringar_fast['data']]
  df_inv_fast = pd.DataFrame({'Time': keys_inv_fast, 'Total': values_inv_fast})

  # Slice the DataFrame to select the last 60 rows
  df_inv_fast = df_inv_fast.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)']
  data_inv_fast = []
  for i, column in enumerate(df_inv_fast.columns[1:]):
      trace = go.Scatter(
          x=df_inv_fast['Time'],
          y=df_inv_fast['Total'],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df_inv_fast['Time'], df_inv_fast['Total'])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[0],  # Use the first color defined in colors
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_inv_fast.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df_inv_fast['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.35,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Mnkr, säsongsrensade fasta priser, senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.2,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T17Kv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_inv_fast = go.Layout(
      title='Bostadsinvesteringar',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
          font=dict(size=12)  # Set font size for legend text
      ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_inv_fast['title']['y'] = 0.89

  bostadsinvesteringar_fasta = go.Figure(data=data_inv_fast, layout=layout_inv_fast, layout_width=700)
  #st.plotly_chart(bostadsinvesteringar_fasta)

  ### Investeringar i övriga byggnader och anläggningar

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Typ",
        "selection": {
          "filter": "item",
          "values": [
            "112"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "0000000H"
          ]
        }
      },
      {
        "code": "Tid",
        "selection": {
          "filter": "item",
          "values": [
            "1993K1",
            "1993K2",
            "1993K3",
            "1993K4",
            "1994K1",
            "1994K2",
            "1994K3",
            "1994K4",
            "1995K1",
            "1995K2",
            "1995K3",
            "1995K4",
            "1996K1",
            "1996K2",
            "1996K3",
            "1996K4",
            "1997K1",
            "1997K2",
            "1997K3",
            "1997K4",
            "1998K1",
            "1998K2",
            "1998K3",
            "1998K4",
            "1999K1",
            "1999K2",
            "1999K3",
            "1999K4",
            "2000K1",
            "2000K2",
            "2000K3",
            "2000K4",
            "2001K1",
            "2001K2",
            "2001K3",
            "2001K4",
            "2002K1",
            "2002K2",
            "2002K3",
            "2002K4",
            "2003K1",
            "2003K2",
            "2003K3",
            "2003K4",
            "2004K1",
            "2004K2",
            "2004K3",
            "2004K4",
            "2005K1",
            "2005K2",
            "2005K3",
            "2005K4",
            "2006K1",
            "2006K2",
            "2006K3",
            "2006K4",
            "2007K1",
            "2007K2",
            "2007K3",
            "2007K4",
            "2008K1",
            "2008K2",
            "2008K3",
            "2008K4",
            "2009K1",
            "2009K2",
            "2009K3",
            "2009K4",
            "2010K1",
            "2010K2",
            "2010K3",
            "2010K4",
            "2011K1",
            "2011K2",
            "2011K3",
            "2011K4",
            "2012K1",
            "2012K2",
            "2012K3",
            "2012K4",
            "2013K1",
            "2013K2",
            "2013K3",
            "2013K4",
            "2014K1",
            "2014K2",
            "2014K3",
            "2014K4",
            "2015K1",
            "2015K2",
            "2015K3",
            "2015K4",
            "2016K1",
            "2016K2",
            "2016K3",
            "2016K4",
            "2017K1",
            "2017K2",
            "2017K3",
            "2017K4",
            "2018K1",
            "2018K2",
            "2018K3",
            "2018K4",
            "2019K1",
            "2019K2",
            "2019K3",
            "2019K4",
            "2020K1",
            "2020K2",
            "2020K3",
            "2020K4",
            "2021K1",
            "2021K2",
            "2021K3",
            "2021K4",
            "2022K1",
            "2022K2",
            "2022K3",
            "2022K4",
            "2023K1",
            "2023K2",
            "2023K3",
            "2023K4",
            "2024K1",
            "2024K2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/NR/NR0103/NR0103B/NR0103ENS2010T17Kv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  investeringar_ovrigt = response_json
  keys_inv_fast_ovr = [entry['key'][1] for entry in investeringar_ovrigt['data']]
  values_inv_fast_ovr = [float(entry['values'][0]) for entry in investeringar_ovrigt['data']]
  df_inv_fast_ovr = pd.DataFrame({'Time': keys_inv_fast_ovr, 'Total': values_inv_fast_ovr})

  # Slice the DataFrame to select the last 60 rows
  df_inv_fast_ovr = df_inv_fast_ovr.iloc[-61:]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)']
  data_inv_ovrigt = []
  for i, column in enumerate(df_inv_fast_ovr.columns[1:]):
      trace = go.Scatter(
          x=df_inv_fast_ovr['Time'],
          y=df_inv_fast_ovr['Total'],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df_inv_fast_ovr['Time'], df_inv_fast_ovr['Total'])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[0],  # Use the first color defined in colors
              width=2.6
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_inv_ovrigt.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df_inv_fast_ovr['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.35,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Mnkr, säsongsrensade fasta priser, senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.25,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T17Kv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_inv_ovrigt = go.Layout(
      title='Investeringar i övriga byggnader och anläggningar',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  ,
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_inv_ovrigt['title']['y'] = 0.89

  investeringar_ovrigt = go.Figure(data=data_inv_ovrigt, layout=layout_inv_ovrigt, layout_width=700)
  #st.plotly_chart(investeringar_ovrigt)

  df_inv_fast.rename(columns={'Year': 'Time'}, inplace=True)

  combined_df = pd.merge(df_inv_fast.iloc[-61:], df_inv_fast_ovr, on="Time", how="outer")
  combined_df.rename(columns={'Total_x': 'Bostadsinvesteringar',
                              'Total_y': 'Övriga byggnader och anläggningar'}, inplace=True)

  # Format the 'Bostadsinvesteringar' column with spaces as thousands separators
  #combined_df['Bostadsinvesteringar'] = combined_df['Bostadsinvesteringar'].apply(lambda x: f"{x:,.0f}".replace(',', ''))

  # Format the 'Övriga investeringar och anläggningar' column similarly
  #combined_df['Övriga byggnader och anläggningar'] = combined_df['Övriga byggnader och anläggningar'].apply(lambda x: f"{x:,.0f}".replace(',', ' '))

  # Determine the number of x-ticks to display
  #desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  #step_size = math.ceil(len(combined_df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  #tick_positions = list(range(0, len(combined_df), step_size))
  #tick_positions.append(len(combined_df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  #tick_labels = combined_df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']  # Colors from the previous code
  data_combined = []
  for i, column in enumerate(combined_df.columns[1:]):
      trace = go.Scatter(
          x=combined_df['Time'],
          y=combined_df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(combined_df['Time'], combined_df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6 if column != 'Total' else 1.5,  # Adjust line width for 'Total' line
              dash='dash' if column == 'Total' else 'solid',  # Set line style to dashed for 'Total' line
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_combined.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = combined_df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.05,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add annotation for the data source below the frame
  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.25,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T17Kv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_combined = layout_inv_ovrigt
  layout_combined["title"] = "Investeringar inom bygg och anläggning"
  layout_combined['title']['y'] = 0.89

  combined = go.Figure(data=data_combined, layout=layout_combined, layout_width=700)

# Configure the Plotly figure to improve download quality
  config = {
    'toImageButtonOptions': {
        'format': 'png',  # Export format
        'filename': 'high_quality_plot',  # Filename for download
        'scale': 2  # Increase scale for higher resolution (scale=2 means 2x the default resolution)
    },
    'displaylogo': False  # Optionally remove the Plotly logo from the toolbar
  }
  st.plotly_chart(combined, config=config)

  st.header("Senaste utfall", divider=True)
  col1, col2, col3 = st.columns(3)
  latest_value = combined_df['Bostadsinvesteringar'].iloc[-1]
  previous_value = combined_df['Bostadsinvesteringar'].iloc[-2]

  # Calculate the percentage change
  percentage_change = (latest_value / previous_value - 1) * 100

  # Format the numeric value with spaces as thousands separators
  latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')

  # Format the percentage change to one decimal place with a comma instead of a dot
  formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'

  # Display the metric with formatted percentage change
  #col3.metric(("Total", combined_df['Bostadsinvesteringar'].iloc[-1] + combined_df['Övriga byggnader och anläggningar'].iloc[-1], (combined_df['Bostadsinvesteringar'].iloc[-1] + combined_df['Övriga byggnader och anläggningar'].iloc[-1]) / (combined_df['Bostadsinvesteringar'].iloc[-2]+combined_df['Övriga byggnader och anläggningar'].iloc[-2])-1)*100)
  col1.metric(f"Bostadsinvesteringar" , latest_formatted_value, formatted_change)
  latest_value_ovrig = combined_df['Övriga byggnader och anläggningar'].iloc[-1]
  previous_value_ovrig = combined_df['Övriga byggnader och anläggningar'].iloc[-2]

  # Calculate the percentage change
  percentage_change_ovrig = (latest_value_ovrig / previous_value_ovrig - 1) * 100

  # Format the numeric value with spaces as thousands separators
  latest_formatted_value = f"{latest_value_ovrig:,.0f}".replace(',', ' ')

  # Format the percentage change to one decimal place with a comma instead of a dot
  formatted_change_ovrig = f"{percentage_change_ovrig:.1f}".replace('.', ',') + '%'
  col2.metric(f"Övriga byggnader och anläggningar", latest_formatted_value, formatted_change_ovrig)

  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  df_xlsx = to_excel(combined_df)
  col1.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="16")

  new_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.15,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_combined['annotations'] = [new_annotation, data_source_annotation]
  layout_combined['title']['x'] = 0.1
  combined = go.Figure(data=data_combined, layout=layout_combined, layout_width=700)

  # Save the figure as a high-resolution image (PNG format by default)
  image_data = save_figure_as_image(combined, format='png')

  # Add a download button for the figure image
  col2.download_button(
      label="📈 Hämta figur",
      data=image_data,
      file_name="figure.png",
      mime="image/png",
      key="20"
  )

with tab7:
  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "vs:RegionRiket99",
          "values": [
            "00"
          ]
        }
      },
      {
        "code": "Hustyp",
        "selection": {
          "filter": "item",
          "values": [
            "FLERBO",
            "SMÅHUS"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "BO0101A3"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101C/LagenhetNyKv16"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  fardig_kv = response_json

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import pandas as pd
  import math

  keys_kv = [entry['key'][2] for entry in fardig_kv['data']]
  values_fle = [float(entry['values'][0]) for entry in fardig_kv['data'] if entry['key'][1] == 'FLERBO']
  values_sma = [float(entry['values'][0]) for entry in fardig_kv['data'] if entry['key'][1] == 'SMÅHUS']

  yearly_totals = {}
  for entry in fardig_kv['data']:
    year_quarter = entry['key'][2]
    value = float(entry['values'][0])
    year = int(year_quarter[:4])

    if year in yearly_totals:
      yearly_totals[year] += value
    else:
      yearly_totals[year] = value

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_sma, values_fle])

  # Trim keys_barometer to match the minimum length
  keys_kv_trimmed = keys_kv[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_kv_trimmed})

  # Add columns for each line plot, ensuring lengths match
  df['Flerbostadshus'] = values_fle[:min_length]
  df['Småhus'] = values_sma[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']  # Colors from the previous code
  data_fardig = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6 if column != 'Total' else 1.5,  # Adjust line width for 'Total' line
              dash='dash' if column == 'Total' else 'solid',  # Set line style to dashed for 'Total' line
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_fardig.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.30,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101C/LagenhetNyKv16/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout
  layout_fardig = go.Layout(
      title='Färdigställda bostäder (kvartal, ej uppräknat)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  ,
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_fardig['title']['y'] = 0.89
  fardig_tot = go.Figure(data=data_fardig, layout=layout_fardig, layout_width=700)

  #csv = convert_df(df)

  st.plotly_chart(fardig_tot, config=config)
  #st.download_button(
  #    label="Hämta data",
  #    data=csv,
  #    file_name="csv.csv",
  #    mime="text/csv",
  #    key="2"
  #)
  st.header("Senaste utfall", divider=True)
  col1, col2, col3 = st.columns(3)
  latest_value = df['Flerbostadshus'].iloc[-1]
  previous_value = df['Flerbostadshus'].iloc[-2]

  # Calculate the percentage change
  percentage_change = (latest_value / previous_value - 1) * 100

  # Format the numeric value with spaces as thousands separators
  latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')

  # Format the percentage change to one decimal place with a comma instead of a dot
  formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'

  # Display the metric with formatted percentage change
  col1.metric(f"Flerbostadshus" , latest_formatted_value, formatted_change)

  latest_value = df['Småhus'].iloc[-1]
  previous_value = df['Småhus'].iloc[-2]
  percentage_change = (latest_value / previous_value - 1) * 100
  latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')
  formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'
  col2.metric(f"Småhus" , latest_formatted_value, formatted_change)

  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  df_xlsx = to_excel(df)
  col1.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="17")

  new_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.15,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_fardig['annotations'] = [new_annotation, data_source_annotation]
  layout_fardig['title']['x'] = 0.1
  fardig_tot = go.Figure(data=data_fardig, layout=layout_fardig)

  @st.cache_data
  def save_figure_as_image(fig, format='png'):
      # Save the figure to a BytesIO object
      img_bytes = BytesIO()
      fig.write_image(img_bytes, format=format, scale=2)  # Increase scale for higher resolution
      img_bytes.seek(0)
      return img_bytes

  # Save the figure as a high-resolution image (PNG format by default)
  image_data = save_figure_as_image(fardig_tot, format='png')

  # Add a download button for the figure image
  col2.download_button(
      label="📈 Hämta figur",
      data=image_data,
      file_name="figure.png",
      mime="image/png",
      key="18"
  )

  ###### Påbörjade bostäder: uppdelat per upplåtelseform

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "vs:RegionRiket99",
          "values": [
            "00"
          ]
        }
      },
      {
        "code": "Hustyp",
        "selection": {
          "filter": "item",
          "values": [
            "FLERBO",
            "SMÅHUS"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "BO0101A4"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101C/LagenhetNyKv16"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  import plotly.graph_objs as go
  import plotly.offline as pyo
  import pandas as pd
  import math

  keys_pkv = [entry['key'][2] for entry in response_json['data']]
  values_pfle = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'FLERBO']
  values_psma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'SMÅHUS']

  yearly_totals = {}
  for entry in response_json['data']:
    year_quarter = entry['key'][2]
    value = float(entry['values'][0])
    year = int(year_quarter[:4])

    if year in yearly_totals:
      yearly_totals[year] += value
    else:
      yearly_totals[year] = value

  # Determine the minimum length among all value lists
  min_length = min(len(values) for values in [values_psma, values_pfle])

  # Trim keys_barometer to match the minimum length
  keys_pkv_trimmed = keys_pkv[:min_length]

  # Create a DataFrame to organize the data with time as the index
  df = pd.DataFrame({'Time': keys_pkv_trimmed})

  # Add columns for each line plot, ensuring lengths match
  df['Flerbostadshus'] = values_pfle[:min_length]
  df['Småhus'] = values_psma[:min_length]

  # Slice the DataFrame to select the last 60 rows
  df = df.iloc[-61:]

  # Determine the number of x-ticks to display
  desired_ticks = 12

  # Calculate the step size for selecting x-ticks
  step_size = math.ceil(len(df) / (desired_ticks - 1))  # Adjusting for the inclusion of the last x-tick

  # Select x-ticks at regular intervals with the last x-tick included
  tick_positions = list(range(0, len(df), step_size))
  tick_positions.append(len(df) - 1)  # Include the last x-tick position

  # Extract the corresponding timestamps for the selected x-ticks
  tick_labels = df['Time'].iloc[tick_positions]

  # Create traces using DataFrame columns
  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']  # Colors from the previous code
  data_pkv = []
  for i, column in enumerate(df.columns[1:]):
      trace = go.Scatter(
          x=df['Time'],
          y=df[column],
          name=column,
          hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
          hoverinfo='text',
          mode='lines',
          line=dict(
              color=colors[i],
              width=2.6 if column != 'Total' else 1.5,  # Adjust line width for 'Total' line
              dash='dash' if column == 'Total' else 'solid',  # Set line style to dashed for 'Total' line
          ),
          opacity=1,  # Set opacity to 1 for solid colors
          selected=dict(marker=dict(color='red')),
          unselected=dict(marker=dict(opacity=0.1))
      )
      data_pkv.append(trace)

  # Add subtopic for the time of the last datapoint
  last_datapoint_time = df['Time'].iloc[-1]
  last_datapoint_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  data_source_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.30,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101C/LagenhetNyKv16/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

 # Layout
  layout_pkv = go.Layout(
      title='Påbörjade bostäder (kvartal, ej uppräknat)',
      titlefont=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          #title='Index',
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      xaxis2=dict(
          showline=True,  # Show top x-axis line
          linewidth=1,  # Set top x-axis line width
          linecolor='black',  # Set top x-axis line color
          mirror=True,  # Show top x-axis line on the bottom side
      ),
      yaxis2=dict(
          showline=True,  # Show right y-axis line
          linewidth=1,  # Set right y-axis line width
          linecolor='black',  # Set right y-axis line color
          mirror=True,  # Show right y-axis line on the left side
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[last_datapoint_annotation, data_source_annotation],  # Add annotation for the last datapoint and data source
      legend=dict(
        x=1.05,  # Position the legend to the right of the chart
        y=1,
        traceorder='normal',
        font=dict(  # Update legend font properties
            family="Monaco, monospace",
            size=12,
            color="black"
        )
    )
  ,
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_pkv['title']['y'] = 0.89

  pkv_tot = go.Figure(data=data_pkv, layout=layout_pkv, layout_width=700)
  st.plotly_chart(pkv_tot, config=config)

  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  df_xlsx = to_excel(df)
  col1.download_button(label='📥 Hämta data',
                                data=df_xlsx,
                                file_name= 'df_test.xlsx',
                                key="30")

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "vs:RegionRiket99",
          "values": []
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "000001O2"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101B/LagenhetOmbNKv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))
  ombyggnad = response_json

  import plotly.graph_objs as go
  import plotly.offline as pyo

 # Prepare the data for the 'ombyggnad' plot
  keys_ombyggnad = [entry['key'][0] for entry in ombyggnad['data']]
  values_ombyggnad = [float(entry['values'][0]) for entry in ombyggnad['data']]

  # Compute yearly totals
  yearly_totals = {}
  for entry in ombyggnad['data']:
      year_quarter = entry['key'][0]
      value = float(entry['values'][0])
      year = int(year_quarter[:4])

      if year in yearly_totals:
          yearly_totals[year] += value
      else:
          yearly_totals[year] = value

  years_ombyggnad = list(yearly_totals.keys())
  totals_ombyggnad = list(yearly_totals.values())

  # Create trace for 'ombyggnad' plot
  data_ombyggnad = [go.Scatter(
      x=keys_ombyggnad,
      y=values_ombyggnad,
      hovertext=[f"Tidpunkt: {key}<br>Påbörjad ombyggnad: {value}" for key, value in zip(keys_ombyggnad, values_ombyggnad)],
      hoverinfo='text',
      mode='lines',
      line=dict(
          color='rgb(8,48,107)',
          width=1.5
      ),
      opacity=0.6,
      selected=dict(
          marker=dict(
              color='red'
          )
      ),
      unselected=dict(
          marker=dict(
              opacity=0.1
          )
      )
  )]

  # Add subtopic for the last datapoint of 'ombyggnad'
  last_datapoint_time_ombyggnad = keys_ombyggnad[-1]
  last_datapoint_annotation_ombyggnad = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time_ombyggnad}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add data source annotation for 'ombyggnad'
  data_source_annotation_ombyggnad = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.30,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101B/LagenhetOmbNKv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout for the 'ombyggnad' plot (same structure as layout_pkv but with appropriate titles)
  # Create layout similar to the first example
  layout_ombyggnad = go.Layout(
      title={
          'text': 'Nettoförändring: lägenheter i påbörjad ombyggnad av flerbostadshus',
          'y': 0.90,  # Adjust vertical title position
          'x': 0.49,  # Adjust horizontal title position
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 16}
      },
      xaxis=dict(
          tickangle=270,  # Rotate x-axis tick labels
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[data_source_annotation_ombyggnad, last_datapoint_annotation_ombyggnad],  # Add the data source annotation
      legend=dict(
          x=1.05,  # Position the legend to the right of the chart
          y=1,
          traceorder='normal',
          font=dict(  # Update legend font properties
              family="Monaco, monospace",
              size=12,
              color="black"
          )
      ),
      margin=dict(
          l=40,  # Adjust left margin
          r=40,  # Adjust right margin
          t=100,  # Adjust top margin
          b=100  # Adjust bottom margin
      ),
      width=600,  # Set the width of the figure
  )

  layout_ombyggnad['title']['y'] = 0.89

  # Create and plot the 'ombyggnad' figure using the updated layout
  ombyggt = go.Figure(data=data_ombyggnad, layout=layout_ombyggnad)
  st.plotly_chart(ombyggt, config=config)

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Region",
        "selection": {
          "filter": "vs:RegionRiket99",
          "values": [
              "00"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "000001O1"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101B/LagenhetOmbNKv"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))
  ombyggnad = response_json

  import plotly.graph_objs as go
  import plotly.offline as pyo

 # Prepare the data for the 'ombyggnad' plot
  keys_ombyggnad = [entry['key'][1] for entry in ombyggnad['data']]
  values_ombyggnad = [float(entry['values'][0]) for entry in ombyggnad['data']]

  # Compute yearly totals
  yearly_totals = {}
  for entry in ombyggnad['data']:
      year_quarter = entry['key'][0]
      value = float(entry['values'][0])
      year = int(year_quarter[:4])

      if year in yearly_totals:
          yearly_totals[year] += value
      else:
          yearly_totals[year] = value

  years_ombyggnad = list(yearly_totals.keys())
  totals_ombyggnad = list(yearly_totals.values())

  # Create trace for 'ombyggnad' plot
  data_ombyggnad = [go.Scatter(
      x=keys_ombyggnad,
      y=values_ombyggnad,
      hovertext=[f"Tidpunkt: {key}<br>Färdigställd ombyggnad: {value}" for key, value in zip(keys_ombyggnad, values_ombyggnad)],
      hoverinfo='text',
      mode='lines',
      line=dict(
          color='rgb(8,48,107)',
          width=1.5
      ),
      opacity=0.6,
      selected=dict(
          marker=dict(
              color='red'
          )
      ),
      unselected=dict(
          marker=dict(
              opacity=0.1
          )
      )
  )]

  # Add subtopic for the last datapoint of 'ombyggnad'
  last_datapoint_time_ombyggnad = keys_ombyggnad[-1]
  last_datapoint_annotation_ombyggnad = dict(
      xref='paper',
      yref='paper',
      x=0.08,
      y=1.03,
      xanchor='center',
      yanchor='bottom',
      text=f'Senaste utfall: {last_datapoint_time_ombyggnad}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Add data source annotation for 'ombyggnad'
  data_source_annotation_ombyggnad = dict(
      xref='paper',
      yref='paper',
      x=0.01,
      y=-0.30,
      xanchor='center',
      yanchor='top',
      text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101B/LagenhetOmbNKv/">SCB</a>',
      font=dict(size=12, color='black'),  # Set font size and color
      showarrow=False,
  )

  # Layout for the 'ombyggnad' plot (same structure as layout_pkv but with appropriate titles)
  # Create layout similar to the first example
  layout_ombyggnad = go.Layout(
      title={
          'text': 'Nettoförändring: lägenheter i färdigställd ombyggnad av flerbostadshus',
          'y': 0.90,  # Adjust vertical title position
          'x': 0.49,  # Adjust horizontal title position
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 16}
      },
      xaxis=dict(
          tickangle=270,  # Rotate x-axis tick labels
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          ticks='outside',  # Place ticks outside the plot
          ticklen=5,  # Length of the ticks
      ),
      yaxis=dict(
          showline=True,  # Show y-axis line
          linewidth=1,  # Set y-axis line width
          linecolor='black',  # Set y-axis line color
          mirror=True,  # Show y-axis line on the top and right side
          tickfont=dict(size=16),  # Set font size for y-axis ticks
          tickformat=",",  # Format y-axis ticks as thousand separator
      ),
      plot_bgcolor='white',
      yaxis_gridcolor='lightgray',
      annotations=[data_source_annotation_ombyggnad, last_datapoint_annotation_ombyggnad],  # Add the data source annotation
      legend=dict(
          x=1.05,  # Position the legend to the right of the chart
          y=1,
          traceorder='normal',
          font=dict(  # Update legend font properties
              family="Monaco, monospace",
              size=12,
              color="black"
          )
      ),
      margin=dict(
          l=40,  # Adjust left margin
          r=40,  # Adjust right margin
          t=100,  # Adjust top margin
          b=100  # Adjust bottom margin
      ),
      width=600,  # Set the width of the figure
  )

  layout_ombyggnad['title']['y'] = 0.89

  # Create and plot the 'ombyggnad' figure using the updated layout
  ombyggt = go.Figure(data=data_ombyggnad, layout=layout_ombyggnad)
  st.plotly_chart(ombyggt, config=config)

with tab8:
  import eurostat
  toc_df = eurostat.get_toc_df()

  import pandas as pd
  import plotly.graph_objects as go
  import eurostat  # Ensure the Eurostat library is installed and imported

  def plot_eurostat_data(data_code, countries_to_keep, year_range, value_name,
                        main_title, sub_heading, xaxis_title, yaxis_title, eurostat_link,
                        additional_filter_pars=None):
      """
      Function to plot Eurostat data with custom titles, countries, and formatting.

      Parameters:
      - data_code: str, the Eurostat data code to retrieve the dataset.
      - countries_to_keep: list, countries' codes to keep in the data (e.g., ['NO', 'DK', 'FI', 'SE', 'EU27_2020']).
      - year_range: tuple, range of years for the columns (e.g., (2010, 2024)).
      - value_name: str, the name for the value column after melting.
      - main_title: str, the main title of the plot.
      - sub_heading: str, the sub-heading text.
      - xaxis_title: str, title for the x-axis.
      - yaxis_title: str, title for the y-axis.
      - eurostat_link: str, the source link for the data.
      - additional_filter_pars: dict, additional filters to be included in the request (optional).

      Returns:
      - A Plotly figure (go.Figure) object with the customized plot.
      """

      # Step 1: Fetch the Data
      filter_pars = {'age': 'TOTAL', 'sex': 'T', 'incgrp': 'TOTAL'}
      data = eurostat.get_data_df(data_code, filter_pars=filter_pars)

      # If additional filters are provided, merge them with the default filter parameters
      if additional_filter_pars:
        filter_pars.update(additional_filter_pars)

      # Fetch the Data
      data = eurostat.get_data_df(data_code, filter_pars=filter_pars)

      # Step 2: Rename 'geo\\TIME_PERIOD' for easier manipulation
      data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

      # Step 3: Filter the Data to Keep Only Specific Countries
      data = data[data['geo'].isin(countries_to_keep)]

      # Step 4: Melt the Data to Long Format
      data_long = pd.melt(
          data,
          id_vars=['geo'],  # Keep the 'geo' column fixed
          value_vars=[str(year) for year in range(*year_range)],  # Columns representing years
          var_name='year',  # Name for the melted 'year' column
          value_name=value_name  # Name for the melted 'value' column
      )

      # Convert 'year' to datetime format for correct plotting on the x-axis
      data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

      # Step 5: Create the Plot
      fig = go.Figure()

      # Add a line for each country
      for country in data_long['geo'].unique():
          country_data = data_long[data_long['geo'] == country]
          country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
          fig.add_trace(go.Scatter(
              x=country_data['year'],
              y=country_data[value_name],
              mode='lines+markers',
              name=country_label  # Set the name to match the x-axis labels
          ))

      # Create sub-heading as an annotation
      sub_heading_annotation = dict(
          xref='paper',
          yref='paper',
          x=0.22,  # Adjust x position as needed
          y=1.03,  # Position just above the plot area, below the main title
          xanchor='center',
          yanchor='bottom',
          text=sub_heading,  # Your sub-heading text
          font=dict(size=14, color='black'),  # Set font size and color
          showarrow=False
      )

      # Customize Layout with a Frame and Improved Title Position
      fig.update_layout(
          title={
              'text': main_title,
              'y': 0.92,  # Adjust to control the exact vertical position of the title
              'x': 0.30,
              'xanchor': 'center',
              'yanchor': 'top',
              'font': {'size': 18}  # Adjust font size as needed
          },
          xaxis=dict(
              title=xaxis_title,
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,  # Create a frame effect
              tickangle=0,
              tickfont=dict(size=14)
          ),
          yaxis=dict(
              title=yaxis_title,
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,  # Create a frame effect
              tickfont=dict(size=16)
          ),
          plot_bgcolor='white',
          margin=dict(
              l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
          ),
          width=700,
          template='plotly_white',
          annotations=[
              sub_heading_annotation,  # Add sub-heading annotation
              dict(
                  text=f'Källa: <a href="{eurostat_link}">Eurostat</a>',
                  xref='paper', yref='paper',
                  x=1, y=-0.15,
                  showarrow=False,
                  font=dict(size=12)
              )
          ],
          # Update the legend settings
          legend=dict(
              orientation="h",  # Set legend orientation to horizontal
              yanchor="top",    # Anchor the legend to the top
              y=-0.2,           # Adjust y position to place it below the plot
              xanchor="center", # Center the legend horizontally
              x=0.5,            # Center it on the x-axis
              traceorder='normal',
              title='',
              font=dict(family="Monaco, monospace", size=12, color="black")
          )

      )

      # Step 6: Return the figure
      return fig

  # Example usage
  fig = plot_eurostat_data(
      data_code='ilc_lvho07a',
      countries_to_keep=['NO', 'DK', 'FI', 'SE', 'EU27_2020'],
      year_range=(2010, 2024),
      value_name='value',
      main_title='Housing cost overburden rate',
      sub_heading='Hela befolkningen, 2010-2023',
      xaxis_title='År',
      yaxis_title='Procent',
      eurostat_link='https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO07A__custom_12778359/default/table?lang=en'
  )

  # Display the plot
  fig.show()

  # Define filter parameters for the dataset
  my_filter_pars = {'startPeriod': '2005-Q1', 'indic_bt': 'BPRM_SQM', 'cpa2_1': 'CPA_F41001', 's_adj': 'NSA', 'unit': 'PCH_SM'}
  data = eurostat.get_data_df('sts_cobp_q', filter_pars=my_filter_pars)

  import plotly.graph_objects as go

  # Select the row with the string "SE" under the column "geo\TIME_PERIOD"
  row = data.loc[data['geo\\TIME_PERIOD'] == 'SE']
  values = row.iloc[0, 2:]

  # Set the x-axis labels and y-axis values for Sweden
  x_labels = values.index[28:]
  y_values = values.values[28:]

  # Select data for EU27, DK, FI, NO
  row_eu27 = data.loc[data['geo\\TIME_PERIOD'] == 'EU27_2020']
  values_eu27 = row_eu27.iloc[0, 2:]
  x_labels_eu27 = values_eu27.index[28:]
  y_values_eu27 = values_eu27.values[28:]

  row_dk = data.loc[data['geo\\TIME_PERIOD'] == 'DK']
  values_dk = row_dk.iloc[0, 2:]
  x_labels_dk = values_dk.index[28:]
  y_values_dk = values_dk.values[28:]

  row_fi = data.loc[data['geo\\TIME_PERIOD'] == 'FI']
  values_fi = row_fi.iloc[0, 2:]
  x_labels_fi = values_fi.index[28:]
  y_values_fi = values_fi.values[28:]

  row_no = data.loc[data['geo\\TIME_PERIOD'] == 'NO']
  values_no = row_no.iloc[0, 2:]
  x_labels_no = values_no.index[28:]
  y_values_no = values_no.values[28:]

  # Create traces for each country
  data_euro_permit = [
      go.Scatter(
          x=x_labels,
          y=y_values,
          name='SE',
          hovertext=[f"År: {year}<br>Building permits - SE: {total}" for year, total in zip(x_labels, y_values)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='#0051BA', line=dict(color='#0051BA', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_eu27,
          y=y_values_eu27,
          name='EU',
          hovertext=[f"År: {year}<br>Building permits - EU27: {total}" for year, total in zip(x_labels_eu27, y_values_eu27)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='dash'),
          marker=dict(color='#FDB813', line=dict(color='#FDB813', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_dk,
          y=y_values_dk,
          name='DK',
          hovertext=[f"År: {year}<br>Building permits - DK: {total}" for year, total in zip(x_labels_dk, y_values_dk)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(250,80,80)', line=dict(color='rgb(250,80,80)', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_no,
          y=y_values_no,
          name='NO',
          hovertext=[f"År: {year}<br>Building permits - NO: {total}" for year, total in zip(x_labels_no, y_values_no)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(200,75,10)', line=dict(color='rgb(200,75,10)', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_fi,
          y=y_values_fi,
          name='FI',
          hovertext=[f"År: {year}<br>Building permits - FI: {total}" for year, total in zip(x_labels_fi, y_values_fi)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(20,200,220)', line=dict(color='rgb(20,200,220)', width=1.5)),
          opacity=0.6
      )
  ]

  # Define layout for the chart
  layout_permit = go.Layout(
      title={
          'text': 'Bygglov: antal kvadratmeter golvyta, bostäder',
          'y': 0.90,  # Adjust vertical title position
          'x': 0.35,  # Adjust horizontal title position
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}
      },
      xaxis=dict(
          title='År',
          tickangle=90,
          tickfont=dict(size=10, color='rgb(107, 107, 107)'),
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True
      ),
      yaxis=dict(
          title='Årlig procentuell förändring',
          titlefont=dict(size=16, color='rgb(107, 107, 107)'),
          tickfont=dict(size=14, color='rgb(107, 107, 107)'),
          tickformat=',d',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True
      ),
      plot_bgcolor='white',
      margin=dict(l=40, r=40, t=100, b=100),  # Adjust margins
      width=700,  # Smaller layout width
      template='plotly_white',
      annotations=[
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/STS_COBP_Q/default/table">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.30,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05, y=1,
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Create figure
  fig_euro_permit = go.Figure(data=data_euro_permit, layout=layout_permit)

  # Display plot in Streamlit
  st.plotly_chart(fig_euro_permit)

 # House price index, Total, Annual rate of change
  my_filter_pars = {'startPeriod': '2011-Q1', 'purchase': 'TOTAL', 'unit': 'I15_Q'}
  data = eurostat.get_data_df('prc_hpi_q', filter_pars=my_filter_pars)

  import plotly.graph_objects as go

  # Select the row with the string "SE" under the column "geo\TIME_PERIOD"
  row = data.loc[data['geo\\TIME_PERIOD'] == 'SE']
  values = row.iloc[0, 2:]

  # Set the x-axis labels and y-axis values
  x_labels = values.index[28:]
  y_values = values.values[28:]

  # Select data for EU27, DK, FI, NO
  row_eu27 = data.loc[data['geo\\TIME_PERIOD'] == 'EU27_2020']
  values_eu27 = row_eu27.iloc[0, 2:]
  x_labels_eu27 = values_eu27.index[28:]
  y_values_eu27 = values_eu27.values[28:]

  row_dk = data.loc[data['geo\\TIME_PERIOD'] == 'DK']
  values_dk = row_dk.iloc[0, 2:]
  x_labels_dk = values_dk.index[28:]
  y_values_dk = values_dk.values[28:]

  row_fi = data.loc[data['geo\\TIME_PERIOD'] == 'FI']
  values_fi = row_fi.iloc[0, 2:]
  x_labels_fi = values_fi.index[28:]
  y_values_fi = values_fi.values[28:]

  row_no = data.loc[data['geo\\TIME_PERIOD'] == 'NO']
  values_no = row_no.iloc[0, 2:]
  x_labels_no = values_no.index[28:]
  y_values_no = values_no.values[28:]

  # Create traces for each country
  data_euro_price = [
      go.Scatter(
          x=x_labels,
          y=y_values,
          name='SE',
          hovertext=[f"År: {year}<br>Index - SE: {total}" for year, total in zip(x_labels, y_values)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='#0051BA', line=dict(color='#0051BA', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_eu27,
          y=y_values_eu27,
          name='EU',
          hovertext=[f"År: {year}<br>Index - EU27: {total}" for year, total in zip(x_labels_eu27, y_values_eu27)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='dash'),
          marker=dict(color='#FDB813', line=dict(color='#FDB813', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_dk,
          y=y_values_dk,
          name='DK',
          hovertext=[f"År: {year}<br>Index - DK: {total}" for year, total in zip(x_labels_dk, y_values_dk)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(250,80,80)', line=dict(color='rgb(250,80,80)', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_no,
          y=y_values_no,
          name='NO',
          hovertext=[f"År: {year}<br>Index - NO: {total}" for year, total in zip(x_labels_no, y_values_no)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(200,75,10)', line=dict(color='rgb(200,75,10)', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_fi,
          y=y_values_fi,
          name='FI',
          hovertext=[f"År: {year}<br>Index - FI: {total}" for year, total in zip(x_labels_fi, y_values_fi)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(20,200,220)', line=dict(color='rgb(20,200,220)', width=1.5)),
          opacity=0.6
      )
  ]

  # Define layout for a smaller width and a similar style as the previous plot
  layout_price = go.Layout(
      title={
          'text': 'Bostadsprisindex, total',
          'y': 0.90,  # Adjust the vertical position of the title
          'x': 0.25,  # Adjust the horizontal position of the title
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}
      },
      xaxis=dict(
          title='År',
          tickangle=90,
          tickfont=dict(size=10, color='rgb(107, 107, 107)'),
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True  # Frame effect
      ),
      yaxis=dict(
          title='Kvartalsindex, 2015=100',
          titlefont=dict(size=16, color='rgb(107, 107, 107)'),
          tickfont=dict(size=14, color='rgb(107, 107, 107)'),
          tickformat=',d',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True  # Frame effect
      ),
      plot_bgcolor='white',
      margin=dict(l=40, r=40, t=100, b=100),  # Adjust margins to match previous layout
      width=700,  # Smaller layout width
      template='plotly_white',
      annotations=[
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/PRC_HPI_Q/default/table">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.30,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05, y=1,
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Create figure
  fig_euro = go.Figure(data=data_euro_price, layout=layout_price)

  # Display plot in Streamlit
  st.plotly_chart(fig_euro)


  my_filter_pars = {'startPeriod': '2011-Q1', 'indic_bt': 'COST', 'unit': 'I15'}
  data = eurostat.get_data_df('sts_copi_q', filter_pars=my_filter_pars)

  import plotly.graph_objects as go

  # Select the row with the string "SE" under the column "geo\TIME_PERIOD"
  row = data.loc[data['geo\\TIME_PERIOD'] == 'SE']
  values = row.iloc[0, 2:]

  # Set the x-axis labels and y-axis values
  x_labels = values.index[28:]
  y_values = values.values[28:]

  # Select data for EU27, DK, FI, NO
  row_eu27 = data.loc[data['geo\\TIME_PERIOD'] == 'EU27_2020']
  values_eu27 = row_eu27.iloc[0, 2:]
  x_labels_eu27 = values_eu27.index[28:]
  y_values_eu27 = values_eu27.values[28:]

  row_dk = data.loc[data['geo\\TIME_PERIOD'] == 'DK']
  values_dk = row_dk.iloc[0, 2:]
  x_labels_dk = values_dk.index[28:]
  y_values_dk = values_dk.values[28:]

  row_fi = data.loc[data['geo\\TIME_PERIOD'] == 'FI']
  values_fi = row_fi.iloc[0, 2:]
  x_labels_fi = values_fi.index[28:]
  y_values_fi = values_fi.values[28:]

  row_no = data.loc[data['geo\\TIME_PERIOD'] == 'NO']
  values_no = row_no.iloc[0, 2:]
  x_labels_no = values_no.index[28:]
  y_values_no = values_no.values[28:]

  # Create traces for each country
  data_euro = [
      go.Scatter(
          x=x_labels,
          y=y_values,
          name='SE',
          hovertext=[f"År: {year}<br>Index - SE: {total}" for year, total in zip(x_labels, y_values)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='#0051BA', line=dict(color='#0051BA', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_eu27,
          y=y_values_eu27,
          name='EU',
          hovertext=[f"År: {year}<br>Index - EU27: {total}" for year, total in zip(x_labels_eu27, y_values_eu27)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='dash'),
          marker=dict(color='#FDB813', line=dict(color='#FDB813', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_dk,
          y=y_values_dk,
          name='DK',
          hovertext=[f"År: {year}<br>Index - DK: {total}" for year, total in zip(x_labels_dk, y_values_dk)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(250,80,80)', line=dict(color='rgb(250,80,80)', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_no,
          y=y_values_no,
          name='NO',
          hovertext=[f"År: {year}<br>Index - NO: {total}" for year, total in zip(x_labels_no, y_values_no)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(200,75,10)', line=dict(color='rgb(200,75,10)', width=1.5)),
          opacity=0.6
      ),
      go.Scatter(
          x=x_labels_fi,
          y=y_values_fi,
          name='FI',
          hovertext=[f"År: {year}<br>Index - FI: {total}" for year, total in zip(x_labels_fi, y_values_fi)],
          hoverinfo='text',
          mode='lines',
          line=dict(dash='solid'),
          marker=dict(color='rgb(20,200,220)', line=dict(color='rgb(20,200,220)', width=1.5)),
          opacity=0.6
      )
  ]

  # Define layout for a smaller width and a similar style as the previous plot
  layout_cost = go.Layout(
      title={
          'text': 'Byggkostnadsindex, nya bostadshus',
          'y': 0.90,  # Adjust the vertical position of the title
          'x': 0.30,  # Adjust the horizontal position of the title
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}
      },
      xaxis=dict(
          title='År',
          tickangle=90,
          tickfont=dict(size=10, color='rgb(107, 107, 107)'),
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True  # Frame effect
      ),
      yaxis=dict(
          title='Index, 2015=100',
          titlefont=dict(size=16, color='rgb(107, 107, 107)'),
          tickfont=dict(size=14, color='rgb(107, 107, 107)'),
          tickformat=',d',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True  # Frame effect
      ),
      plot_bgcolor='white',
      margin=dict(l=40, r=40, t=100, b=100),  # Adjust margins to match previous layout
      width=700,  # Smaller layout width
      template='plotly_white',
      annotations=[
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/STS_COPI_Q/default/table">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.30,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05, y=1,
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Create figure
  fig_euro = go.Figure(data=data_euro, layout=layout_cost)

  # Display plot in Streamlit
  st.plotly_chart(fig_euro)


  my_filter_pars = {'startPeriod': '2010', 'deg_urb': 'DEG1'}
  data = eurostat.get_data_df('ilc_lvho07d', filter_pars=my_filter_pars)

  import pandas as pd
  import plotly.graph_objects as go

  # Step 2: Prepare the Data
  # Rename 'geo\\TIME_PERIOD' for easier manipulation
  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

  # Step 3: Filter the Data to Keep Only Specific Countries
  # List of countries to keep
  countries_to_keep = ['NO', 'DK', 'FI', 'SE', 'EU27_2020']
  data = data[data['geo'].isin(countries_to_keep)]

  # Step 4: Melt the Data to Long Format
  # Melt the data to have a 'year' column and a 'value' column
  data_long = pd.melt(
      data,
      id_vars=['geo'],  # Keep the 'geo' column fixed
      value_vars=[str(year) for year in range(2010, 2024)],  # Columns representing years
      var_name='year',  # Name for the melted 'year' column
      value_name='value'  # Name for the melted 'value' column
  )

  # Convert 'year' to datetime format for correct plotting on the x-axis
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  # Step 5: Plot with Plotly
  fig = go.Figure()

  # Add a line for each country
  for country in data_long['geo'].unique():
      country_data = data_long[data_long['geo'] == country]
      country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
      fig.add_trace(go.Scatter(
          x=country_data['year'],
          y=country_data['value'],
          mode='lines+markers',
          name=country_label  # Set the name to match the legend entries
      ))

  # Create sub-heading as an annotation
  sub_heading_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.12,  # Adjust x position as needed
      y=1.03,  # Position just above the plot area, below the main title
      xanchor='center',
      yanchor='bottom',
      text='I städer, 2010-2023',  # Your sub-heading text
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False
  )

  # Customize Layout with a Frame and Improved Title Position
  fig.update_layout(
      title={
          'text': 'Hushåll med boendeutgifter över 40% av disponibel inkomst',
          'y': 0.90,  # Adjust to control the exact vertical position of the title
          'x': 0.43,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}  # Adjust font size as needed
      },
      xaxis=dict(
          title='År',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickangle=0,
          tickfont=dict(size=14)
      ),
      yaxis=dict(
          title='Procent',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickfont=dict(size=16)
      ),
      plot_bgcolor='white',
      margin=dict(
          l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
      ),
      width=700,
      template='plotly_white',
      annotations=[
          sub_heading_annotation,  # Add sub-heading annotation
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO07D__custom_7140801/bookmark/table?lang=en&bookmarkId=411e17fd-9b03-4729-8ad9-ea4844481e08">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.20,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05,
          y=1,
          traceorder='normal',
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Display the plot
  st.plotly_chart(fig)

  my_filter_pars = {'age': 'TOTAL', 'sex': 'T', 'incgrp': 'TOTAL'}
  data = eurostat.get_data_df('ilc_lvho07a', filter_pars=my_filter_pars)

  #plot_eurostat_data('ilc_mdes06', ['NO', 'DK', 'FI', 'SE', 'EU27_2020'], (2010, 2024), 'Housing cost overburden rate',
         #               'Housing cost overburden rate', 'Hela befolkningen, 2010-2023', 'År', 'Procent', 'https://ec.europa.eu/eurostat/databrowser/view/ILC_MDES06/default/table', additional_filter_pars={'age': 'TOTAL', 'sex': 'T'})

  import pandas as pd
  import plotly.graph_objects as go

  # Step 2: Prepare the Data
  # Rename 'geo\\TIME_PERIOD' for easier manipulation
  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

  # Step 3: Filter the Data to Keep Only Specific Countries
  # List of countries to keep
  countries_to_keep = ['NO', 'DK', 'FI', 'SE', 'EU27_2020']
  data = data[data['geo'].isin(countries_to_keep)]

  # Step 4: Melt the Data to Long Format
  # Melt the data to have a 'year' column and a 'value' column
  data_long = pd.melt(
      data,
      id_vars=['geo'],  # Keep the 'geo' column fixed
      value_vars=[str(year) for year in range(2010, 2024)],  # Columns representing years
      var_name='year',  # Name for the melted 'year' column
      value_name='value'  # Name for the melted 'value' column
  )

  # Convert 'year' to datetime format for correct plotting on the x-axis
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  # Step 5: Plot with Plotly
  fig = go.Figure()

  # Add a line for each country
  for country in data_long['geo'].unique():
      country_data = data_long[data_long['geo'] == country]
      country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
      fig.add_trace(go.Scatter(
          x=country_data['year'],
          y=country_data['value'],
          mode='lines+markers',
          name=country_label  # Set the name to match the x-axis labels
      ))

  # Create sub-heading as an annotation
  sub_heading_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.16,  # Adjust x position as needed
      y=1.03,  # Position just above the plot area, below the main title
      xanchor='center',
      yanchor='bottom',
      text='Hela befolkningen, 2010-2023',  # Your sub-heading text
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False
  )

  # Customize Layout with a Frame and Improved Title Position
  fig.update_layout(
      title={
          'text': 'Hushåll med boendeutgifter över 40% av disponibel inkomst',
          'y': 0.90,  # Adjust to control the exact vertical position of the title
          'x': 0.40,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}  # Adjust font size as needed
      },
      xaxis=dict(
          title='År',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickangle=0,
          tickfont=dict(size=14)
      ),
      yaxis=dict(
          title='Procent',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickfont=dict(size=16)
      ),
      plot_bgcolor='white',
      margin=dict(
          l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
      ),
      width=700,
      template='plotly_white',
      annotations=[
          sub_heading_annotation,  # Add sub-heading annotation
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO07A__custom_12778359/default/table?lang=en">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.20,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05,
          y=1,
          traceorder='normal',
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Display the plot
  fig.show()
  st.plotly_chart(fig)

  my_filter_pars = {'startPeriod': '2010', 'deg_urb': 'DEG1', 'hhtyp': 'TOTAL', 'incgrp': 'B_MD60'}
  data = eurostat.get_data_df('ilc_mded01', filter_pars=my_filter_pars)

  import pandas as pd
  import plotly.graph_objects as go

  # Step 2: Prepare the Data
  # Rename 'geo\\TIME_PERIOD' for easier manipulation
  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

  # Step 3: Filter the Data to Keep Only Specific Countries
  # List of countries to keep
  countries_to_keep = ['NO', 'DK', 'FI', 'SE', 'EU27_2020']
  data = data[data['geo'].isin(countries_to_keep)]

  # Step 4: Melt the Data to Long Format
  # Melt the data to have a 'year' column and a 'value' column
  data_long = pd.melt(
      data,
      id_vars=['geo'],  # Keep the 'geo' column fixed
      value_vars=[str(year) for year in range(2010, 2024)],  # Columns representing years
      var_name='year',  # Name for the melted 'year' column
      value_name='value'  # Name for the melted 'value' column
  )

  # Convert 'year' to datetime format for correct plotting on the x-axis
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  # Step 5: Plot with Plotly
  fig = go.Figure()

  # Add a line for each country
  for country in data_long['geo'].unique():
      country_data = data_long[data_long['geo'] == country]
      country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
      fig.add_trace(go.Scatter(
          x=country_data['year'],
          y=country_data['value'],
          mode='lines+markers',
          name=country_label  # Set the name to match the x-axis labels
      ))

  # Create sub-heading as an annotation
  sub_heading_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.30,  # Adjust x position as needed
      y=1.03,  # Position just above the plot area, below the main title
      xanchor='center',
      yanchor='bottom',
      text='Personer som kan anses vara i riskzonen för fattigdom, 2010-2023',  # Your sub-heading text
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False
  )

  # Customize Layout with a Frame and Improved Title Position
  fig.update_layout(
      title={
          'text': 'Boendekostnadens andel av hushållens disponibla inkomst',
          'y': 0.90,  # Adjust to control the exact vertical position of the title
          'x': 0.38,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}  # Adjust font size as needed
      },
      xaxis=dict(
          title='År',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickangle=0,
          tickfont=dict(size=14)
      ),
      yaxis=dict(
          title='Procent',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickfont=dict(size=16)
      ),
      plot_bgcolor='white',
      margin=dict(
          l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
      ),
      width=700,
      template='plotly_white',
      annotations=[
          sub_heading_annotation,  # Add sub-heading annotation
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_MDED01__custom_7140904/bookmark/table?lang=en&bookmarkId=659e8061-cde5-4ddb-b633-7cff3c16b7bd">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.20,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05,
          y=1,
          traceorder='normal',
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Display the plot
 #fig.show()
  st.plotly_chart(fig)

  my_filter_pars = {'startPeriod': '2010', 'deg_urb': 'DEG1', 'hhtyp': 'TOTAL', 'incgrp': 'TOTAL'}
  data = eurostat.get_data_df('ilc_mded01', filter_pars=my_filter_pars)

  import pandas as pd
  import plotly.graph_objects as go

  # Step 2: Prepare the Data
  # Rename 'geo\\TIME_PERIOD' for easier manipulation
  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

  # Step 3: Filter the Data to Keep Only Specific Countries
  # List of countries to keep
  countries_to_keep = ['NO', 'DK', 'FI', 'SE', 'EU27_2020']
  data = data[data['geo'].isin(countries_to_keep)]

  # Step 4: Melt the Data to Long Format
  # Melt the data to have a 'year' column and a 'value' column
  data_long = pd.melt(
      data,
      id_vars=['geo'],  # Keep the 'geo' column fixed
      value_vars=[str(year) for year in range(2010, 2024)],  # Columns representing years
      var_name='year',  # Name for the melted 'year' column
      value_name='value'  # Name for the melted 'value' column
  )

  # Convert 'year' to datetime format for correct plotting on the x-axis
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  # Step 5: Plot with Plotly
  fig = go.Figure()

  # Add a line for each country
  for country in data_long['geo'].unique():
      country_data = data_long[data_long['geo'] == country]
      country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
      fig.add_trace(go.Scatter(
          x=country_data['year'],
          y=country_data['value'],
          mode='lines+markers',
          name=country_label  # Set the name to match the x-axis labels
      ))

  # Create sub-heading as an annotation
  sub_heading_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.18,  # Adjust x position as needed
      y=1.03,  # Position just above the plot area, below the main title
      xanchor='center',
      yanchor='bottom',
      text='Hela befolkningen, 2010-2023',  # Your sub-heading text
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False
  )

  # Customize Layout with a Frame and Improved Title Position
  fig.update_layout(
      title={
          'text': 'Boendekostnadens andel av hushållens disponibla inkomst',
          'y': 0.90,  # Adjust to control the exact vertical position of the title
          'x': 0.38,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}  # Adjust font size as needed
      },
      xaxis=dict(
          title='År',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickangle=0,
          tickfont=dict(size=14)
      ),
      yaxis=dict(
          title='Procent',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickfont=dict(size=16)
      ),
      plot_bgcolor='white',
      margin=dict(
          l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
      ),
      width=700,
      template='plotly_white',
      annotations=[
          sub_heading_annotation,  # Add sub-heading annotation
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_MDED01__custom_7140904/bookmark/table?lang=en&bookmarkId=659e8061-cde5-4ddb-b633-7cff3c16b7bd">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.20,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05,
          y=1,
          traceorder='normal',
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Display the plot
  #fig.show()
  st.plotly_chart(fig)

  my_filter_pars = {'startPeriod': '2010', 'age': 'TOTAL', 'incgrp': 'TOTAL', 'sex': 'T'}
  data = eurostat.get_data_df('ilc_lvho05a', filter_pars=my_filter_pars)

  import pandas as pd
  import plotly.graph_objects as go

  # Step 2: Prepare the Data
  # Rename 'geo\\TIME_PERIOD' for easier manipulation
  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

  # Step 3: Filter the Data to Keep Only Specific Countries
  # List of countries to keep
  countries_to_keep = ['NO', 'DK', 'FI', 'SE', 'EU27_2020']
  data = data[data['geo'].isin(countries_to_keep)]

  # Step 4: Melt the Data to Long Format
  # Melt the data to have a 'year' column and a 'value' column
  data_long = pd.melt(
      data,
      id_vars=['geo'],  # Keep the 'geo' column fixed
      value_vars=[str(year) for year in range(2010, 2024)],  # Columns representing years
      var_name='year',  # Name for the melted 'year' column
      value_name='value'  # Name for the melted 'value' column
  )

  # Convert 'year' to datetime format for correct plotting on the x-axis
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  # Step 5: Plot with Plotly
  fig = go.Figure()

  # Add a line for each country
  for country in data_long['geo'].unique():
      country_data = data_long[data_long['geo'] == country]
      country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
      fig.add_trace(go.Scatter(
          x=country_data['year'],
          y=country_data['value'],
          mode='lines+markers',
          name=country_label  # Set the name to match the legend entries
      ))

  # Create sub-heading as an annotation
  sub_heading_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.18,  # Adjust x position as needed
      y=1.03,  # Position just above the plot area, below the main title
      xanchor='center',
      yanchor='bottom',
      text='Hela befolkningen, 2010-2023',  # Your sub-heading text
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False
  )

  # Customize Layout with a Frame and Improved Title Position
  fig.update_layout(
      title={
          'text': 'Andel av befolkningen i trångbodda hushåll',
          'y': 0.90,  # Adjust to control the exact vertical position of the title
          'x': 0.35,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}  # Adjust font size as needed
      },
      xaxis=dict(
          title='År',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickangle=0,
          tickfont=dict(size=14)
      ),
      yaxis=dict(
          title='Procent',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickfont=dict(size=16)
      ),
      plot_bgcolor='white',
      margin=dict(
          l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
      ),
      width=700,
      template='plotly_white',
      annotations=[
          sub_heading_annotation,  # Add sub-heading annotation
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO05A__custom_7141011/bookmark/table?lang=en&bookmarkId=ac6efb37-3f2f-4b65-9cd6-88e05c335bc1">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.20,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05,
          y=1,
          traceorder='normal',
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Display the plot
  fig.show()
  st.plotly_chart(fig)
  #The overcrowding rate is defined as the percentage of the population living in an overcrowded household.
  #A person is considered as living in an overcrowded household if the household does not have at its disposal a minimum number of rooms equal to:
  #one room for the household;
  #one room per couple in the household;
  #one room for each single person aged 18 or more;
  #one room per pair of single people of the same gender between 12 and 17 years of age;
  #one room for each single person between 12 and 17 years of age and not included in the previous category;
  #one room per pair of children under 12 years of age.

  my_filter_pars = {'startPeriod': '2010', 'age': 'TOTAL', 'incgrp': 'B_MD60', 'sex': 'T'}
  data = eurostat.get_data_df('ilc_lvho05a', filter_pars=my_filter_pars)

  import pandas as pd
  import plotly.graph_objects as go

  # Step 2: Prepare the Data
  # Rename 'geo\\TIME_PERIOD' for easier manipulation
  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

  # Step 3: Filter the Data to Keep Only Specific Countries
  # List of countries to keep
  countries_to_keep = ['NO', 'DK', 'FI', 'SE', 'EU27_2020']
  data = data[data['geo'].isin(countries_to_keep)]

  # Step 4: Melt the Data to Long Format
  # Melt the data to have a 'year' column and a 'value' column
  data_long = pd.melt(
      data,
      id_vars=['geo'],  # Keep the 'geo' column fixed
      value_vars=[str(year) for year in range(2010, 2024)],  # Columns representing years
      var_name='year',  # Name for the melted 'year' column
      value_name='value'  # Name for the melted 'value' column
  )

  # Convert 'year' to datetime format for correct plotting on the x-axis
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  # Step 5: Plot with Plotly
  fig = go.Figure()

  # Add a line for each country
  for country in data_long['geo'].unique():
      country_data = data_long[data_long['geo'] == country]
      country_label = 'EU' if country == 'EU27_2020' else country  # Rename 'EU27_2020' to 'EU'
      fig.add_trace(go.Scatter(
          x=country_data['year'],
          y=country_data['value'],
          mode='lines+markers',
          name=country_label  # Set the name to match the legend entries
      ))

  # Create sub-heading as an annotation
  sub_heading_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.35,  # Adjust x position as needed
      y=1.03,  # Position just above the plot area, below the main title
      xanchor='center',
      yanchor='bottom',
      text='Personer som kan anses vara i riskzonen för fattigdom, 2010-2023',  # Your sub-heading text
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False
  )

  # Customize Layout with a Frame and Improved Title Position
  fig.update_layout(
      title={
          'text': 'Andel av befolkningen i trångbodda hushåll',
          'y': 0.90,  # Adjust to control the exact vertical position of the title
          'x': 0.35,
          'xanchor': 'center',
          'yanchor': 'top',
          'font': {'size': 18}  # Adjust font size as needed
      },
      xaxis=dict(
          title='År',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickangle=0,
          tickfont=dict(size=14)
      ),
      yaxis=dict(
          title='Procent',
          showline=True,
          linewidth=1,
          linecolor='black',
          mirror=True,  # Create a frame effect
          tickfont=dict(size=16)
      ),
      plot_bgcolor='white',
      margin=dict(
          l=40, r=40, t=100, b=100  # Adjust margins to fit the titles and annotations
      ),
      width=700,
      template='plotly_white',
      annotations=[
          sub_heading_annotation,  # Add sub-heading annotation
          dict(
              text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ilc_lvho05a/default/table?lang=en">Eurostat</a>',
              xref='paper', yref='paper',
              x=1, y=-0.20,
              showarrow=False,
              font=dict(size=12)
          )
      ],
      legend=dict(
          x=1.05,
          y=1,
          traceorder='normal',
          title='Land',
          font=dict(family="Monaco, monospace", size=12, color="black")
      )
  )

  # Display the plot
  #fig.show()
  st.plotly_chart(fig)


  chat_completion = groq_client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": f"Based on the yearly data in the dataframe {data_long}, describe the data (look through {fig} to understand what the data is about; it shows the housing overcrowding rate for people in risk of poverty) with trends. Focus on Sweden and how its data relates to the other countries.",
          }
      ],
      model=llama_70B,
  )

  #st.write(chat_completion.choices[0].message.content)
  st.button("Reset", type="primary")
  if st.button("Analysera med AI"):
      st.write(chat_completion.choices[0].message.content)
  else:
      st.write(" ")

