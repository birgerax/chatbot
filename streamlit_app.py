import streamlit as st

st.title("Statistik!!! 📊")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Statistikuppdateringar", "BP", "Konkurser", "Byggkostnadsindex", "KI", "Investeringar", "Ny- och ombyggnad", "Internationella jämförelser"])

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

    table.title = "Statistikuppdateringar sorterat efter datum"

    for _, category, variable in sorterade_datum:
      table.add_row([category, _.strftime("%Y-%m-%d")])
    st.write(table)
  display_data()
with tab2:
  # Data till diagram 3.3
  import requests
  import json

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
            "2024K1"
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
