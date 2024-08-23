import streamlit as st

st.title("Statistik!!! 📊")
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

    table.title = "Statistikuppdateringar sorterat efter datum ⏲"

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
          traceorder='normal'
      )
  )
  layout_bki_tot['title']['y'] = 0.89

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
        traceorder='normal'
    )
  )

  layout_bki_sma['title']['y'] = 0.89

  bki_sma = go.Figure(data=data_bki_sma, layout=layout_bki_sma, layout_width=1000)
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
        traceorder='normal'
    )
  )

  layout_bki_fler['title']['y'] = 0.89
  bki_fler = go.Figure(data=data_bki_fler, layout=layout_bki_fler, layout_width=1000)
  #pyo.iplot(bki_fler, filename='line-mode')

  bki_tot = go.Figure(data=data_bki_tot, layout=layout_bki_tot, layout_width=1000)
  st.plotly_chart(bki_tot)
  st.plotly_chart(bki_sma)
  st.plotly_chart(bki_fler)

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
      y=-0.2,
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
        traceorder='normal'
    )
  )

  # Create the figure
  barometerindikatorn = go.Figure(data=data_barometerindikatorn, layout=layout_barometerindikatorn, layout_width=1000)
  #pyo.iplot(barometerindikatorn, filename='line-mode')
  st.plotly_chart(barometerindikatorn)

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
      y=-0.2,
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
        traceorder='normal'
    )
  )

  layout_anbudspriser['title']['y'] = 0.89

  anbudspriser = go.Figure(data=data_anbudspriser, layout=layout_anbudspriser, layout_width=1000)
  #pyo.iplot(anbudspriser, filename='line-mode')
  st.plotly_chart(anbudspriser)

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
      y=-0.2,
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
        traceorder='normal'
    )
  )

  layout_orderstock['title']['y'] = 0.89

  orderstock = go.Figure(data=data_orderstock, layout=layout_orderstock, layout_width=1000)
  #pyo.iplot(orderstock, filename='line-mode')
  st.plotly_chart(orderstock)

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
      y=-0.2,
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
        traceorder='normal'
    )
  )

  layout_anstallningsplaner['title']['y'] = 0.89

  planer = go.Figure(data=data_anstallningsplaner, layout=layout_anstallningsplaner, layout_width=1000)
 #pyo.iplot(planer, filename='line-mode')
  st.plotly_chart(planer)

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
        traceorder='normal'
    ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )


  layout_hinder['title']['y'] = 0.89

  hinder = go.Figure(data=data_hinder, layout=layout_hinder, layout_width=1000)
 # pyo.iplot(hinder, filename='line-mode')
  st.plotly_chart(hinder)

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
        traceorder='normal'
    ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_hinder_hus['title']['y'] = 0.89

  hinder_hus = go.Figure(data=data_hinder_hus, layout=layout_hinder_hus, layout_width=1000)
  #pyo.iplot(hinder_hus, filename='line-mode')
  st.plotly_chart(hinder_hus)

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
        traceorder='normal'
    ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_hinder_anlaggning['title']['y'] = 0.89

  hinder_anlaggning = go.Figure(data=data_hinder_anlaggning, layout=layout_hinder_anlaggning, layout_width=1100)
  #pyo.iplot(hinder_anlaggning, filename='line-mode')
  st.plotly_chart(hinder_anlaggning, width=1100, height=900)

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
            "2024K1"
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
          font=dict(size=14)  # Set font size for legend text
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
            "2024K1"
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
          font=dict(size=14)  # Set font size for legend text
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
            "2024K1"
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
      x=0.30,
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
          font=dict(size=14)  # Set font size for legend text
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
            "2024K1"
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
      x=0.30,
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
        traceorder='normal'
    ),
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
  layout_combined["title"] = "Fasta bruttoinvesteringar"
  layout_combined['title']['y'] = 0.89

  combined = go.Figure(data=data_combined, layout=layout_combined, layout_width=900)
  st.plotly_chart(combined)

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
      y=-0.25,
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
        traceorder='normal'
    ),
      margin=dict(
          b=100  # Increase the bottom margin to provide more space for annotations
      )
  )

  layout_fardig['title']['y'] = 0.89
  fardig_tot = go.Figure(data=data_fardig, layout=layout_fardig)
  st.plotly_chart(fardig_tot)
