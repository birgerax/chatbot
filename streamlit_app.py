import numpy as np
import streamlit as st
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from groq import Groq

st.title("Statistik 📊")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Datum", "Bostadsbestånd", "Hushållens boende", "BKI", "KI", "Investeringar", "Ny- och ombyggnad", "Eurostat", "BP"])

with tab1:
    #### Få datum för nästa SCB-publicering: byggande
  import requests
  from bs4 import BeautifulSoup

  url = "https://www.scb.se/hitta-statistik/statistik-efter-amne/boende-bebyggelse-och-mark/byggande-och-ombyggnad/bygglov-nybyggnad-och-ombyggnad/"
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

  url_bki = "https://www.scb.se/hitta-statistik/statistik-efter-amne/priser-och-ekonomiska-tendenser/priser/byggkostnadsindex-bki/"
  response_bki = requests.get(url_bki)
  soup_bki = BeautifulSoup(response_bki.text, "html.parser")
  datum_bki = soup_bki.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

  #### Få datum för nästa SCB-publicering: konkurser
  import requests
  from bs4 import BeautifulSoup

  url_k = "https://www.scb.se/hitta-statistik/statistik-efter-amne/naringsverksamhet-och-utrikeshandel/foretagens-demografi/konkurser-och-offentliga-ackord/"
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
  #datum_tj = soup_tj.find("div", {"class": "row topIntro"}).find("a", {"class": "link-large link-icon-calendar link-icon"}).find("span").text

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
  #datum_t = datum_tj.split(": ")[1].split("-")
  #datum_t = "-".join([datum_t[0], datum_t[1], datum_t[2]])

  # Printa kommande statistikuppdateringar i ordning efter närmast datum
  from datetime import datetime

  alla_datum = [(datetime.strptime(datum_bygg, "%Y-%m-%d"), "Bygglov, nybyggnad och ombyggnad", next_publication),
                (datetime.strptime(datum_fin, "%Y-%m-%d"), "Finansmarknadsstatistik", datum_finans),
                (datetime.strptime(datum_b, "%Y-%m-%d"), "Byggkostnadsindex", datum_bki),
                (datetime.strptime(datum_ko, "%Y-%m-%d"), "Konkurser", datum_k),
                (datetime.strptime(datum_n, "%Y-%m-%d"), "Nationalräkenskaper (bl.a. bostadsinvesteringar)", datum_nr)]
                #(datetime.strptime(datum_t, "%Y-%m-%d"), "Prisindex i producent- och importled (bl.a. tjänsteprisindex)", datum_tj)

  #alla_datum = [(datetime.strptime(datum_fin, "%Y-%m-%d"), "Finansmarknadsstatistik", datum_finans),
                #(datetime.strptime(datum_n, "%Y-%m-%d"), "Nationalräkenskaper (bl.a. bostadsinvesteringar)", datum_nr)]

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

  from datetime import datetime
  from io import BytesIO
  import base64

  def skapa_ics_b64(kategori, datum_str):
      """Skapar en base64-kodad .ics-sträng för ett möte kl. 08:00–09:00."""
      dtstart = datum_str.replace("-", "") + "T080000"
      dtend   = datum_str.replace("-", "") + "T090000"
      ics = (
          "BEGIN:VCALENDAR\n"
          "VERSION:2.0\n"
          "PRODID:-//SCB Stats//SV\n"
          "BEGIN:VEVENT\n"
          f"SUMMARY:{kategori}\n"
          f"DTSTART:{dtstart}\n"
          f"DTEND:{dtend}\n"
          f"DESCRIPTION:SCB publicerar ny statistik: {kategori}\n"
          "END:VEVENT\n"
          "END:VCALENDAR\n"
      )
      return base64.b64encode(ics.encode("utf-8")).decode("utf-8")

  # --- HTML-tabell med klickbara datum ---
  rader = ""
  for datum_obj, kategori, _ in sorterade_datum:
      datum_str = datum_obj.strftime("%Y-%m-%d")
      b64 = skapa_ics_b64(kategori, datum_str)
      rader += f"""
      <tr>
          <td>{kategori}</td>
          <td><a href="data:text/calendar;base64,{b64}" download="{kategori}.ics">{datum_str}</a></td>
      </tr>
      """

  html = f"""
  <style>
      .scb-table {{ border-collapse: collapse; width: 100%; font-size: 15px; }}
      .scb-table th {{ background-color: #f0f2f6; text-align: left; padding: 10px 14px; border: 1px solid #ddd; }}
      .scb-table td {{ padding: 9px 14px; border: 1px solid #ddd; }}
      .scb-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
      .scb-table tr:hover {{ background-color: #eef2fb; }}
      .scb-table a {{ color: #0068c9; text-decoration: none; }}
      .scb-table a:hover {{ text-decoration: underline; }}
  </style>
  <table class="scb-table">
      <thead>
          <tr>
              <th>Kategori</th>
              <th>Nästa publicering</th>
          </tr>
      </thead>
      <tbody>
          {rader}
      </tbody>
  </table>
  """

  import streamlit.components.v1 as components

  st.subheader("Statistikuppdateringar sorterat efter datum 📆")
  components.html(html, height=55 + len(sorterade_datum) * 45)

with tab2:
  import plotly.graph_objs as go
  import plotly.offline as pyo
  import pandas as pd
  import math

  # Counter to ensure unique keys for each download button
  download_counter = 0
  download_counter_excel = 0

  import sqlite3
  from datetime import datetime

  # Connect to SQLite database (it will create the file if it doesn't exist)
  #conn = sqlite3.connect('/content/plot.db')
  #cursor = conn.cursor()

  # Create a table for storing plot data
  #cursor.execute('''
  #CREATE TABLE IF NOT EXISTS plot_data (
  #    id INTEGER PRIMARY KEY AUTOINCREMENT,
  #    plot_title TEXT,
  #    x_values TEXT,
  #    y_values TEXT,
  #    source_url TEXT,
  #    created_at TIMESTAMP
  #)
  #''')

  # Commit changes to the database
  #conn.commit()

  #def store_plot_data(values_dict, keys, title, source_url):
  #  # Convert values_dict (y-values) and keys (x-values) to strings
  #  y_values = str(values_dict)
  #  x_values = str(keys)

    # Get the current timestamp
   # created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Insert data into the database
    #cursor.execute('''
    #INSERT INTO plot_data (plot_title, x_values, y_values, source_url, created_at)
    #VALUES (?, ?, ?, ?, ?)
    #''', (title, x_values, y_values, source_url, created_at))

    # Commit the changes
    #conn.commit()

  import streamlit as st
  import plotly.graph_objects as go
  import pandas as pd
  from io import StringIO

  FIG_HEIGHT = 500  # Standardhöjd – justera efter behov

  def create_bki_plot(values_dict, keys_kv, colors, title, source_url, underrubrik, bredd, source, rader_data, y_axis_label=None, index=False, bki=False, skip_ticks=False):
      min_length = min(len(values) for values in values_dict.values())
      keys_kv_trimmed = keys_kv[:min_length]

      df = pd.DataFrame({'Time': keys_kv_trimmed})

      # --- Tidsformatering ---
      if 'M' in df['Time'].iloc[0]:
          month_map = {
              '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
              '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
          }
          df['Month'] = df['Time'].str[5:7]
          df['Year']  = df['Time'].str[:4]
          df['Tid']   = df['Month'].map(month_map) + '-' + df['Year']

      elif 'K' in df['Time'].iloc[0]:
          quarter_map = {'K1': 'Q1', 'K2': 'Q2', 'K3': 'Q3', 'K4': 'Q4'}
          df['Quarter'] = df['Time'].str[4:6]
          df['Year']    = df['Time'].str[:4]
          df['Tid']     = df['Quarter'].map(quarter_map) + '-' + df['Year']

      else:
          df['Tid'] = df['Time']

      for label, values in values_dict.items():
          df[label] = values[:min_length]

      old_df = df
      df = df.iloc[rader_data:].reset_index(drop=True)

      # --- Traces ---
      data_bki_tot = []
      for i, (label, _) in enumerate(values_dict.items()):
          trace = go.Scatter(
              x=df['Tid'],
              y=df[label],
              name=label,
              hovertext=[f"Tidpunkt: {t}<br>{label}: {v}" for t, v in zip(df['Tid'], df[label])],
              hoverinfo='text',
              mode='lines',
              line=dict(
                  color=colors[i],
                  width=2.6,
                  dash='dash' if label == "Total" else None
              ),
              opacity=1,
              selected=dict(marker=dict(color='red')),
              unselected=dict(marker=dict(opacity=0.1))
          )
          data_bki_tot.append(trace)

      # --- Tick-positioner ---
      if 'K' in df['Time'].iloc[0]:
          tick_positions_all = [i for i, q in enumerate(df['Quarter']) if q == 'K1']
          tick_labels_all    = [df['Year'].iloc[i] for i in tick_positions_all]
      elif 'M' in df['Time'].iloc[0]:
          tick_positions_all = [i for i, m in enumerate(df['Month']) if m == '01']
          tick_labels_all    = [df['Year'].iloc[i] for i in tick_positions_all]
      else:
          tick_positions_all = df['Time'].tolist()
          tick_labels_all    = df['Time'].tolist()

      step_size = max(1, len(tick_positions_all) // 8)
      step = 2 if skip_ticks else step_size
      final_tick_positions = tick_positions_all[::step]
      final_tick_labels    = tick_labels_all[::step]

      # --- Y-axelns intervall ---
      visible_values = [v for label in values_dict for v in df[label].dropna()]
      y_min = min(visible_values) if visible_values else 0
      y_max = max(visible_values) if visible_values else 100
      y_margin = (y_max - y_min) * 0.05

      if index:
          y_axis_range = [y_min - 3, y_max + y_margin]
      elif bki:
          y_axis_range = [97, y_max + y_margin]
      else:
          y_axis_range = [min(0, y_min), y_max + y_margin]

      # --- Titel & underrubrik ---
      last_datapoint_time = df['Tid'].iloc[-1]
      subtitle_text = (
          f'Senaste utfall: {last_datapoint_time}'
          if underrubrik == "Senaste utfall"
          else underrubrik
      )

      # Titel + underrubrik i samma textsträng med HTML.
      # Underrubriken hamnar alltid direkt under titeln, oavsett figurbredd.
      combined_title = (
          f'{title}<br>'
          f'<span style="font-size:14px; color:#444; font-weight:normal;">'
          f'{subtitle_text}</span>'
      )

      # --- Layout ---
      layout = go.Layout(
          title=dict(
              text=combined_title,
              font=dict(size=18),
              x=0.07,
              xanchor='left',
              y=0.84,
              yanchor='top',
          ),
          height=FIG_HEIGHT,
          font=dict(size=14),
          xaxis=dict(
              tickvals=final_tick_positions,
              ticktext=final_tick_labels,
              tickangle=0,
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,
              tickfont=dict(size=14),
              tickcolor="#646464",
              ticks='outside',
              ticklen=5,
              **({'range': [0, len(df) - 1]} if ('K' in df['Time'].iloc[0] or 'M' in df['Time'].iloc[0]) else {}),
          ),
          yaxis=dict(
              range=y_axis_range,
              zeroline=(y_min <= 0 <= y_max),
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,
              title=y_axis_label,
              tickfont=dict(size=16),
          ),
          plot_bgcolor='white',
          yaxis_gridcolor='lightgray',
          legend=dict(
              x=1.05,
              y=1,
              traceorder='normal',
              font=dict(family="Monaco, monospace", size=12, color="black")
          ),
          margin=dict(t=120, b=70, r=80, l=60),
          annotations=[
              dict(
                  xref='paper', yref='paper',
                  x=0.0, y=-0.12,
                  xanchor='left', yanchor='top',
                  text=f'Källa: <a href="{source_url}">{source}</a>',
                  font=dict(size=12, color='black'),
                  showarrow=False,
              ),
          ],
      )

      config = {
          'toImageButtonOptions': {
              'format': 'png',
              'width': None,
              'height': None,
              'filename': 'high_quality_plot',
              'scale': 2
          },
          'displaylogo': False
      }

      fig = go.Figure(data=data_bki_tot, layout=layout)
      fig.update_layout(width=bredd)

      # --- Visa figur ---
      st.plotly_chart(fig, config=config)

      # --- Nedladdningsknappar ---
      def save_as_html(fig):
          buf = StringIO()
          fig.write_html(buf, include_plotlyjs='cdn', config=config)
          return buf.getvalue().encode('utf-8')

      def display_download_button(fig):
          global download_counter
          col1.download_button(
              label="📈 Hämta figur",
              data=save_as_html(fig),
              file_name="figure.html",
              mime="text/html",
              key=f"download_button_{download_counter}"
          )
          download_counter += 1

      def display_download_button_excel(df):
          global download_counter_excel
          col2.download_button(
              label='📥 Hämta data',
              data=to_excel(df.iloc[:, 1:]),
              file_name='data.xlsx',
              mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
              key=f"download_button_excel_{download_counter_excel}"
          )
          download_counter_excel += 1

      st.header("Hämta", divider=True)
      col1, col2, col3 = st.columns(3)
      display_download_button(fig)
      display_download_button_excel(old_df)
      return df


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
              "SMÅHUS",
              "FLERBOST",
              "ÖVRHUS",
              "SPEC"
            ]
          }
        },
        {
          "code": "Upplatelseform",
          "selection": {
            "filter": "item",
            "values": [
              "1",
              "2",
              "3",
              "ÖVRIGT"
            ]
          }
        }
      ],
      "response": {
        "format": "json"
      }
    }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0104/BO0104D/BO0104T04"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  keys = [entry['key'][3] for entry in response_json['data'] if entry['key'][1] == 'SMÅHUS' and entry['key'][2] == '1']
  values_hr = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][2] == '1']
  values_br = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][2] == '2']
  values_ar = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][2] == '3']
  values_saknas = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][2] == 'ÖVRIGT']

    # Define a function to sum values for each year given the repeated time series data
  def sum_yearly_values(values, num_years):
      series_count = 4       # Number of series per year in the values list

      # Reshape values into series_count sublists
      values_reshaped = [values[i * num_years:(i + 1) * num_years] for i in range(series_count)]

      # Sum the values for each year across the 4 series
      summed_values = [sum(year_values) for year_values in zip(*values_reshaped)]

      return summed_values  # Return only the list of summed values

  num_years = len(keys)  # Number of unique years

  # Compute the summed values for each list
  summed_values_hr = sum_yearly_values(values_hr, num_years)
  summed_values_br = sum_yearly_values(values_br, num_years)
  summed_values_ar = sum_yearly_values(values_ar, num_years)
  summed_values_saknas = sum_yearly_values(values_saknas, num_years)

  # Create a DataFrame with each column for the summed values
  df_summed_values = pd.DataFrame({
      'HR': summed_values_hr,
      'BR': summed_values_br,
      'AR': summed_values_ar,
      'Saknas': summed_values_saknas
  })

  values_dict = {
    'Hyresrätt': df_summed_values['HR'],
    'Bostadsrätt': df_summed_values['BR'],
    'Äganderätt': df_summed_values['AR'],
    #'Uppgift saknas': df_summed_values['Saknas'],
  }

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0,128,0)']
  title = "Antal lägenheter efter upplåtelseform och år"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0104__BO0104D/BO0104T04/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0)

  # Create a total column summing across the three categories for each year
  df_summed_values['Total'] = (
      df_summed_values['HR'] +
      df_summed_values['BR'] +
      df_summed_values['AR']
  )

  # Calculate the proportion for each category
  df_summed_values['Hyresrätt_Proportion'] = (df_summed_values['HR'] / df_summed_values['Total']) * 100
  df_summed_values['Bostadsrätt_Proportion'] = (df_summed_values['BR'] / df_summed_values['Total']) * 100
  df_summed_values['Äganderätt_Proportion'] = (df_summed_values['AR'] / df_summed_values['Total']) * 100

  # Organize proportions in a dictionary if needed
  values_dict_proportions = {
      'Hyresrätt': df_summed_values['Hyresrätt_Proportion'],
      'Bostadsrätt': df_summed_values['Bostadsrätt_Proportion'],
      'Äganderätt': df_summed_values['Äganderätt_Proportion']
  }

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0,128,0)']
  title = "Andel lägenheter efter upplåtelseform och år"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0104__BO0104D/BO0104T04/"
  create_bki_plot(values_dict_proportions, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0, y_axis_label = "Procent")

  keys = [entry['key'][3] for entry in response_json['data'] if entry['key'][1] == 'SMÅHUS' and entry['key'][2] == '1']
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'SMÅHUS']
  values_fler = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'FLERBOST']
  values_ovrig = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'ÖVRHUS']
  values_special = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'SPEC']

    # Define a function to sum values for each year given the repeated time series data
  def sum_yearly_values(values, num_years):
      series_count = 4       # Number of series per year in the values list

      # Reshape values into series_count sublists
      values_reshaped = [values[i * num_years:(i + 1) * num_years] for i in range(series_count)]

      # Sum the values for each year across the 4 series
      summed_values = [sum(year_values) for year_values in zip(*values_reshaped)]

      return summed_values  # Return only the list of summed values

  num_years = len(keys)  # Number of unique years

  # Compute the summed values for each list
  summed_values_sma = sum_yearly_values(values_sma, num_years)
  summed_values_fler = sum_yearly_values(values_fler, num_years)
  summed_values_ovrig = sum_yearly_values(values_ovrig, num_years)
  summed_values_special = sum_yearly_values(values_special, num_years)

  # Create a DataFrame with each column for the summed values
  df_summed_values = pd.DataFrame({
      'Småhus': summed_values_sma,
      'Flerbostadshus': summed_values_fler,
      'Övriga hus': summed_values_ovrig,
      'Specialbostäder': summed_values_special
  })

  values_dict = {
    'Flerbostadshus': df_summed_values['Flerbostadshus'],
    'Småhus': df_summed_values['Småhus'],
    'Övriga hus': df_summed_values['Övriga hus'],
    'Specialbostäder': df_summed_values['Specialbostäder'],
  }

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0,128,0)', 'rgb(255,165,0)']
  title = "Antal lägenheter efter hustyp och år"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0104__BO0104D/BO0104T04/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0)

    # Create a total column summing across the three categories for each year
  df_summed_values['Total'] = (
      df_summed_values['Småhus'] +
      df_summed_values['Flerbostadshus'] +
      df_summed_values['Övriga hus'] +
      df_summed_values['Specialbostäder']
  )

  # Calculate the proportion for each category
  df_summed_values['Småhus_Proportion'] = (df_summed_values['Småhus'] / df_summed_values['Total']) * 100
  df_summed_values['Flerbostadshus_Proportion'] = (df_summed_values['Flerbostadshus'] / df_summed_values['Total']) * 100
  df_summed_values['Övriga hus_Proportion'] = (df_summed_values['Övriga hus'] / df_summed_values['Total']) * 100
  df_summed_values['Specialbostäder_Proportion'] = (df_summed_values['Specialbostäder'] / df_summed_values['Total']) * 100

  # Organize proportions in a dictionary if needed
  values_dict_proportions = {
      'Flerbostadshus': df_summed_values['Flerbostadshus_Proportion'],
      'Småhus': df_summed_values['Småhus_Proportion'],
      'Övriga hus': df_summed_values['Övriga hus_Proportion'],
      'Specialbostäder': df_summed_values['Specialbostäder_Proportion']
  }

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0,128,0)', 'rgb(255,165,0)']
  title = "Andel lägenheter efter hustyp och år"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0104__BO0104D/BO0104T04/"
  create_bki_plot(values_dict_proportions, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0, y_axis_label = "Procent")

  # Extracting data for each house type and tenure status combination
  keys = [
      entry['key'][3]
      for entry in response_json['data']
      if entry['key'][2] == '1'  # Tenure status 1
  ]

  # Collecting values for each category of tenure status (1, 2, 3, 'ÖVRIGT') while considering the house type
  # SMÅHUS
  values_hr_småhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SMÅHUS' and entry['key'][2] == '1'
  ]

  values_br_småhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SMÅHUS' and entry['key'][2] == '2'
  ]

  values_ar_småhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SMÅHUS' and entry['key'][2] == '3'
  ]

  values_saknas_småhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SMÅHUS' and entry['key'][2] == 'ÖVRIGT'
  ]

  # FLERBOST
  values_hr_flerbost = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'FLERBOST' and entry['key'][2] == '1'
  ]

  values_br_flerbost = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'FLERBOST' and entry['key'][2] == '2'
  ]

  values_ar_flerbost = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'FLERBOST' and entry['key'][2] == '3'
  ]

  values_saknas_flerbost = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'FLERBOST' and entry['key'][2] == 'ÖVRIGT'
  ]

  # ÖVRHUS
  values_hr_ovrhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'ÖVRHUS' and entry['key'][2] == '1'
  ]

  values_br_ovrhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'ÖVRHUS' and entry['key'][2] == '2'
  ]

  values_ar_ovrhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'ÖVRHUS' and entry['key'][2] == '3'
  ]

  values_saknas_ovrhus = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'ÖVRHUS' and entry['key'][2] == 'ÖVRIGT'
  ]

  # SPEC
  values_hr_spec = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SPEC' and entry['key'][2] == '1'
  ]

  values_br_spec = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SPEC' and entry['key'][2] == '2'
  ]

  values_ar_spec = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SPEC' and entry['key'][2] == '3'
  ]

  values_saknas_spec = [
      float(entry['values'][0])
      for entry in response_json['data']
      if entry['key'][1] == 'SPEC' and entry['key'][2] == 'ÖVRIGT'
  ]

  # Create the dictionary with the data for each house type
  values_dict = {
      'Småhus': {
          'HR': values_hr_småhus,
          'BR': values_br_småhus,
          'ÄR': values_ar_småhus,
          'SAKNAS': values_saknas_småhus
      },
      'Flerbo': {
          'HR': values_hr_flerbost,
          'BR': values_br_flerbost,
          'ÄR': values_ar_flerbost,
          'SAKNAS': values_saknas_flerbost
      },
      'Övriga hus': {
          'HR': values_hr_ovrhus,
          'BR': values_br_ovrhus,
          'ÄR': values_ar_ovrhus,
          'SAKNAS': values_saknas_ovrhus
      },
      'Specialbostäder': {
          'HR': values_hr_spec,
          'BR': values_br_spec,
          'ÄR': values_ar_spec,
          'SAKNAS': values_saknas_spec
      }
  }

  # Define the colors for each house type
  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0,128,0)', 'rgb(255,165,0)']

  # Create the Plotly plot
  fig = go.Figure()

  # Add traces for each house type and tenure status
  for idx, (house_type, values) in enumerate(values_dict.items()):
      fig.add_trace(go.Bar(
          x=keys,
          y=values['HR'],
          name=f'{house_type} HR',
          marker_color=colors[idx]
      ))
      fig.add_trace(go.Bar(
          x=keys,
          y=values['BR'],
          name=f'{house_type} BR',
          marker_color=colors[idx]
      ))
      fig.add_trace(go.Bar(
          x=keys,
          y=values['ÄR'],
          name=f'{house_type} ÄR',
          marker_color=colors[idx]
      ))
      fig.add_trace(go.Bar(
          x=keys,
          y=values['SAKNAS'],
          name=f'{house_type} SAKNAS',
          marker_color=colors[idx]
      ))

  # Update the layout
  fig.update_layout(
      barmode='stack',  # Stack the bars
      title="Antal lägenheter efter hustyp och upplåtelseform",
      xaxis_title="År",
      yaxis_title="Antal lägenheter",
      legend_title="Hustyp och upplåtelseform",
      showlegend=True,
      xaxis={'categoryorder': 'category ascending'},
      template="plotly_white"
  )

  # Add a source URL (if applicable)
  fig.add_annotation(
      x=0,
      y=-0.25,
      xref="paper",
      yref="paper",
      text="Källa: <a href='https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0104__BO0104D/BO0104T04/' target='_blank'>SCB</a>",
      showarrow=False,
      font=dict(size=12),
      align="right"
  )

  # Display the plot in Streamlit
  st.plotly_chart(fig)

  import io
  # Prepare the data for the DataFrame
  data = []

  # Loop through the keys (x-values) and populate rows with the corresponding values
  for year_idx, year in enumerate(keys):
      for house_type, tenure_values in values_dict.items():
          # Safely access each list in tenure_values using a fallback (e.g., 0.0 if index is out of bounds)
          data.append({
              "År": year,
              "Hustyp": house_type,
              "HR": tenure_values["HR"][year_idx] if year_idx < len(tenure_values["HR"]) else 0.0,
              "BR": tenure_values["BR"][year_idx] if year_idx < len(tenure_values["BR"]) else 0.0,
              "ÄR": tenure_values["ÄR"][year_idx] if year_idx < len(tenure_values["ÄR"]) else 0.0,
              "SAKNAS": tenure_values["SAKNAS"][year_idx] if year_idx < len(tenure_values["SAKNAS"]) else 0.0,
          })

  # Convert the collected data into a DataFrame
  df = pd.DataFrame(data)

  # Filter out rows where all values are 0
  df_filtered = df.loc[~((df['HR'] == 0) & (df['BR'] == 0) & (df['ÄR'] == 0) & (df['SAKNAS'] == 0))]

  # Filter out rows where the year is repeated and all values are 0 (for the year 1990 or others if needed)
  # You can change 1990 to another year if the issue affects other years as well
  df_filtered = df_filtered[~((df_filtered['År'] == 1990) & (df_filtered[['HR', 'BR', 'ÄR', 'SAKNAS']] == 0).all(axis=1))]

  # Convert the filtered DataFrame into an Excel file and save it in memory
  output = io.BytesIO()
  with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
      df_filtered.to_excel(writer, index=False, sheet_name="Data")

  # No need to call save() explicitly, as `ExcelWriter` automatically saves when exiting the context

  # Retrieve the byte content of the Excel file
  df_xlsx = output.getvalue()

  # Add the download button for the Excel file
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)

  col1.download_button(
      label='📥 Hämta data',
      data=df_xlsx,
      file_name='data_filtered.xlsx',
      mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      key=f"download_button_excel_{download_counter_excel}"  # Unique key based on the counter
  )
  download_counter_excel += 1

  # Calculate total values per year for normalization
  totals_per_year = {year: 0 for year in keys}

  # Accumulate totals only for valid entries
  for house_type, tenures in values_dict.items():
      for tenure, values in tenures.items():
          for idx, year in enumerate(keys):
              if idx < len(values):  # Ensure valid index
                  totals_per_year[year] += values[idx]

  # Normalize the values to calculate proportions
  values_dict_normalized = {}
  for house_type, tenures in values_dict.items():
      values_dict_normalized[house_type] = {}
      for tenure, values in tenures.items():
          values_dict_normalized[house_type][tenure] = [
              (value / totals_per_year[year]) * 100 if idx < len(values) and totals_per_year[year] > 0 else 0
              for idx, (value, year) in enumerate(zip(values, keys))
          ]

  # Create the Plotly plot for proportions
  fig_proportions = go.Figure()

  # Add traces for each house type and tenure status (normalized)
  for idx, (house_type, values) in enumerate(values_dict_normalized.items()):
      fig_proportions.add_trace(go.Bar(
          x=keys,
          y=values['HR'],
          name=f'{house_type} HR',
          marker_color=colors[idx]
      ))
      fig_proportions.add_trace(go.Bar(
          x=keys,
          y=values['BR'],
          name=f'{house_type} BR',
          marker_color=colors[idx]
      ))
      fig_proportions.add_trace(go.Bar(
          x=keys,
          y=values['ÄR'],
          name=f'{house_type} ÄR',
          marker_color=colors[idx]
      ))
      fig_proportions.add_trace(go.Bar(
          x=keys,
          y=values['SAKNAS'],
          name=f'{house_type} SAKNAS',
          marker_color=colors[idx]
      ))

  # Update the layout for the proportions plot
  fig_proportions.update_layout(
      barmode='stack',  # Stack the bars
      title="Andel lägenheter efter hustyp och upplåtelseform",
      xaxis_title="År",
      yaxis_title="Andel (%)",
      legend_title="Hustyp och upplåtelseform",
      showlegend=True,
      xaxis={'categoryorder': 'category ascending'},
      yaxis=dict(tickformat=".0f"),  # Format percentages
      template="plotly_white"
  )

  # Add a source URL (if applicable)
  fig_proportions.add_annotation(
      x=0,
      y=-0.25,
      xref="paper",
      yref="paper",
      text="Källa: <a href='https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0104__BO0104D/BO0104T04/' target='_blank'>SCB</a> och egna beräkningar",
      showarrow=False,
      font=dict(size=12),
      align="right"
  )

  # Display the plot in Streamlit
  st.plotly_chart(fig_proportions)

  # Extract data for the last year
  last_year = keys[-1]  # Assuming keys are sorted and the last entry is the latest year

  # Prepare the data for the detailed pie chart
  pie_data_detailed = []
  labels = []
  colors = []  # Assign distinct colors for each label

  # Define base colors for house types
  house_type_colors = {
      'Småhus': 'rgb(8,48,107)',
      'Flerbo': 'rgb(204, 0, 0)',
      'Övriga hus': 'rgb(0,128,0)',
      'Specialbostäder': 'rgb(255,165,0)'
  }

 # Loop through house types and tenures to populate labels and values
  for house_type, tenures in values_dict.items():
      for tenure, values in tenures.items():
          labels.append(f"{house_type} - {tenure}")
          pie_data_detailed.append(values[-1])  # Get the last year's value
          colors.append(house_type_colors[house_type])

  # Create the pie chart
  fig_pie_detailed = go.Figure(
      data=[go.Pie(
          labels=labels,
          values=pie_data_detailed,
          marker=dict(
              colors=colors,
              line=dict(color='white', width=2)  # Add white border with 2px width
          ),
          hole=0.3  # Optional, makes it a donut chart
      )]
  )

  # Update layout for the pie chart
  fig_pie_detailed.update_layout(
      title=f"Andel lägenheter efter hustyp och upplåtelseform, år {last_year}",
      template="plotly_white",
      annotations=[
          dict(
              text="",
              x=0.5,
              y=0.5,
              font_size=15,
              showarrow=False
          )
      ]
  )

  # Display the pie chart in Streamlit
  st.plotly_chart(fig_pie_detailed)

with tab3:
  import requests
  import json
  import math

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
        "code": "Boendeform",
        "selection": {
          "filter": "item",
          "values": [
            "SMAG",
            "SMBO",
            "SMHY0",
            "FBBO",
            "FBHY0",
            "SPBO",
            "OB",
            "ÖVRIGT",
            "TOT"
          ]
        }
      },
      {
        "code": "Alder",
        "selection": {
          "filter": "item",
          "values": [
            "total"
          ]
        }
      },
      {
        "code": "Kon",
        "selection": {
          "filter": "item",
          "values": [
            "4"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/HE/HE0111/HE0111A/HushallT31"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  keys = []
  values_smag = []
  values_smbo = []
  values_smhyo = []
  values_fbbo = []
  values_fbhy0 = []
  values_spbo = []
  values_ob = []
  values_ovr = []
  values_tot = []

  # Loopa igenom data
  for entry in response_json['data']:
      key = entry['key'][4]  # År
      keys.append(key)
      value = entry['values'][0]

      if value != '..' and not math.isnan(float(value)):
          value = float(value)

          if entry['key'][1] == 'SMAG':
              values_smag.append(value)
          elif entry['key'][1] == 'SMBO':
              values_smbo.append(value)
          elif entry['key'][1] == 'SMHY0':
              values_smhyo.append(value)
          elif entry['key'][1] == 'FBBO':
              values_fbbo.append(value)
          elif entry['key'][1] == 'FBHY0':
              values_fbhy0.append(value)
          elif entry['key'][1] == 'SPBO':
              values_spbo.append(value)
          elif entry['key'][1] == 'OB':
              values_ob.append(value)
          elif entry['key'][1] == 'ÖVRIGT':
              values_ovr.append(value)
          elif entry['key'][1] == 'TOT':
              values_tot.append(value)

  # Beräkna andelen för varje kategori per år
  shares_smag = [v / t if t != 0 else 0 for v, t in zip(values_smag, values_tot)]
  shares_smbo = [v / t if t != 0 else 0 for v, t in zip(values_smbo, values_tot)]
  shares_smhyo = [v / t if t != 0 else 0 for v, t in zip(values_smhyo, values_tot)]
  shares_fbbo = [v / t if t != 0 else 0 for v, t in zip(values_fbbo, values_tot)]
  shares_fbhy0 = [v / t if t != 0 else 0 for v, t in zip(values_fbhy0, values_tot)]
  shares_spbo = [v / t if t != 0 else 0 for v, t in zip(values_spbo, values_tot)]
  shares_ob = [v / t if t != 0 else 0 for v, t in zip(values_ob, values_tot)]
  shares_ovr = [v / t if t != 0 else 0 for v, t in zip(values_ovr, values_tot)]
  # Extrahera unika år
  keys = sorted(set(keys))  # Skapa en lista med unika år

  values_dict = {
      'Småhus, äganderätt': [v * 100 for v in shares_smag],
      'Småhus, bostadsrätt': [v * 100 for v in shares_smbo],
      'Småhus, hyresrätt': [v * 100 for v in shares_smhyo],
      'Flerbostadshus, bostadsrätt': [v * 100 for v in shares_fbbo],
      'Flerbostadshus, hyresrätt': [v * 100 for v in shares_fbhy0],
      'Specialbostad': [v * 100 for v in shares_spbo],
      'Övrigt boende': [v * 100 for v in shares_ob],
      'Uppgift saknas': [v * 100 for v in shares_ovr]
  }


  colors = [
      'rgb(8,48,107)',    # Dark Blue
      'rgb(204, 0, 0)',   # Red
      'rgb(0,128,0)',     # Green
      'rgb(255, 165, 0)', # Orange
      'rgb(0,0,255)',     # Blue
      'rgb(255, 255, 0)', # Yellow
      'rgb(128,0,128)',   # Purple
      'rgb(255, 105, 180)' # Hot Pink
  ]

  title = "Andel personer efter boendeform, riket"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__HE__HE0111__HE0111A/HushallT31/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0, y_axis_label = "Procent")

  values_dict = {
      'Småhus': [sum(x) * 100 for x in zip(shares_smag, shares_smbo, shares_smhyo)],
      'Flerbostadshus': [sum(x) * 100 for x in zip(shares_fbbo, shares_fbhy0)],
      'Specialbostad': [v * 100 for v in shares_spbo],
      'Övrigt boende': [v * 100 for v in shares_ob],
      'Uppgift saknas': [v * 100 for v in shares_ovr]
  }

  colors = [
      'rgb(8,48,107)',    # Dark Blue
      'rgb(204, 0, 0)',   # Red
      'rgb(0,128,0)',     # Green
      'rgb(255, 165, 0)', # Orange
      'rgb(255, 105, 180)' # Hot Pink
  ]

  title = "Andel personer efter hustyp, riket"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__HE__HE0111__HE0111A/HushallT31/"

  create_bki_plot(values_dict, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0, y_axis_label = "Procent")

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
        "code": "Hushallstyp",
        "selection": {
          "filter": "item",
          "values": [
            #"LEK",
            #"LEKP",
            #"LEM",
            #"LEMP",
            #"LEKB",
            #"LEMB",
            #"LS",
            #"LS1",
            #"LS2",
            #"LS3",
            #"LÖ",
            #"LÖMB",
            "SAMTLH"
          ]
        }
      },
      {
        "code": "Boendeform",
        "selection": {
          "filter": "item",
          "values": [
            "SMAG",
            "SMBO",
            "SMHY0",
            "FBBO",
            "FBHY0",
            "SPBO",
            "OB",
            "OVR",
            "TOT"
          ]
        }
      },
      {
        "code": "ContentsCode",
        "selection": {
          "filter": "item",
          "values": [
            "HE0111EE"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/HE/HE0111/HE0111A/HushallT22"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  import math

  keys = []
  values_smag = []
  values_smbo = []
  values_smhyo = []
  values_fbbo = []
  values_fbhy0 = []
  values_spbo = []
  values_ob = []
  values_ovr = []
  values_tot = []

  # Loopa igenom data
  for entry in response_json['data']:
      key = entry['key'][3]  # År
      keys.append(key)
      value = entry['values'][0]

      if value != '..' and not math.isnan(float(value)):
          value = float(value)

          if entry['key'][2] == 'SMAG':
              values_smag.append(value)
          elif entry['key'][2] == 'SMBO':
              values_smbo.append(value)
          elif entry['key'][2] == 'SMHY0':
              values_smhyo.append(value)
          elif entry['key'][2] == 'FBBO':
              values_fbbo.append(value)
          elif entry['key'][2] == 'FBHY0':
              values_fbhy0.append(value)
          elif entry['key'][2] == 'SPBO':
              values_spbo.append(value)
          elif entry['key'][2] == 'OB':
              values_ob.append(value)
          elif entry['key'][2] == 'OVR':
              values_ovr.append(value)
          elif entry['key'][2] == 'TOT':
              values_tot.append(value)

  keys = sorted(set(keys))

  values_dict = {
    'Småhus, äganderätt': values_smag,
    'Småhus, bostadsrätt': values_smbo,
    'Småhus, hyresrätt': values_smhyo,
    'Flerbostadshus, bostadsrätt': values_fbbo,
    'Flerbostadshus, hyresrätt': values_fbhy0,
    'Specialbostad': values_spbo,
    'Övrigt boende': values_ob,
    'Uppgift saknas': values_ovr
  }

  colors = [
      'rgb(8,48,107)',    # Dark Blue
      'rgb(204, 0, 0)',   # Red
      'rgb(0,128,0)',     # Green
      'rgb(255, 165, 0)', # Orange
      'rgb(0,0,255)',     # Blue
      'rgb(255, 255, 0)', # Yellow
      'rgb(128,0,128)',   # Purple
      'rgb(255, 105, 180)' # Hot Pink
  ]

  title = "Andel hushåll efter boendeform, riket"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__HE__HE0111__HE0111A/HushallT22/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0, y_axis_label = "Procent")

  values_dict = {
    'Småhus': [sum(x) for x in zip(values_smag, values_smbo, values_smhyo)],
    'Flerbostadshus': [sum(x) for x in zip(values_fbbo, values_fbhy0)],
    'Specialbostad': values_spbo,
    'Övrigt boende': values_ob,
    'Uppgift saknas': values_ovr
  }

  colors = [
      'rgb(8,48,107)',    # Dark Blue
      'rgb(204, 0, 0)',   # Red
      'rgb(0,128,0)',     # Green
      'rgb(255, 165, 0)', # Orange
      'rgb(255, 105, 180)' # Hot Pink
  ]

  title = "Andel hushåll efter hustyp, riket"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__HE__HE0111__HE0111A/HushallT22/"

  create_bki_plot(values_dict, keys, colors, title, source_url, "", 600, 'SCB och egna beräkningar', 0, y_axis_label = "Procent")

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
            "BYGGKOST",
            "TRANS"
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

  keys_kv = [entry['key'][2] for entry in response_json['data']]
  values_fle = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TOTAL']
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'TOTAL']

  # Find the minimum length among all arrays
  min_length = min(len(keys_kv), len(values_fle), len(values_sma))

  # Trim each array to match the minimum length
  keys_kv_trimmed = keys_kv[:min_length]
  values_fle_trimmed = values_fle[:min_length]
  values_sma_trimmed = values_sma[:min_length]

  # Create a dictionary with the trimmed values
  values_dict = {
      'Flerbostadshus': values_fle_trimmed,
      'Gruppbyggda småhus': values_sma_trimmed
  }

  # Find the index for January 2021 in keys_kv (format 'YYYYMXX', i.e., '2021M01')
  if '2021M01' in keys_kv:
      january_2021_index = keys_kv.index('2021M01')
  else:
      raise ValueError("January 2021 ('2021M01') not found in keys_kv.")

  # Normalize each series in values_dict based on the value for January 2021
  for label in values_dict.keys():
      january_2021_value = values_dict[label][january_2021_index]

      # Normalize the entire series so that the value for January 2021 equals 100
      values_dict[label] = [(value / january_2021_value) * 100 for value in values_dict[label]]

  # Convert to DataFrame
  combined_df = pd.DataFrame({'Time': keys_kv_trimmed, **values_dict})
  rader = 72
  combined_df = combined_df.iloc[rader:]

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
  title = "Byggkostnadsindex för bostäder exkl. löneglidning och moms"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/"
  create_bki_plot(values_dict, keys_kv, colors, title, source_url, 'Index, januari 2021 = 100', 600, 'SCB och egna beräkningar', 72, bki=True)
  #df_bki = create_bki_plot(values_dict, keys_kv, colors, title, source_url, 'Index, januari 2021 = 100', 600, 'SCB och egna beräkningar', 72, bki=True)

  st.header("Senaste utfall", divider=True)
  col1, col2, col3 = st.columns(3)
  #latest_value = values_fle[-1]
  latest_value = values_dict['Flerbostadshus'][-1]
  previous_value = values_dict['Flerbostadshus'][-2]
  #previous_value = values_fle[-2]

  # Calculate the percentage change
  percentage_change = (latest_value / previous_value - 1) * 100

  # Format the numeric value with spaces as thousands separators
  latest_formatted_value = f"{latest_value:,.1f}".replace(',', ' ')

  # Format the percentage change to one decimal place with a comma instead of a dot
  formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'

  #latest_value_sma = values_sma[-1]
  #previous_value_sma = values_sma[-2]
  latest_value_sma = values_dict['Gruppbyggda småhus'][-1]
  previous_value_sma = values_dict['Gruppbyggda småhus'][-2]

  percentage_change_sma = (latest_value_sma / previous_value_sma - 1) * 100
  latest_formatted_value_sma = f"{latest_value_sma:,.1f}".replace(',', ' ')
  formatted_change_sma = f"{percentage_change_sma:.1f}".replace('.', ',') + '%'

  col1.metric(f"Flerbostadshus" , latest_formatted_value, formatted_change)
  col2.metric(f"Gruppbyggda småhus" , latest_formatted_value_sma, formatted_change_sma)

  from io import StringIO

  def create_bki_yoy_plot(values_dict, keys_kv, colors, title, source_url, source):
      min_length = min(len(values) for values in values_dict.values())
      keys_kv_trimmed = keys_kv[:min_length]

      df = pd.DataFrame({'Time': keys_kv_trimmed})

      month_map = {
          '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
          '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
      }
      df['Month'] = df['Time'].str[5:7]
      df['Year']  = df['Time'].str[:4]
      df['Tid']   = df['Month'].map(month_map) + '-' + df['Year']

      for label, values in values_dict.items():
          df[label] = values[:min_length]

      for label in values_dict.keys():
          df[f'{label} YoY'] = df[label].pct_change(periods=12) * 100

      df = df.iloc[72:]

      tick_positions_all = [i for i, month in enumerate(df['Month']) if month == '01']
      tick_labels_all    = [df['Year'].iloc[i] for i in tick_positions_all]
      step_size          = max(1, len(tick_positions_all) // 10)
      final_tick_positions = tick_positions_all[::step_size]
      final_tick_labels    = tick_labels_all[::step_size]

      data_bki_tot = []
      for idx, (label, _) in enumerate(values_dict.items()):
          data_bki_tot.append(go.Scatter(
              x=df['Tid'],
              y=df[f'{label} YoY'],
              mode='lines',
              name=label,
              line=dict(color=colors[idx], width=2)
          ))

      # Titel + underrubrik ihopslagen precis som i create_bki_plot
      combined_title = (
          f'{title}<br>'
          f'<span style="font-size:14px; color:#444; font-weight:normal;">'
          f'Årlig procentuell förändring</span>'
      )

      layout = go.Layout(
          title=dict(
              text=combined_title,
              font=dict(size=18),
              x=0.07,
              xanchor='left',
              y=0.84,
              yanchor='top',
          ),
          height=500,
          font=dict(size=18),
          xaxis=dict(
              tickvals=final_tick_positions,
              ticktext=final_tick_labels,
              nticks=len(final_tick_labels),
              tickangle=0,
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,
              tickfont=dict(size=14),
              tickcolor="#646464",
              ticks='outside',
              ticklen=5,
          ),
          yaxis=dict(
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,
              tickfont=dict(size=16),
              nticks=12,
          ),
          plot_bgcolor='white',
          yaxis_gridcolor='lightgray',
          legend=dict(
              x=1.05,
              y=1,
              traceorder='normal',
              font=dict(family="Monaco, monospace", size=12, color="black"),
              tracegroupgap=50,
          ),
          margin=dict(t=120, b=70, r=80, l=60),
          annotations=[
              dict(
                  xref='paper', yref='paper',
                  x=0.0, y=-0.15,
                  xanchor='left', yanchor='top',
                  text=f'Källa: <a href="{source_url}">{source}</a>',
                  font=dict(size=12, color='black'),
                  showarrow=False,
              ),
          ],
      )

      config = {
          'toImageButtonOptions': {
              'format': 'png',
              'filename': 'high_quality_plot',
              'scale': 2,
          },
          'displaylogo': False
      }

      fig = go.Figure(data=data_bki_tot, layout=layout)
      fig.update_layout(width=600)
      st.plotly_chart(fig, config=config)

      # --- Nedladdningsknappar ---
      def save_as_html(fig):
          buf = StringIO()
          fig.write_html(buf, include_plotlyjs='cdn', config=config)
          return buf.getvalue().encode('utf-8')

      def display_download_button(fig):
          global download_counter
          col1.download_button(
              label="📈 Hämta figur",
              data=save_as_html(fig),
              file_name="figure.html",
              mime="text/html",
              key=f"download_button_{download_counter}"
          )
          download_counter += 1

      def display_download_button_excel(df):
          global download_counter_excel
          col2.download_button(
              label='📥 Hämta data',
              data=to_excel(df.iloc[:, :]),
              file_name='data.xlsx',
              mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
              key=f"download_button_excel_{download_counter_excel}"
          )
          download_counter_excel += 1

      st.header("Hämta", divider=True)
      col1, col2, col3 = st.columns(3)
      display_download_button(fig)
      display_download_button_excel(df)


  keys_kv = [entry['key'][2] for entry in response_json['data']]
  values_fle = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TOTAL']
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'TOTAL']

  values_dict = {
      'Flerbostadshus': values_fle,
      'Gruppbyggda småhus': values_sma
  }

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
  title = "Byggkostnadsindex för bostäder exkl. löneglidning och moms"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/"
  create_bki_yoy_plot(values_dict, keys_kv, colors, title, source_url, 'SCB och egna beräkningar')

  min_length = min(len(values) for values in [values_sma, values_fle])
  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'TOTAL']
  values_arbl = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'ARBL']
  values_mask = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'MASK']
  values_mate = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'MATERIAL']
  values_kost = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'BYGGKOST']
  values_etti = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'GRUPPSMÅ' and entry['key'][1] == 'ETTILLFEM']
  values_trans = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TRANS']

  values_dict = {
    'Total': values_sma[:min_length],
    #'Löner': values_arbl[:min_length],
    #'Maskiner': values_mask[:min_length],
    #'Entreprenörernas kostnad': values_etti[:min_length],
    'Material': values_mate[:min_length],
    'Byggherrekostnad': values_kost[:min_length],
    'Transport': values_trans[:min_length]
  }

  # Find the index for January 2021 in keys_kv (format 'YYYYMXX', i.e., '2021M01')
  if '2021M01' in keys_kv:
      january_2021_index = keys_kv.index('2021M01')
  else:
      raise ValueError("January 2021 ('2021M01') not found in keys_kv.")

  # Normalize each series in values_dict based on the value for January 2021
  for label in values_dict.keys():
      january_2021_value = values_dict[label][january_2021_index]

      # Normalize the entire series so that the value for January 2021 equals 100
      values_dict[label] = [(value / january_2021_value) * 100 for value in values_dict[label]]

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'orange', '#509D00', '#004B84', '#E87502']
  title = "Byggkostnadsindex för gruppbyggda småhus"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/"
  create_bki_plot(values_dict, keys_kv, colors, title, source_url, 'Index, januari 2021 = 100', 600, 'SCB och egna beräkningar', 72, index=False, bki=True)

  latest_value_sma = values_sma[-1]
  previous_value_sma = values_sma[-2]

  values_sma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TOTAL']
  values_arbl = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'ARBL']
  values_mask = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'MASK']
  values_mate = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'MATERIAL']
  values_kost = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'BYGGKOST']
  values_etti = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'ETTILLFEM']
  values_trans = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'FLERBO' and entry['key'][1] == 'TRANS']

  values_dict = {
    'Total': values_sma[:min_length],
    #'Löner': values_arbl[:min_length],
    #'Maskiner': values_mask[:min_length],
    #'Entreprenörernas kostnad': values_etti[:min_length],
    'Material': values_mate[:min_length],
    'Byggherrekostnad': values_kost[:min_length],
    'Transport': values_trans[:min_length]
  }

  # Find the index for January 2021 in keys_kv (format 'YYYYMXX', i.e., '2021M01')
  if '2021M01' in keys_kv:
      january_2021_index = keys_kv.index('2021M01')
  else:
      raise ValueError("January 2021 ('2021M01') not found in keys_kv.")

  # Normalize each series in values_dict based on the value for January 2021
  for label in values_dict.keys():
      january_2021_value = values_dict[label][january_2021_index]

      # Normalize the entire series so that the value for January 2021 equals 100
      values_dict[label] = [(value / january_2021_value) * 100 for value in values_dict[label]]

  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'orange', '#509D00', '#004B84', '#E87502']
  title = "Byggkostnadsindex för flerbostadshus"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/FPIBOM2015/"
  create_bki_plot(values_dict, keys_kv, colors, title, source_url, 'Index, januari 2021 = 100', 600, 'SCB och egna beräkningar', 72, index=False, bki=True)
  create_bki_yoy_plot(values_dict, keys_kv, colors, title, source_url, 'SCB och egna beräkningar')

  # Convert `keys_kv` to a DataFrame
  keys_df_år = pd.DataFrame({'Key': keys_kv})

  # Convert `values_dict` to a DataFrame
  values_df_år = pd.DataFrame(values_dict)

  # Combine the DataFrames
  combined_df_år = pd.concat([keys_df_år, values_df_år], axis=1)
  combined_df_år = combined_df_år.iloc[72:120]

  month_map = {
          '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'maj', '06': 'jun',
          '07': 'jul', '08': 'aug', '09': 'sep', '10': 'okt', '11': 'nov', '12': 'dec'
  }

  combined_df_år['Key'] = combined_df_år['Key'].str[5:].map(month_map) + '-' + combined_df_år['Key'].str[2:4]
  combined_df_år = combined_df_år.rename(columns={"Key": "Time"})  # Rename column

  import requests
  import json

  session = requests.Session()

  query = {
    "query": [],
    "response": {
      "format": "json"
    }
  }

  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/PR/PR0502/PR0502A/BKIMAM"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  # Step 1: Get all unique time periods
  keys = sorted(set(entry['key'][1] for entry in response_json['data']))

  custom_names = {
    'TRA': 'Trävaror',
    'BET': 'Betong',
    'SNICK': 'Snickerier',
    'JOS': 'Järn- och stålvaror',
    'DARM': 'Armeringsstål',
    'VIT': 'Vita varor',
    'GOLV': 'Golvmaterial',
    'MAL': 'Material för målning',
    'OVR': 'Övrigt byggmaterial',
    'BTOT': 'Byggmästerivaror total',
    'VS': 'VVS-material',
    'EL': 'El-material'
  }

  # Step 2: Group values by key[0]
  values_dict = {}
  for entry in response_json['data']:
      original_category = entry['key'][0]  # This is 'BET', 'P010', or other values
      category_name = custom_names.get(original_category, original_category)  # Use mapping, default to original name

      if category_name not in values_dict:
          values_dict[category_name] = []

      values_dict[category_name].append(float(entry['values'][0]))

  colors = [
      'rgb(8,48,107)',   # Deep Blue
      'rgb(204, 0, 0)',  # Strong Red
      'rgb(255,140,0)',  # Dark Orange
      'rgb(80,157,0)',   # Deep Green
      'rgb(0,75,132)',   # Navy Blue
      'rgb(232,117,2)',  # Warm Orange
      'rgb(128,0,128)',  # Purple
      'rgb(0,128,128)',  # Teal
      'rgb(75,0,130)',   # Indigo
      'rgb(220,20,60)',  # Crimson
      'rgb(34,139,34)',  # Forest Green
      'rgb(70,130,180)'  # Steel Blue
  ]

  title = "Byggkostnadsindex för flerbostadshus efter byggnadsmaterial"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__PR__PR0502__PR0502A/BKIMAM/"
  create_bki_plot(values_dict, keys, colors, title, source_url, 'Index, 1968 = 100', 600, 'SCB', 72, index=False, bki=True)
  create_bki_yoy_plot(values_dict, keys, colors, title, source_url, 'SCB och egna beräkningar')

  #@st.cache_data
  #def save_figure_as_image(fig, format='png'):
      # Save the figure to a BytesIO object
  #    img_bytes = BytesIO()
  #    fig.write_image(img_bytes, format=format, scale=2)  # Increase scale for higher resolution
  #    img_bytes.seek(0)
  #    return img_bytes

  #st.header("Hämta", divider=True)
  # Save the figure as a high-resolution image (PNG format by default)
  #image_data = save_figure_as_image(bki_tot, format='png')

  # Add a download button for the figure image
  #st.download_button(
  #    label="📈 Hämta figur",
  #    data=image_data,
  #    file_name="figure.png",
  #    mime="image/png",
  #    key="1"
  #)

  # Extract columns from df_tot and df_sma, excluding the first column ('Time')
  #df_tot_columns = df_tot.iloc[:, 1:]
  #df_sma_columns = df_sma.iloc[:, 1:]
  # Create gap columns filled with NaN to place between DataFrames
  #gap_column_1 = pd.DataFrame(np.nan, index=df.index, columns=[''])
  #gap_column_2 = pd.DataFrame(np.nan, index=df.index, columns=[''])
  # Concatenate df, gap_column_1, df_tot_columns, gap_column_2, and df_sma_columns horizontally
  #df_combined = pd.concat([df, gap_column_1, df_tot_columns, gap_column_2, df_sma_columns], axis=1)

  #df_xlsx = to_excel(df_combined)
  #col2.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="10")

with tab5:

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

  values_dict = {
    'Byggindustri (SNI 41-43)': values_bbyg,
    'Bygg & anläggning (SNI 41-42)': values_bboa,
    'Husbyggande (SNI 41)': values_b41,
    'Anläggningsverksamhet (SNI 42)': values_b42,
    'Specialiserad byggverksamhet (SNI 43)': values_b43,
    'Totala näringslivet': values_btot,
  }

  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']
  title = "Konfidensindikatorn"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__indikatorer/Indikatorm.px/"
  df_kbarometern = create_bki_plot(values_dict, keys_barometer, colors, title, source_url, "Index medelvärde = 100", 700, 'Konjunkturinstitutet', -61, index=True)

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

  barometer_bbyg = pd.DataFrame({'År': keys_barometer[:len(values_bbyg)], 'Byggindustri (SNI 41-43)': values_bbyg})
  barometer_bboa = pd.DataFrame({'År2': keys_barometer[:len(values_bbyg)], 'Bygg & anläggning (SNI 41-42)': values_bboa})
  barometer_b41 = pd.DataFrame({'År3': keys_barometer[:len(values_bbyg)], 'Husbyggande (SNI 41)': values_b41})
  barometer_b42 = pd.DataFrame({'År4': keys_barometer[:len(values_bbyg)], 'Anläggningsverksamhet (SNI 42)': values_b42})
  barometer_b43 = pd.DataFrame({'År5': keys_barometer[:len(values_bbyg)], 'Specialiserad byggverksamhet (SNI 43)': values_b43})
  barometer_btot = pd.DataFrame({'År6': keys_barometer[:len(values_bbyg)], 'Totala näringslivet': values_btot})

  df_barometer = pd.concat([barometer_bbyg, barometer_bboa, barometer_b41, barometer_b42, barometer_b43, barometer_btot], axis=1)

  # Remove the third column
  df_barometer = df_barometer.drop(['År2', 'År3', 'År4', 'År5', 'År6'], axis=1)

### Byggandet

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
            "101",
            #"201"
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

  keys = []
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
          keys.append(entry['key'][3])

  values_dict = {
    'Byggindustri (SNI 41-43)': values_bbyg,
    'Bygg & anläggning (SNI 41-42)': values_bboa,
    'Husbyggande (SNI 41)': values_b41,
    'Anläggningsverksamhet (SNI 42)': values_b42,
    'Specialiserad byggverksamhet (SNI 43)': values_b43,
  }

  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']
  title = "Byggandet, utfall (säsongsrensat)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  df_byggande_utfall = create_bki_plot(values_dict, keys, colors, title, source_url, "Senaste utfall", 700, 'Konjunkturinstitutet', -61, index=True)

  session = requests.Session()

  query = {
    "query": [
      {
        "code": "Fråga",
        "selection": {
          "filter": "item",
          "values": [
            #"101",
            "201"
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

  keys = []
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
          keys.append(entry['key'][3])

  values_dict = {
    'Byggindustri (SNI 41-43)': values_bbyg,
    'Bygg & anläggning (SNI 41-42)': values_bboa,
    'Husbyggande (SNI 41)': values_b41,
    'Anläggningsverksamhet (SNI 42)': values_b42,
    'Specialiserad byggverksamhet (SNI 43)': values_b43,
  }

  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']
  title = "Byggandet, förväntningar (säsongsrensat)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  df_byggande = create_bki_plot(values_dict, keys, colors, title, source_url, "Senaste utfall", 700, 'Konjunkturinstitutet', -61, index=True)

  #st.header("Hämta", divider=True)
  #df_xlsx = to_excel(df)
  #st.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="8")

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

  values_dict = {
    'Byggindustri (SNI 41-43)': values_bbyg,
    'Bygg & anläggning (SNI 41-42)': values_bboa,
    'Husbyggande (SNI 41)': values_b41,
    'Anläggningsverksamhet (SNI 42)': values_b42,
    'Specialiserad byggverksamhet (SNI 43)': values_b43,
  }

  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']
  title = "Orderstock, nulägesomdöme (säsongsrensat)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  create_bki_plot(values_dict, keys_barometer, colors, title, source_url, "Senaste utfall", 700, 'Konjunkturinstitutet', -61, index=True)

  #st.header("Hämta", divider=True)
  #df_xlsx = to_excel(df)
  #st.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="7")

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

  values_dict = {
    'Byggindustri (SNI 41-43)': values_bbyg,
    'Bygg & anläggning (SNI 41-42)': values_bboa,
    'Husbyggande (SNI 41)': values_b41,
    'Anläggningsverksamhet (SNI 42)': values_b42,
    'Specialiserad byggverksamhet (SNI 43)': values_b43,
  }

  min_length = min(len(keys_planer), len(values_bbyg), len(values_bboa),
                 len(values_b41), len(values_b42), len(values_b43))

  keys_planer_trimmed = keys_planer[:min_length]
  values_dict_trimmed = {
    'Byggindustri': values_bbyg[:min_length],
    'Bygg & anläggning': values_bboa[:min_length],
    'Husbyggande': values_b41[:min_length],
    'Anläggningsverksamhet': values_b42[:min_length],
    'Specialiserad byggverksamhet': values_b43[:min_length],
}

  # Convert to DataFrame
  combined_df_planer = pd.DataFrame({'Time': keys_planer_trimmed, **values_dict_trimmed})

  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']
  title = "Anställningsplaner (säsongsrensat)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  create_bki_plot(values_dict, keys_barometer, colors, title, source_url, "Senaste utfall", 700, 'Konjunkturinstitutet', -61, index=True)

  #st.header("Hämta", divider=True)
  #df_xlsx = to_excel(df)
  #st.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="6")

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

  values_dict = {
    'Byggindustri (SNI 41-43)': values_bbyg,
    'Bygg & anläggning (SNI 41-42)': values_bboa,
    'Husbyggande (SNI 41)': values_b41,
    'Anläggningsverksamhet (SNI 42)': values_b42,
    'Specialiserad byggverksamhet (SNI 43)': values_b43,
  }

  colors = ['#A2B1B3', '#D9CC00', '#2BB2F7', '#509D00', '#004B84', '#E87502']
  title = "Anbudspriser, utfall (säsongsrensat)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  create_bki_plot(values_dict, keys_barometer, colors, title, source_url, "Senaste utfall", 700, 'Konjunkturinstitutet', -61, index=True)

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

  values_dict = {
    'Efterfrågan': values_efterfragan,
    'Material och/eller utrustning': values_material,
    'Arbetskraft': values_arbetskraft,
    'Finansiella restriktioner': values_finans,
    'Annat': values_annat,
  }

  # Ensure all lists are of equal length
  min_length = min(len(keys_hinder), len(values_efterfragan), len(values_material),
                 len(values_arbetskraft), len(values_finans), len(values_annat))

  keys_hinder_trimmed = keys_hinder[:min_length]
  values_dict_trimmed = {
    'Efterfrågan': values_efterfragan[:min_length],
    'Material och/eller utrustning': values_material[:min_length],
    'Arbetskraft': values_arbetskraft[:min_length],
    'Finansiella restriktioner': values_finans[:min_length],
    'Annat': values_annat[:min_length],
}

  # Convert to DataFrame
  combined_df = pd.DataFrame({'Time': keys_hinder_trimmed, **values_dict_trimmed})

  colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']
  title = "Främsta hinder för byggande (hela byggindustrin)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  create_bki_plot(values_dict, keys_hinder, colors, title, source_url, "Senaste utfall", 650, 'Konjunkturinstitutet', -61)

  #st.header("Hämta", divider=True)
  #df_xlsx = to_excel(df)
  #st.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="5")

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

  values_dict = {
    'Efterfrågan': values_efterfragan,
    'Material och/eller utrustning': values_material,
    'Arbetskraft': values_arbetskraft,
    'Finansiella restriktioner': values_finans,
    'Annat': values_annat,
  }

  keys_hinder_trimmed = keys_hinder[:min_length]
  values_dict_trimmed = {
    'Efterfrågan': values_efterfragan[:min_length],
    'Material och/eller utrustning': values_material[:min_length],
    'Arbetskraft': values_arbetskraft[:min_length],
    'Finansiella restriktioner': values_finans[:min_length],
    'Annat': values_annat[:min_length],
}

  # Convert to DataFrame
  combined_df_hus = pd.DataFrame({'Time': keys_hinder_trimmed, **values_dict_trimmed})

  colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']
  title = "Främsta hinder för husbyggande (SNI 41)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  create_bki_plot(values_dict, keys_hinder, colors, title, source_url, "Senaste utfall", 650, 'Konjunkturinstitutet', -61)

  #st.header("Hämta", divider=True)
  #df_xlsx = to_excel(df)
  #st.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="4")

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

  values_dict = {
    'Efterfrågan': values_efterfragan,
    'Material och/eller utrustning': values_material,
    'Arbetskraft': values_arbetskraft,
    'Finansiella restriktioner': values_finans,
    'Annat': values_annat,
  }

  keys_hinder_trimmed = keys_hinder[:min_length]
  values_dict_trimmed = {
    'Efterfrågan': values_efterfragan[:min_length],
    'Material och/eller utrustning': values_material[:min_length],
    'Arbetskraft': values_arbetskraft[:min_length],
    'Finansiella restriktioner': values_finans[:min_length],
    'Annat': values_annat[:min_length],
}

  # Convert to DataFrame
  combined_df_anl = pd.DataFrame({'Time': keys_hinder_trimmed, **values_dict_trimmed})

  colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']
  title = "Främsta hinder för anläggningsverksamhet (SNI 42)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__ftgmanad/Barboam.px/"
  create_bki_plot(values_dict, keys_hinder, colors, title, source_url, "Senaste utfall", 650, 'Konjunkturinstitutet', -61)

  #st.header("Hämta", divider=True)
  #df_xlsx = to_excel(df)
  #st.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="15")

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
            "Q010",
            "Q020"
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
      },
      {
        "code": "Grupp",
        "selection": {
          "filter": "item",
          "values": [
            "100"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/hushall/hushall.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  # Extract keys[3] (date values) and values[0] (numeric values) only when keys[0] == 'Q010'
  keys = [entry['key'][3] for entry in response_json['data'] if entry['key'][0] == 'Q010']
  # Convert values_dict into a dictionary
  values_dict = {'Q010': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q010']}

  colors = ['#0072B2', '#FFC20A']
  title = "Hushållets ekonomi nu jämfört med 12 månader sedan"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__hushall/hushall.px/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 650, 'Konjunkturinstitutet', -61)

  keys = [entry['key'][3] for entry in response_json['data'] if entry['key'][0] == 'Q020']
  # Convert values_dict into a dictionary
  values_dict = {'Q020': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q020']}

  title = "Hushållets ekonomi om 12 månader"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__hushall/hushall.px/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 650, 'Konjunkturinstitutet', -61)

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
            "Q010",
            "Q020"
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
      },
      {
        "code": "Grupp",
        "selection": {
          "filter": "item",
          "values": [
            "701",
            "702",
            "703"
          ]
        }
      }
    ],
    "response": {
      "format": "json"
    }
  }

  url = "https://statistik.konj.se:443/PxWeb/api/v1/sv/KonjBar/hushall/hushall.px"

  response = session.post(url, json=query)
  response_json = json.loads(response.content.decode('utf-8-sig'))

  # Extract data for each value of key[2]
  keys = [entry['key'][3] for entry in response_json['data'] if entry['key'][0] == 'Q010' and entry['key'][2] == '701']
  values_dict = {
      'Hyresrätt': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q010' and entry['key'][2] == '701'],
      'Bostadsrätt': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q010' and entry['key'][2] == '702'],
      'Småhus': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q010' and entry['key'][2] == '703']
  }

  colors = ['#0072B2', '#FFC20A', '#DC4405']
  title = "Hushållets ekonomi nu jämfört med 12 månader sedan (bostadstyp)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__hushall/hushall.px/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 650, 'Konjunkturinstitutet', -61)

  keys = [entry['key'][3] for entry in response_json['data'] if entry['key'][0] == 'Q020' and entry['key'][2] == '701']
  values_dict = {
      'Hyresrätt': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q020' and entry['key'][2] == '701'],
      'Bostadsrätt': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q020' and entry['key'][2] == '702'],
      'Småhus': [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][0] == 'Q020' and entry['key'][2] == '703']
  }

  title = "Hushållets ekonomi om 12 månader (bostadstyp)"
  source_url = "https://statistik.konj.se/PxWeb/pxweb/sv/KonjBar/KonjBar__hushall/hushall.px/"
  create_bki_plot(values_dict, keys, colors, title, source_url, "", 650, 'Konjunkturinstitutet', -61)

  #st.write(df_byggande)
  #st.write(df_byggande_utfall)

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
            "2024K2",
            "2024K3",
            "2024K4",
            "2025K1",
            "2025K2",
            "2025K3"
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
      font=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          tickcolor="#646464",
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
            "2024K2",
            "2024K3",
            "2024K4",
            "2025K1",
            "2025K2",
            "2025K3"
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
      font=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          tickcolor="#646464",
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
            "2024K2",
            "2024K3",
            "2024K4",
            "2025K1",
            "2025K2",
            "2025K3"
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
      font=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          tickcolor="#646464",
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
            "2024K2",
            "2024K3",
            "2024K4",
            "2025K1",
            "2025K2",
            "2025K3"
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
      font=dict(size=18),  # Adjust font size to fit the title within the available space
      xaxis=dict(
          # Remove tickvals and ticktext properties
          tickangle=270,  # Rotate x-axis tick labels 180 degrees
          showline=True,  # Show x-axis line
          linewidth=1,  # Set x-axis line width
          linecolor='black',  # Set x-axis line color
          mirror=True,  # Show x-axis line on the top and right side
          tickfont=dict(size=14),  # Set font size for x-axis ticks
          tickcolor="#646464",
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

  #colors = ['#0072B2', '#FFC20A', '#DC4405', '#009E73', '#9850D5', '#F79F1F']
  #title = "Investeringar inom bygg och anläggning"
  #source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T18Kv/"
  #create_bki_plot(combined_df, keys_inv, colors, title, source_url, "Mnkr, säsongsrensade löpande priser", 600, 'SCB')

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
  layout_combined.update(
          yaxis=dict(
        nticks=10,  # Increase the number of ticks on y-axis for more granularity
    ),
      legend=dict(
    #orientation="h",  # Horizontal orientation
    yanchor="top",    # Anchor at the top of the legend box
    y=-0.28,           # Position it below the plot area
    xanchor="center", # Center horizontally
    x=0.5,            # Place the center of the legend at the center of the x-axis
    font=dict(size=12)
    )
  )
  layout_combined["title"] = "Investeringar inom bygg och anläggning"
  layout_combined['title']['y'] = 0.89

  combined = go.Figure(data=data_combined, layout=layout_combined, layout_width=450)

# Configure the Plotly figure to improve download quality
  config = {
    'toImageButtonOptions': {
        'format': 'png',  # Export format
        'filename': 'high_quality_plot',  # Filename for download
        'scale': 2  # Increase scale for higher resolution (scale=2 means 2x the default resolution)
    },
    'displaylogo': False  # Optionally remove the Plotly logo from the toolbar
  }
  #st.plotly_chart(combined, config=config)

  # Prepare values_dict and keys_kv
  values_dict = {
      "Bostadsinvesteringar": combined_df['Bostadsinvesteringar'].tolist(),
      "Övriga byggnader och anläggningar": combined_df['Övriga byggnader och anläggningar'].tolist()
  }
  keys_kv = combined_df['Time'].tolist()

  # Define colors and title
  colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
  title = "Investeringar inom bygg och anläggning"
  source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__NR__NR0103__NR0103B/NR0103ENS2010T17Kv/"

  # Call the create_bki_plot function
  create_bki_plot(
      values_dict=values_dict,
      keys_kv=keys_kv,
      colors=colors,
      title=title,
      source_url=source_url,
      underrubrik="Mnkr, säsongsrensade fasta priser",
      bredd=700,
      source="SCB",
      rader_data=0,
      skip_ticks=True
  )

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

  new_annotation = dict(
      xref='paper',
      yref='paper',
      x=0.39,
      y=1,
      xanchor='center',
      yanchor='bottom',
      text=f'Mnkr, säsongsrensade fasta priser, senaste utfall: {last_datapoint_time}',
      font=dict(size=14, color='black'),  # Set font size and color
      showarrow=False,
  )

  layout_combined['annotations'] = [new_annotation, data_source_annotation]
  layout_combined['title']['x'] = 0.1
  combined = go.Figure(data=data_combined, layout=layout_combined, layout_width=500)

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

  keys_kv = [entry['key'][2] for entry in fardig_kv['data']]
  values_fle = [float(entry['values'][0]) for entry in fardig_kv['data'] if entry['key'][1] == 'FLERBO']
  values_sma = [float(entry['values'][0]) for entry in fardig_kv['data'] if entry['key'][1] == 'SMÅHUS']

  values_dict = {
      'Flerbostadshus': values_fle,
      'Småhus': values_sma
  }

  #st.title("Färdigställda bostäder")
  # Create a dropdown menu to select a title
  selected_title = st.selectbox("Välj ett alternativ:", ["Påbörjade", "Färdigställda", "Ombyggnad"])

  if selected_title == "Färdigställda":
    #st.title("Färdigställda")
    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
    title = "Färdigställda bostäder per kvartal (ej uppräknat)"
    source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101C/LagenhetNyKv16/"
    create_bki_plot(values_dict, keys_kv, colors, title, source_url, "Senaste utfall", 600, 'SCB', 0)

    st.header("Senaste utfall", divider=True)
    col1, col2, col3 = st.columns(3)
    latest_value = values_fle[-1]
    previous_value = values_fle[-2]

    # Calculate the percentage change
    percentage_change = (latest_value / previous_value - 1) * 100

    # Format the numeric value with spaces as thousands separators
    latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')

    # Format the percentage change to one decimal place with a comma instead of a dot
    formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'

    # Display the metric with formatted percentage change
    col1.metric(f"Flerbostadshus" , latest_formatted_value, formatted_change)

    latest_value = values_sma[-1]
    previous_value = values_sma[-2]
    percentage_change = (latest_value / previous_value - 1) * 100
    latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')
    formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'
    col2.metric(f"Småhus" , latest_formatted_value, formatted_change)

    #st.header("Hämta", divider=True)
    #col1, col2, col3 = st.columns(3)
    #df_xlsx = to_excel(df)
    #col1.download_button(label='📥 Hämta data',
    #                              data=df_xlsx,
    #                              file_name= 'df_test.xlsx',
    #                              key="17")

    #new_annotation = dict(
    #    xref='paper',
    #    yref='paper',
    #    x=0.15,
    #    y=1,
    #    xanchor='center',
    #    yanchor='bottom',
    #    text=f'Senaste utfall: {last_datapoint_time}',
    #    font=dict(size=14, color='black'),  # Set font size and color
    #    showarrow=False,
    #)

    #layout_fardig['annotations'] = [new_annotation, data_source_annotation]
    #layout_fardig['title']['x'] = 0.1
    #fardig_tot = go.Figure(data=data_fardig, layout=layout_fardig)

    # Aggregate yearly totals
    yearly_totals = {}
    for entry in response_json['data']:
        year_quarter = entry['key'][2]
        value = float(entry['values'][0])
        year = int(year_quarter[:4])
        yearly_totals[year] = yearly_totals.get(year, 0) + value

    # Trim data to ensure same lengths for plotting
    min_length = min(len(values) for values in [values_sma, values_fle])
    keys_kv_trimmed = keys_kv[:min_length]
    df = pd.DataFrame({'Time': keys_kv_trimmed, 'Flerbostadshus': values_fle[:min_length], 'Småhus': values_sma[:min_length]})

    # Rolling annual totals
    df['Småhus_rolling'] = df['Småhus'].rolling(window=4).sum()
    df['Flerbostadshus_rolling'] = df['Flerbostadshus'].rolling(window=4).sum()
    df['Total_rolling'] = df['Småhus_rolling'] + df['Flerbostadshus_rolling']
    df = df.dropna(subset=['Småhus_rolling', 'Flerbostadshus_rolling', 'Total_rolling'])

    # X-axis labels for every fifth year
    df_first_quarters = df[df['Time'].str.endswith('K1')].copy()
    df_first_quarters['Year'] = df_first_quarters['Time'].str[:4]
    tick_positions = list(range(0, len(df_first_quarters), math.ceil(len(df_first_quarters) / 9)))
    selected_years = df_first_quarters.iloc[tick_positions]

    # Use create_bki_plot to create the plot
    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0, 128, 0)']
    values_dict = {
        'Flerbostadshus': df['Flerbostadshus_rolling'].tolist(),
        'Småhus': df['Småhus_rolling'].tolist(),
        'Totalt': df['Total_rolling'].tolist()
    }

    # Define source and title
    source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101C/LagenhetNyKv16/"
    title = 'Färdigställda bostäder (rullande årstal)'

    # Create the plot
    create_bki_plot(
        values_dict,
        df['Time'].tolist(),
        colors,
        title,
        source_url,
        "Senaste utfall",
        600,
        'SCB och egna beräkningar',
        0
    )

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
              "BO0101A5"
            ]
          }
        }
      ],
      "response": {
        "format": "json"
      }
    }

    url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101A/LghReHustypAr"

    response = session.post(url, json=query)
    response_json = json.loads(response.content.decode('utf-8-sig'))

    fardig_ar = response_json

    import plotly.graph_objs as go
    import plotly.offline as pyo
    import pandas as pd
    import math

    keys_ar = [entry['key'][2] for entry in fardig_ar['data']]
    values_fle = [float(entry['values'][0]) for entry in fardig_ar['data'] if entry['key'][1] == 'FLERBO']
    values_sma = [float(entry['values'][0]) for entry in fardig_ar['data'] if entry['key'][1] == 'SMÅHUS']

    yearly_totals = {}
    for entry in fardig_ar['data']:
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
    keys_kv_trimmed = keys_ar[:min_length]

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
        text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101A/LghReHustypAr/">SCB</a>',
        font=dict(size=12, color='black'),  # Set font size and color
        showarrow=False,
    )

    # Layout
    layout_fardig = go.Layout(
        title='Färdigställda bostäder per år',
        font=dict(size=18),  # Adjust font size to fit the title within the available space
        xaxis=dict(
            # Remove tickvals and ticktext properties
            tickangle=0,  # Rotate x-axis tick labels 180 degrees
            showline=True,  # Show x-axis line
            linewidth=1,  # Set x-axis line width
            linecolor='black',  # Set x-axis line color
            mirror=True,  # Show x-axis line on the top and right side
            tickfont=dict(size=14),  # Set font size for x-axis ticks
            tickcolor="#646464",
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
    fardig_ar = go.Figure(data=data_fardig, layout=layout_fardig, layout_width=600)
    st.plotly_chart(fardig_ar, config=config)

    st.header("Hämta", divider=True)
    col1, col2, col3 = st.columns(3)
    df_xlsx = to_excel(df)
    col1.download_button(label='📥 Hämta data',
                                  data=df_xlsx,
                                  file_name= 'df_test.xlsx',
                                  key="57")

    # Step 1: Calculate the total for each time period
    df['Total'] = df['Flerbostadshus'] + df['Småhus']

    # Step 2: Calculate the percentage for each category
    df['Flerbostadshus'] = (df['Flerbostadshus'] / df['Total']) * 100
    df['Småhus'] = (df['Småhus'] / df['Total']) * 100

    # Step 3: Create traces for percentage values
    data_percentage = []
    for i, column in enumerate(['Flerbostadshus', 'Småhus']):
        trace = go.Scatter(
            x=df['Time'],
            y=df[column],
            name=column,
            hovertext=[f"Tidpunkt: {time}<br>{column}: {value:.2f}%" for time, value in zip(df['Time'], df[column])],
            hoverinfo='text',
            mode='lines',
            line=dict(
                color=colors[i],
                width=2.6 if column != 'Total' else 1.5,
                dash='dash' if column == 'Total' else 'solid',
            ),
            opacity=1,
            selected=dict(marker=dict(color='red')),
            unselected=dict(marker=dict(opacity=0.1))
        )
        data_percentage.append(trace)

    # Step 4: Update layout for the new plot (title change)
    layout_percentage = go.Layout(
        title='Andel färdigställda bostäder per år och hustyp',
        font=dict(size=18),
        xaxis=dict(
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=14),
            tickcolor="#646464",
            ticks='outside',
            ticklen=5,
        ),
        yaxis=dict(
            title='Procent',
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=16),
            tickformat=",",
        ),
        plot_bgcolor='white',
        yaxis_gridcolor='lightgray',
        annotations=[last_datapoint_annotation, data_source_annotation],
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
            b=100
        )
    )

    # Update the title position
    layout_percentage['title']['y'] = 0.89

    # Create the figure for percentage values
    percentage_fig = go.Figure(data=data_percentage, layout=layout_percentage, layout_width=600)

    # Display the figure
    st.plotly_chart(percentage_fig, config=config)

    st.header("Hämta", divider=True)
    col1, col2, col3 = st.columns(3)
    df_xlsx = to_excel(df)
    col1.download_button(label='📥 Hämta data',
                                  data=df_xlsx,
                                  file_name= 'df_test.xlsx',
                                  key="58")

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
          "code": "Upplatelseform",
          "selection": {
            "filter": "item",
            "values": [
              "1",
              "2",
              "3"
            ]
          }
        }
      ],
      "response": {
        "format": "json"
      }
    }

    url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BO/BO0101/BO0101A/LghReHtypUfAr"

    response = session.post(url, json=query)
    response_json = json.loads(response.content.decode('utf-8-sig'))

    fardig_upplatelseform = response_json

    import plotly.graph_objs as go
    import pandas as pd
    import math

    # Assuming fardig_upplatelseform is the new dataset replacing fardig_ar

    # Extract relevant data from fardig_upplatelseform
    keys_up = [entry['key'][3] for entry in fardig_upplatelseform['data']]
    values_fle = {1: [], 2: [], 3: []}  # Tenure statuses for Flerbostadshus: Hyresrätt(1), Bostadsrätt(2), Äganderätt(3)
    values_sma = {1: [], 2: [], 3: []}  # Tenure statuses for Småhus: Hyresrätt(1), Bostadsrätt(2), Äganderätt(3)

    for entry in fardig_upplatelseform['data']:
        house_type = entry['key'][1]  # House type: 'FLERBO' or 'SMÅHUS'
        tenure_status = int(entry['key'][2])  # Tenure status: 1=Hyresrätt, 2=Bostadsrätt, 3=Äganderätt
        value = float(entry['values'][0])

        if house_type == 'FLERBO':
            values_fle[tenure_status].append(value)
        elif house_type == 'SMÅHUS':
            values_sma[tenure_status].append(value)

    # Determine the minimum length for each group
    min_length_fle = min(len(values) for values in values_fle.values())
    min_length_sma = min(len(values) for values in values_sma.values())

    # Trim the time keys to match the minimum lengths
    keys_fle_trimmed = keys_up[:min_length_fle]
    keys_sma_trimmed = keys_up[:min_length_sma]

    # Create DataFrames for each house type (Flerbostadshus and Småhus)
    df_fle = pd.DataFrame({'Time': keys_fle_trimmed})
    df_fle['Hyresrätt'] = values_fle[1][:min_length_fle]
    df_fle['Bostadsrätt'] = values_fle[2][:min_length_fle]
    df_fle['Äganderätt'] = values_fle[3][:min_length_fle]

    df_sma = pd.DataFrame({'Time': keys_sma_trimmed})
    df_sma['Hyresrätt'] = values_sma[1][:min_length_sma]
    df_sma['Bostadsrätt'] = values_sma[2][:min_length_sma]
    df_sma['Äganderätt'] = values_sma[3][:min_length_sma]

    # Slice the DataFrames to select the last 60 rows
    df_fle = df_fle.iloc[-61:]
    df_sma = df_sma.iloc[-61:]

    # Define color scheme for the three tenure statuses
    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0,128,0)']  # Different colors for Hyresrätt, Bostadsrätt, Äganderätt

    # Plot function to reuse for both house types
    def plot_housing_data(df, house_type, colors):
        data_traces = []
        for i, column in enumerate(df.columns[1:]):
            trace = go.Scatter(
                x=df['Time'],
                y=df[column],
                name=f'{column}',
                hovertext=[f"Tidpunkt: {time}<br>{column}: {value}" for time, value in zip(df['Time'], df[column])],
                hoverinfo='text',
                mode='lines',
                line=dict(
                    color=colors[i],
                    width=2.6,  # Set line width
                    dash='solid',
                ),
                opacity=1,
            )
            data_traces.append(trace)

        # Determine the tick positions for every fifth year
        tick_positions = [year for year in df['Time'] if int(year) % 5 == 0]

        # Add annotation for the time of the last datapoint
        last_datapoint_time = df['Time'].iloc[-1]
        last_datapoint_annotation = dict(
            xref='paper',
            yref='paper',
            x=0.08,
            y=1.03,
            xanchor='center',
            yanchor='bottom',
            text=f'Senaste utfall: {last_datapoint_time}',
            font=dict(size=14, color='black'),
            showarrow=False,
        )

        # Data source annotation
        data_source_annotation = dict(
            xref='paper',
            yref='paper',
            x=0.01,
            y=-0.2,
            xanchor='center',
            yanchor='top',
            text='Källa: <a href="https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101A/LghReHtypUfAr/">SCB</a>',
            font=dict(size=12, color='black'),
            showarrow=False,
        )

        # Layout
        layout = go.Layout(
            title=f'Färdigställda {house_type} per år och upplåtelseform',
            font=dict(size=18),
            xaxis=dict(
                tickvals=tick_positions,
                tickangle=0,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                tickfont=dict(size=14),
                tickcolor="#646464",
                ticks='outside',
                ticklen=5,
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                tickfont=dict(size=16),
                tickformat=",",
            ),
            plot_bgcolor='white',
            yaxis_gridcolor='lightgray',
            annotations=[last_datapoint_annotation, data_source_annotation],
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
                b=100
            )
        )

        layout['title']['y'] = 0.89

        # Create figure
        fig = go.Figure(data=data_traces, layout=layout, layout_width=600)

        return fig

    # Create and show the plots
    fig_fle = plot_housing_data(df_fle, 'flerbostadshus', colors)
    fig_sma = plot_housing_data(df_sma, 'småhus', colors)

    # Display the figures
    st.plotly_chart(fig_fle, config=config)
    st.plotly_chart(fig_sma, config=config)

    st.header("Hämta", divider=True)
    col1, col2, col3 = st.columns(3)
    df_xlsx = to_excel(df_fle)
    col1.download_button(label='📥 Hämta data',
                                  data=df_xlsx,
                                  file_name= 'df_test.xlsx',
                                  key="59")
    df_xlsx = to_excel(df_sma)
    col2.download_button(label='📥 Hämta data',
                                  data=df_xlsx,
                                  file_name= 'df_test.xlsx',
                                  key="60")

  ###### Påbörjade bostäder: uppdelat per upplåtelseform
  elif selected_title == "Påbörjade":
    #st.title("Påbörjade")
  #st.title("Påbörjade bostäder")
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

    keys_pkv = [entry['key'][2] for entry in response_json['data']]
    values_pfle = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'FLERBO']
    values_psma = [float(entry['values'][0]) for entry in response_json['data'] if entry['key'][1] == 'SMÅHUS']

    values_dict = {
        'Flerbostadshus': values_pfle,
        'Småhus': values_psma
    }

    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
    title = "Påbörjade bostäder per kvartal (ej uppräknat)"
    source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101C/LagenhetNyKv16/"
    create_bki_plot(values_dict, keys_pkv, colors, title, source_url, "Senaste utfall", 600, 'SCB', 0)

    st.header("Senaste utfall", divider=True)
    col1, col2, col3 = st.columns(3)
    latest_value = values_pfle[-1]
    previous_value = values_pfle[-2]

    # Calculate the percentage change
    percentage_change = (latest_value / previous_value - 1) * 100

    # Format the numeric value with spaces as thousands separators
    latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')

    # Format the percentage change to one decimal place with a comma instead of a dot
    formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'

    # Display the metric with formatted percentage change
    col1.metric(f"Flerbostadshus" , latest_formatted_value, formatted_change)

    latest_value = values_psma[-1]
    previous_value = values_psma[-2]
    percentage_change = (latest_value / previous_value - 1) * 100
    latest_formatted_value = f"{latest_value:,.0f}".replace(',', ' ')
    formatted_change = f"{percentage_change:.1f}".replace('.', ',') + '%'
    col2.metric(f"Småhus" , latest_formatted_value, formatted_change)

    # Aggregate yearly totals
    yearly_totals = {}
    for entry in response_json['data']:
        year_quarter = entry['key'][2]
        value = float(entry['values'][0])
        year = int(year_quarter[:4])
        yearly_totals[year] = yearly_totals.get(year, 0) + value

    # Trim data to ensure same lengths for plotting
    min_length = min(len(values) for values in [values_psma, values_pfle])
    keys_pkv_trimmed = keys_pkv[:min_length]
    df = pd.DataFrame({'Time': keys_pkv_trimmed, 'Flerbostadshus': values_pfle[:min_length], 'Småhus': values_psma[:min_length]})

    # Rolling annual totals
    df['Småhus_rolling'] = df['Småhus'].rolling(window=4).sum()
    df['Flerbostadshus_rolling'] = df['Flerbostadshus'].rolling(window=4).sum()
    df['Total_rolling'] = df['Småhus_rolling'] + df['Flerbostadshus_rolling']
    df = df.dropna(subset=['Småhus_rolling', 'Flerbostadshus_rolling', 'Total_rolling'])

    # X-axis labels for every fifth year
    df_first_quarters = df[df['Time'].str.endswith('K1')].copy()
    df_first_quarters['Year'] = df_first_quarters['Time'].str[:4]
    tick_positions = list(range(0, len(df_first_quarters), math.ceil(len(df_first_quarters) / 9)))
    selected_years = df_first_quarters.iloc[tick_positions]

    # Use create_bki_plot to create the plot
    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)', 'rgb(0, 128, 0)']
    values_dict = {
        'Flerbostadshus': df['Flerbostadshus_rolling'].tolist(),
        'Småhus': df['Småhus_rolling'].tolist(),
        'Totalt': df['Total_rolling'].tolist()
    }

    # Define source and title
    source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101C/LagenhetNyKv16/"
    title = 'Påbörjade bostäder (rullande årstal)'

    # Create the plot
    create_bki_plot(
        values_dict,
        df['Time'].tolist(),
        colors,
        title,
        source_url,
        "Senaste utfall",
        600,
        'SCB och egna beräkningar',
        0
    )

    st.header("Uppräknade värden:", divider=True)
    st.write("https://www.scb.se/hitta-statistik/statistik-efter-amne/boende-bebyggelse-och-mark/byggande-och-ombyggnad/bygglov-nybyggnad-och-ombyggnad/pong/tabell-och-diagram/nybyggnad/paborjade-nybyggda-bostadslagenheter/")
    st.write("https://www.scb.se/hitta-statistik/statistik-efter-amne/boende-bebyggelse-och-mark/byggande-och-ombyggnad/bygglov-nybyggnad-och-ombyggnad/pong/tabell-och-diagram/nybyggnad/nybyggnad-av-bostader-oversiktstabell-preliminara-siffror/")
    
  #st.header("Hämta", divider=True)
  #col1, col2, col3 = st.columns(3)
  #df_xlsx = to_excel(df)
  #col1.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="30")

  elif selected_title == "Ombyggnad":
    #st.title("Ombyggnad")
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

    values_dict = {
        'Ombyggnad': values_ombyggnad
    }

    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
    title = "Nettoförändring: påbörjad ombyggnad av flerbostadshus"
    source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101B/LagenhetOmbNKv/"
    create_bki_plot(values_dict, keys_ombyggnad, colors, title, source_url, "Senaste utfall", 500, 'SCB', 0)

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

    values_dict = {
        'Ombyggnad': values_ombyggnad
    }

    colors = ['rgb(8,48,107)', 'rgb(204, 0, 0)']
    title = "Nettoförändring: färdigställd ombyggnad av flerbostadshus"
    source_url = "https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/START__BO__BO0101__BO0101B/LagenhetOmbNKv/"
    create_bki_plot(values_dict, keys_ombyggnad, colors, title, source_url, "Senaste utfall", 500, 'SCB', 0)

with tab8:
  import eurostat
  toc_df = eurostat.get_toc_df()

  import pandas as pd
  import plotly.graph_objects as go
  import eurostat  # Ensure the Eurostat library is installed and imported
  from io import StringIO

  def create_bki_plot_eurostat(values_dict, keys_kv, colors, title, source_url, sub_heading, sub_heading_x):
      min_length = min(len(values) for values in values_dict.values())
      keys_kv_trimmed = keys_kv[:min_length]

      df = pd.DataFrame({'Time': keys_kv_trimmed})

      if 'Q' in df['Time'].iloc[0]:
          df['Quarter'] = df['Time'].str[5:]
          df['Year']    = df['Time'].str[:4]

      for label, values in values_dict.items():
          df[label] = values[:min_length]

      # --- Traces ---
      data_bki_tot = []
      for i, (label, values) in enumerate(values_dict.items()):
          trace = go.Scatter(
              x=df['Time'],
              y=df[label],
              name=label,
              hovertext=[f"Tidpunkt: {time}<br>{label}: {value}" for time, value in zip(df['Time'], df[label])],
              hoverinfo='text',
              mode='lines',
              line=dict(color=colors[i], width=2.6, dash='dash' if label == "Total" else None),
              opacity=1
          )
          data_bki_tot.append(trace)

      # --- Tick-positioner (endast Q1) ---
      final_tick_positions = []
      final_tick_labels    = []
      for i, (quarter, year) in enumerate(zip(df['Quarter'], df['Year'])):
          if quarter == 'Q1':
              final_tick_positions.append(i)
              final_tick_labels.append(year)

      # --- Titel + underrubrik ihopslagen ---
      combined_title = (
          f'{title}<br>'
          f'<span style="font-size:14px; color:#444; font-weight:normal;">'
          f'{sub_heading}</span>'
      )

      layout = go.Layout(
          title=dict(
              text=combined_title,
              font=dict(size=18),
              x=0.07,
              xanchor='left',
              y=0.84,
              yanchor='top',
          ),
          height=500,
          font=dict(size=18),
          xaxis=dict(
              tickvals=final_tick_positions,
              ticktext=final_tick_labels,
              tickangle=0,
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,
              tickfont=dict(size=14),
              tickcolor="#646464",
              ticks='outside',
              ticklen=5,
          ),
          yaxis=dict(
              showline=True,
              linewidth=1,
              linecolor='black',
              mirror=True,
              tickfont=dict(size=16),
          ),
          plot_bgcolor='white',
          yaxis_gridcolor='lightgray',
          legend=dict(
              x=1.05,
              y=1,
              traceorder='normal',
              font=dict(family="Monaco, monospace", size=12, color="black")
          ),
          margin=dict(t=120, b=70, r=80, l=60),
          annotations=[
              dict(
                  xref='paper', yref='paper',
                  x=0.0, y=-0.15,
                  xanchor='left', yanchor='top',
                  text=f'Källa: <a href="{source_url}">Eurostat</a>',
                  font=dict(size=12, color='black'),
                  showarrow=False,
              ),
          ],
      )

      config = {
          'toImageButtonOptions': {
              'format': 'png',
              'filename': 'high_quality_plot',
              'scale': 2,
          },
          'displaylogo': False
      }

      fig = go.Figure(data=data_bki_tot, layout=layout)
      fig.update_layout(width=600)
      st.plotly_chart(fig, config=config)

      # --- Nedladdningsknappar ---
      def save_as_html(fig):
          buf = StringIO()
          fig.write_html(buf, include_plotlyjs='cdn', config=config)
          return buf.getvalue().encode('utf-8')

      def display_download_button(fig):
          global download_counter
          col1.download_button(
              label="📈 Hämta figur",
              data=save_as_html(fig),
              file_name="figure.html",
              mime="text/html",
              key=f"download_button_{download_counter}"
          )
          download_counter += 1

      def display_download_button_excel(df):
          global download_counter_excel
          col2.download_button(
              label='📥 Hämta data',
              data=to_excel(df),
              file_name='data.xlsx',
              mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
              key=f"download_button_excel_{download_counter_excel}"
          )
          download_counter_excel += 1

      st.header("Hämta", divider=True)
      col1, col2, col3 = st.columns(3)
      display_download_button(fig)
      display_download_button_excel(df)

  # Example usage
  #fig = plot_eurostat_data(
  #    data_code='ilc_lvho07a',
  #    countries_to_keep=['NO', 'DK', 'FI', 'SE', 'EU27_2020'],
  #    year_range=(2010, 2024),
  #    value_name='value',
  #    main_title='Housing cost overburden rate',
  #    sub_heading='Hela befolkningen, 2010-2023',
  #    xaxis_title='År',
  #    yaxis_title='Procent',
  #    eurostat_link='https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO07A__custom_12778359/default/table?lang=en'
  #)

  #st.plotly_chart(fig)

  # Define filter parameters for the dataset
  my_filter_pars = {'startPeriod': '2008-Q3', 'indic_bt': 'BPRM_SQM', 'cpa2_1': 'CPA_F41001', 's_adj': 'NSA', 'unit': 'PCH_SM'}
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

  values_dict = {
        'SE': y_values,
        'EU': y_values_eu27,
        'DK': y_values_dk,
        'NO': y_values_no,
        'FI': y_values_fi,
  }

  colors = ['#0051BA', '#FDB813', 'rgb(250,80,80)', 'rgb(200,75,10)', 'rgb(20,200,220)']
  title = "Kvm beviljade bygglov för bostäder (golvyta)"
  source_url = "https://ec.europa.eu/eurostat/databrowser/view/STS_COBP_Q/default/table"
  create_bki_plot_eurostat(values_dict, x_labels, colors, title, source_url, 'Årlig procentuell förändring', 0.15)

  #st.header("Hämta", divider=True)
  #col1, col2, col3 = st.columns(3)
  #df_xlsx = to_excel(data)
  #col1.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="50")

 # House price index, Total, Annual rate of change
  my_filter_pars = {'startPeriod': '2008-Q3', 'purchase': 'TOTAL', 'unit': 'I15_Q'}
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

  values_dict = {
        'SE': y_values,
        'EU': y_values_eu27,
        'DK': y_values_dk,
        'NO': y_values_no,
        'FI': y_values_fi,
  }

  colors = ['#0051BA', '#FDB813', 'rgb(250,80,80)', 'rgb(200,75,10)', 'rgb(20,200,220)']
  title = "Bostadsprisindex, total"
  source_url = "https://ec.europa.eu/eurostat/databrowser/view/PRC_HPI_Q/default/table"
  create_bki_plot_eurostat(values_dict, x_labels, colors, title, source_url, 'Index 2015=100', 0.08)

  #st.header("Hämta", divider=True)
  #col1, col2, col3 = st.columns(3)
  #df_xlsx = to_excel(data)
  #col1.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="49")

  my_filter_pars = {'startPeriod': '2011-Q1', 'indic_bt': 'COST', 'unit': 'I15'}
  data = eurostat.get_data_df('sts_copi_q', filter_pars=my_filter_pars)

  import plotly.graph_objects as go

  # Select the row with the string "SE" under the column "geo\TIME_PERIOD"
  row = data.loc[data['geo\\TIME_PERIOD'] == 'SE']
  values = row.iloc[0, 2:]

  # Set the x-axis labels and y-axis values
  x_labels = values.index[20:]
  y_values = values.values[20:]

  # Select data for EU27, DK, FI, NO
  row_eu27 = data.loc[data['geo\\TIME_PERIOD'] == 'EU27_2020']
  values_eu27 = row_eu27.iloc[0, 2:]
  x_labels_eu27 = values_eu27.index[20:]
  y_values_eu27 = values_eu27.values[20:]

  row_dk = data.loc[data['geo\\TIME_PERIOD'] == 'DK']
  values_dk = row_dk.iloc[0, 2:]
  x_labels_dk = values_dk.index[20:]
  y_values_dk = values_dk.values[20:]

  row_fi = data.loc[data['geo\\TIME_PERIOD'] == 'FI']
  values_fi = row_fi.iloc[0, 2:]
  x_labels_fi = values_fi.index[20:]
  y_values_fi = values_fi.values[20:]

  row_no = data.loc[data['geo\\TIME_PERIOD'] == 'NO']
  values_no = row_no.iloc[0, 2:]
  x_labels_no = values_no.index[20:]
  y_values_no = values_no.values[20:]

  values_dict = {
        'SE': y_values,
        'EU': y_values_eu27,
        'DK': y_values_dk,
        'NO': y_values_no,
        'FI': y_values_fi,
  }

  colors = ['#0051BA', '#FDB813', 'rgb(250,80,80)', 'rgb(200,75,10)', 'rgb(20,200,220)']
  title = "Byggkostnadsindex, nya bostadshus"
  source_url = "https://ec.europa.eu/eurostat/databrowser/view/STS_COPI_Q/default/table"
  create_bki_plot_eurostat(values_dict, x_labels, colors, title, source_url, 'Index 2015=100', 0.08)

  #st.header("Hämta", divider=True)
  #col1, col2, col3 = st.columns(3)
  #df_xlsx = to_excel(data)
  #col1.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="48")

  my_filter_pars = {'startPeriod': '2011-Q1', 'indic_bt': 'COST', 'unit': 'PCH_SM'}
  data = eurostat.get_data_df('sts_copi_q', filter_pars=my_filter_pars)

  import plotly.graph_objects as go

  # Select the row with the string "SE" under the column "geo\TIME_PERIOD"
  row = data.loc[data['geo\\TIME_PERIOD'] == 'SE']
  values = row.iloc[0, 2:]

  # Set the x-axis labels and y-axis values
  x_labels = values.index[20:]
  y_values = values.values[20:]

  # Select data for EU27, DK, FI, NO
  row_eu27 = data.loc[data['geo\\TIME_PERIOD'] == 'EU27_2020']
  values_eu27 = row_eu27.iloc[0, 2:]
  x_labels_eu27 = values_eu27.index[20:]
  y_values_eu27 = values_eu27.values[20:]

  row_dk = data.loc[data['geo\\TIME_PERIOD'] == 'DK']
  values_dk = row_dk.iloc[0, 2:]
  x_labels_dk = values_dk.index[20:]
  y_values_dk = values_dk.values[20:]

  row_fi = data.loc[data['geo\\TIME_PERIOD'] == 'FI']
  values_fi = row_fi.iloc[0, 2:]
  x_labels_fi = values_fi.index[20:]
  y_values_fi = values_fi.values[20:]

  row_no = data.loc[data['geo\\TIME_PERIOD'] == 'NO']
  values_no = row_no.iloc[0, 2:]
  x_labels_no = values_no.index[20:]
  y_values_no = values_no.values[20:]

  values_dict = {
        'SE': y_values,
        'EU': y_values_eu27,
        'DK': y_values_dk,
        'NO': y_values_no,
        'FI': y_values_fi,
  }

  colors = ['#0051BA', '#FDB813', 'rgb(250,80,80)', 'rgb(200,75,10)', 'rgb(20,200,220)']
  title = "Byggkostnadsindex, nya bostadshus"
  source_url = "https://ec.europa.eu/eurostat/databrowser/view/STS_COPI_Q/default/table"
  create_bki_plot_eurostat(values_dict, x_labels, colors, title, source_url, 'Årlig procentuell förändring', 0.15)

  #st.header("Hämta", divider=True)
  #col1, col2, col3 = st.columns(3)
  #df_xlsx = to_excel(data)
  #col1.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="51")

  my_filter_pars = {'startPeriod': '2011-Q1', 'indic_bt': 'COST', 'unit': 'I21'}
  data = eurostat.get_data_df('sts_copi_q', filter_pars=my_filter_pars)

  import plotly.graph_objects as go

  # Select the row with the string "SE" under the column "geo\TIME_PERIOD"
  row = data.loc[data['geo\\TIME_PERIOD'] == 'SE']
  values = row.iloc[0, 2:]

  # Set the x-axis labels and y-axis values
  x_labels = values.index[44:]
  y_values = values.values[44:]

  # Select data for EU27, DK, FI, NO
  row_eu27 = data.loc[data['geo\\TIME_PERIOD'] == 'EU27_2020']
  values_eu27 = row_eu27.iloc[0, 2:]
  x_labels_eu27 = values_eu27.index[44:]
  y_values_eu27 = values_eu27.values[44:]

  row_dk = data.loc[data['geo\\TIME_PERIOD'] == 'DK']
  values_dk = row_dk.iloc[0, 2:]
  x_labels_dk = values_dk.index[44:]
  y_values_dk = values_dk.values[44:]

  row_fi = data.loc[data['geo\\TIME_PERIOD'] == 'FI']
  values_fi = row_fi.iloc[0, 2:]
  x_labels_fi = values_fi.index[44:]
  y_values_fi = values_fi.values[44:]

  row_no = data.loc[data['geo\\TIME_PERIOD'] == 'NO']
  values_no = row_no.iloc[0, 2:]
  x_labels_no = values_no.index[44:]
  y_values_no = values_no.values[44:]

  values_dict = {
        'SE': y_values,
        'EU': y_values_eu27,
        'DK': y_values_dk,
        'NO': y_values_no,
        'FI': y_values_fi,
  }

  colors = ['#0051BA', '#FDB813', 'rgb(250,80,80)', 'rgb(200,75,10)', 'rgb(20,200,220)']
  title = "Byggkostnadsindex, nya bostadshus"
  source_url = "https://ec.europa.eu/eurostat/databrowser/view/STS_COPI_Q/default/table"
  create_bki_plot_eurostat(values_dict, x_labels, colors, title, source_url, 'Index 2021=100', 0.08)

  #st.header("Hämta", divider=True)
  #col1, col2, col3 = st.columns(3)
  #df_xlsx = to_excel(data)
  #col1.download_button(label='📥 Hämta data',
  #                              data=df_xlsx,
  #                              file_name= 'df_test.xlsx',
  #                              key="52")


  # ── ilc_lvho07d – Hushåll med boendeutgifter > 40%, i städer ───────────────
  my_filter_pars = {'startPeriod': '2010', 'deg_urb': 'DEG1'}
  data = eurostat.get_data_df('ilc_lvho07d', filter_pars=my_filter_pars)

  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)
  data = data[data['geo'].isin(['NO', 'DK', 'FI', 'SE', 'EU27_2020'])]
  data_long = pd.melt(data, id_vars=['geo'],
      value_vars=[str(y) for y in range(2010, 2025)],
      var_name='year', value_name='value')
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  fig = go.Figure()
  colors = ['rgb(250,80,80)', '#FDB813', 'rgb(20,200,220)', 'rgb(200,75,10)', '#0051BA']
  for idx, country in enumerate(data_long['geo'].unique()):
      cd = data_long[data_long['geo'] == country]
      fig.add_trace(go.Scatter(
          x=cd['year'], y=cd['value'],
          mode='lines',
          name='EU' if country == 'EU27_2020' else country,
          line=dict(color=colors[idx], width=2.6)
      ))

  fig.update_layout(
      title=dict(
          text='Hushåll med boendeutgifter över 40% av disponibel inkomst'
              '<br><span style="font-size:14px; color:#444; font-weight:normal;">I städer, 2010–2024</span>',
          font=dict(size=18), x=0.07, xanchor='left', y=0.84, yanchor='top'
      ),
      font=dict(size=18), height=500, width=600,
      xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True,
                tickangle=0, tickfont=dict(size=14), tickcolor="#646464", ticks='outside', ticklen=5),
      yaxis=dict(title='Procent', showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=16)),
      plot_bgcolor='white', yaxis_gridcolor='lightgray',
      margin=dict(t=120, b=70, r=80, l=60),
      legend=dict(x=1.05, y=1, traceorder='normal',
                  font=dict(family="Monaco, monospace", size=12, color="black")),
      annotations=[dict(
          xref='paper', yref='paper', x=0.0, y=-0.15,
          xanchor='left', yanchor='top', showarrow=False,
          text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO07D__custom_7140801/bookmark/table?lang=en&bookmarkId=411e17fd-9b03-4729-8ad9-ea4844481e08">Eurostat</a>',
          font=dict(size=12, color='black')
      )]
  )
  st.plotly_chart(fig)
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  col1.download_button(label='📥 Hämta data', data=to_excel(data), file_name='df_test.xlsx', key="47")


  # ── ilc_lvho07a – Hushåll med boendeutgifter > 40%, hela befolkningen ───────
  my_filter_pars = {'age': 'TOTAL', 'sex': 'T', 'incgrp': 'TOTAL'}
  data = eurostat.get_data_df('ilc_lvho07a', filter_pars=my_filter_pars)

  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)
  data = data[data['geo'].isin(['NO', 'DK', 'FI', 'SE', 'EU27_2020'])]
  data_long = pd.melt(data, id_vars=['geo'],
      value_vars=[str(y) for y in range(2010, 2025)],
      var_name='year', value_name='value')
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  fig = go.Figure()
  for idx, country in enumerate(data_long['geo'].unique()):
      cd = data_long[data_long['geo'] == country]
      fig.add_trace(go.Scatter(
          x=cd['year'], y=cd['value'],
          mode='lines',
          name='EU' if country == 'EU27_2020' else country,
          line=dict(color=colors[idx], width=2.6)
      ))

  fig.update_layout(
      title=dict(
          text='Hushåll med boendeutgifter över 40% av disponibel inkomst'
              '<br><span style="font-size:14px; color:#444; font-weight:normal;">Hela befolkningen, 2010–2024</span>',
          font=dict(size=18), x=0.07, xanchor='left', y=0.84, yanchor='top'
      ),
      font=dict(size=18), height=500, width=600,
      xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True,
                tickangle=0, tickfont=dict(size=14), tickcolor="#646464", ticks='outside', ticklen=5),
      yaxis=dict(title='Procent', showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=16)),
      plot_bgcolor='white', yaxis_gridcolor='lightgray',
      margin=dict(t=120, b=70, r=80, l=60),
      legend=dict(x=1.05, y=1, traceorder='normal',
                  font=dict(family="Monaco, monospace", size=12, color="black")),
      annotations=[dict(
          xref='paper', yref='paper', x=0.0, y=-0.15,
          xanchor='left', yanchor='top', showarrow=False,
          text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO07A__custom_12778359/default/table?lang=en">Eurostat</a>',
          font=dict(size=12, color='black')
      )]
  )
  st.plotly_chart(fig)
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  col1.download_button(label='📥 Hämta data', data=to_excel(data), file_name='df_test.xlsx', key="46")


  # ── ilc_mded01 – Boendekostnad andel, riskzon för fattigdom ─────────────────
  my_filter_pars = {'startPeriod': '2010', 'deg_urb': 'DEG1', 'hhtyp': 'TOTAL', 'incgrp': 'B_MD60'}
  data = eurostat.get_data_df('ilc_mded01', filter_pars=my_filter_pars)

  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)
  data = data[data['geo'].isin(['NO', 'DK', 'FI', 'SE', 'EU27_2020'])]
  data_long = pd.melt(data, id_vars=['geo'],
      value_vars=[str(y) for y in range(2010, 2025)],
      var_name='year', value_name='value')
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  fig = go.Figure()
  for idx, country in enumerate(data_long['geo'].unique()):
      cd = data_long[data_long['geo'] == country]
      fig.add_trace(go.Scatter(
          x=cd['year'], y=cd['value'],
          mode='lines',
          name='EU' if country == 'EU27_2020' else country,
          line=dict(color=colors[idx], width=2.6)
      ))

  fig.update_layout(
      title=dict(
          text='Boendekostnadens andel av hushållens disponibla inkomst'
              '<br><span style="font-size:14px; color:#444; font-weight:normal;">I riskzonen för fattigdom, 2010–2024</span>',
          font=dict(size=18), x=0.07, xanchor='left', y=0.84, yanchor='top'
      ),
      font=dict(size=18), height=500, width=600,
      xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True,
                tickangle=0, tickfont=dict(size=14), tickcolor="#646464", ticks='outside', ticklen=5),
      yaxis=dict(title='Procent', showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=16)),
      plot_bgcolor='white', yaxis_gridcolor='lightgray',
      margin=dict(t=120, b=70, r=80, l=60),
      legend=dict(x=1.05, y=1, traceorder='normal',
                  font=dict(family="Monaco, monospace", size=12, color="black")),
      annotations=[dict(
          xref='paper', yref='paper', x=0.0, y=-0.15,
          xanchor='left', yanchor='top', showarrow=False,
          text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_MDED01__custom_7140904/bookmark/table?lang=en&bookmarkId=659e8061-cde5-4ddb-b633-7cff3c16b7bd">Eurostat</a>',
          font=dict(size=12, color='black')
      )]
  )
  st.plotly_chart(fig)
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  col1.download_button(label='📥 Hämta data', data=to_excel(data), file_name='df_test.xlsx', key="45")


  # ── ilc_mded01 – Boendekostnad andel, hela befolkningen ─────────────────────
  my_filter_pars = {'startPeriod': '2010', 'deg_urb': 'DEG1', 'hhtyp': 'TOTAL', 'incgrp': 'TOTAL'}
  data = eurostat.get_data_df('ilc_mded01', filter_pars=my_filter_pars)

  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)
  data = data[data['geo'].isin(['NO', 'DK', 'FI', 'SE', 'EU27_2020'])]
  data_long = pd.melt(data, id_vars=['geo'],
      value_vars=[str(y) for y in range(2010, 2025)],
      var_name='year', value_name='value')
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  fig = go.Figure()
  for idx, country in enumerate(data_long['geo'].unique()):
      cd = data_long[data_long['geo'] == country]
      fig.add_trace(go.Scatter(
          x=cd['year'], y=cd['value'],
          mode='lines',
          name='EU' if country == 'EU27_2020' else country,
          line=dict(color=colors[idx], width=2.6)
      ))

  fig.update_layout(
      title=dict(
          text='Boendekostnadens andel av hushållens disponibla inkomst'
              '<br><span style="font-size:14px; color:#444; font-weight:normal;">Hela befolkningen, 2010–2024</span>',
          font=dict(size=18), x=0.07, xanchor='left', y=0.84, yanchor='top'
      ),
      font=dict(size=18), height=500, width=600,
      xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True,
                tickangle=0, tickfont=dict(size=14), tickcolor="#646464", ticks='outside', ticklen=5),
      yaxis=dict(title='Procent', showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=16)),
      plot_bgcolor='white', yaxis_gridcolor='lightgray',
      margin=dict(t=120, b=70, r=80, l=60),
      legend=dict(x=1.05, y=1, traceorder='normal',
                  font=dict(family="Monaco, monospace", size=12, color="black")),
      annotations=[dict(
          xref='paper', yref='paper', x=0.0, y=-0.15,
          xanchor='left', yanchor='top', showarrow=False,
          text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_MDED01__custom_7140904/bookmark/table?lang=en&bookmarkId=659e8061-cde5-4ddb-b633-7cff3c16b7bd">Eurostat</a>',
          font=dict(size=12, color='black')
      )]
  )
  st.plotly_chart(fig)
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  col1.download_button(label='📥 Hämta data', data=to_excel(data), file_name='df_test.xlsx', key="40")


  # ── ilc_lvho05a – Trångboddhet, hela befolkningen ────────────────────────────
  my_filter_pars = {'startPeriod': '2010', 'age': 'TOTAL', 'incgrp': 'TOTAL', 'sex': 'T'}
  data = eurostat.get_data_df('ilc_lvho05a', filter_pars=my_filter_pars)

  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)
  data = data[data['geo'].isin(['NO', 'DK', 'FI', 'SE', 'EU27_2020'])]
  data_long = pd.melt(data, id_vars=['geo'],
      value_vars=[str(y) for y in range(2010, 2025)],
      var_name='year', value_name='value')
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  fig = go.Figure()
  for idx, country in enumerate(data_long['geo'].unique()):
      cd = data_long[data_long['geo'] == country]
      fig.add_trace(go.Scatter(
          x=cd['year'], y=cd['value'],
          mode='lines',
          name='EU' if country == 'EU27_2020' else country,
          line=dict(color=colors[idx], width=2.6)
      ))

  fig.update_layout(
      title=dict(
          text='Andel av befolkningen i trångbodda hushåll'
              '<br><span style="font-size:14px; color:#444; font-weight:normal;">Hela befolkningen, 2010–2024</span>',
          font=dict(size=18), x=0.07, xanchor='left', y=0.84, yanchor='top'
      ),
      font=dict(size=18), height=500, width=600,
      xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True,
                tickangle=0, tickfont=dict(size=14), tickcolor="#646464", ticks='outside', ticklen=5),
      yaxis=dict(title='Procent', showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=16)),
      plot_bgcolor='white', yaxis_gridcolor='lightgray',
      margin=dict(t=120, b=70, r=80, l=60),
      legend=dict(x=1.05, y=1, traceorder='normal',
                  font=dict(family="Monaco, monospace", size=12, color="black")),
      annotations=[dict(
          xref='paper', yref='paper', x=0.0, y=-0.15,
          xanchor='left', yanchor='top', showarrow=False,
          text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ILC_LVHO05A__custom_7141011/bookmark/table?lang=en&bookmarkId=ac6efb37-3f2f-4b65-9cd6-88e05c335bc1">Eurostat</a>',
          font=dict(size=12, color='black')
      )]
  )
  st.plotly_chart(fig)
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  col1.download_button(label='📥 Hämta data', data=to_excel(data), file_name='df_test.xlsx', key="41")


  # ── ilc_lvho05a – Trångboddhet, riskzon för fattigdom ───────────────────────
  my_filter_pars = {'startPeriod': '2010', 'age': 'TOTAL', 'incgrp': 'B_MD60', 'sex': 'T'}
  data = eurostat.get_data_df('ilc_lvho05a', filter_pars=my_filter_pars)

  data.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)
  data = data[data['geo'].isin(['NO', 'DK', 'FI', 'SE', 'EU27_2020'])]
  data_long = pd.melt(data, id_vars=['geo'],
      value_vars=[str(y) for y in range(2010, 2025)],
      var_name='year', value_name='value')
  data_long['year'] = pd.to_datetime(data_long['year'], format='%Y')

  fig = go.Figure()
  for idx, country in enumerate(data_long['geo'].unique()):
      cd = data_long[data_long['geo'] == country]
      fig.add_trace(go.Scatter(
          x=cd['year'], y=cd['value'],
          mode='lines',
          name='EU' if country == 'EU27_2020' else country,
          line=dict(color=colors[idx], width=2.6)
      ))

  fig.update_layout(
      title=dict(
          text='Andel av befolkningen i trångbodda hushåll'
              '<br><span style="font-size:14px; color:#444; font-weight:normal;">I riskzonen för fattigdom, 2010–2024</span>',
          font=dict(size=18), x=0.07, xanchor='left', y=0.84, yanchor='top'
      ),
      font=dict(size=18), height=500, width=600,
      xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True,
                tickangle=0, tickfont=dict(size=14), tickcolor="#646464", ticks='outside', ticklen=5),
      yaxis=dict(title='Procent', showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=16)),
      plot_bgcolor='white', yaxis_gridcolor='lightgray',
      margin=dict(t=120, b=70, r=80, l=60),
      legend=dict(x=1.05, y=1, traceorder='normal',
                  font=dict(family="Monaco, monospace", size=12, color="black")),
      annotations=[dict(
          xref='paper', yref='paper', x=0.0, y=-0.15,
          xanchor='left', yanchor='top', showarrow=False,
          text='Källa: <a href="https://ec.europa.eu/eurostat/databrowser/view/ilc_lvho05a/default/table?lang=en">Eurostat</a>',
          font=dict(size=12, color='black')
      )]
  )
  st.plotly_chart(fig)
  st.header("Hämta", divider=True)
  col1, col2, col3 = st.columns(3)
  col1.download_button(label='📥 Hämta data', data=to_excel(data), file_name='df_test.xlsx', key="42")

# ── Filter & hämta data ────────────────────────────────────────────────────────
  gdp_filter_pars = {
      'freq':    'A',
      'unit':    'PC_GDP',
      'asset10': 'N111G',        # Dwellings gross (bostäder, brutto)
  }
  data_gdp = eurostat.get_data_df('nama_10_an6', filter_pars=gdp_filter_pars)

  # ── Länder att visa ────────────────────────────────────────────────────────────
  country_map = {
      'SE':       'SE',
      'EU27_2020':'EU',
      'DK':       'DK',
      'NO':       'NO',
      'FI':       'FI',
  }

  time_col = 'geo\\TIME_PERIOD'   # kolumnnamn i Eurostat-df

  # Hämta tidsaxel (kolumner efter den första icke-tidskolumnen)
  first_country = list(country_map.keys())[0]
  ref_row   = data_gdp.loc[data_gdp[time_col] == first_country]
  year_cols = ref_row.columns[ref_row.columns.get_loc(time_col) + 1:]

  # Välj ett startår (matcha ungefär med sts-serien ovan)
  start_year = '2008'
  year_cols_trimmed = [c for c in year_cols if str(c) >= start_year]

  # ── Bygg values_dict ──────────────────────────────────────────────────────────
  values_dict_gdp = {}
  for eurostat_code, label in country_map.items():
      row = data_gdp.loc[data_gdp[time_col] == eurostat_code]
      if row.empty:
          continue
      vals = row.iloc[0][year_cols_trimmed].values.tolist()
      values_dict_gdp[label] = vals

  x_labels_gdp = list(year_cols_trimmed)   # ['2008', '2009', …]

  # ── Plotta ────────────────────────────────────────────────────────────────────
  colors_gdp  = ['#0051BA', '#FDB813', 'rgb(250,80,80)', 'rgb(200,75,10)', 'rgb(20,200,220)']
  title_gdp   = "Andel bostadsinvesteringar av BNP"
  source_gdp  = "https://ec.europa.eu/eurostat/databrowser/view/nama_10_an6__custom_12695327/bookmark/table?lang=en&bookmarkId=58012838-134a-4845-8e0b-0444203bc9a0"

  # Anpassad plotfunktion för årsdata (ingen Q-logik behövs)
  def create_bki_plot_eurostat_annual(values_dict, keys_kv, colors, title, source_url, sub_heading):
      min_length     = min(len(v) for v in values_dict.values())
      keys_kv_trimmed = keys_kv[:min_length]

      df = pd.DataFrame({'Time': keys_kv_trimmed})
      for label, values in values_dict.items():
          df[label] = values[:min_length]

      # Traces
      data_traces = []
      for i, (label, _) in enumerate(values_dict.items()):
          trace = go.Scatter(
              x=df['Time'],
              y=df[label],
              name=label,
              hovertext=[
                  f"År: {t}<br>{label}: {v:.2f} %"
                  for t, v in zip(df['Time'], df[label])
              ],
              hoverinfo='text',
              mode='lines',
              line=dict(color=colors[i], width=2.6,
                        dash='dash' if label == 'Total' else None),
              opacity=1,
          )
          data_traces.append(trace)

      combined_title = (
          f'{title}<br>'
          f'<span style="font-size:14px; color:#444; font-weight:normal;">'
          f'{sub_heading}</span>'
      )

      layout = go.Layout(
          title=dict(
              text=combined_title,
              font=dict(size=18),
              x=0.07, xanchor='left',
              y=0.84, yanchor='top',
          ),
          height=500,
          font=dict(size=18),
          xaxis=dict(
              tickvals=df['Time'].tolist()[::2],   # vartannat år
              ticktext=df['Time'].tolist()[::2],
              tickangle=0,
              showline=True, linewidth=1, linecolor='black', mirror=True,
              tickfont=dict(size=14),
              tickcolor='#646464',
              ticks='outside', ticklen=5,
          ),
          yaxis=dict(
              showline=True, linewidth=1, linecolor='black', mirror=True,
              tickfont=dict(size=16),
              ticksuffix=' %',
          ),
          plot_bgcolor='white',
          yaxis_gridcolor='lightgray',
          legend=dict(
              x=1.05, y=1,
              traceorder='normal',
              font=dict(family='Monaco, monospace', size=12, color='black'),
          ),
          margin=dict(t=120, b=70, r=80, l=70),
          annotations=[
              dict(
                  xref='paper', yref='paper',
                  x=0.0, y=-0.15,
                  xanchor='left', yanchor='top',
                  text=f'Källa: <a href="{source_url}">Eurostat</a>',
                  font=dict(size=12, color='black'),
                  showarrow=False,
              ),
          ],
      )

      config = {
          'toImageButtonOptions': {'format': 'png', 'filename': 'bostadsinv_bnp', 'scale': 2},
          'displaylogo': False,
      }

      fig = go.Figure(data=data_traces, layout=layout)
      fig.update_layout(width=600)
      st.plotly_chart(fig, config=config)

      # --- Nedladdningsknappar ---
      def save_as_html(fig):
          buf = StringIO()
          fig.write_html(buf, include_plotlyjs='cdn', config=config)
          return buf.getvalue().encode('utf-8')

      def display_download_button(fig):
          global download_counter
          col1.download_button(
              label='📈 Hämta figur',
              data=save_as_html(fig),
              file_name='figure_bnp.html',
              mime='text/html',
              key=f'download_button_{download_counter}',
          )
          download_counter += 1

      def display_download_button_excel(df):
          global download_counter_excel
          col2.download_button(
              label='📥 Hämta data',
              data=to_excel(df),
              file_name='data_bnp.xlsx',
              mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
              key=f'download_button_excel_{download_counter_excel}',
          )
          download_counter_excel += 1

      st.header('Hämta', divider=True)
      col1, col2, col3 = st.columns(3)
      display_download_button(fig)
      display_download_button_excel(df)


  create_bki_plot_eurostat_annual(
      values_dict_gdp,
      x_labels_gdp,
      colors_gdp,
      title_gdp,
      source_gdp,
      'Bruttoinvesteringar',
  )

  # ── Andel bygginvesteringar av BNP (N11KG) ────────────────────────────────────
  bygg_filter_pars = {
      'freq':    'A',
      'unit':    'PC_GDP',
      'asset10': 'N11KG',        # Total construction (bygginvesteringar, brutto)
  }
  data_bygg = eurostat.get_data_df('nama_10_an6', filter_pars=bygg_filter_pars)

  ref_row_bygg   = data_bygg.loc[data_bygg[time_col] == first_country]
  year_cols_bygg = ref_row_bygg.columns[ref_row_bygg.columns.get_loc(time_col) + 1:]
  year_cols_bygg_trimmed = [c for c in year_cols_bygg if str(c) >= start_year]

  values_dict_bygg = {}
  for eurostat_code, label in country_map.items():
      row = data_bygg.loc[data_bygg[time_col] == eurostat_code]
      if row.empty:
          continue
      vals = row.iloc[0][year_cols_bygg_trimmed].values.tolist()
      values_dict_bygg[label] = vals

  x_labels_bygg = list(year_cols_bygg_trimmed)

  create_bki_plot_eurostat_annual(
      values_dict_bygg,
      x_labels_bygg,
      colors_gdp,
      'Andel bygginvesteringar av BNP',
      source_gdp,
      'Bruttoinvesteringar',
  )

with tab9:
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
            "2024K2",
            "2024K3"
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
  #st.pyplot(plt)
  st.write(df_combined_kpi.round(1))
  st.subheader("** Tabell 3.3 Påbörjade bostäder och befolkningsökning **")
  st.write(df_combined_tot.round(0).fillna(0).astype(int))



