import streamlit as st

st.title("Test!!!! 😎🐴")

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
