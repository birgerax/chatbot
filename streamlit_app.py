import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

###### Påbörjade bostäder: uppdelat per upplåtelseform

import requests
import json
import pandas as pd
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
    y=1,
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
    y=-0.2,
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
        font=dict(size=14)  # Set font size for legend text
    ),
    margin=dict(
        b=100  # Increase the bottom margin to provide more space for annotations
    )
)


layout_pkv['title']['y'] = 0.89

pkv_tot = go.Figure(data=data_pkv, layout=layout_pkv, layout_width=800)
pkv_tot.show()




layout_p_kvartal['title']['y'] = 0.89
p_kvartal_tot = go.Figure(data=data_p_kvartal, layout=layout_p_kvartal, layout_width=700)

p_kvartal_tot.show()
