import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rainfall_model.pkl")

st.title("🌧️ Rainfall Classification")

month = st.number_input("Month", 1, 12, 1)

season = st.selectbox(
    "Season",
    ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
)

state = st.text_input("State", "Telangana")

district = st.text_input("District", "Hyderabad")

avg_temp = st.number_input("Average Temperature", value=25.0)
min_temp = st.number_input("Minimum Temperature", value=20.0)
max_temp = st.number_input("Maximum Temperature", value=30.0)
wind_speed = st.number_input("Wind Speed", value=5.0)
air_pressure = st.number_input("Air Pressure", value=1013.0)
elevation = st.number_input("Elevation", value=500.0)
latitude = st.number_input("Latitude", value=17.38)
longitude = st.number_input("Longitude", value=78.48)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "month": [month],
        "season": [season],
        "state": [state],
        "district": [district],
        "avg_temp": [avg_temp],
        "min_temp": [min_temp],
        "max_temp": [max_temp],
        "wind_speed": [wind_speed],
        "air_pressure": [air_pressure],
        "elevation": [elevation],
        "latitude": [latitude],
        "longitude": [longitude]
    })
    input_data=pd.get_dummies(input_data)
    input_data= input_data.reindex(columns=model.feature_names_in_,fill_value=0)
    prediction = model.predict(input_data)

    st.success(f"🌧️ Predicted Rainfall Category: {prediction[0]}")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
# -----------------------------------------
# RAINFALL PREDICTION - GRAPHS
# -----------------------------------------

st.header("Rainfall Prediction – Graphs")

# Load Excel data
df = pd.read_excel("india_weather_rainfall_data.xlsx")

# Remove empty rows
df = df.dropna()

# -----------------------------------------
# Graph 1: Rainfall by Month
# -----------------------------------------

st.subheader("1. Average Rainfall by Month")

monthly_rainfall = df.groupby("month")["rainfall"].mean()

fig, ax = plt.subplots()
monthly_rainfall.plot(kind="bar", ax=ax)

ax.set_xlabel("Month")
ax.set_ylabel("Average Rainfall")
ax.set_title("Average Rainfall by Month")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# -----------------------------------------
# Graph 2: Temperature
# -----------------------------------------

st.subheader("2. Average Temperature")

temperature = df.groupby("month")["avg_temp"].mean()

fig, ax = plt.subplots()
temperature.plot(kind="line", marker="o", ax=ax)

ax.set_xlabel("Month")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Average Temperature by Month")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

st.pyplot(fig)


# -----------------------------------------
# Graph 3: Wind Speed
# -----------------------------------------

st.subheader("3. Average Wind Speed")

wind = df.groupby("month")["wind_speed"].mean()

fig, ax = plt.subplots()
wind.plot(kind="bar", ax=ax)

ax.set_xlabel("Month")
ax.set_ylabel("Wind Speed")
ax.set_title("Average Wind Speed by Month")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# -----------------------------------------
# Graph 4: Air Pressure
# -----------------------------------------

st.subheader("4. Average Air Pressure")

pressure = df.groupby("month")["air_pressure"].mean()

fig, ax = plt.subplots()
pressure.plot(kind="line", marker="o", ax=ax)

ax.set_xlabel("Month")
ax.set_ylabel("Air Pressure")
ax.set_title("Average Air Pressure by Month")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

st.pyplot(fig)


# -----------------------------------------
# Graph 5: Rainfall by State
# -----------------------------------------

st.subheader("5. Average Rainfall by State")

state_rainfall = df.groupby("state")["rainfall"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
state_rainfall.plot(kind="bar", ax=ax)

ax.set_xlabel("State")
ax.set_ylabel("Average Rainfall")
ax.set_title("Average Rainfall by State")

plt.xticks(rotation=90)
plt.tight_layout()

st.pyplot(fig)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🌧️ Indian Rainfall Prediction")

# Load dataset
df = pd.read_excel("india_weather_rainfall_data.xlsx")

# -------- EDA --------
st.header("📊 Exploratory Data Analysis")

st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Statistical Summary")
st.write(df.describe())

st.subheader("Correlation")
fig, ax = plt.subplots()
sns.heatmap(df.select_dtypes("number").corr(),
            annot=True, ax=ax)
st.pyplot(fig)
