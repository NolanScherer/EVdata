import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

x = pd.read_csv("/Users/nolanscherer/Downloads/CS/Projects/EV Project/Electric_Vehicle_Population_Data.csv")

#part 1: visualizing the data 
#create two dataframes for PHEVs(plug in hybrid electric vehicles) and BEVs (battery elevtric vehicles)
batt = x.loc[x["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)"] 
batt = batt.loc[batt["Electric Range"] > 0]
plug = x.loc[x["Electric Vehicle Type"] == "Plug-in Hybrid Electric Vehicle (PHEV)"] 

#remove redundant columns
batt.drop(inplace=True, columns=["VIN (1-10)", "County", "State", "Postal Code", 'Make', 'Model', 'DOL Vehicle ID','2020 Census Tract'])
plug.drop(inplace=True, columns=["VIN (1-10)", "County", "State", "Postal Code", 'Make', 'Model', 'DOL Vehicle ID','2020 Census Tract'])


plt.scatter(batt["Model Year"], batt["Electric Range"])
plt.title("Electric range over time of EV vehicles")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")

z = np.polyfit(batt["Model Year"], batt["Electric Range"],1)
p = np.poly1d(z) 

plt.plot(batt["Model Year"], p(batt["Model Year"]), "r--", label='Regression Line')
plt.show() #scatterplot


city_counts = plug["City"].value_counts()
city_proportions = city_counts / len(plug)
city_proportions_list = list(city_proportions.items())

plt.pie([i[1] for i in city_proportions_list[:20]],labels=[j[0] for j in city_proportions_list[:20]])
plt.title("top 20 cities in washington for PHEVs")
plt.show() #pie chart


#part 2: ML model to predict make

relevant = ["City", "Electric Vehicle Type", "Electric Range", "Model Year"]
df = x[relevant]
target = "Make"

encodedLabels = {}
for i in relevant:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
    encodedLabels[i] = le

targetLE = LabelEncoder()
df[target] = targetLE.fit_transform(x[target])

i = df[relevant]
j = df[target]
i_train, i_test, j_train, j_test = train_test_split(i,j,test_size=0.2, random_state=35)

clas=DecisionTreeClassifier()
clas.fit(i_train,j_train)

j_prediction = clas.predict(i_test)
print(classification_report(j_test,j_prediction))
