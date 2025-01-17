import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv("/Users/nolanscherer/Downloads/CS/Projects/EV Project/Electric_Vehicle_Population_Data.csv")
batt = x.loc[x["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)"] #batt is a data frame with only battery vehicles
batt = batt.loc[batt["Electric Range"] > 0]


plt.scatter(batt["Model Year"], batt["Electric Range"])
plt.xlabel("Model Year")
plt.ylabel("Electric Range")

z = np.polyfit(batt["Model Year"], batt["Electric Range"],1)
p = np.poly1d(z) 

plt.plot(batt["Model Year"], p(batt["Model Year"]), "r--", label='Regression Line')
plt.show()
