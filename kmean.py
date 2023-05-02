import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import mysql.connector

#Menghubungkan progam dengan Database MySql
mydb = mysql.connector.connect(host='localhost',
                        #Nama Database
						database='prostat',
						user='root',
						password='')

mycursor = mydb.cursor()

#Membaca data pada Tabel 'data4'
data = pd.read_sql("SELECT * FROM data4", mydb)

#Menggunakan data pada Kolom Time spent on fitness dan Time spent on sleep sebagai inputan
plt.scatter(data['Time spent on fitness'], data['Time spent on sleep'] )
plt.xlabel('Time spent on fitness')
plt.ylabel('Time spent on sleep')

scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)

data_scaled = pd.DataFrame(data_scaled, columns=['Time spent on fitness','Time spent on sleep'])

km = KMeans(n_clusters=2, init='k-means++', n_init=20, verbose=1)
 
y_predicted = km.fit_predict(data_scaled[['Time spent on fitness','Time spent on sleep']])
y_predicted
 
data['Health_Issue'] = y_predicted

data1 = data[data.Health_Issue==0]
data2 = data[data.Health_Issue==1]

plt.scatter(data1['Time spent on fitness'],data1['Time spent on sleep'],color='green')
plt.scatter(data2['Time spent on fitness'],data2['Time spent on sleep'],color='blue')

plt.xlabel('Time spent on fitness')
plt.ylabel('Time spent on sleep')
plt.grid()

conditions = [     
    (data['Health_Issue']==0),
    (data['Health_Issue']==1)]
choices = ['0','1']
data['Health_Issue'] = np.select(conditions, choices)


# Define a function to predict the cluster for a given input
def predict_cluster():
    # Get the input values from the Entry widgets
    fitness_time = int(fitness_entry.get())
    sleep_time = int(sleep_entry.get())

    # Scale the input values
    input_scaled = scaler.transform([[fitness_time, sleep_time]])

    # Predict the cluster for the input
    cluster = km.predict(input_scaled)

    # Display the result in the output Text widget
    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, f"The input belongs to cluster {cluster[0]}")

# Create a Tkinter window
root = tk.Tk()


# Create Entry widgets for input
fitness_label = tk.Label(root, text="Time spent on fitness")
fitness_label.pack()
fitness_entry = tk.Entry(root)
fitness_entry.pack()

sleep_label = tk.Label(root, text="Time spent on sleep")
sleep_label.pack()
sleep_entry = tk.Entry(root)
sleep_entry.pack()

# Create a button to predict the cluster for the input
predict_button = tk.Button(root, text="Predict", command=predict_cluster)
predict_button.pack()

# Create an output Text widget to display the result
output_label = tk.Label(root, text="Cluster prediction")
output_label.pack()
output_text = tk.Text(root, height=1)
output_text.pack()

# Create a Text widget to display the table
table = tk.Text(root)

# Insert the data into the Text widget
table.insert(tk.END, data.to_string(index=False))

# Pack the Text widget into the window
table.pack()
plt.show()

# Run the Tkinter event loop
root.mainloop()
