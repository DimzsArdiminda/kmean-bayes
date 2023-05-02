import tkinter as tk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
import mysql.connector


# Load the data from the database
mydb = mysql.connector.connect(
    host="localhost", 
    #Nama Database
    database="prostat",
    user="root",
    password=""
)

# Load the training data from the 'training_set' table
df_train = pd.read_sql("SELECT * FROM training_set", mydb)

# Load the testing data from the 'testingset' table
df_test = pd.read_sql("SELECT * FROM testingset", mydb)

# Create a scatter plot of Time spent on fitness and Time spent on sleep
sns.scatterplot(data=df_train, x="Time spent on fitness", y="Time spent on sleep", hue="Health issue")

# Separate the features and target variables for training and testing data
X_train = df_train[["Time spent on fitness", "Time spent on sleep"]]
y_train = df_train["Health issue"]
X_test = df_test[["Time spent on fitness", "Time spent on sleep"]]
y_test = df_test["Health issue"]

# Train the model
model = GaussianNB()
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate the accuracy and F1 score of the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the accuracy and F1 score
print("Predicted class labels:", y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 score: {f1:.2f}")

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Plot the classification report
print(classification_report(y_test, y_pred))

# Define function to make predictions using the trained model
def predict_health_issue(fitness_time, sleep_time):
    data = {"Time spent on fitness": fitness_time, "Time spent on sleep": sleep_time}
    X_new = pd.DataFrame(data, index=[0])
    y_pred = model.predict(X_new)
    return y_pred[0]

# Define function to handle button click event
def on_predict():
    fitness_time = float(entry_fitness.get())
    sleep_time = float(entry_sleep.get())
    health_issue = predict_health_issue(fitness_time, sleep_time)
    result_label.config(text=f"Predicted health issue: {health_issue}")
    health_score = round(model.predict_proba([[fitness_time, sleep_time]])[0][1]*100)
    health_score_label.config(text=f"Health score: {health_score}%")

# Define function to handle button click event for adding new data
def on_add_data():
    fitness_time = float(entry_fitness.get())
    sleep_time = float(entry_sleep.get())
    new_data = {"Time spent on fitness": [fitness_time], "Time spent on sleep": [sleep_time], "Health issue": [on_predict.health_issue]}
    df_new = pd.DataFrame(new_data)
    global df
    df = pd.concat([df, df_new], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df[["Time spent on fitness", "Time spent on sleep", "Health issue"]], df["Health issue"], test_size=0.2
    )
    model.fit(X_train[["Time spent on fitness", "Time spent on sleep"]], y_train)
    result_label.config(text=f"Data added successfully!")
    health_score_label.config(text="")


# Create a Tkinter window
root = tk.Tk()

# Create a Text widget to display the data
table = tk.Text(root)

# Create the input widgets
frame_input = tk.Frame(root)
frame_input.pack()
label_fitness = tk.Label(frame_input, text="Time spent on fitness:")
label_fitness.pack(side=tk.LEFT)
entry_fitness = tk.Entry(frame_input)
entry_fitness.pack(side=tk.LEFT)
label_sleep = tk.Label(frame_input, text="Time spent on sleep:")
label_sleep.pack(side=tk.LEFT)

# Create the input
entry_sleep = tk.Entry(frame_input)
entry_sleep.pack(side=tk.LEFT)


# Create the buttons
frame_buttons = tk.Frame(root)
frame_buttons.pack()
predict_button = tk.Button(frame_buttons, text="Predict", command=on_predict)
predict_button.pack(side=tk.LEFT)

# Create labels to display the results
result_label = tk.Label(root, text="")
result_label.pack()
health_score_label = tk.Label(root, text="")
health_score_label.pack()

# Insert the data into the Text widget
table.insert(tk.END, df_train.to_string(index=False))

# Pack the Text widget into the window
table.pack()

# Run the Tkinter event loop
root.mainloop()
