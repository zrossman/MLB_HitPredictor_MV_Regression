import kaggle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_file('s903124/mlb-statcast-data', file_name = 'Statcast_2021.csv')

import zipfile
with zipfile.ZipFile('Statcast_2021.csv.zip', 'r') as zipref:
    zipref.extractall()

#Read in our data, and extract relevant columns
df = pd.read_csv('Statcast_2021.csv', index_col = 0)
df = df[['events', 'launch_speed', 'launch_angle']]

#Here, we delete rows that have no result. We care only about balls put into the field of play (or caught foul balls).
df.dropna(axis = 0, inplace = True)

#Now that our data only consists of pitches that produce events, lets take a look at these possible events.
unique_plays = df['events'].unique()
print('Unique Play Results')
print(unique_plays)
print()
#We are only interested in hits and outs. Singles, doubles, triples, and home runs are all considered hits. Field outs,
#grounded into double plays, force outs, field_errors, fielders choice out, and triple plays are all considered outs,
#and go against a players batting average.

#Here we change all outs to 0, and all hits to 1
df['events'].replace('field_out', 0, inplace = True)
df['events'].replace('grounded_into_double_play', 0, inplace = True)
df['events'].replace('force_out', 0, inplace = True)
df['events'].replace('field_error', 0, inplace = True)
df['events'].replace('fielders_choice_out', 0, inplace = True)
df['events'].replace('double_play', 0, inplace = True)
df['events'].replace('triple_play', 0, inplace = True)
df['events'].replace('single', 1, inplace = True)
df['events'].replace('double', 1, inplace = True)
df['events'].replace('triple', 1, inplace = True)
df['events'].replace('home_run', 1, inplace = True)

#Lets take a look at all of our remaining unique events
results = df['events'].value_counts()
print('Event Totals')
print(results)
print()
#We see that we still have some extra events left over. These events don't count as at bats, so we don't want them

#Getting rid of the events that don't count as at bats, by selecting only the rows that resulted in a 1 or 0
df = df.loc[(df['events'] == 1) | (df['events'] == 0)]
results = df['events'].value_counts()
print('Event Totals')
print(results)
print()

#Converting our feature and label column/s into arrays
X = np.array(df[['launch_speed', 'launch_angle']])
y = np.array(df['events'])

#Splitting our data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)

#Using our train data to find a line of best fit
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Using line of best fit on our test data
y_pred = regressor.predict(X_test)

#Comparing our predictions to the y_test data
comparison = []
for i in range(len(y_test)):
    a_list = []
    a_list.append(y_test[i])
    a_list.append(y_pred[i])
    comparison.append(a_list)
print('Comparison [y_test, y_pred]')
print(comparison[:50])
print()

#Calculating our average prediction score for hits and outs
hit_total = 0
hit_total_pred = 0
out_total = 0
out_total_pred = 0
for row in comparison:
    if row[0] == 0:
        out_total += 1
        out_total_pred += row[1]
    else:
        hit_total += 1
        hit_total_pred += row[1]
avg_hit_pred = hit_total_pred / hit_total
avg_out_pred = out_total_pred / out_total
print('Average Prediction Scores for Outs, Hits')
print(avg_out_pred, avg_hit_pred)
print()

#Calculating total number of hits and outs in y_test
y_test_hits = 0
y_test_outs = 0
for element in y_test:
    if element == 0:
        y_test_outs += 1
    else:
        y_test_hits += 1
print('y_test Hit, Out Totals:', y_test_hits,',', y_test_outs)

hit_threshold = 0.416

#Calculating total number of hits and outs in y_pred
y_pred_hits = 0
y_pred_outs = 0
for element in y_pred:
    if element >= hit_threshold:
        y_pred_hits += 1
    else:
        y_pred_outs += 1
print('y_pred Hit, Out Totals:', y_pred_hits, ',', y_pred_outs)

#Calculating the model's accuracy on actual hits
hit_list = []
for row in comparison:
    if row[0] == 1:
        hit_list.append(row)
hit_correct = 0
for element in hit_list:
    if element[1] >= hit_threshold:
        hit_correct += 1
hit_pred_accuracy = hit_correct / len(hit_list)
print('Prediction accuracy on hits:', hit_pred_accuracy)

#Calculating the model's accuracy on actual outs
out_list = []
for row in comparison:
    if row[0] == 0:
        out_list.append(row)
out_correct = 0
for element in out_list:
    if element[1] <= hit_threshold:
        out_correct += 1
out_pred_accuracy = out_correct / len(out_list)
print('Prediction accuracy on outs:', out_pred_accuracy)

#Calculating the total accuracy of the model
total_correct = 0
for row in comparison:
    if row[0] == 1 and row[1] >= hit_threshold:
        total_correct += 1
    elif row[0] == 0 and row[1] < hit_threshold:
        total_correct += 1
accuracy = total_correct / len(comparison)
print('Total Accuracy:', accuracy)

#Creating a function that will predict outs/hits
def hit_or_out(input):
    input = np.array(input)
    input = np.reshape(input, (1, -1))
    prediction = regressor.predict(input)
    if prediction > hit_threshold:
        return 'Hit'
    else:
        return 'Out'


print(hit_or_out([100, 20]))