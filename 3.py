import numpy as np
import pandas as pd
from nose.tools import *
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


league_data = pd.read_csv("data/results.csv")
stats = pd.read_csv('data/stats.csv')
standings = pd.read_csv('data/EPL Standings 2000-2022.csv')
with_goalscorers = pd.read_csv('data/with_goalscorers.csv')
assert_is_not_none(league_data)
assert_is_not_none(stats)
assert_is_not_none(standings)
assert_is_not_none(with_goalscorers)

league_data.to_excel("league_data.xlsx", index=False)

print(league_data)


expected_columns = ['home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'season']
expected_shape = (4560, 6)

assert_equal(list(league_data.columns), expected_columns)
assert_equal(league_data.shape, expected_shape)

new_standings = standings[standings['Season'].between('2006-07','2017-18')]
new_standings.to_excel("new_standings.xlsx", index=False)
print(new_standings[new_standings.Pos == 1])


new_standings.reset_index(drop=True,inplace=True)
new_standings = new_standings.drop('Qualification or relegation',axis=1)
print(new_standings)
assert_equal(new_standings.shape, (240, 11))

with_goalscorers[['Top Scorer Goals', 'Top Scorer Team']] = with_goalscorers['Top Scorer'].str.extract(r'(\d+)\((.*?)\)$')
with_goalscorers.drop('Top Scorer', axis=1, inplace=True)
with_goalscorers['Top Scorer Goals'] = with_goalscorers['Top Scorer Goals'].astype(int)
with_goalscorers.drop('# Squads',axis = 1,inplace=True)
with_goalscorers_sorted = with_goalscorers[with_goalscorers.Season.between('2006-2007','2017-2018')]
with_goalscorers.to_excel("with_goalscorers.xlsx", index=False)
with_goalscorers['Top Scorer Goals'] = with_goalscorers['Top Scorer Goals'].astype(int)
assert_equal(with_goalscorers_sorted.shape, (12, 5))

#Data Analysis
champions = new_standings[new_standings.Pos==1]
champions.reset_index(drop=True,inplace=True)

assert_equal(champions.shape, (12, 11))
assert_is_not_none(champions)
assert_is_not_none(stats)
assert_is_not_none(with_goalscorers_sorted)
assert_is_not_none(league_data)
assert_is_not_none(new_standings)
assert_equal(with_goalscorers_sorted.columns.tolist(), ['Season', 'Competition Name', 'Champion', 'Top Scorer Goals','Top Scorer Team'])

most_goals_season = stats.loc[stats.goals.idxmax()][['team', 'goals', 'season']]
print(most_goals_season)
non_champions = new_standings[new_standings['Pos'] != 1]
non_champions.reset_index(drop=True, inplace=True)
most_goals_non_champion = non_champions.nlargest(1, 'GF')
print(most_goals_non_champion[['Team', 'GF', 'Season','Pts']])
champions_scored_less_1 = champions[champions['GF'] < non_champions['GF'].max()]
var = champions_scored_less_1.shape[0]
champions_scored_more = champions[champions.GF>non_champions.GF.max()]
print(champions_scored_more)

with_goalscorers_sorted = with_goalscorers.copy() # To avoid a warning
with_goalscorers_sorted.sort_values(by='Season', inplace=True)
with_goalscorers_sorted.reset_index(drop=True, inplace=True)
print(with_goalscorers_sorted)


goalscorer_champions = with_goalscorers_sorted[with_goalscorers_sorted.apply(lambda row: str(row['Champion']) in str(row['Top Scorer Team']), axis=1)]


percentage_tg_ch = len(goalscorer_champions) / len(with_goalscorers_sorted[14:26]) * 100
percentage_formatted = f"{percentage_tg_ch:.1f}%"
print(percentage_formatted)

second_placed = new_standings[new_standings.Pos==2]
second_placed.reset_index(drop=True,inplace=True)
winning_margin=champions.Pts-second_placed.Pts
champions = champions.assign(Winning_margin=winning_margin.values)
print(champions)


#Plots

assert_is_not_none(non_champions)
assert_is_not_none(goalscorer_champions)
assert_is_not_none(second_placed)
assert_is_not_none(percentage_tg_ch)
assert_is_not_none(league_data)
assert_is_not_none(new_standings)
assert_is_not_none(winning_margin)

plt.plot(champions.Season,champions.Winning_margin)
plt.xlabel('Season')
plt.ylabel('Winning margin')
plt.xticks(rotation=45)
plt.show()

times_champion = champions.Team.value_counts()
teams = times_champion.index.to_list()
plt.pie(times_champion,
    labels = teams,
    autopct ='%1.1f%%',
    colors = ['red','royalblue','skyblue','white'])
plt.title('Titles won from 2006-07 to 2017-18')
plt.show()

#modeling
y = stats['wins']

X = stats.drop(['wins','season','team'],axis = 1)

print(y.shape)
print(X.shape)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


# Now split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Fit the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

actual_predicted = pd.DataFrame({'Actual wins': y_test.squeeze(), 'Predicted wins': y_pred.squeeze()}).reset_index(drop=True)
actual_predicted.head()

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}\n\n')

# Fit Logistic Regression model
logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)

# Predict on the test set using Logistic Regression
y_pred_logistic = logistic_regressor.predict(X_test)

# Calculate performance metrics for Logistic Regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, average='weighted')
recall_logistic = recall_score(y_test, y_pred_logistic, average='weighted')
f1_logistic = f1_score(y_test, y_pred_logistic, average='weighted')

# Print evaluation metrics for Logistic Regression
print("Logistic Regression Evaluation Metrics:")
print(f"Accuracy: {accuracy_logistic:.2f}")
print(f"Precision: {precision_logistic:.2f}")
print(f"Recall: {recall_logistic:.2f}")
print(f"F1 Score: {f1_logistic:.2f}")
print()

# Calculate performance metrics for Linear Regression (already calculated)
# Print evaluation metrics for Linear Regression
print("Linear Regression Evaluation Metrics:")
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}\n\n')

# Create a line plot for actual wins
plt.plot(actual_predicted.index, actual_predicted['Actual wins'], label='Actual Wins')

# Create a line plot for predicted wins
plt.plot(actual_predicted.index, actual_predicted['Predicted wins'], label='Predicted Wins')

plt.xlabel('Index')
plt.ylabel('Wins')
plt.title('Actual vs. Predicted Wins')
plt.legend()
plt.show()

# Plot the scatter plot
sns.scatterplot(x='Predicted wins', y='Actual wins', data=actual_predicted)

# Add the regression line
sns.regplot(x='Predicted wins', y='Actual wins', data=actual_predicted, scatter=False, color='red')

# Set the axis labels and title
plt.xlabel('Predicted wins')
plt.ylabel('Actual wins')
plt.title('Actual vs Predicted Wins')

# Show the plot
plt.show()

feature_importance = pd.Series(regressor.coef_, index=X.columns).sort_values(ascending=False)
print(feature_importance)