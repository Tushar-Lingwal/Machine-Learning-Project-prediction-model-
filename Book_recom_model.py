import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('books.csv', error_bad_lines=False)
print(df.head())

print(df.isnull().sum())
print(df.describe())

top_ten = df[df['ratings_count'] > 1000000]

# making a top_ten var for storing book that contain only number of rating > 1000000
# arranging the values of the books according to the 1-5 avg rating in decending order in list top_ten
top_ten.sort_values(by='average_rating', ascending=False)

# plotting the bar graph for the above list made

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 5))
data = top_ten.sort_values(by='average_rating', ascending=False).head(10)
sns.barplot(x="average_rating", y="title", data=data, palette='inferno')
plt.show()

# first we group by thw books authors and sort them in descending order then we extract the top 10 authors form the list
# and then we reassign the index of the table we got

most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(
    10).set_index('authors')
plt.figure(figsize=(10, 5))

# we plot the bar graph of the following data we extracted
ax = sns.barplot(most_books['title'], most_books.index, palette='inferno')
ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")

# this is for marking the values in front of the bar graph.
# we calculated the pixel width of the plot then assign it to a variable and finally put values in front of the graph
# patches function is used to count no of pixel in the 2d drawing
totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width() + .2, i.get_y() + .2, str(round(i.get_width())), fontsize=15, color='black')
plt.show()

# we are calculating the moet number of times rated book in the whole data set
# we arranged top 10 most rated books together using "rating count" column as the reference form data set
most_rated = df.sort_values('ratings_count', ascending=False).head(10).set_index('title')
plt.figure(figsize=(15, 6))
ax = sns.barplot(most_rated['ratings_count'], most_rated.index, palette='inferno')

totals = []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width() + .2, i.get_y() + .2, str(round(i.get_width())), fontsize=15, color='black')
plt.show()

# we are now plotting a distribution graph for the "average_rating"
# we change the type for the "average_rating to float using .astype()

df.average_rating = df.average_rating.astype(float)

fig, ax = plt.subplots(figsize=[12, 6])

# making a distribution plot
sns.distplot(df['average_rating'], ax=ax)
ax.set_title('Average rating distribution for all books', fontsize=20)
ax.set_xlabel('Average rating', fontsize=13)
plt.show()

# next we draw the relation plot between the rating Count and average rating
ax = sns.relplot(data=df, x="average_rating", y="ratings_count", color = 'red', sizes=(100, 200), height=7, marker='o')
plt.title("Relation between Rating counts and Average Ratings",fontsize = 15)
ax.set_axis_labels("Average Rating", "Ratings Count")
plt.show()

# next we calculated the "number of pages" vs "average rating"
plt.figure(figsize=(15,10))
ax = sns.relplot(x="average_rating", y="  num_pages", data = df, color = 'red',sizes=(100, 200), height=7, marker='o')
ax.set_axis_labels("Average Rating", "Number of Pages")
plt.show()

df2 = df.copy()
df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])

features = pd.concat([rating_df,
                      language_df,
                      df2['average_rating'],
                      df2['ratings_count']], axis=1)


min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# making the KNN (k near    aest neighbour) predictive model for the data frame features
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)

# making a book recommender function that will store best rating books name and recommend it to the people
def BookRecommender(book_name):
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df2.loc[newid].title)
    return book_list_name


BookNames = BookRecommender('Harry Potter and the Half-Blood Prince (Harry Potter  #6)')
print(BookNames)
