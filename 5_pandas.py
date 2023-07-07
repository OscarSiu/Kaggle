import pandas as pd

#Create variables
fruit_sales =  pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index = ['2017 Sales', '2018 Sales'])
print(fruit_sales)

ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
print(ingredients)
""""""""""""""""""""""""""""""""""""""""""""""""

#Read csv
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)
#reviews.shape()
#reviews.head()


#Write csv
fruit_sales.to_csv('cows_and_goats.csv')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#index based selection [Row-first, column-second]
sample_reviews = reviews.iloc[:3, 0]

#label based selection
df =  reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
# reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])] #is in
reviews.loc[reviews.price.notnull()]

#Assign values
reviews['index_backwards'] = range(len(reviews), 0, -1)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Summary functions
reviews.points.describe()
reviews.points.mean()
reviews.points.median()
reviews.taster_name.unique()
reviews.taster_name.value_counts()


bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
num_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

#Map function
reviews.points.map(lambda p: p - review_points_mean)

#Apply function
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Grouping
reviews_written = reviews.groupby('taster_twitter_handle').size()
reviews.groupby(['country']).price.agg([len, min, max])

#Sorting
countries_reviewed.reset_index() # multi index
countries_reviewed.sort_values(by=['country', 'len'])
best_rating_per_price =reviews.groupby('price')['points'].max().sort_index()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
reviews_per_region = reviews.region_1.fillna('Unknown')
reviews_per_region = reviews_per_region.value_counts().sort_value(by=value_counts(), ascending=False)
#dtype
#astype()
#pd.isnull()
#pd.fillna()
#replace() 
#rename()
#pd.concat()
#join()