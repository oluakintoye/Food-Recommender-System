from flask import Flask, redirect, url_for, request
from flask import render_template
app = Flask(__name__)


import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

Brand = ""
Category = []
Manufacturer = []
Message = []



@app.route('/display_recommendations')
def display_recommendations():

    return render_template('Recommendations_.html')



@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    ingredient_list=""
    if request.method == 'POST':
        brand = request.form['brand']
        print(brand)
        category = request.form['category']
        manufacturer = request.form['manufacturer']
        message = request.form['message']


        #Appending new data
        ingredient_list += brand + " " + category + " " + manufacturer + " " + message
        print(ingredient_list)

        pd.set_option('display.max_columns', 10)
        df = pd.read_csv('Food_Ingredients.csv',  encoding='latin-1')

        print('=' * 90)

        #Choosing the Vector features to base my recommendations on
        print('---------------------------FEATURE VECTOR TABLE---------------------------')
        df = df[['brand','categories','features.value','manufacturer','name']]

        df = pd.DataFrame(np.array([['new',ingredient_list]]), columns=['name','features.value']).append(df, ignore_index=True, sort=True)
        # df.loc[0].replace(np.nan, "a")
        df.fillna("")
        print(df.head())
        print(df.shape)


        print('--------------------Feature Vector Information--------------------')
        print(df.info())
        print()

        print('=' * 90)

        print("--------------------LIST OF UNIQUE VALUES IN THE FEATURE VECTORS--------------------")
        unique_values_brand = df['brand'].unique().tolist()
        unique_values_categories = df['categories'].unique().tolist()
        unique_values_ingredients = df['features.value'].unique().tolist()
        unique_values_manufacturer = df['manufacturer'].unique().tolist()


        print('=' * 90)

        print()
        print("--------------------NUMBER OF UNIQUE VALUES FOR THE FEATURE VECTORS--------------------")
        print('Brands: ',unique_values_brand)
        print('Categories: ', unique_values_categories)
        print('Ingredients: ', unique_values_ingredients)
        print('Manufacturers: ', unique_values_manufacturer)

        print('=' * 90)

        print()
        print('--------------------FEATURE VECTORS UNIQUE VALUES(NO)--------------------')
        print("Brand: ", len(unique_values_brand))
        print('Categories: ', len(unique_values_categories))
        print('Ingredients: ', len(unique_values_ingredients))
        print('manufacturers: ', len(unique_values_manufacturer))


        print('=' * 90)

        print()
        print('--------------------FEATURE VECTORS UNIQUE VALUES COUNT(TOP 20)\n--------------------')
        print("BRANDS: \n", df['brand'].value_counts().head(20))
        print()
        print('CATEGORIES: \n', df['categories'].value_counts().head(20))
        print()
        # print('INGREDIENTS: \n', df['features.value'].value_counts())
        print()
        print('MANUFACTURERS: \n', df['manufacturer'].value_counts().head(20))
        print()
        # print('NAMES: \n', df['name'].encode("utf-8").value_counts().head(20 ))
        print()


        print('=' * 90)


        #Transforming the values of the FV individual columns into single words so they are considered as unique values.

        # discarding the commas between the actors' full names and getting only the first three names
        df['categories'] = df['categories'].astype('str')
        df['categories'] = df['categories'].map(lambda x: x.split(',')[:3])
        df['manufacturer'] = df['manufacturer'].astype('str')
        df['manufacturer'] = df['manufacturer'].map(lambda x: x.split(',')[:3])
        df['brand'] = df['brand'].astype('str')
        df['brand'] = df['brand'].map(lambda x: x.split(' '))

        # print('huh??')
        # print(df['categories'].head())

        print('=' * 90)
        print()

        # merging together first and last name for each categories and manufacturers, so it's considered as one word
        # and there is no mix up between people sharing a first name

        for index, row in df.iterrows():
            row['categories'] = [x.lower().replace(' ','') for x in row['categories']]
            row['manufacturer'] = [x.lower().replace(' ','') for x in row['manufacturer']]
            row['brand'] = ''.join(row['brand']).lower()

        print(df['categories'].head())
        print(df['brand'].head())
        print(df['manufacturer'].head())

        print('=' * 90)

        # initializing the new column
        df['Key_words'] = ""
        df['features.value'] = df['features.value'].astype('str')

        for index, row in df.iterrows():
            ingredients = row['features.value']

            # instantiating Rake, by default is uses english stopwords from NLTK
            # and discard all puntuation characters
            r = Rake()

            # extracting the words by passing the text
            r.extract_keywords_from_text(ingredients)

            # getting the dictionary whith key words and their scores
            key_words_dict_scores = r.get_word_degrees()

            # assigning the key words to the new column
            row['Key_words'] = list(key_words_dict_scores.keys())

        # dropping the Features.value column
        df.drop(columns = ['features.value'], inplace = True)

        df.set_index('name', inplace = True)
        print(df.head())

        print('=' * 90)

        df['bag_of_words'] = ''
        columns = df.columns
        for index, row in df.iterrows():
            words = ''
            for col in columns:
                if col != 'brand':
                    words = words + ' '.join(row[col])+ ' '
                else:
                    words = words + row[col]+ ' '
            row['bag_of_words'] = words

        df.drop(columns = [col for col in df.columns if col != 'bag_of_words'], inplace = True)
        print('bag of word right?')
        # df = pd.DataFrame(np.array([['new','just a joke']]), columns=['name','bag_of_words']).append(df, ignore_index=True)
        # print('MY OWN RATINGS')
        # f = df.loc[0].dropna()
        # print(f.head())

        print(df.shape)
        print(df.head())


        # instantiating and generating the count matrix
        count = CountVectorizer()
        count_matrix = count.fit_transform(df['bag_of_words'])

        # creating a Series for the Food names so they are associated to an ordered numerical
        # list I will use later to match the indexes
        indices = pd.Series(df.index)
        print('is thiss??')
        print(indices[:5])



        # generating the cosine similarity matrix
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        print("cosine simm")
        print(cosine_sim)


        # function that takes in food as input and returns the top 10 recommended foods
        def recommendations(title, cosine_sim = cosine_sim):

            recommended_food = []

            # gettin the index of the food that matches the name
            idx = indices[indices == title].index[0]

            # creating a Series with the similarity scores in descending order
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

            # getting the indexes of the 10 most similar foods
            top_10_indexes = list(score_series.iloc[1:11].index)

            # populating the list with the names of the best 10 matching foods
            for i in top_10_indexes:
                recommended_food.append(list(df.index)[i])
            print("wait")
            for r in range(len(recommended_food)):
                print(recommended_food[r])
            return recommended_food
    foods = recommendations('new')
    comma_separated = ','.join(foods)
    return redirect(url_for('display_recommendations', food = comma_separated))


app.run(debug = True)