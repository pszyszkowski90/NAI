import argparse
import json
import numpy as np
from imdb import IMDb

ia = IMDb()


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True,
                        help='First user')
    parser.add_argument('--user2', dest='user2', required=True,
                        help='Second user')
    parser.add_argument("--score-type", dest="score_type", required=True,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users, 
    # then the score is 0 
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


# Compute the Pearson correlation score between user1 and user2
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0 
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies 
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


def movie_description_get(title):
    items = ia.search_movie(title)
    i = 0
    while i < len(items):
        print('\t[' + str(i) + '] ' + items[i]['title'] + ' (' + str(items[i]['year']) + ')')
        i += 1

    movieIndex = int(input('\nKtóry film masz na myśli? '))
    movie = ia.get_movie(items[movieIndex].movieID)

    print(movie['plot'][0])
    # print(ia.get_movie_main(items[0].movieID))
    # print(items)

def print_result(maxScore, userDict):
    print('Na podstawie danych uzytkownika ' + maxScore['user'])
    new_dict = {}
    for key, value in userDict.items():
        if key not in data[user].keys():
            new_dict[key] = value
    sorted_dict = dict(sorted(new_dict.items(), key=lambda element: element[1]))
    best7 = dict(list(sorted_dict.items())[-7:])
    worse7 = dict(list(sorted_dict.items())[:7])
    counter = 0
    movie_all_list = []
    print('\nPolecam filmy:')
    for itm in best7:
        print('\t[' + str(counter) + '] ' + itm + ' ' + str(best7[itm]))
        counter += 1
        movie_all_list.append(itm)

    print('\nNie polecam filmów:')
    for itm in worse7:
        print('\t[' + str(counter) + '] ' + itm + ' ' + str(worse7[itm]))
        counter += 1
        movie_all_list.append(itm)
    return movie_all_list
if __name__ == '__main__':
    # user = input()
    # score_type = input()
    user = 'Pawel Czapiewski'
    # score_type = 'Euclidean'
    score_type = 'Pearson'
    ratings_file = 'ratings.json'
    movie_description_show = True
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
    score_array = []
    for item in data:
        if item != user:
            if score_type == 'Euclidean':
                score_array.append({'score': euclidean_score(data, user, item), 'user': item})
            else:
                score_array.append({'score': pearson_score(data, user, item), 'user': item})

    maxScore = max(score_array, key=lambda x: x['score'])
    userDict = data[maxScore['user']]
    movie_all_list = print_result(maxScore, userDict)

    while movie_description_show:
        movie = int(input('\nw celu pobrania opisu filmu podaj numer zaprezentowany obok tytułu: '))
        movie_description_get(movie_all_list[movie])
        print_result(maxScore, userDict)
