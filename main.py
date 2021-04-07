# Import pandas and numpy
import pandas as pd
import numpy as np
import operator
import gim
from genetic import genetic_optimize, cost_function

import warnings

warnings.simplefilter('ignore')

# Constants
NO_OF_FEATURES = 21
NO_OF_GENRES = 19
WEIGHTS = np.random.rand(NO_OF_FEATURES)
NO_OF_ITERATIONS=10
NO_OF_NEIGHBOURS=20


# Loading the data from the dataset
# Load users
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# Load ratings
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

# Load genres
i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

# Merge users and ratings on user_id
movies_users_ratings = pd.merge(users, ratings, on='user_id')

# Merge movies_users_ratings and items on movie_id
items_merged = pd.merge(movies_users_ratings, items, on='movie_id')







# Fuzzy Sets
class Age:
    """Define methods to get fuzzy values of given age in three sets i.e. young, middle, and old."""

    def __int__(self):
        pass

    def young(self, age):
        """Get value for young fuzzy set for given age."""
        if age < 20.0:
            return 1.0
        elif 20.0 <= age < 35.0:
            return float((35 - age) / 15.0)
        else:
            return 0.0

    def middle(self, age):
        """Get value for middle fuzzy set for given age."""
        if age <= 20 or age > 60:
            return 0.0
        elif 20 < age <= 35:
            return float(age - 20) / 15
        elif 35 < age <= 45:
            return 1.0
        elif 45 < age <= 60:
            return (60 - age) / 15.0

    def old(self, age):
        """get value for old fuzzy set for given age."""
        if age <= 45:
            return 0.0
        elif 45 < age <= 60:
            return (age - 45.0) / 15
        else:
            return 1.0

    def get_fuzzy_set(self, age):
        """Get fuzzy set values of given age."""
        return [self.young(age),
                self.middle(age),
                self.old(age)]


class GIM:
    """GIM- Genre Interestingness Measure"""

    def __init__(self):
        pass

    def gim_a(self, gim, i):
        """Method to get fuzzy set value for very_bad, bad, average, good."""
        if gim <= i - 2 or gim > i:
            return 0.0
        elif i - 2 < gim <= i - 1:
            return gim - i + 2.0
        elif i - 1 < gim <= i:
            return float(i - gim)

    def very_bad(self, gim):
        if gim <= 1.0:
            return 1.0
        else:
            return 0.0

    def bad(self, gim):
        return self.gim_a(gim, 2.0)

    def average(self, gim):
        return self.gim_a(gim, 3.0)

    def good(self, gim):
        return self.gim_a(gim, 4.0)

    def very_good(self, gim):
        return self.gim_a(gim, 5.0)

    def excellent(self, gim):
        if gim <= 4.0:
            return 0.0
        else:
            return (gim - 4.0)

    def get_fuzzy_set(self, gim_value):
        """Get fuzzy set of gim(list of values) based on given gim value."""
        return [self.very_bad(gim_value),
                self.bad(gim_value),
                self.average(gim_value),
                self.good(gim_value),
                self.very_good(gim_value),
                self.excellent(gim_value)]






# Create objects for Age and GIM to use for fuzzy sets
age = Age()
gim_obj = GIM()

m_cols = ['unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'age',
          'user_id']

# Initialize empty dataFrames for active and passive users
model_data_active_users = pd.DataFrame(columns=m_cols)
model_data_passive_users = pd.DataFrame(columns=m_cols)


def euclidean_dist(list_a, list_b):
    """Return the Euclidean distance between two array elements."""
    return np.linalg.norm(np.array(list_a) - np.array(list_b))


def fuzzy_dist(first_point, second_point, fuzzy_set_first_point, fuzzy_set_second_point):
    """Returns fuzzy distance between two values and their fuzzy sets."""
    return abs(first_point - second_point) * euclidean_dist(fuzzy_set_first_point, fuzzy_set_second_point)


def fuzzy_distance(ui, uj):
    """Returns fuzzy distance between given points."""

    fuzzy_dis = [0] * NO_OF_FEATURES

    # Get fuzzy set values for movie genres
    for i in range(0, NO_OF_GENRES):
        ui_gim = gim_obj.get_fuzzy_set(ui[i])
        uj_gim = gim_obj.get_fuzzy_set(uj[i])
        fuzzy_dis[i] = fuzzy_dist(ui[i], uj[i], ui_gim, uj_gim)

    # Get fuzzy set values for age
    ui_gim = age.get_fuzzy_set(ui[i])
    uj_gim = age.get_fuzzy_set(uj[i])
    fuzzy_dis[i] = fuzzy_dist(ui[i], uj[i], ui_gim, uj_gim)

    # adding user_id of second user
    fuzzy_dis[NO_OF_FEATURES-1] = uj['user_id']
    return fuzzy_dis


def get_neighbours(model_active_users, model_passive_users):

    # Save active users and its neighbours in a data-frame with active users' id as column name
    user_neighbours= pd.DataFrame(columns=model_active_users['user_id'])

    # Iterate over active users model and save neighbours of each active users in user_neighbours
    for _, value in model_active_users.iterrows():
        j = 0
        fuzzy_vec = []
        for _, value_p in model_passive_users.iterrows():
            fuzzy_vec.append(fuzzy_distance(value, value_p))

            fuzzy_gen_dist = np.sum(np.multiply(WEIGHTS[:-1], np.array(fuzzy_vec[j][:-1]))) ** 0.5

            fuzzy_vec[j] = [fuzzy_gen_dist, fuzzy_vec[j][-1]]

            j = j + 1

        user_neighbours[value[-1]] = [n[1] for n in sorted(fuzzy_vec, key=operator.itemgetter(0), reverse=True)][:NO_OF_NEIGHBOURS]
    return user_neighbours


def model_for_users(users_data):
    """Create model for given users data i.e. merged movies, items, and users

    Args:
        users_data: DataFrame of merged movies, items, and users based on movie_id
    """

    i = 0
    model_data_for_users = pd.DataFrame(columns=m_cols)

    for _, curr_value in users_data.iterrows():

        # Get user movies based on user
        user_movies = items_merged.loc[items_merged['user_id'] == curr_value['user_id']]

        # Get feature list for all movies of one user
        feature_array = gim.gim_final(user_movies, curr_value['user_id'])
        feature_array[NO_OF_GENRES] = curr_value['age']
        feature_array[NO_OF_GENRES + 1] = curr_value['user_id']

        # Save current feature values in model data
        model_data_for_users.loc[i] = feature_array
        i = i + 1
    return model_data_for_users


def recommend(nearest_neighbours, test_users_data):
    """Recommend rating for given movies i.e. test_examples based on nearest neighbours.

    Also return actual and predicated ratings for testing users
    """
    predicated_rat=[]
    actual_rat = []
    for key, item in test_users_data.iterrows():
        m_id = item['movie_id']
        n_ratings = []
        for i in nearest_neighbours:

            # Get items or movie details reviewed by neighbour i with given m_id
            temp = items_merged.loc[items_merged['user_id'] == i].loc[items_merged['movie_id'] == m_id]
            for k, it in temp.iterrows():
                n_ratings.append(it['rating'])
        predicated_rat.append(float(sum(n_ratings)) / len(n_ratings) if len(n_ratings) else 0)
        actual_rat.append(item['rating'])
    return actual_rat, predicated_rat

# Users who has rated movies at least 60 movies
top_users = items_merged.groupby('user_id').size().sort_values(ascending=False)[:497]

model_error = []

# Train model for given iterations
for i in range(0, NO_OF_ITERATIONS):

    # Get random 10% of the top_users as active users and remaining are passive users
    active_users = top_users.sample(frac=0.10)

    # Random 34% of active users will be used for training and 66% users for testing purpose.
    training_active_users = active_users.sample(frac=0.34)
    #testing_active_users = active_users.drop(training_active_users.index)

    # passive_users will be used as training examples
    passive_users = top_users.drop(active_users.index)

    # Get active and passive users' data from merged movies, items, and users
    training_active_users_data = items_merged.loc[items_merged['user_id'].isin(training_active_users)][:10]
    test_active_users_data = items_merged.loc[items_merged['user_id'].isin(training_active_users)][10:]
    passive_users_data = items_merged.loc[items_merged['user_id'].isin(passive_users)][:10]

    # Get model for active users
    model_data_active_users = model_for_users(training_active_users_data)

    # Get model for passive users
    model_data_passive_users = model_for_users(passive_users_data)

    # Get neighbour users of active users
    active_users_neighbours = get_neighbours(model_data_active_users, model_data_passive_users)

    # Recommend users based on neighbours
    actual_ratings, predicated_ratings=recommend(active_users_neighbours, test_active_users_data)

    # Get error for predictions of test users
    cur_error = cost_function(actual_ratings, predicated_ratings)

    # Optimize weights using genetic algorithm approach and update weighs
    WEIGHTS = genetic_optimize(actual_ratings, predicated_ratings).flatten()

    # Add current iteration error to model error for MAE of the model
    model_error.append(cost_function(actual_ratings, predicated_ratings))

    # Log details for current iteration
    print("Iteration: ", i)
    print("Error: ", cur_error)
    print('Weights: ', WEIGHTS)
print("Mean Absolute Error for all iterations: ", sum(model_error)/len(model_error))
