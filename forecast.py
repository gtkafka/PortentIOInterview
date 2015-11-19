__author__ = 'kafka'

from pandas import *
import numpy as np
import matplotlib.pyplot as mplt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import datetime
from decimal import Decimal


def get_data():
    temp = '/Users/kafka/PycharmProjects/PortentIO/%s.csv'
    data =  read_csv(temp % 'movie_wiki_twitter_by_days_to_release')
    return(data)

def group_data(data):

    df = data.drop(['title', 'imdb_movie_id', 'number_of_views',
                 'days_to_release'], axis=1)
    df1 = df[np.isfinite(data['theatres'])]
    tiny_data = df1.groupby('imdb_id').first()
    full_data= data.groupby('imdb_id')

    return(tiny_data, full_data)

def get_views(grouped, show_sample_plots=True):

    count = 0
    imdbviews_list={}
    for name, group in grouped:

        #get information for plots
        num       = np.array(group['number_of_views'])
        dtr       = np.array(group['days_to_release'])
        imdbviews = Series(num, index=dtr)

        #get info from first instances of group
        doy   = int(group['day_of_year'].iloc[0])
        yr    = int(group['year'].iloc[0])
        title = group['title'].iloc[0]
        bor   = '%.2E' % Decimal(str(group['opening_we_bor'].iloc[0]))
        date  = str(datetime.datetime.strptime(str(yr)+' '+str(doy),
                                           '%Y %j'))
        count +=1

        if(show_sample_plots==True and count%414==0):
            get_plot(imdbviews, dtr, title, bor, date)

        if(len(num)==100 and bor!="NAN" ):
            imdbviews_list[bor] = imdbviews
    return(imdbviews_list)

def get_plot(data, index, title, bor, date):
    #we can plot momentum curves to predidct what a movie will do towards
    #opening day. this kind of analysis might be better fitted for
    #the cum-sum of twitter sentiment

    ax = data.plot( title='Momentum Plot \n'+title+\
                                      '  BOR($):'+bor+'  OpDay:'+date )
    ax.set_xlabel("days to release")
    ax.set_ylabel("number of views")

    rolling_means ={}
    for i in np.linspace(1,80, 10):
        X=Series(rolling_mean(data, i), index=index)
        rolling_means[i] = X
        X.plot(alpha = 0.7)

    mplt.show()

#Get the data and drop unwanted columns:

data = get_data()
print(list(data.keys()))
gdf,fdf = group_data(data)
views = get_views(fdf)


X1 = np.array(list(views.values()))
y1 = np.array(list(views.keys()))
#Regularize the data:
X2 = (X1-np.mean(X1))/np.std(X1)

pca_ = PCA().fit(X2, y1)
mplt.plot(np.cumsum(pca_.explained_variance_ratio_))

#The goal function (opening weekend BOR) is selected out:
y = gdf['opening_we_bor']
X = gdf[['weekend_num', 'imdb_rating', 'budget', 'theatres']]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                random_state = 12, test_size=0.1)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(Xtrain, ytrain)

#we can generate the importance of each column on determining the BOR:
print(sorted(zip(map(lambda x: round(x, 4),
                     rf.feature_importances_), X), reverse=True))

#Let's see how well we did at predicting the BOR on the test set:
y_predict = rf.predict(Xtest)
r2 = r2_score(ytest, y_predict)
print(r2)

actors = factorize(gdf['lead_actor'])
genre = factorize(gdf['genre'])[0]
gen = DataFrame(genre)
gen.columns=['genre']