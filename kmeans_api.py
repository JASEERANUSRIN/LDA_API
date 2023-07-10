# for importing dataset
import numpy as np
import pandas as pd
    
# for performing text clustering    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# for providing the path
import os
# print(os.listdir('../input/'))

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# reading the data in csv format
from datetime import datetime
from http.client import OK
import pandas as pd
import sklearn


from sklearn.decomposition import LatentDirichletAllocation

class get_token():

  def token_results(self,max_df,min_df,k_clusters,random_state):
        data = pd.read_csv('data_1.csv')

    #    getting the shape
        data.shape
        data.head()
        data.info()

        col = ['short_description']
        dataset= data[col]

        # getting the length of the text as another feature

        data['Length'] = data['short_description'].apply(len)

        # looking at the head of the data

        data.head()
        import re
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        import re
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        from nltk.corpus import stopwords
        import nltk
        from nltk.stem import WordNetLemmatizer
        lemmatizer=WordNetLemmatizer()

        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_ ]')
        REMOVE_NUM = re.compile('[\d+]')
        STOPWORDS = set(stopwords.words('english'))

        def clean_text(text):
         """
        text: a string
        return: modified initial string
        """
            # lowercase text
         text = text.lower() 

            # replace REPLACE_BY_SPACE_RE symbols by space in text
         text = REPLACE_BY_SPACE_RE.sub(' ', text) 
                    
                    # Remove the XXXX values
         text = text.replace('x', '') 
                    
                    # Remove white space
         text = REMOVE_NUM.sub('', text)

                    #  delete symbols which are in BAD_SYMBOLS_RE from text
         text = BAD_SYMBOLS_RE.sub('', text) 

                    # delete stopwords from text
         text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
                    
                    # removes any words composed of less than 2 or more than 21 letters
         text = ' '.join(word for word in text.split() if (len(word) >=2 and len(word) <= 21))

                    # Stemming the words
                #     text = ' '.join([stemmer.stem(word) for word in text.split()])
                    
         text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
                    
         return text

        # dataset["short_description"] = dataset["short_description"].apply(clean_text)
        # dataset["short_description"]
            # print(dataset)
            # vectorizing the data using Tfidf Vectorizer

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words='english', max_features = 2000)
        X = vectorizer.fit_transform(dataset['short_description'])

            # getting the shape of X
        print("Shape of X :", X.shape)

        k_clusters = 10
        from sklearn.cluster import KMeans
        print('Starting....')
        score = []
        for i in range(1,k_clusters + 1):
                kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=600,n_init=5,random_state=0)
                kmeans.fit(X)
                score.append(kmeans.inertia_)
        print('Fit Complete....')

        # plt.plot(range(1,k_clusters + 1 ),score)
        # plt.title('The Elbow Method')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('Score')
        #         # plt.savefig('elbow3.png')
        # plt.show()
        true_k = 6
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=600, n_init=1)
        model.fit(X)

        print("Top terms per cluster:")

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        for i in range(true_k):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :2]:
                print(' %s' % terms[ind]),
            print

        model = KMeans(n_clusters=true_k, init='k-means++', n_init =10, max_iter=300, tol=0.000001, random_state=0)
        model.fit(X)

        clusters = model.predict(X)
        # Create a new column to display the predicted result
        dataset["ClusterName"] = clusters
        x=dataset.head(10)
        print(x)

        mynewarticle=pd.DataFrame(x)

        current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
        str_current_time= str(current_datetime)


        print("Creating csv report")
        mynewarticle.to_csv("kmeans"+ str_current_time + ".csv",sep="\t")
        print("article.csv created") 

        print(mynewarticle)

        return OK