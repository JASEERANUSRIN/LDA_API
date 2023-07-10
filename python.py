
from datetime import datetime
from http.client import OK
import pandas as pd
import sklearn


from sklearn.decomposition import LatentDirichletAllocation

class get_token():


     def token_results(self,max_df,min_df,n_components):
        npr = pd.read_csv('npr.csv')  # to read the csv file

        npr['Article'][0] # returns the text of that article - first row

        import sklearn.feature_extraction.text as text



        vectorizer = text.TfidfVectorizer(max_df=max_df,min_df=min_df, stop_words='english', ngram_range=(1, 2), max_features=4000)

        a = vectorizer.fit_transform(npr['Article'])

        #print(a)

        tf_idf = pd.DataFrame(a.todense()).iloc[:5]  
        tf_idf.columns = vectorizer.get_feature_names_out()
        tfidf_matrix = tf_idf.T
        tfidf_matrix.columns = ['response'+ str(i) for i in range(1, 6)]
        tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)

        # Top 10 words 
        tfidf_matrix = tfidf_matrix.sort_values(by ='count', ascending=False)[:15] 

        # Print the first 10 words 
        print(tfidf_matrix.drop(columns=['count']).head(15))

        print(vectorizer.vocabulary_)
        print(vectorizer.idf_)



        LDA = LatentDirichletAllocation(n_components,random_state = 42)
        # many parameters

        LDA.fit(a)

        # Grab the vocabulary -

        len(vectorizer.get_feature_names_out()) # holds instance of every single word

        type(vectorizer.get_feature_names_out()) # list of all the words 

        vectorizer.get_feature_names_out()[3333] #  prints words in that index

        # print random words from list from doc

        import random

        random_word_id = random.randint(0,4000)
        vectorizer.get_feature_names_out()[random_word_id]

        len(LDA.components_) # no of comp 

        type(LDA.components_)  #  numpy array with prob for each word

        (LDA.components_)

        single_topic = LDA.components_[0] # first topic

        single_topic.argsort()

        # argsort ---- index positions sorted from least - greates
        # here, top 10 values (10 greatest values)
        # last 10 values of argsort

        single_topic.argsort()[-10:] 

        top_ten_words = single_topic.argsort()[-10:]

        # high prob words in single topic
        for index in top_ten_words:
            print(vectorizer.get_feature_names_out()[index])

        for i,topic in enumerate(LDA.components_):
            print(f"The top 15 words for topic #{i}")
            print([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-15:]])
            print('\n')
            print('\n')

        a


        # new column with topic no
        topic_results = LDA.transform(a)

        topic_results.shape

        topic_results[0].round(2) # prob of topics in doc


        topic_results[0,1].argmax() # returns index position with high prob

        npr['Topic'] = topic_results.argmax(axis=1)

        npr

        mynewarticle=pd.DataFrame(npr)

        current_datetime=datetime.now().strftime("%m-%d-%Y %H-%M")
        str_current_time= str(current_datetime)


        print("Creating csv report")
        mynewarticle.to_csv("Doc"+ str_current_time + ".csv",sep="\t")
        print("article.csv created") 

        print(mynewarticle)
        return OK
        
            



        