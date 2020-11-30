class Credibility( object ):
   
    def predict( self, cls, text ):
        vector = self.__convert2vector(text)
        predictTestCD = cls.predict(vector)
        predictTestCD = int(predictTestCD[0])
        if predictTestCD == 0: 
          resultsCD = 'Non-Credible'
          factorCD = 0.2
        elif predictTestCD == 1 :
          resultsCD = 'Credible'
          factorCD = 0.8
        return resultsCD, factorCD


    def load( self, model2load ):
        import pickle  # Import pickle Packag
        # Load the Model back from file
        with open(model2load, 'rb') as file:  
            Pickled_Model = pickle.load(file)

        # msg = "load a model." + model2load
        return Pickled_Model


    def save(self, *args, **kwargs):
        import pickle  # Import pickle Packag
        from sklearn.ensemble import RandomForestClassifier
        modelName = kwargs.get('modelName', 'MalicousAccount.pkl')
        model2save = kwargs.get('model2save', RandomForestClassifier(criterion='gini', min_samples_leaf=50, min_samples_split=10)) #self.dfCls['classifier'].values[0])
        with open(modelName, 'wb') as file:  
            pickle.dump(model2save, file)
        msg = "saved model " + modelName 
        return msg

    def __convert2vector(self, tweetToPredict): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        n_tokens=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  vector_tweet += (nlp.vocab[token.text].vector)
                  n_tokens+=1 
        if n_tokens != 0:
            vector_tweet = vector_tweet / n_tokens
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text    