class NetworkBasedPredictor():

  def __init__(self):
    self.model = None

  def __convert2vector(self, tweetToPredict): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  vector_tweet += (nlp.vocab[token.text].vector)
        if len(review) != 0:
            vector_tweet = vector_tweet / len(review)
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text
  
  def __convert_prediction(self, prediction):
    r = [0.16, 0.33, 0.49, 0.66, 0.83, 0.96]
    return r[prediction[0]-1]

  def load(self, path):
    import pickle  # Import pickle Package
    # Load the Model back from file
    with open(path, 'rb') as file:  
        self.model = pickle.load(file)

  def predict(self, text, source=0):
    df = self.__convert2vector(text)
    df['node_rank'] = 0
    return self.__convert_prediction(self.model.predict(df))