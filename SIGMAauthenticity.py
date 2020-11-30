true_venue_labels = ['news','interview','television','show', 'speech', 'reporters', 'debate', 'newsletter', 'press', 'CNN', 'ABC', 'CBS', 'video', 'conference', 'official', 'book']
false_venue_labels = ['website', 'tweet', 'mail', 'e-mail', 'mailer', 'web', 'site', 'meme', 'comic', 'advertisement', 'ad', 'blog', 'flier', 
                'letter', 'social', 'tweets', 'internet', 'message', 'campaign', 'post', 'facebook', 'handout', 'leaflet', 'letter' ]

true_statement_labels = ['original','true','mostly-true','half-true']
false_statement_labels = ['barely-true','false','pants-fire']


class VerifiableAuthenticity:

  #Intialising and loading the pickled model
  def __init__(self):
    #logClassifier = linear_model.LogisticRegression(solver='liblinear', C=1, random_state=111)

    # Load Data
    global dataTrain
    global dataTest
    urlTrain = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRddGgs_vs5jOS0nAqGrWiY_7rF4DqbxktBHe4RCJoccK8p9c1k1jLXsXJ5DppAoLhb7dic_auG7jmn/pub?gid=550184183&single=true&output=csv"
    urlTest = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSTLq913ZT2GdLqRYIP34Wzr6hNNqpswM9yGoTk77w_MSuDTg5rCVmZnlkhmJRZ2Fz3UJtx7XbnIvO1/pub?gid=576608418&single=true&output=csv"
    rTrain= requests.get(urlTrain)
    rTest= requests.get(urlTest)
    dataTrain = pd.read_csv(BytesIO(rTrain.content))
    dataTest = pd.read_csv(BytesIO(rTest.content))
        
    #dataTrain = pd.read_csv('/content/real_politifact_combined.csv', sep=',')
    #dataTest = pd.read_csv('/content/polifact-test.csv', sep=',')
    #print(dataTrain.columns)

    # Clean Data 
    self.cleanDataset(dataTrain)
    self.cleanDataset(dataTest)

    # Calculating speakerscore and adding speakerscore to the the dataset
    val1 = (dataTrain['true_score']+dataTrain['mostly_true']+dataTrain['half_true']).astype(int)
    val2 = (dataTrain['mostly_false']+dataTrain['false_score']+dataTrain['pants_on_fire']).astype(int)
    dataTrain['speaker_score'] = np.where(val1 > val2, 1, 0)

    # Calculating statement_score and adding statement_score to the dataset
    labelcolname = 'statement_score'
    dataTrain[labelcolname] = dataTrain.apply(lambda row: self.simplify_statement_label(row['Label']), axis=1)

    # Calculating venue_score and adding venue score to the dataset
    labelcolname = 'venue_score'
    dataTrain[labelcolname] = dataTrain.apply(lambda row: self.simplify_venue_label(row['venue']), axis=1)

    # Load Model
    filename = "/content/cmpe257_AlternusVera_SIGMA/VerifiableAuthenticity_PickledModel.pkl"
    with open(filename, 'rb') as file:
      global Pickled_Model  
      Pickled_Model = pickle.load(file)
     # print(Pickled_Model)

    # Train test Split
    X = dataTrain[['speaker_score']]
    Y = dataTrain['statement_score'] 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    # # Pickled Model Fit and Predict
    # Pickled_Model.fit(x_train, y_train)
    # y_pred = Pickled_Model.predict(x_test)

    dataTrain['Statement'] = dataTrain['Statement'].map(str)
    dataTest['Statement'] = dataTest['Statement'].map(str)
    
    # Convert text to word count vectors with CountVectorizer
    # create the transform
    cvec = CountVectorizer()

    # tokenize, build vocab and encode training data
    traindata_cvec = cvec.fit_transform(dataTrain['Statement'].values)
    # Calculate inverse document frequencies
    # create the transform
    tfidf_vec = TfidfTransformer()
    # tokenize, build vocab and encode training data
    traindata_tfidf_vec = tfidf_vec.fit_transform(traindata_cvec)
    # tfidf score
    tfidf_vec.transform(traindata_cvec) 
    # tfidf + ngrams
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), use_idf=True, smooth_idf=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(dataTrain['Statement'].values)
    # tfidf.vocabulary_
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
    #print(df.sort_values(by=["tfidf"],ascending=False))
    # nltk.download('treebank')
    # nltk.download('universal_tagset') 
    # POS tagging using CRF
    tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')

    # load training sentences
    training_sentences = dataTrain['Statement'] 

    global logReg_pipeline_cv    
    logReg_pipeline_cv = Pipeline([
      ('LogRCV', cvec),
      ('LogR_model', LogisticRegression())
    ])
    logReg_pipeline_cv.fit(dataTrain['Statement'], dataTrain['Label'])
    predictions_logReg = logReg_pipeline_cv.predict(dataTest['Statement'])
    statement_accuracy = metrics.accuracy_score(dataTest['Label'], predictions_logReg)
    #statement_precision = metrics.precision_score(dataTest['Label'], predictions_logReg)
    #statement_recall = metrics.recall_score(dataTest['Label'], predictions_logReg)

    # #Calculate accuracy using Pickled_Model and Print
    # model_accuracy = accuracy_score(y_test, y_pred)
    # model_precision = precision_score(y_test, y_pred)
    # model_recall = recall_score(y_test, y_pred) 

    # print("Accuracy of the Pickled model is {:2.2f}% " .format(model_accuracy  * 100))
    # print("Precision of the Pickled model is {:2.2f}% " .format(model_precision * 100))
    # print("Recall of the Pickled model is {:2.2f}% " .format(model_recall * 100))

  #Function to clean the dataset
  def cleanDataset(self, dataTrain):
    dataTrain['true_score']=dataTrain['true_score'].str.split(" ", 1, expand=True)
    dataTrain['mostly_true']=dataTrain['mostly_true'].str.split(" ", 1, expand=True)
    dataTrain['half_true']=dataTrain['half_true'].str.split(" ", 1, expand=True)
    dataTrain['mostly_false']=dataTrain['mostly_false'].str.split(" ", 1, expand=True)
    dataTrain['false_score']=dataTrain['false_score'].str.split(" ", 1, expand=True)
    dataTrain['pants_on_fire']=dataTrain['pants_on_fire'].str.split(" ", 1, expand=True)
    dataTrain['venue']= dataTrain.venue.str.split('in').str[1]
    dataTrain["venue"]= dataTrain.venue.str.replace(":", " ") 
    #print(dataTrain.iloc[9])

  # Function to calculate the venue score
  def simplify_venue_label(self, venue_label):
    if venue_label is np.nan:
      return 0;
    words = venue_label.split(" ")
    for s in words:
      if s in true_venue_labels:
        return 1
      elif s in false_venue_labels:
        return 0
    else:
        return 1

  # Function to calculate the statement score
  def simplify_statement_label(self,input_label):
      if input_label is np.nan:
        return 0;
      if input_label in true_statement_labels:
          return 1
      else:
          return 0

  #Function to calculate the speaker score
  def simplify_speaker_score(self, speakerScore):
    trueScore = speakerScore[0] + speakerScore[1] + speakerScore[2]
    falseScore = speakerScore[3] + speakerScore[4] + speakerScore[5]
    if trueScore > falseScore:
      return 1
    else:
      return 0

  def getAuthenticityScoreBySpeaker(self, speakerScore):
    x = self.simplify_speaker_score(speakerScore)
    xTrain = np.array(x).reshape(-1, 1)
    xPredicted = Pickled_Model.predict(xTrain)
    xPredicedProb = Pickled_Model.predict_proba(xTrain)[:,1]
    return float(xPredicedProb)

  def getAuthenticityScoreByVenue(self, src):
    x = self.simplify_venue_label(src)
    xTrain = np.array(x).reshape(-1, 1)
    xPredicted = Pickled_Model.predict(xTrain)
    xPredicedProb = Pickled_Model.predict_proba(xTrain)[:,1]
    return float(xPredicedProb)

  def getAuthenticityScoreByStatement(self, text):
    predicted = logReg_pipeline_cv.predict([text])
    predicedProb = logReg_pipeline_cv.predict_proba([text])[:,1]
    return bool(predicted), float(predicedProb)

  def predict(self, statement='',venue=''):
    concatStatement = ''
    for str1 in statement:
      concatStatement += str1+ ' '
    venueAuth = self.getAuthenticityScoreByVenue(venue)
    binaryValue, probValue  = self.getAuthenticityScoreByStatement(concatStatement)
    #print(" values ", venueAuth, probValue)
    score = 0.7 * venueAuth + 0.3 * probValue
    #print("score =  ", score)
    return float(score)
     
