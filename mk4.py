import yfinance as yf
import praw as pw
import tweepy as tw
import pandas as pd
import numpy as np
import nltk 
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import fbprophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from datetime import date
import streamlit as st

st.write("""
# Staver
Your one-stop solution for prediction and forecasting of Bitcoin 
""")

st.text('')
st.text('')
st.text('')

def time_series():
     
     st.text('')
     st.text('')
     
     st.write("""
     Please be patient this could take a while as the data is being fetched and processed in real-time
     """)
     
     st.text('')
     st.text('')
     
     # Time Series Forecasting using FB-Prophet 

     st.write("""
     ## Time Series Forecasting using FB-Prophet
     """)
          
     st.text('')
     st.text('')
    
     # importing the time series dataset of bitcoin prices
     today = date.today()
     tickerSymbol = 'BTC-INR'
     tickerData = yf.Ticker(tickerSymbol)
     df = tickerData.history(period='1d', start='2000-01-01', end=today)
     df.reset_index(inplace=True)
     
     st.write("""
     Ticker chart of BTC-INR
     """)
     fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
     st.plotly_chart(fig)
     
     st.write("""
     The forecasted price of Bitcoin in INR
     """)
        
     # changing column headings
     Date = df['Date']
     Close = df['Close']
     df_prophet = pd.DataFrame()
     df_prophet['ds'] = Date
     df_prophet['y'] = Close
     
     
     # making the model
     model = Prophet()
     
     # fitting the data to the model
     model.fit(df_prophet)
     
     # predict future prices for the next one year
     future_dates = model.make_future_dataframe(periods=365);
     prediction = model.predict(future_dates)
     
     # plotting the predictions
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                         mode='lines',
                         name='Daily Close'))
     fig.add_trace(go.Scatter(x=prediction['ds'], y=prediction['yhat'],
                         mode='lines',
                         name='Prediction'))
     fig.add_trace(go.Scatter(x=prediction['ds'], y=prediction['yhat_upper'],
                         mode='lines',
                         name='Upper limit of predicted values'))
     fig.add_trace(go.Scatter(x=prediction['ds'], y=prediction['yhat_lower'],
                         mode='lines',
                         name='Lower limit of predicted values'))
     st.plotly_chart(fig)
     
     # input fields
     df_final = pd.DataFrame()
     df_final['Date'] = prediction['ds']
     df_final['Lower limit of Prediction'] = prediction['yhat_lower']
     df_final['Upper limit of Prediction'] = prediction['yhat_upper']
     df_final['Prediction'] = prediction['yhat']

     user_input = st.text_input("Enter Date")
     a = df_final.loc[df_final['Date'] == user_input]

     st.write(a['Date'], '\n', a['Lower limit of Prediction'], '\n', a['Upper limit of Prediction'], '\n', a['Prediction'])
     



     
     
def sentiment_analysis():
   
   st.text('')
   st.text('')
   
   st.write("""
   Please be patient this could take a while as the data is being fetched and processed in real-time
   """)
 
   st.text('')
   st.text('')
   
   st.write("""
   ## Twitter Sentiment Analysis
   """)

   # twitter credentials
   consumer_key= 'uLPC3KfMtGFcEeq4CxEOohZeg'
   consumer_secret= 'tywsJRvcr2zz5ICg7bkadbSIIjhGFmAlOLjJECjPqMfaRuwc1T'
   access_token= '1300465599823314944-VkC6tWnEUrbxTZ1wYpWIxbc8LQCPNL'
   access_token_secret= 'DDiF0cmidxoQlT2rgEUCGkP4E2DI8PBwz6WMS5QL51zOG'
   auth = tw.OAuthHandler(consumer_key, consumer_secret)
   auth.set_access_token(access_token, access_token_secret)
   api = tw.API(auth, wait_on_rate_limit=True)

   search_words = ['crypto', 'bitcoin']
   date_since = '2021-04-20'
   tweet_text = []
   date_time = []
   location = []
   
   # extracting tweet text, datetime and location
   for words in search_words:
     #st.write(words)
     tweets = tw.Cursor(api.search,
                 q=search_words,
                 lang="en",
                 since=date_since).items(100)
     for tweet in tweets:
       str1 = tweet.text
       str2 = tweet.created_at
       str3 = tweet.user.location
       tweet_text.append(str1)
       date_time.append(str2)
       location.append(str3)

   df_twitter = pd.DataFrame()
   df_twitter['Tweets'] = tweet_text
   df_twitter['Created at'] = date_time
   df_twitter['Location'] = location

   sia = SIA()
   results = []
   for line in tweet_text:
     pol_score = sia.polarity_scores(line)
     pol_score['Tweets'] = line
     results.append(pol_score)
   #st.write(results)
   df_twitter_nlp = pd.DataFrame(results)

   # compound is taken as the deciding factor is classifying the sentiment

   # positive
   df_twitter_nlp.loc[df_twitter_nlp['compound'] > 0, 'Sentiment'] = '1'

   # negative
   df_twitter_nlp.loc[df_twitter_nlp['compound'] < 0, 'Sentiment'] = '-1'

   # neutral
   df_twitter_nlp.loc[df_twitter_nlp['compound'] == 0.0, 'Sentiment'] = '0'

   df_twitter_nlp['Created at'] = date_time
   df_twitter_nlp['Location'] = location

   df_twitter_groupby = df_twitter_nlp.groupby('Sentiment').count()
   lst2 = df_twitter_groupby['Tweets']
   sentiment_lst = ['Negative', 'Neutral', 'Positive']
   dict2 = {'Sentiment': sentiment_lst, 'Number of Tweets': lst2}
   twitter_sent = pd.DataFrame(dict2)
   fig = px.bar(twitter_sent, x='Sentiment', y='Number of Tweets')
   st.plotly_chart(fig)
   if lst2[0] > lst2[2]:
      st.write('## Sentiment is Negative')
   elif lst2[0] < lst2[2]:
      st.write('## Sentiment is Positive')
  
   
   # Reddit Sentiment Analysis 

   st.write("""
   ## Reddit Sentiment Analysis
   """)

   # reddit credentials
   reddit = pw.Reddit(client_id = 'UYBiraXAwH8bcw',
                        client_secret = 'RMg2VFM9ncuAwLl61YB301SBfTZkUQ',
                        user_agent = 'MyAPI/0.0.1',
                        check_for_async=False
                       )

   # getting posts from the subreddits
   lst_reddit = []
   sentiment_lst = ['Negative', 'Neutral', 'Positive']
     
   
   # bitcoin subreddit
   subreddit = reddit.subreddit('bitcoin')
   # hot posts
   for post in subreddit.hot(limit=100):
     lst_reddit.append(post.title)
   # new posts
   for post in subreddit.new(limit=100):
     lst_reddit.append(post.title)
   

   # CryptoCurrency subreddit
   # subreddit = reddit.subreddit('CryptoCurrency')
   # hot posts
   # for post in subreddit.hot(limit=500):
   #  lst_reddit.append(post.title)
   # new posts
   # for post in subreddit.new(limit=500):
   #  lst_reddit.append(post.title)

   # btc subreddit
   subreddit = reddit.subreddit('btc')
   # hot posts
   for post in subreddit.hot(limit=100):
     lst_reddit.append(post.title)
   # new posts
   for post in subreddit.new(limit=100):
     lst_reddit.append(post.title)

   # Crypto_General subreddit
   # subreddit = reddit.subreddit('Crypto_General')
   # hot posts
   # for post in subreddit.hot(limit=500):
   #  lst_reddit.append(post.title)
   # new posts
   # for post in subreddit.new(limit=500):
   #  lst_reddit.append(post.title)

   # Coinbase subreddit
   # subreddit = reddit.subreddit('Coinbase')
   # hot posts
   # for post in subreddit.hot(limit=500):
   #  lst_reddit.append(post.title)
   # new posts
   # for post in subreddit.new(limit=500):
   #  lst_reddit.append(post.title)

   # Binance subreddit
   # subreddit = reddit.subreddit('Binance')
   # hot posts
   # for post in subreddit.hot(limit=500):
   #  lst_reddit.append(post.title)
   # new posts
   # for post in subreddit.new(limit=500):
   #  lst_reddit.append(post.title)

   
   # converting the list into a dataframe and displaying it 
   df_reddit = pd.DataFrame(lst_reddit, columns=['Post Titles'])

   # classifying the post as positive, negative or neutral and displaying the results
   sia = SIA()
   results = []
   for line in lst_reddit:
     pol_score = sia.polarity_scores(line)
     pol_score['Post Titles'] = line
     results.append(pol_score)

   df_reddit_nlp = pd.DataFrame(results)

   # compound is taken as the deciding factor is classifying the sentiment

   # positive
   df_reddit_nlp.loc[df_reddit_nlp['compound'] > 0, 'Sentiment'] = '1'

   # negative
   df_reddit_nlp.loc[df_reddit_nlp['compound'] < 0, 'Sentiment'] = '-1'

   # neutral
   df_reddit_nlp.loc[df_reddit_nlp['compound'] == 0.0, 'Sentiment'] = '0'

   # grouping post by sentiment
   df_reddit_groupby = df_reddit_nlp.groupby('Sentiment').count()
   lst1 = df_reddit_groupby['Post Titles']
   dict1 = {'Sentiment': sentiment_lst, 'Number of Posts': lst1}
   reddit_sent = pd.DataFrame(dict1)
   fig = px.bar(reddit_sent, x='Sentiment', y='Number of Posts')
   st.plotly_chart(fig)
   if lst1[0] > lst1[2]:
      st.write('## Sentiment is Negative')
   elif lst1[0] < lst1[2]:
      st.write('## Sentiment is Positive')
   
   st.text('')
   st.text('')
   
   # Twitter Sentiment Analysis 

   

     

   
# dropdown menu 
option = st.selectbox(
     'Please select your preferred method',
     ('None', 'Sentiment Analysis', 'Time series'))

if option == 'Time series':
   time_series()
elif option == 'Sentiment Analysis':
   sentiment_analysis()




