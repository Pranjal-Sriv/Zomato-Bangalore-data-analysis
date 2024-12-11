#!/usr/bin/env python
# coding: utf-8

# # EDA ZOMATO 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_rows = 50


# In[6]:


# Downloading Dataset
import opendatasets as od
data='https://www.kaggle.com/datasets/manojgharge07/zomato-dataset'
od.download(data)


# In[3]:


import os
data_dir='.\zomato-dataset'
os.listdir(data)


# In[3]:


zomato=pd.read_csv('zomato.csv')


# In[4]:

zomato #viewing dataset

# In[5]:

zomato.info()


# # Our Questions:(KPIs)
#     
# 1.  Calculate Average Rating of each restaurant.
# 2.  Get distribution of Rating column & try to find out what distribution this feature support?
# 3.  Top restaurant chains in Bengalore.
# 4.  How many of the restaurant do not accept online orders?
# 5.  What is the ratio b/w restaurants that provide and do not provide table booking?
# 6.  How many types of restaurants we have?
# 7.  Highest voted restaurant.
# 8.  Total restaurant at different locations of Bengalore.
# 9.  Total number of variety of restaurants ie north indian,south Indian.
# 10. Analyse Approx cost for 2 people.
# 11. Cost vs Rating.
# 12. Is there any difference b/w votes of restaurants accepting and not accepting online orders?
# 13. Is there any difference b/w price of restaurants accepting and not accepting online orders?
# 14. Distribution of cost for 2 people.
# 15. Top 10 Most Expensive restaurant with approx cost for 2 people.
# 16. Top 10 Cheapest restaurant with approx cost for 2 people.
# 17. All the restautant that are below than 500(budget hotel).
# 18. Restaurants that have better rating >4 and that are under budget too.
# 19. Total numbers of Restaurants that have better rating >4 and that are under budget too ie less than 500.
# 20. Total such various affordable hotels at different location.
# 21. Which are the foodie areas?
# 22. Geographical Analysis.

# In[6]:


zomato.isnull()

# In[7]:

# Counting no. of null values for each feature
zomato.isnull().sum()


# In[1]:

sns.heatmap(zomato.isnull())


# In[9]:

zomato.head()


# In[10]:


zomato.describe()


# In Python, the df.describe() method is used to generate descriptive statistics of the columns of a DataFrame. It returns a new DataFrame with columns containing the mean, standard deviation, minimum, maximum, and quartiles of the data in the original DataFrame.
# 
# For example, if you have a DataFrame named df that contains numerical columns, you can use df.describe() to get a summary of the statistics of the numerical columns, such as the mean, standard deviation, minimum, maximum and quartiles of the data in the original 
# df.describe() only shows data for numerical features, not categorical features. in above dataset only 'votes' is numerical.

# ##  EDA STEPS
# 1.  Missing values
# 2.  Explore about Numerical Variables
# 3.  Explore about categorical Variables
# 4.  Finding Relationships between features 
# 5.  Statistical analysis of data
# 6.  Find which distribution does dataset follows
# 7.  Explore different plots for viewing relationships
# 8.  Decide which analysis is best required
# 9.  Handling Categorical features and explore different Ways
# 10. Feature Engineering and selection
# 11. Learn to apply different prediction models 
# 12. learn about different parameters to check models
# 13. Check if model is accurate by testing on random data
# 14. Improve accuracy by using better models
# 15. Understand which model best suits for which dataset
# 16. Remove duplicates
# 17. Explore Numpy() functions for statistical analysis
# 18. Explore Queries in EDA
# 19. List Comprehensions
# 20. Data transformation (converting categorical to numerical)
# 21. Data Encoding (To handle dataset if not readable ia a format)
# 22. Merging datasets on a particular column if required
# 23. Determine key performance indicators
# 24. Determine which plot is suitable for visualization

# In[11]:


[features for features in zomato.columns if zomato[features].isnull().sum()>0]


# In this statement, zomato is a DataFrame and the code is trying to find the columns (or "features") in the DataFrame that have more than one missing value (i.e. NaN value).
# 
# The list comprehension.
# [features for features in zomato.columns if zomato[features].isnull().sum()>1] iterates over the column names of the DataFrame and for each column, it checks if the number of missing values in that column is greater than 1 by using zomato[features].isnull().sum() which returns the number of missing values in that column, then the if condition checks whether the number of missing values is greater than 1. If it is, the column name is added to the list, otherwise it is skipped.
# 
# The end result is a list of column names that have more than one missing value.
# 
# It could be helpful in data cleaning, to identify the columns that have more missing values which may need more attention while cleaning the data.

# In[12]:


zomato.shape  # To know no. of rows and columns in dataframe


# In[13]:


zomato.dtypes  # To get data types of columns


# In[14]:


Restraunts_name=zomato.name.value_counts()

# value_counts() gives count of a particular record in a feature


# In[15]:


Restraunts_name


# In[16]:


zomato.name.value_counts().values


# In[17]:


Restraunts_name_index=zomato.name.value_counts().index


# In[18]:


Restraunts_name_index


# In[19]:


zomato['name'].unique() # gives unique categories in feature


# In[98]:


zomato['name'].nunique()  # gives count of unique entries in a feature


# So, there are 8792 different restraunts in the dataset

# In[104]:


# Creating pie chart to see which restraunts get more orders
plt.pie(Restraunts_name,labels='name')


# In Python, when working with data, you may encounter an error message that says
# 
# "'label' must be of length 'x'"
# 
# which typically occurs when working with pandas DataFrame and Series objects. This error message is thrown when the length of the label is not the same as the length of the data you are trying to label.
# 
# For example, when you are trying to add a new column to a DataFrame or trying to set the index of a DataFrame or Series to a certain column, but the length of the column you are trying to use as the label does not match the length of the data.
# 
# The error message "'label' must be of length 'x'" will tell you the expected length of the label and the actual length of the label that you provided. To solve this problem, you can check the length of the label and the data, and make sure they match.

# Since there are many restraunts, its not possible to visualize all in the pie chart
# So, lets see for top 10 restraunts getting more orders

# In[181]:


plt.pie(Restraunts_name[:10],rotatelabels=True,labels=Restraunts_name_index[:10],autopct='%.1f%%')
plt.axis('equal')


# The autopct parameter in the plt.pie() function in matplotlib is used to format the value (i.e., the percentage) that appears inside each pie slice. It takes in a string format, which specifies how the values should be displayed. For example, passing '%.1f%%' as the autopct parameter would display the values as percentages with one decimal place. If a function is passed, the function will be passed to the pie slices. It allows to format the string that will be used to label the wedges in the pie chart.
# 
# 
# The plt.axis('equal') function in matplotlib is used to set the aspect ratio of the x-axis and y-axis to be equal, meaning that one unit on the x-axis is the same length as one unit on the y-axis. This is particularly useful when creating pie charts or other types of plots where the shapes of the elements should be preserved.
# 
# This ensures that the circles in a pie chart are circular and not oval-shaped, and it also ensures that the x- and y-scales are the same, so that the plot is not distorted.

# OBSERVATION: Zomato maximum records orders from cafe coffee day followed by onesta

# In[4]:


zomato.head()


# In[20]:


zomato.groupby(['name','rate']).size()


# This line of code is using the groupby() method on a DataFrame called zomato to group the rows based on the values in the 'rate' and 'name' columns. The size() method is then called on the resulting groups, which returns the number of rows in each group.
# 
# The groupby() method takes one or more column names as arguments, and groups the rows of the DataFrame based on the unique values in those columns. In this case, the rows will be grouped based on the unique combinations of values in the 'rate' and 'name' columns.
# 
# The size() method returns the number of items in each group, effectively counting the number of rows that belong to each group. The resulting output will be a Series with the group keys as the index and the count of the group as the values.
# 
# It should be noted that this line of code assumes that the DataFrame 'zomato' exists and that the columns 'rate' and 'name' exist in it.

# In[ ]:


zomato.groupby(['name','rate']).size().reset_index()


# In[21]:


# To change column name from 0 to 'rate count'
zomato_ratings=zomato.groupby(['name','rate']).size().reset_index().rename(columns={0:'rate count'})


# In[22]:


zomato_ratings


# In[ ]:


# Checking how many customers have not rated 
zomato['rate'].value_counts()


# In a Jupyter notebook, you can use the
# 
# pd.options.display.max_rows 
# pd.options.display.max_columns 
# 
# options to control the maximum number of rows and columns that will be displayed when displaying a DataFrame. By default, Jupyter notebooks display a maximum of 60 rows and 20 columns.

# ## Transforming / Encoding 'rate' into numerical

# how to convert categorical feature into numerical
# 
# There are multiple ways to convert a categorical feature (a feature with non-numerical values) into numerical values in a DataFrame. One common approach is to use the LabelEncoder class from the sklearn.preprocessing module to assign a unique integer value to each category in the feature. Here's an example:
# 
# Copy code
# 
# from sklearn.preprocessing import LabelEncoder
# 
# #Create a sample DataFrame
# data = {'colors': ['red', 'green', 'blue', 'green', 'red']}
# df = pd.DataFrame(data)
# 
# #Create an instance of the LabelEncoder
# encoder = LabelEncoder()
# 
# #Fit the encoder to the DataFrame column
# encoder.fit(df['colors'])
# 
# #Transform the column and store the result in a new column
# df['colors_encoded'] = encoder.transform(df['colors'])
# 
# 
# In this example, the LabelEncoder is first fitted to the 'colors' column of the DataFrame df, which learns the unique categories in the column. The transform() method is then called on the column to convert the categories into numerical values, and the result is stored in a new column called 'colors_encoded'.
# 
# Another approach is to use the get_dummies() function from pandas library to create a binary representation of each category. Here's an example:
# 
# Copy code
# df_encoded = pd.get_dummies(df, columns=['colors'])
# 
# This will create a new DataFrame with binary columns for each category in the 'colors' column, where each cell will have value 0 or 1 indicating the presence or absence of that category in the original row.
# 
# A third approach is to use the map() function to map the categorical values to numerical values. Here's an example:
# 
# Copy code
# 
# df['colors_encoded'] = df['colors'].map({'red': 0, 'green': 1, 'blue': 2})
# 
# In this example, the map() function is used to create a dictionary that maps each category in the 'colors' column to a numerical value, and then applies this mapping to the column to create a new column 'colors_encoded' with the numerical values.
# 
# It's worth noting that the choice of the approach depends on the specific use case and the requirements of the model you are trying to build.

# In[ ]:


# Using sklearn.preprocessing to rectify rate column
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
#Transforming in a different column
zomato['rating']=encoder.fit_transform(np.ravel(zomato['rate']))

# np.ravel() converts a 2-D array to 1-D array.


# In[ ]:


zomato.keys()


# In[ ]:


zomato.head()


# In[ ]:


zomato['rating'].unique()


# sklearn.preprocessing did not workout 

# In[ ]:


zomato=zomato.drop(columns='rating')


# In[ ]:


zomato.info()


# In[ ]:


# Using Mapping to rectify 'rate' column
zomato['rating']=pd.factorize(zomato['rate'])[0]


# In[ ]:


zomato.info()


# In[ ]:


zomato.head()


# In[ ]:


zomato['rating'].unique()


# Factorize method didnt work out

# In[ ]:


zomato=zomato.drop(columns='rating')


# In[ ]:


zomato.info()


# In[ ]:


zomato['rate'].unique()


# In[ ]:


# Removing nan value for encoding
zomato['rate'].fillna('0',inplace=True)
zomato['rate'].unique()


# In[ ]:


rate_map={'4.1/5':4.1, '3.8/5':3.8, '3.7/5':3.7, '3.6/5':3.6, '4.6/5':4.6, '4.0/5':4.0, '4.2/5':4.2,
       '3.9/5':3.9, '3.1/5':3.9, '3.0/5':3.0, '3.2/5':3.2, '3.3/5':3.3, '2.8/5':2.8, '4.4/5':4.4,
       '4.3/5':4.3, 'NEW':0, '2.9/5':2.9, '3.5/5':3.5, '0':0, '2.6/5':2.6, '3.8 /5':3.8, '3.4/5':3.4,
       '4.5/5':4.5, '2.5/5':2.5, '2.7/5':2.7, '4.7/5':4.7, '2.4/5':2.4, '2.2/5':2.2, '2.3/5':2.3,
       '3.4 /5':3.4, '-':0, '3.6 /5':3.6, '4.8/5':4.8, '3.9 /5':3.9, '4.2 /5':4.2, '4.0 /5':4.0,
       '4.1 /5':4.1, '3.7 /5':3.7, '3.1 /5':3.1, '2.9 /5':2.9, '3.3 /5':3.3, '2.8 /5':2.8,
       '3.5 /5':3.5, '2.7 /5':2.7, '2.5 /5':2.5, '3.2 /5':3.2, '2.6 /5':2.6, '4.5 /5':4.5,
       '4.3 /5':4.3, '4.4 /5':4.4, '4.9/5':4.9, '2.1/5':2.1, '2.0/5':2.0, '1.8/5':1.8, '4.6 /5':4.6,
       '4.9 /5':4.9, '3.0 /5':3.0, '4.8 /5':4.8, '2.3 /5':2.3, '4.7 /5':4.7, '2.4 /5':2.4,
       '2.1 /5':2.1, '2.2 /5':2.2, '2.0 /5':2.0, '1.8 /5':1.8}

zomato['rating']=zomato['rate'].map(rate_map)


# In[ ]:


zomato.info()


# In[ ]:


zomato['rating'].unique()


# In[ ]:


sns.heatmap(zomato.isnull())


# In[ ]:


zomato=zomato.drop(columns='rating')


# Mapping didnt work out as well 

# # Data cleaning

# In[11]:


zomato.shape


# In[12]:


zomato.isnull().sum()


# # Dropping unnecessary columns
# url , address, phone, menu item , reviews list,dish_liked
# 

# In[13]:


zomato=zomato.drop(['url','address','phone','menu_item','reviews_list','dish_liked'],axis=1)


# In[14]:


zomato.info()


# # Check for Duplicates
# This code is removing duplicate rows from a DataFrame called "zomato" and the changes are being made inplace, meaning the DataFrame "zomato" is being modified directly without creating a new DataFrame. The drop_duplicates() method is used to remove duplicate rows from a DataFrame. The inplace parameter is used to make the changes to the DataFrame in place, without creating a new DataFrame. If inplace=False, the function will return a new DataFrame with duplicates removed, but the original DataFrame will not be modified.

# In[15]:


zomato.drop_duplicates(inplace=True)
zomato.shape


# # Removing null values 

# In[16]:


zomato.isnull().sum()


# In[23]:


zomato['rate'].unique()


# # converting rate column to numerical and fixing unknown values
# 
# A function is defined which splits value on the given dilimiter
# value=str(value).split('/) 
# You can use the python string method split() to split the string on the '/' character and take the first element of the resulting list, then convert it into float.
# 
# It will split the string '3.3/5' into a list ['3.3', '5'] and take the first element of that list, which is '3.3', and convert it to a float.

# In[24]:


# converting rate column to numerical and fixing unknown values
def rate_handle(value):
    if(value=='NEW'or value=='-'):
        return np.nan
    else:
        #'3.3/5' will split into 3.3 and /5 float value
        # value=['3.3',''/5']
        value=str(value).split('/')
        #3.3 is first element of the list so we'll access it by value[0]
        value=value[0]
        #retuen it by converting it from string to float
        return float(value)


# In[25]:


#Applying fn in 'rate' column
zomato['rate']=zomato['rate'].apply(rate_handle)


# "unexpected EOF while parsing" is a common error message that occurs when the Python interpreter reaches the end of the file while it is still in the process of parsing the code. This often happens when there is a problem with the syntax of the code, such as a missing parenthesis, bracket, or quotation mark. It can also happen when there is an indentation error, such as an unexpected indent or unindent.
# 
# Here are some common causes of this error and possible solutions:
# 
# Missing parenthesis, bracket, or quotation mark: Make sure all your parenthesis, brackets, and quotation marks are properly closed and match up.
# Indentation error: Make sure your code is properly indented. Python uses indentation to define code blocks, and an unexpected indent or unindent can cause this error.
# Incorrectly placed colon: Make sure that colons are only used in the correct context, such as at the end of a statement that starts a code block (e.g. if, for, def, etc.).
# Multiline statement: Make sure that if a statement spans over multiple lines, it is properly enclosed with parenthesis, brackets or quotes.
# It is also important to make sure that the code is being run with the correct version of Python, as some syntax may not be compatible with older versions of Python.
# 
# You can also use a code editor that has a built-in linter that checks for syntax errors before running the code, which will make it easier to detect syntax errors.

# In[26]:


zomato['rate']=zomato['rate'].apply(rate_handle)


# In[27]:


zomato.rate


# In[26]:


zomato.info()


# In[27]:


# handling null values in 'rate' column


# In[28]:


sns.boxplot(data=zomato,y=zomato.rate)


# In[29]:


# From box plot, we can observe that mean value of rate is around 37
# mean value of rate excluding nan values
mean_rate=zomato['rate'].mean(skipna=True)
mean_rate


# In[30]:


# filling mean rate value in place of nan values
zomato['rate'].fillna(mean_rate,inplace=True)


# In[31]:


zomato.rate.isnull().any()


# In[32]:


zomato.describe()


# In[ ]:


zomato.head(50)


# In[28]:


# Removing column 'listed_in(city)'as not required
zomato.drop(['listed_in(city)'],axis=1,inplace=True)


# In[29]:


zomato.info()


# In[35]:


zomato.head()


# In[36]:


zomato.isnull().sum()


# NULL values of other columns can't be predicted as they are strings so we drop them

# In[40]:


zomato.dropna(inplace=True)
zomato.head()


# In[42]:


zomato.isnull().any()


# ## Cost column analysis

# In[34]:


#Renaming few columns 
zomato.rename(columns={'approx_cost(for two people)':'cost2plates'},inplace=True)


# In[35]:


zomato.info()


# In[76]:


zomato.cost2plates.unique()


#  values greater than 999 have ',' in them.
#  this data has to be cleaned

# Function to clean cost values

# In[79]:


def handlecost(value):
    value=str(value)
    if ',' in value:
        value=value.replace(',','')
        return float(value)
    else:
        return float(value)


# In[81]:


zomato.cost2plates=zomato.cost2plates.apply(handlecost)
zomato.cost2plates.unique()


# In[85]:


zomato.head(10)


# ## Restraunt column analysis

# In[ ]:





# In[86]:


zomato.rest_type.value_counts()


# Clubbing restraunt types occuring less than 1000 time as 'others'

# In[90]:


rest_types=zomato.rest_type.value_counts()
rest_type_1000=rest_types[rest_types<1000]
rest_type_1000


# In[91]:


def handle_rest(value):
    if(value in rest_type_1000):
        return'others'
    else:
        return value


# In[110]:


zomato.rest_type=zomato.rest_type.apply(handle_rest)
zomato.rest_type.value_counts()
rest_types=zomato.rest_type.value_counts()
rest_types


# In[112]:


rest_type_index=zomato.rest_type.value_counts().index
rest_type_index


# In[182]:


#visualizing through pie chart 
plt.pie(rest_types,pctdistance=0.8,rotatelabels=True,explode=(0.2,0.2,0.2,0.2,0.2,0.2,0.2),labels=rest_type_index,autopct='%1.1f%%',shadow=True)
plt.axis('equal')


# # Location Column

# In[122]:


zomato.location.value_counts()


# clubbing locations occuring less than 500 times

# In[134]:


locations=zomato.location.value_counts()
locationindex=zomato.location.value_counts().index


# In[125]:


locations


# In[135]:


locationindex


# In[127]:


location500=locations[locations<500]
location500


# In[128]:


def handleloc(value):
    if value in location500:
        return 'others'
    else:
        return value


# In[130]:


zomato.location=zomato.location.apply(handleloc)
zomato.location.value_counts()


# In[142]:


zomato.location.nunique()


# visualizing using bar plot

# In[178]:


x=locationindex
y=locations
plt.bar(x,y,align='center',width=0.5)
plt.xticks(rotation=90)


# In[190]:


explode=(1.5,2.5,3.5,1.5,1.5,1.5,1.5,3.5,3.5,3.5,6.5,8.5,6.5,8.5,4.5,8.5,2.5,5.5,7.5,6.5,5.5,4.5,3.5,3.5,2.5)
plt.pie(locations,pctdistance=0.7,rotatelabels=True,labels=locationindex,autopct='%1.1f%%',radius=5,explode=explode,shadow=True)
plt.axis('equal')


# Since there is too much smudge in the plot , lets see top 10 locations

# In[153]:


plt.pie(locations[:10],explode=(0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,),labels=locationindex[:10],autopct='%1.1f%%',radius=5,shadow=True)
plt.axis('equal')


# ## Analysing cuisines column

# In[154]:


zomato.info()


# In[157]:


zomato.cuisines.unique()


# In[158]:


zomato.cuisines.nunique()


# In[161]:


cuisines=zomato.cuisines.value_counts()
cuisines


# In[162]:


cuisines100=cuisines[cuisines<100]


# In[163]:


def handlecuisines(value):
    if value in cuisines100:
        return 'others'
    else:
        return value


# In[164]:


zomato.cuisines=zomato.cuisines.apply(handlecuisines)


# In[166]:


zomato.cuisines.value_counts()


# ## Analysing types column

# In[168]:


zomato.Type.value_counts()


# In[169]:


zomato.head()


# ## Data Visualization

# Countplot is used to count the total values for a row

# In[173]:


plt.figure(figsize=(22,8))


# In[175]:


ax=sns.countplot(zomato['location'])
plt.xticks(rotation=90) #rotate label so the dont overlap


# In[191]:


#Showing how many restraunt have online order facility
plt.figure(figsize=(6,6))


# In[192]:


sns.countplot(zomato.online_order)


# In[193]:


#Showing how many restraunt have Book Table Facility
plt.figure(figsize=(6,6))
sns.countplot(zomato.book_table)


# ## Online order facility vs rating

# In[195]:


plt.figure(figsize=(6,6))
sns.boxplot(x='online_order',y='rate',data=zomato)


# # Table booking vs rate

# In[198]:


plt.figure(figsize=(6,6))
sns.boxplot(x='book_table',y='rate',data=zomato)


# # Online order vs location

# In[201]:


zomato.head()


# I want to see how many resturent provide online order based on location
# 
# step 1: First Thing I will do is to group the dataframe with location and online order===> df.grpupby['loaction','online_order'])
# 
# Step 2: After grouping we will select the parameter on which we will apply function such as count(),avg() ===> df.grpupby['loaction','online_order'])['name']
# 
# Step 3: At last we select the function we want to apply, In this case we need to find no. of resturents so we will use count() ===> df.grpupby['loaction','online_order'])['name'].count()

# In[37]:


zomato1=zomato.groupby(['location','online_order'])['name'].count()


# In[38]:


zomato1.head(5)


# In[39]:


# to find avg rating of data based on location
zomato1=zomato.groupby(['location','online_order'])['name'].count()


# In[40]:


zomato1.to_csv('location_online.csv')


# In[41]:


zomato1=pd.read_csv('location_online.csv')


# In[42]:


zomato1.head(5)


# In[43]:


#create a pivot table with location as index and online order as column
zomato1=pd.pivot_table(zomato1,values=None,index='location',columns='online_order',fill_value=0,aggfunc=np.sum)


# In[44]:


zomato1.head(5)


# In[215]:


# using barplot to see online order facility at different locations
zomato1.plot(kind='bar',figsize=(15,6))


# # Types of restraunt vs rate

# In[218]:


plt.figure(figsize=(14,8))
sns.boxplot(x='Type',y='rate',data=zomato)


# # Types of restraunts vs location

# In[45]:


zomato2=zomato.groupby(['Type','location'])['name'].count()
zomato2.to_csv('book_table.csv')
zomato2=pd.read_csv('book_table.csv')
zomato2=pd.pivot_table(zomato2,values=None,index='location',columns='Type',fill_value=0,aggfunc=np.sum)


# zomato2=zomato.groupby(['Type','location'])['name'].count()
# 
# The code is grouping the data from a pandas dataframe called "zomato" by the 'Type' and 'location' columns, and then counting the number of 'name' values in each group. The result is assigned to the variable 'zomato2'.

# zomato2=pd.pivot_table(zomato2,values=None,index='location',columns='Type',fill_value=0,aggfunc=np.sum)
# 
# This line of code creates a pivot table from the grouped data stored in 'zomato2'. The pivot table will have 'location' as the index, 'Type' as the columns, and the sum of the values in each group as the values. 'fill_value' is set to 0, which means that if a cell in the pivot table has no value, it will be filled with 0. The result is assigned to the variable 'zomato2'.
# 
# When 'values' is set to 'None', it means that all the values in the original dataframe will be used to create the pivot table. By setting 'values' to 'None', the pivot table will be created using all the columns in the original dataframe.

# In[46]:


zomato2.head(5)


# In[237]:


zomato2.plot(kind='bar',figsize=(10,6),width=1.5)


# # votes vs location

# In[49]:


zomato3=zomato[['votes','location']]


# This line of code creates a new dataframe 'zomato3' from the original dataframe 'zomato' by selecting only the 'votes' and 'location' columns. The new dataframe 'zomato3' will only contain the values from these two columns.

# In[50]:


zomato3.drop_duplicates()


# In[51]:


zomato4=zomato3.groupby(['location'])['votes'].sum()


# This line of code groups the data in 'zomato3' by the 'location' column, and then calculates the sum of the 'votes' values in each group. The result is stored in a new dataframe 'zomato4'. This line of code will create a new dataframe with one row for each unique location, and the sum of the votes for that location.

# In[52]:


zomato4


# In[53]:


zomato4=zomato4.to_frame()


# This line of code converts the series 'zomato4' (result of the previous grouping) into a dataframe with one column. The new dataframe 'zomato4' will have the same values as the original series, but will now have a column label.

# In[54]:


zomato4=zomato4.sort_values('votes',ascending=False)
zomato4.head()


# In[253]:


plt.figure(figsize=(10,5))
sns.barplot(zomato4.index,zomato4.votes)
plt.xticks(rotation=90)


# # Cuisines vs votes

# In[278]:


zomato5=zomato[['votes','cuisines']]
zomato5.drop_duplicates()
zomato6=zomato5.groupby(['cuisines'])['votes'].sum()
zomato6=zomato6.to_frame()
zomato6
zomato6=zomato6.sort_values('votes',ascending=False)
zomato6.head()


# In[281]:


zomato6=zomato6.iloc[1:,:]


# This line of code selects the second row to the last row of the dataframe 'zomato6' and stores it in a new dataframe 'zomato6'. The first row of the original dataframe is excluded. The '.iloc' method is used to select rows based on their position in the dataframe, and [1:,:] means to select all rows starting from the second row (index 1) to the end.

# In[282]:


zomato6.head()


# In[284]:


plt.figure(figsize=(10,5))
sns.barplot(zomato6.index,zomato6.votes)
plt.xticks(rotation=90)


# # Book table vs location
# 
# 

# In[55]:


zomato7=zomato.groupby(['book_table','location'])['name'].count()
zomato7.to_csv('book_table.csv')
zomato7=pd.read_csv('book_table.csv')
zomato7=pd.pivot_table(zomato7, values=None,index='location',columns='book_table',fill_value=0,aggfunc=np.sum)
zomato7.head(10)


# In[288]:


zomato7.plot(kind='bar', figsize=(12,6))


# # EXTRA DOCUMENTATION

# Distribution Plots
# These plots help us to visualize the distribution of data. We can use these plots to understand the mean, median, range, variance, deviation, etc of the data.
# 
# a. Dist-Plot
# Dist plot gives us the histogram of the selected continuous variable.
# It is an example of a univariate analysis.
# We can change the number of bins i.e. number of vertical bars in a histogram
# 
# 
# b. Joint Plot
# It is the combination of the distplot of two variables.
# It is an example of bivariate analysis.
# We additionally obtain a scatter plot between the variable to reflecting their linear relationship. We can customize the scatter plot into a hexagonal plot, where, more the color intensity, the more will be the number of observations.
# 
# import seaborn as sns
# For Plot 1
# sns.jointplot(x = df['age'], y = df['Fare'], kind = 'scatter')
# For Plot 2
# sns.jointplot(x = df['age'], y = df['Fare'], kind = 'hex')
# 
# kind = ‘hex’ provides the hexagonal plot and kind = ‘reg’ provides a regression line on the graph.
# 
# c. Pair Plot
# It takes all the numerical attributes of the data and plot pairwise scatter plot for two different variables and histograms from the same variables.
# 
# 
# d. Rug Plot
# It draws a dash mark instead of a uniform distribution as in distplot.
# It is an example of a univariate analysis.

# Categorical Plots
# These plots help us understand the categorical variables. We can use them for both univariate and bivariate analysis.
# 
# a. Bar Plot
# It is an example of bivariate analysis.
# On the x-axis, we have a categorical variable and on the y-axis, we have a continuous variable.
# 
# b. Count Plot
# It counts the number of occurrences of categorical variables.
# It is an example of a univariate analysis.
# 
# c. Box Plot
# It is a 5 point summary plot. It gives the information about the maximum, minimum, mean, first quartile, and third quartile of a continuous variable. Also, it equips us with knowledge of outliers.
# We can plot this for a single continuous variable or can analyze different categorical variables based on a continuous variable.
# 
# d. Violin Plots
# It is similar to the Box plot, but it gives supplementary information about the distribution too.
# 

# Advanced Plots
# As the name suggests, they are advanced because they ought to fuse the distribution and categorical encodings.
# 
# a. Strip Plot
# It’s a plot between a continuous variable and a categorical variable.
# It plots as a scatter plot but supplementarily uses categorical encodings of the categorical variable.
# 
# import seaborn as sns
# sns.stripplot(y = df['Age'], x = df['Pclass'])
# 
# b. Swarm Plot
# It is the combination of a strip plot and a violin plot.
# Along with the number of data points, it also provides their respective distribution.

# Matrix Plots
# These are the special types of plots that use two-dimensional matrix data for visualization. It is difficult to analyze and generate patterns from matrix data because of its large dimensions. So, this makes the process easier by providing color coding to matrix data.
# 
# 
# a. Heat Map
# In the given raw dataset ‘df’, we have seven numeric variables. So, let us generate a correlation matrix between these seven variables.
# 
# df.corr()
# 
# Another very obvious example is to use heatmaps to understand the missing value patterns. 
# 
# 
# b. Cluster Map
# If we have a matrix data and want to group some features according to their similarity, cluster maps can assist us.
# 
# Cluster maps use Hierarchical clustering to form different clusters.
# 
# sns.clustermap(tran.corr(), annot='True',cmap='viridis')
# 
# 

# Grids
# Grid plots provide us more control over visualizations and plots various assorted graphs with a single line of code.
# 
# a. Facet Grid
# Suppose we want to plot the age distribution of males and females in all the three classes of tickets. Hence, we would be having in a total of 6 graphs.
# 
# The Facet grids provide very clear graphs as per requirements.
# 
# sns.FacetGrid( col = ‘col’, row = ‘row’, data = data) provides an empty grid of all unique categories in the col and row. Later, we can use different plots and common variables for peculiar variations.

# Regression Plot
# This is a more advanced statistical plot that provides a scatter plot along with a linear fitting on the data.
# 
# sns.lmplot(x = 'Age', y = 'PassengerId', data = df, hue = 'Sex)
# 
# 

# In[ ]:

# In[ ]:




