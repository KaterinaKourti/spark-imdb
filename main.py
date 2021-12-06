from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import *

from pyspark.sql.functions import col,avg,round
from os import path,getcwd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# spark = SparkSession.builder.appName('Spark-IMDb-Project').config('spark.sql.analyzer.failAmbiguousSelfJoin',False).getOrCreate()
spark = SparkSession.builder.appName('Spark-IMDb-Project').getOrCreate()

movies_genre_file = "data/MOVIES_GENRE.txt"
user_movies_file = "data/USER_MOVIES.txt"
users_file = "data/USERS.txt"

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Creating the schemas of the tables based on the datasets given
movies_genre_schema = StructType(
    [StructField("mid", IntegerType(), True), StructField("genre", StringType(), True)]
)

user_movies_schema = StructType(
    [
        StructField("userid", IntegerType(), True),
        StructField("mid", IntegerType(), True),
        StructField("rating", IntegerType(), True),
    ]
)

users_schema = StructType(
    [
        StructField("userid", IntegerType(), True),
        StructField("uname", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("age", IntegerType(), True),
    ]
)

# Creating the dataframes
movies_genre_df = (
    spark.read.schema(movies_genre_schema)
    .option("header", "true")
    .option("delimiter", "|")
    .csv(movies_genre_file)
)
# movies_genre_df.show()

user_movies_df = (
    spark.read.schema(user_movies_schema)
    .option("header", "true")
    .option("delimiter", "|")
    .csv(user_movies_file)
)
# user_movies_df.show()

users_df = (
    spark.read.schema(users_schema)
    .option("header", "true")
    .option("delimiter", "|")
    .csv(users_file)
)
# users_df.show()

### Join the appropriate tables in order to create one table in which the cube operator will be applied
# After having the Users table joined with the UserMovies table,
# we can join their result with the MoviesGenre table on mid column,
# to create the final table for the cube
final_joined = (
    user_movies_df.join(users_df, user_movies_df.userid == users_df.userid, how="inner")
    .drop(user_movies_df.userid)
    .join(movies_genre_df, user_movies_df.mid == movies_genre_df.mid, how="inner")
    .drop(movies_genre_df.mid)
)
# final_joined.show()


# Creating the cube for the final joined table on dimensions 'genre' and 'gender',
# aggregated data base on average of 'rating' column
final_cubed = (
    final_joined.cube(col('genre'), col('gender'))
    .agg(round(avg(col("rating")),2).alias("avg_rating"))
    .orderBy("genre", "gender")
)
final_cubed.show()

#### ----------------------------------------------------------- Query 1 -------------------------------------------------------------------
folder = 'cube-results'
filepath = path.join(folder,'{}')

# Calculating each Group By produced by the cube
# ----------------------------------------------------------------------------------------
# Group By 'genre'
gb_genre = final_cubed.filter(final_cubed.genre.isNotNull() & final_cubed.gender.isNull())
pd_gb_genre = gb_genre.toPandas()
pd_gb_genre.to_csv(filepath.format('Genre.csv'),index=False)
gb_genre.show(truncate=False)

# ----------------------------------------------------------------------------------------
# Group By 'gender'
gb_gender = final_cubed.filter(final_cubed.gender.isNotNull() & final_cubed.genre.isNull())
pd_gb_gender = gb_gender.toPandas()
pd_gb_gender.to_csv(filepath.format('Gender.csv'),index=False)
gb_gender.show(truncate=False)

# ----------------------------------------------------------------------------------------
# Group By ('genre','gender')
gb_genre_gender = final_cubed.filter(final_cubed.gender.isNotNull() & final_cubed.genre.isNotNull())
pd_gb_genre_gender = gb_genre_gender.toPandas()
pd_gb_genre_gender.to_csv(filepath.format('Genre_Gender.csv'),index=False)
gb_genre_gender.show(truncate=False)

# ----------------------------------------------------------------------------------------
# Group By 'none'
gb_none = final_cubed.filter(final_cubed.gender.isNull() & final_cubed.genre.isNull())
pd_gb_none = gb_none.toPandas()
pd_gb_none.to_csv(filepath.format('None.csv'),index=False)
gb_none.show(truncate=False)


#### ----------------------------------------------------------- Query 2 ---------------------------------------------------------------

# We will use the Group By ('genre','gender') 
female_ratings = gb_genre_gender.withColumnRenamed('avg_rating', 'female_avg_rating').filter(gb_genre_gender.gender == 'F')
male_ratings = gb_genre_gender.withColumnRenamed('avg_rating', 'male_avg_rating').filter(gb_genre_gender.gender == 'M')
joined = male_ratings.join(female_ratings, ['genre'])
joined.createOrReplaceTempView('GenreRatingsPerGender')

data = spark.sql('SELECT genre FROM GenreRatingsPerGender WHERE female_avg_rating > male_avg_rating')
data.show()


# ### ---------------------------------------------------------- Query 3 -------------------------------------------------------------
# select count() group by rating, the chosen genre is 'Adventure 
rating_per_genre = final_joined.where(col('genre') == 'Adventure').groupBy(col('rating')).count().withColumnRenamed("count","total_ratings").orderBy(col('rating'))
rating_per_genre.show()

ratings_df = rating_per_genre.toPandas()

labels =  ratings_df.rating
sizes = ratings_df.total_ratings

# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#F9E79F']
colors = ['#F3A935','#2586a4','#6ebe9f','#55596a','#c73558']
fig, ax = plt.subplots()

def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(sizes,labels=labels, colors=colors, autopct=lambda pct: func(pct, sizes),
                                  textprops=dict(color="w"))

ax.legend(wedges, labels,
          title="Ratings",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("Total count number per rating for genre 'Adventure'")

# plt.show()
plt.savefig('ratings_pie.png')

# Clear plt
plt.clf()

##################################################
# Chart bars

fig1 = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(labels, sizes, color = colors,
        width = 0.4)
 
plt.xlabel("Rating")
plt.ylabel("Total Number of Ratings")
plt.title("Total count number per rating for genre 'Adventure'")
plt.savefig('rating_bars.png')