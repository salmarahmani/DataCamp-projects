# This script analyzes Netflix movie data to investigate whether movies are getting shorter over the years.
# It loads a CSV file containing Netflix data, filters out TV shows, and then examines movies with a duration
# of less than 60 minutes. The script assigns colors to different genre groups and creates a scatter plot
# showing movie duration against release year. Finally, it prompts a question based on the observation from
# the plot to determine whether movies are indeed getting shorter.


# Importing pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Start coding!

# Step 1: Load the CSV file and store as netflix_df
netflix_df = pd.read_csv('netflix_data.csv')

# Step 2: Filter the data to remove TV shows and store as netflix_subset
netflix_subset = netflix_df[netflix_df['type'] == 'Movie']

# Step 3: Investigate the Netflix movie data
netflix_movies = netflix_subset[["title", "country", "genre", "release_year", "duration"]]

# Step 4: Filter netflix_movies to find movies shorter than 60 minutes
short_movies = netflix_movies[netflix_movies['duration'] < 60]

# Inspect the result
print(short_movies.head())

# Step 5: Assign colors to genre groups and create a scatter plot
colors = []
for index, row in netflix_movies.iterrows():
    if row['genre'] == 'Children':
        colors.append('blue')
    elif row['genre'] == 'Documentaries':
        colors.append('green')
    elif row['genre'] == 'Stand-Up':
        colors.append('red')
    else:
        colors.append('gray')

# Initialize a matplotlib figure object
fig, ax = plt.subplots()

# Create a scatter plot for movie duration by release year
ax.scatter(netflix_movies['release_year'], netflix_movies['duration'], c=colors)

# Set labels and title
ax.set_xlabel("Release year")
ax.set_ylabel("Duration (min)")
ax.set_title("Movie Duration by Year of Release")

# Show the plot
plt.show()

# Step 6: Answer the question "Are we certain that movies are getting shorter?"
# Inspecting the plot can provide insights
answer = "no"  # Adjust based on the observation from the plot
