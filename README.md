# world-cup-2026
This project is for the submission of the [FIFA World Cup 2026 Prediction Competition on DataCamp.](https://app.datacamp.com/learn/competitions/world-cup-prediction)

## Datasets
### 1. Matches
To download the dataset of all international matches, run `scrape_data.ipynb` in the `notebooks` directory.
The data is taken from a compiled set from user `martj42` on [GitHub](https://github.com/martj42/international_results)
It is also available for downloading on [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Note: If the full data is downloaded from Kaggle, then filter the date accordingly

### 2. Team Records (Head-2-Head stats)


### 3. FIFA Rankings
This data was taken from [FIFA](https://inside.fifa.com/fifa-world-ranking/men). Since FIFA does not have public APIs, and we cannot use `requests` or traditional scraping methods, the data was extracted manually and compiled using Gemini (or any other genAI chatbot).

### 4. Elo Ratings
This data was taken from [World Football Elo Ratings](https://www.eloratings.net). Although Elo Ratings does have one `.tsv` file that can be called using requests, the dataset lacks column headers to know what the values stand for. Because of that I decided to also manually extract and compile this data using Claude (or any other genAI chatbot). 
- Note: Testing this with Gemini, it generated a previous version of the dataset and not the most recent one so I decided to switch to Claude.



