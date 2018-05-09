# Web Ranking
## In order to run the Code
1. please make sure that you replace the path in `trustlist = pd.read_csv("/yourpath/newsCorpora.csv", sep='\t', names = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP" ])`
2. The size of file `newsCorpora.csv` is over 100MB (which might be too big for uploading to github) and the push is always rejected, please download it from https://archive.ics.uci.edu/ml/datasets/News+Aggregator
3. I have merged the original two scraping scripts together with the testing script, thus now we have a single script which will do all works. The first half scrapes and creates data files. The former one get contents under `<body>` tag, the second one finds all paragraph tags `<p>`. Either of these 2 ways cost arround FOUR hours to generate all data I need to train and test, with size of 1000 instances.
4. Now I can upload all datasets I scraped and tested in the project, thanks to recommend from Diego. All of the data are stored in `Data` folder.
5. After the data scraping scripts finish working, there will be totally 8 data sets created (4 sets each). Then continue to the second half of the script(remember to check the path) to see the results
6. The tutorial To extract content from a URL can be found here: https://www.dataquest.io/blog/web-scraping-tutorial-python/
7. To learn how to detect fake news with Scikit-Learn: https://www.datacamp.com/community/tutorials/scikit-learn-fake-news
