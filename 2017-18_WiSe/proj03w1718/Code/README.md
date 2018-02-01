# Web Ranking
## In order to run the Code
1. please make sure that you replace the path in `trustlist = pd.read_csv("/yourpath/newsCorpora.csv", sep='\t', names = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP" ])`
2. the size of file `newsCorpora.csv` is over 100MB and the push is always rejected, please download it from https://archive.ics.uci.edu/ml/datasets/News+Aggregator
3. There are two scripts `Read url list by _body_.ipynb` and `Read url list by _body_.ipynb`
```
cd horus-api
npm install
cd ../horus-web
npm install
```
3. Configure the parameters in `horus-api/config.js` and `horus-web/src/config.js`
4. run the api
```
cd horus-api
node app.js
```
5. Run the web interface
```
cd horus-web
npm start
```

