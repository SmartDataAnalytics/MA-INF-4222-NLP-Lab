# Web Ranking
## In order to run the project
1. please make sure to replace the path in `trustlist = pd.read_csv("/home/xiaotianzhou/Downloads/newsCorpora.csv", sep='\t', names = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP" ])`
2. Install dependencies
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

