# horus-interface
	## Steps to run the project
	1. Copy the `horus.db` to desired path. (default is the root directory of horus-api)
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
	