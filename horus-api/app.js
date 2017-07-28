const express = require('express'),
      epilogue = require('epilogue'),
	  bodyParser = require('body-parser'),
	  models = require('./models');

	
let app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

const APP_PORT = 3001;

// Add headers
app.use(function (req, res, next) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE'); // If needed
    res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,contenttype'); // If needed
    res.setHeader('Access-Control-Allow-Credentials', true); // If needed


    // Pass to next layer of middleware
    next();
});

app.use('/assets', express.static('assets'));

app.get('/', (req,res) => {
	res.json({test: 'hello'});
});

app.listen(APP_PORT, _ => {
	console.log('server started at port', APP_PORT);
});


// Initialize epilogue
epilogue.initialize({
  app: app,
  sequelize: models.sequelize
});

// Create REST resource
let termResource = epilogue.resource({
	model: models.HORUS_TERM_SEARCH,
	endpoints: ['/terms', '/terms/:id']
});

let resultTextResource = epilogue.resource({
	model: models.HORUS_SEARCH_RESULT_TEXT,
	endpoints: ['/result_text', '/result_text/:id']
});

let resultImageResource = epilogue.resource({
	model: models.HORUS_SEARCH_RESULT_IMG,
	endpoints: ['/result_image', '/result_image/:id']
});

