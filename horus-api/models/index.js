const Sequelize = require('sequelize');


const db = new Sequelize('horus.db', '', '', {
	dialect: 'sqlite',
	storage: './horus.db'
});

const HORUS_NER_TYPES = require('./HORUS_NER_TYPES')(db, Sequelize);
const HORUS_SEARCH_ENGINE = require('./HORUS_SEARCH_ENGINE')(db, Sequelize);
const HORUS_SEARCH_RESULT_IMG = require('./HORUS_SEARCH_RESULT_IMG')(db, Sequelize);
const HORUS_SEARCH_RESULT_TEXT = require('./HORUS_SEARCH_RESULT_TEXT')(db, Sequelize);
const HORUS_SEARCH_TYPES = require('./HORUS_SEARCH_TYPES')(db, Sequelize);
const HORUS_SENTENCES = require('./HORUS_SENTENCES')(db, Sequelize);
const HORUS_TERM = require('./HORUS_TERM')(db, Sequelize);
const HORUS_TERM_SEARCH = require('./HORUS_TERM_SEARCH')(db, Sequelize);




module.exports = {
	HORUS_NER_TYPES,
	HORUS_SEARCH_ENGINE,
	HORUS_SEARCH_RESULT_IMG,
	HORUS_SEARCH_RESULT_TEXT,
	HORUS_SEARCH_TYPES, 
	HORUS_SENTENCES, 
	HORUS_TERM, 
	HORUS_TERM_SEARCH,
	sequelize: db
}
