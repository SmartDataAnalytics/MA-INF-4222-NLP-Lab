/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_TERM_SEARCH', {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      primaryKey: true
    },
    term: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    language: {
      type: DataTypes.TEXT,
      allowNull: true,
      defaultValue: 'en'
    },
    id_search_engine: {
      type: DataTypes.DOUBLE,
      allowNull: true,
      defaultValue: '1'
    },
    search_engine_features: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    id_search_type: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    metaquery: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    query_date: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    query_tot_resource: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    id_term: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    tot_results_returned: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: '0'
    }
  }, {
	  timestamps: false,
      tableName: 'HORUS_TERM_SEARCH'
  });
};
