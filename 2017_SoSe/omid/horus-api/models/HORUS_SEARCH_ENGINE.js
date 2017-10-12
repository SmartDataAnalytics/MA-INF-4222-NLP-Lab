/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_SEARCH_ENGINE', {
    id: {
      type: DataTypes.INTEGER,
      allowNull: true,
      primaryKey: true
    },
    name: {
      type: DataTypes.TEXT,
      allowNull: true
    }
  }, {
	  timestamps: false,
    tableName: 'HORUS_SEARCH_ENGINE'
  });
};
