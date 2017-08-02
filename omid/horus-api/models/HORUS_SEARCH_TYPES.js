/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_SEARCH_TYPES', {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      primaryKey: true
    },
    desc: {
      type: DataTypes.TEXT,
      allowNull: false
    }
  }, {
	  timestamps: false,
      tableName: 'HORUS_SEARCH_TYPES'
  });
};
