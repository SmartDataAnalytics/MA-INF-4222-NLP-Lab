/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_TERM', {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      primaryKey: true
    },
    term: {
      type: DataTypes.TEXT,
      allowNull: false
    }
  }, {
    tableName: 'HORUS_TERM'
  });
};
