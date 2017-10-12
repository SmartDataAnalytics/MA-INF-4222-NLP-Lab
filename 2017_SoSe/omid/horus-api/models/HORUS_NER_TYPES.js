/* jshint indent: 2 */

module.exports = function(sequelize, dataType) {
  return sequelize.define('HORUS_NER_TYPES', {
    id: {
      type: dataType.INTEGER,
      allowNull: true,
      primaryKey: true
    },
    type: {
      type: dataType.TEXT,
      allowNull: true
    },
    desc: {
      type: dataType.TEXT,
      allowNull: true
    }
  }, {
      tableName: 'HORUS_NER_TYPES',
      timestamps: false
  });
};
