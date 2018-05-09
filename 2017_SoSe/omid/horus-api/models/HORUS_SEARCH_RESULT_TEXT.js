/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_SEARCH_RESULT_TEXT', {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      primaryKey: true
    },
    id_term_search: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    id_ner_type: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    search_engine_resource_id: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_seq: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    result_url: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_title: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_description: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_html_text: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    text_1_klass: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    text_2_klass: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    text_3_klass: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    text_4_klass: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    text_5_klass: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    result_title_en: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_description_en: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    processed: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: '0'
    }
  }, {
      tableName: 'HORUS_SEARCH_RESULT_TEXT',
	  timestamps: false
  });
};
