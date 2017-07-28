/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_SEARCH_RESULT_IMG', {
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
    result_media_url: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_media_title: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_media_content_type: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_media_height: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_media_width: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_media_thumb_media_url: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    result_media_thumb_media_content_type: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    nr_faces: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_logos: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    filename: {
      type: DataTypes.TEXT,
      allowNull: true,
      defaultValue: '0'
    },
    nr_place_1: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_2: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_3: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_4: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_5: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_6: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_7: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_8: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_9: {
      type: DataTypes.INTEGER,
      allowNull: false,
      defaultValue: '0'
    },
    nr_place_10: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: '0'
    },
    processed: {
      type: DataTypes.INTEGER,
      allowNull: true,
      defaultValue: '0'
    }
  }, {
	  timestamps: false,
      tableName: 'HORUS_SEARCH_RESULT_IMG'
  });
};
