/* jshint indent: 2 */

module.exports = function(sequelize, DataTypes) {
  return sequelize.define('HORUS_SENTENCES', {
    id: {
      type: DataTypes.INTEGER,
      allowNull: false,
      primaryKey: true
    },
    sentence_has_NER: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    sentence: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    same_tokenization_nltk: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    same_tokenization_stanford: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    same_tokenization_tweetNLP: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    corpus_name: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    corpus_tokens: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    corpus_ner_y: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    corpus_pos_y: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    corpus_pos_uni_y: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_nltk_tokens: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_nltk_ner: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_nltk_pos: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_nltk_pos_universal: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_nltk_compounds: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_stanford_tokens: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_stanford_ner: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_stanford_pos: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_stanford_pos_universal: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_stanford_compounds: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_tweetNLP_tokens: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_tweetNLP_ner: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_tweetNLP_pos: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_tweetNLP_pos_universal: {
      type: DataTypes.TEXT,
      allowNull: true
    },
    annotator_tweetNLP_compounds: {
      type: DataTypes.TEXT,
      allowNull: true
    }
  }, {
	  timestamps: false,
      tableName: 'HORUS_SENTENCES'
  });
};
