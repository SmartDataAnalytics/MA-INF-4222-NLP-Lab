# Tools

----



## Folder overview

### Code and Notebooks

| File/Folder                                                | short description                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [EmojiCounting.ipynb](EmojiCounting.ipynb)                 | short notebook to count emoji occurences from train logfiles |
| [Emoji_Distance.py](Emoji_Distance.py)                     | provides functions to convert emojis to sentiment vectors and to find closest Emojis to given sentiment vectors |
| [Evaluation_with_csv.ipynb](Evaluation_with_csv.ipynb)     | test predictions of merged Approach                          |
| [Preprocessing.ipynb](Preprocessing.ipynb)                 | preprocess table with Emoji Descriptions                     |
| [User_Interface.ipynb](User_Interface.ipynb)               | Main User interface containing merged approach with predictions. Needs a pretrained pipeline contained in [./clf/](./clf/) (for instructions see there) |
| [emoji_plotting.ipynb](emoji_plotting.ipynb)               | plot sentiment space of emojis                               |
| [emoji_table.ipynb](emoji_table.ipynb)                     | just playing around with the emoji description table         |
| [kmeans_on_Emojis.ipynb](kmeans_on_Emojis.ipynb)           | here we experimented with emoji labelings based on clusters in sentiment space, but didn't investigate that approach further |
| [sklearn_doc2vec.py](sklearn_doc2vec.py)                   | wrapper to integrate doc2vec as vectorizer into a sklearn pipeline |
| [stream_language_detector.py](stream_language_detector.py) | little helper script for data preprocessing                  |
| [twitter2messages.sh](twitter2messages.sh)                 | little helper script for data preprocessing                  |
| [json_stream_filter](json_stream_filter)                   | helper tools for filtering json streams (used for preprocessing twitter data) |

### Data Files

| File/Folder                                                  | short description                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [emoji_counts](emoji_counts)                                 | folder containing logfiles of emoji usage, used by [EmojiCounting.ipynb](EmojiCounting.ipynb) |
| [emoji-data.txt](emoji-data.txt)                             | official emoji specifications                                |
| [emoji-list.txt](emoji-list.txt)                             | exported sequence of emojis                                  |
| [emoji_descriptions.csv](emoji_descriptions.csv), [emoji_descriptions_preprocessed.csv](emoji_descriptions_preprocessed.csv) | processed emoji descriptions                                 |

