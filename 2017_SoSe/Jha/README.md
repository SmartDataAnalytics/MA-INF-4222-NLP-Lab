# KATANA
KATANA is a tool (in progress) that helps us add labels to RDF knowledge base.

There are three packages at the moment:
* KBUtils handle the data required from the Sparql Endpoint.
* NLUtils take care of the processing of Natural Language Text. ComparisonUtils class takes care of the comparison
* Models: Our definition of RDF Resource and property.


## Usage
* Use mavens command to package and clean the project.
* Use the SparqlHandler generateSampleDataset by feeding it a list of classes. (Its a one time process and takes some time). This will result in a generation of Natural Language text.
* Once you the Text is in place run the ComparisonUtils class.

_Important:_  Since Lemma matching is very limited. There are cases when the resources with functional properties (from the KB) do not occur in text or not parsed correctly by OpenIE, resulting in empty results. The only solution is to make sure that NL text is large/diverse enough. One easy hack is to check the list of resources associated with the properties and use their abstracts.


## System Architecture
![alt text](https://github.com/Kunal-Jha/MA-INF-4222-Lab-SoSe2017/blob/master/Jha/KATANA/SysArc-1.jpg)
