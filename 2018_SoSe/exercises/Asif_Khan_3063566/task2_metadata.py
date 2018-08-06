from rdflib import Namespace, Graph, Literal
from rdflib.namespace import FOAF, OWL, XSD, RDFS, DCTERMS, DOAP, DC


prov = Namespace('http://www.w3.org/ns/prov#')
dcat = Namespace('http://www.w3.org/ns/dcat#')
mexalgo = Namespace('http://mex.aksw.org/mex-algo#')
mexperf = Namespace('http://mex.aksw.org/mex-perf#')
mexcore = Namespace('http://mex.aksw.org/mex-core#')
this = Namespace('http://mex.aksw.org/examples/')

g = Graph()
# Create Binding
g.bind('dct',DCTERMS)
g.bind('owl',OWL)
g.bind('foaf',FOAF)
g.bind('xsd', XSD)
g.bind('rdfs', RDFS)
g.bind('doap', DOAP)
g.bind('dc', DC)
g.bind('prov', prov)
g.bind('dcat', dcat)
g.bind('mexalgo',mexalgo)
g.bind('mexperf',mexperf)
g.bind('mexcore',mexcore)
g.bind('this',this)

g.add((this.khan_task2,mexcore.Experiment, prov.Entity))
g.add((this.khan_task2,mexcore.ApplicationContext, prov.Entity))
g.add((this.khan_task2,RDFS.label, Literal('2255383')))
g.add((this.khan_task2,dct.date, Literal('2018-05-15',datatype=XSD.date)))
g.add((this.khan_task2,FOAF.givenName, Literal('Asif')))
g.add((this.khan_task2,FOAF.mbox, Literal('mak4086@gmail.com')))

#Configuration-1
g.add((this.configuration1,mexcore.ExperimentConfiguration, prov.Entity))
g.add((this.configuration1,prov.used, this.model1))
g.add((this.configuration1,prov.wasStartedBy, this.khan_task2))

#Configuration-2
g.add((this.configuration2,mexcore.ExperimentConfiguration, prov.Entity))
g.add((this.configuration1,prov.used, this.model2))
g.add((this.configuration1,prov.wasStartedBy, this.khan_task2))

#Configuration-3
g.add((this.configuration1,mexcore.ExperimentConfiguration, prov.Entity))
g.add((this.configuration1,prov.used, this.model2))
g.add((this.configuration1,prov.used, this.model1))
g.add((this.configuration1,prov.wasStartedBy, this.khan_task2))

#Configuration-4
g.add((this.configuration1,mexcore.ExperimentConfiguration, prov.Entity))
g.add((this.configuration1,prov.used, this.model4))
g.add((this.configuration1,prov.wasStartedBy, this.khan_task2))

g.add((this.test,mexcore.Test,prov.Entity))
g.add((this.test,RDFS.label,Literal('Test')))

g.add((this.hyerparameter_model1,mexalgo.HyperParameterCollection,prov.Entity))
g.add((this.hyerparameter1,RDFS.label,Literal('HyperParameterCollection')))
g.add((this.hyerparameter_model1,prov.hadMember,this.hyerparameter1))

g.add((this.hyerparameter_model2,mexalgo.HyperParameterCollection,prov.Entity))
g.add((this.hyerparameter_model2,RDFS.label,Literal('HyperParameterCollection')))
g.add((this.hyerparameter_model2,prov.hadMember,this.hyerparameter2))
g.add((this.hyerparameter_model2,prov.hadMember,this.hyerparameter3))
g.add((this.hyerparameter_model2,prov.hadMember,this.hyerparameter4))
g.add((this.hyerparameter_model2,prov.hadMember,this.hyerparameter5))


g.add((this.hyerparameter_model4,mexalgo.HyperParameterCollection,prov.Entity))
g.add((this.hyerparameter_model4,RDFS.label,Literal('HyperParameterCollection')))
g.add((this.hyerparameter_model4,prov.hadMember,this.hyerparameter6))
g.add((this.hyerparameter_model4,prov.hadMember,this.hyerparameter7))
g.add((this.hyerparameter_model4,prov.hadMember,this.hyerparameter8))



g.add((this.hyerparameter1,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter1,RDFS.label, Literal('alpha')))
g.add((this.hyerparameter1,DCTERMS.identifier, Literal('alpha')))
g.add((this.hyerparameter1,prov.value, Literal('1.0',datatype=XSD.float)))

g.add((this.hyerparameter2,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter2,RDFS.label, Literal('min_samples_split')))
g.add((this.hyerparameter2,DCTERMS.identifier, Literal('min_samples_split')))
g.add((this.hyerparameter2,prov.value, Literal('2',datatype=XSD.integer)))

g.add((this.hyerparameter3,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter3,RDFS.label, Literal('min_samples_leaf')))
g.add((this.hyerparameter3,DCTERMS.identifier, Literal('min_samples_leaf')))
g.add((this.hyerparameter3,prov.value, Literal('1',datatype=XSD.integer)))

g.add((this.hyerparameter4,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter4,RDFS.label, Literal('splitter')))
g.add((this.hyerparameter4,DCTERMS.identifier, Literal('splitter')))
g.add((this.hyerparameter4,prov.value, Literal('best')))

g.add((this.hyerparameter5,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter5,RDFS.label, Literal('criterion')))
g.add((this.hyerparameter5,DCTERMS.identifier, Literal('criterion')))
g.add((this.hyerparameter5,prov.value, Literal('gini')))

g.add((this.hyerparameter6,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter6,RDFS.label, Literal('max_iter')))
g.add((this.hyerparameter6,DCTERMS.identifier, Literal('max_iter')))
g.add((this.hyerparameter6,prov.value, Literal('100',datatype=XSD.integer)))

g.add((this.hyerparameter7,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter7,RDFS.label, Literal('penalty')))
g.add((this.hyerparameter7,DCTERMS.identifier, Literal('penalty')))
g.add((this.hyerparameter7,prov.value, Literal('l2')))


g.add((this.hyerparameter8,mexalgo.HyperParameter,prov.Entity))
g.add((this.hyerparameter8,RDFS.label, Literal('C')))
g.add((this.hyerparameter8,DCTERMS.identifier, Literal('C')))
g.add((this.hyerparameter8,prov.value, Literal('1.0',datatype=XSD.float)))

g.add((this.dataset1,mexcore.Dataset,prov.Entity))
g.add((this.dataset1,RDFS.label,Literal('Fake-News')))
g.add((this.dataset1,DCTERMS.landingPage,Literal('https://github.com/GeorgeMcIntire/fake_real_news_dataset')))

g.add((this.dataset2,mexcore.Dataset,prov.Entity))
g.add((this.dataset2,RDFS.label,Literal('Liar-Liar')))
g.add((this.dataset2,DCTERMS.landingPage,Literal('https://www.cs.ucsb.edu/william/data/liar_dataset.zip')))

g.add((this.dataset3,mexcore.Dataset,prov.Entity))
g.add((this.dataset3,RDFS.label,Literal('Fake-News+Liar-Liar')))
g.add((this.dataset3,DCTERMS.landingPage,Literal('https://www.cs.ucsb.edu/william/data/liar_dataset.zip')))
g.add((this.dataset3,DCTERMS.landingPage,Literal('https://github.com/GeorgeMcIntire/fake_real_news_dataset')))


g.add((this.cross_validation,mexcore.crossValidation,prov.Entity))
g.add((this.cross_validation,RDFS.label,Literal('cross validation')))
g.add((this.cross_validation,mexcore.folds,Literal('5',datatype=XSD.integer)))
g.add((this.cross_validation,mexcore.random_state,Literal('4222',datatype=XSD.integer)))

g.add((this.execution1,mexcore.ExecutionOverall,prov.Entity))
g.add((this.execution1,prov.generated,this.performance_measures1))
g.add((this.execution1,prov.used,this.test))
g.add((this.execution1,prov.used,this.hyerparameter_model1))
g.add((this.execution1,prov.used,this.model1))

g.add((this.execution2,mexcore.ExecutionOverall,prov.Entity))
g.add((this.execution2,prov.generated,this.performance_measures2))
g.add((this.execution2,prov.used,this.test))
g.add((this.execution2,prov.used,this.hyerparameter_model2))
g.add((this.execution2,prov.used,this.model2))

g.add((this.execution3,mexcore.ExecutionOverall,prov.Entity))
g.add((this.execution3,prov.generated,this.performance_measures3))
g.add((this.execution3,prov.used,this.test))
g.add((this.execution3,prov.used,this.model2))
g.add((this.execution3,prov.used,this.model3))

g.add((this.execution2,mexcore.ExecutionOverall,prov.Entity))
g.add((this.execution2,prov.generated,this.performance_measures4))
g.add((this.execution2,prov.used,this.test))
g.add((this.execution2,prov.used,this.hyerparameter_model4))
g.add((this.execution2,prov.used,this.model4))

g.add((this.performance_measures1,mexcore.PerformanceMeasure,prov.Entity))
g.add((this.performance_measures1,mexperf.precision,Literal('0.98',datatype=XSD.float)))
g.add((this.performance_measures1,mexperf.recall,Literal('0.76',datatype=XSD.float)))
g.add((this.performance_measures1,mexperf.accuracy,Literal('0.84',datatype=XSD.float)))
g.add((this.performance_measures1,prov.wasGeneratedBy,this.execution1))

g.add((this.performance_measures2,mexcore.PerformanceMeasure,prov.Entity))
g.add((this.performance_measures2,mexperf.precision,Literal('0.85',datatype=XSD.float)))
g.add((this.performance_measures2,mexperf.recall,Literal('0.81',datatype=XSD.float)))
g.add((this.performance_measures2,mexperf.accuracy,Literal('0.72',datatype=XSD.float)))
g.add((this.performance_measures2,prov.wasGeneratedBy,this.execution2))

g.add((this.performance_measures3,mexcore.PerformanceMeasure,prov.Entity))
g.add((this.performance_measures3,mexperf.precision,Literal('0.88',datatype=XSD.float)))
g.add((this.performance_measures3,mexperf.recall,Literal('0.81',datatype=XSD.float)))
g.add((this.performance_measures3,mexperf.accuracy,Literal('0.74',datatype=XSD.float)))
g.add((this.performance_measures3,prov.wasGeneratedBy,this.execution3))


g.add((this.performance_measures4,mexcore.PerformanceMeasure,prov.Entity))
g.add((this.performance_measures4,mexperf.precision,Literal('0.98',datatype=XSD.float)))
g.add((this.performance_measures4,mexperf.recall,Literal('0.80',datatype=XSD.float)))
g.add((this.performance_measures4,mexperf.accuracy,Literal('0.81',datatype=XSD.float)))
g.add((this.performance_measures4,prov.wasGeneratedBy,this.execution4))


g.add((this.model1,mexalgo.Algorithm,prov.Entity))
g.add((this.model1,RDFS.label,Literal('MultinomialNB')))
g.add((this.model1,DCTERMS.identifier,Literal('MultinomialNB')))
g.add((this.model1,mexalgo.hasHyperParameter,this.hyerparameter1))

g.add((this.model2,mexalgo.Algorithm,prov.Entity))
g.add((this.model2,RDFS.label,Literal('DecisionTree')))
g.add((this.model2,DCTERMS.identifier,Literal('DecisionTree')))
g.add((this.model2,mexalgo.hasHyperParameter,this.hyerparameter2))
g.add((this.model2,mexalgo.hasHyperParameter,this.hyerparameter3))
g.add((this.model2,mexalgo.hasHyperParameter,this.hyerparameter4))
g.add((this.model2,mexalgo.hasHyperParameter,this.hyerparameter5))

g.add((this.model4,mexalgo.Algorithm,prov.Entity))
g.add((this.model4,RDFS.label,Literal('LogisticRegression')))
g.add((this.model4,DCTERMS.identifier,Literal('LogisticRegression')))
g.add((this.model4,mexalgo.hasHyperParameter,this.hyerparameter6))
g.add((this.model4,mexalgo.hasHyperParameter,this.hyerparameter7))
g.add((this.model4,mexalgo.hasHyperParameter,this.hyerparameter8))

with open('task2_metadata.ttl','wb') as f:
	f.write(g.serialize(format='turtle'))