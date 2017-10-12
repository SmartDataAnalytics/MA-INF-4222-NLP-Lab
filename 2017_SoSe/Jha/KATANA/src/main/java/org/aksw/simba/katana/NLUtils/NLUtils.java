package org.aksw.simba.katana.NLUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import org.aksw.simba.katana.model.RDFProperty;
import org.aksw.simba.katana.model.RDFResource;

import com.google.common.io.Files;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.dcoref.Mention;
import edu.stanford.nlp.hcoref.data.CorefChain.CorefMention;
import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

public class NLUtils {

	protected StanfordCoreNLP pipeline;

	public NLUtils() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma,ner, depparse,parse,natlog,openie, mention,dcoref");
		this.pipeline = new StanfordCoreNLP(props);
	}

	public Annotation getAnnotatedText(String text) {
		Annotation document = new Annotation(text);
		this.pipeline.annotate(document);
		return document;
	}

	public void corefResoultion(Annotation document) {
		System.out.print(document.toString());
		 Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
		 System.out.println(graph);
		
	}

	public List<CoreMap> filterSentences(Annotation document,
			Map<RDFProperty, ArrayList<RDFResource>> kbPropResourceMap) {
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		List<CoreMap> nlText = new ArrayList<CoreMap>();
		Set<RDFProperty> rp = kbPropResourceMap.keySet();
		for (RDFProperty ele : rp) {
			for (CoreMap sentence : sentences) {
				if (sentence.get(CoreAnnotations.TextAnnotation.class).contains(ele.getLabel())) {
					nlText.add(sentence);
				}
			}
		}
		return nlText;
	}

	public List<String> lemmatize(Annotation document) {
		List<String> lemmas = new LinkedList<String>();
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				lemmas.add(token.get(LemmaAnnotation.class));
			}
		}
		return lemmas;
	}

	public List<RelationTriple> getTriplesfromNL(List<CoreMap> sentences) {

		List<RelationTriple> triples = new ArrayList<>();
		for (CoreMap sentence : sentences) {
			triples.addAll(sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class));
		}
		
		return triples;
	}

	
}
