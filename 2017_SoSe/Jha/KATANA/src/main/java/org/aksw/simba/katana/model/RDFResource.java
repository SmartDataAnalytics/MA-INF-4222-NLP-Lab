package org.aksw.simba.katana.model;

import java.util.ArrayList;
import java.util.List;

import org.aksw.simba.katana.NLUtils.NLUtils;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

public class RDFResource {

	String uri;
	String kbLabel;
	ArrayList<String> lemma;
	NLUtils nl = new NLUtils();

	public String getKbLabel() {
		return kbLabel;
	}

	public void generateLemmafromLabel() {
		Annotation doc = nl.getAnnotatedText(kbLabel);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				lemma.add(token.get(LemmaAnnotation.class));
			}
		}
	}

	public void setKbLabel(String kbLabel) {
		this.kbLabel = kbLabel;
	}

	public RDFResource(String uri, String label) {

		this.uri = uri;
		this.kbLabel = label;
		this.lemma = new ArrayList<String>();
		this.generateLemmafromLabel();
		
	}

	public RDFResource(String uri) {

		this.uri = uri;

	}

	public String getUri() {
		return uri;
	}

	public void setUri(String uri) {
		this.uri = uri;
	}

	public ArrayList<String> getLabels() {
		return lemma;
	}

	public void setLabels(ArrayList<String> labels) {
		this.lemma = labels;
	}

}
