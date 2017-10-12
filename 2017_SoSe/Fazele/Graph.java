package com.company;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import gr.forth.ics.graph.Edge;
import gr.forth.ics.graph.Node;
import gr.forth.ics.graph.PrimaryGraph;
import gr.forth.ics.graph.algo.KCoreDecomposition;

import java.util.*;

public class Graph implements Cloneable {
    private com.company.Graph internalGraph;
    private static int windowSize = 4;
    private Map<String,Map<String,Integer>> weightedAdjGraph = new HashMap<>();
    private static interviews.graphs.Graph g1;
    private static List<String> stemmedTermList = new ArrayList<>();

    protected Object clone() throws CloneNotSupportedException {

        return super.clone();
    }

    public interviews.graphs.Graph G1() {
        if (g1 == (null))
            g1 = new interviews.graphs.Graph(stemmedTermList.size());
        return g1;
    }

    public void renew_G1(){
        g1 = null;
    }
    public void renew_stemmedtermlist(){
        stemmedTermList = new ArrayList<>();
    }

    public Graph(){

    }


    public Map<String,Map<String,Integer>> graphCreator() {
        gr.forth.ics.graph.Graph g = new PrimaryGraph();
        Node[] n = g.newNodes(stemmedTermList.toArray());
        if (weightedAdjGraph.isEmpty()) {
            Map<String, Integer> neighboursMap;
            for (int i = 0; i < stemmedTermList.size(); i++) {
                if (i + windowSize < stemmedTermList.size()) {
                    List<String> stemmedTermSubList = new ArrayList<>(stemmedTermList.subList(i, i + windowSize));
                    for (int i1 = i; i1 < i + windowSize; i1++) {
                        String currentStem = stemmedTermList.get(i1);
                        if (!currentStem.equals(null)) {
                            if (weightedAdjGraph.containsKey(currentStem))
                                neighboursMap = new HashMap<>(weightedAdjGraph.get(currentStem));
                            else
                                neighboursMap = new HashMap<>();

                            for (int i2 = 0; i2 < windowSize; i2++) {
                                String sameWindowStem = stemmedTermSubList.get(i2); //this stem is in the same window that currentStem is.
                                if (!currentStem.equals(sameWindowStem) && !sameWindowStem.equals("NA") && !currentStem.equals("NA")) {

                                    if (!neighboursMap.isEmpty()) {
                                        if (neighboursMap.containsKey(sameWindowStem)) {
                                            neighboursMap.put(sameWindowStem, neighboursMap.get(sameWindowStem) + 1);
                                            G1().raiseDegree(stemmedTermList.indexOf(currentStem), stemmedTermList.indexOf(sameWindowStem));

                                        } else {
                                            neighboursMap.put(sameWindowStem, 1);
                                            G1().addEdge(stemmedTermList.indexOf(currentStem), stemmedTermList.indexOf(sameWindowStem), 1);
                                        }
                                    } else {
                                        neighboursMap.put(sameWindowStem, 1);
                                        G1().addEdge(stemmedTermList.indexOf(currentStem), stemmedTermList.indexOf(sameWindowStem), 1);
                                    }
                                }
                            }
                            if (!currentStem.equals("NA")) {
                                weightedAdjGraph.put(currentStem, neighboursMap);
                            }
                        } else
                            break;
                    }
                } else
                    break;
            }
        }
        interviews.graphs.KCore kc = new interviews.graphs.KCore(G1());
        kc.computeWeighted();
        return weightedAdjGraph;
    }


    public Map<String,Map<String,Integer>> getWeightedAdjGraph (){
        return  weightedAdjGraph;
    }


    public Map<String,Map<String,Integer>> KCoreNodesExtractor(SingletoneKCore singletoneKCore,String primaryVertex,String vertex,Map<String,Map<String,Integer>>kcoreNodesMap,List<String> visited){
        singletoneKCore.KC(this);
        //KC();
        visited.add(vertex);
        //int i = 0;
        Map<String,Integer> neighboursMap = new HashMap<>();
        try {
            neighboursMap = new HashMap<>(weightedAdjGraph.get(vertex));
        }catch (Exception e){
            String e1 = vertex;
        }
        Map<String,Integer> map = new HashMap<>(computeVertexDegree());
        Map<String,Integer> partialNeighbour = new HashMap<>();
        for (String s: neighboursMap.keySet()){
                if (singletoneKCore.KC(this).core(stemmedTermList.indexOf(s)) == singletoneKCore.KC(this).core(stemmedTermList.indexOf(primaryVertex))
                        || singletoneKCore.KC(this).core(stemmedTermList.indexOf(s)) > singletoneKCore.KC(this).core(stemmedTermList.indexOf(primaryVertex))){
                    partialNeighbour.put(s,neighboursMap.get(s));
                    if (!visited.contains(s)) {
                        KCoreNodesExtractor(singletoneKCore,primaryVertex,s, kcoreNodesMap, visited);

                    }
                }
        }
        kcoreNodesMap.put(vertex,partialNeighbour);
        return kcoreNodesMap;
    }



    public void setStemmedTermList_English(String sentence) {
        String tagged = "";
        Spliter spliter = new Spliter();
        Stemmer stanfordStemmer = new Stemmer();
        MaxentTagger tagger = new MaxentTagger("/Users/fazeletavakoli/IdeaProjects/NLPLabFirstProject/POSTaggerModels/models/english-left3words-distsim.tagger");
        tagged = tagger.tagString(sentence);
        spliter.sentenceSplitter(tagged);
        for (String s : spliter.getSpliter()) {
            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) == '_') {
                    String wordSubString = s.substring(i + 1, s.length());
                    wordSubString = wordSubString.trim();
                    if (wordSubString.equals("JJ") || wordSubString.equals("JJR")
                            || wordSubString.equals("JJS") ||
                            wordSubString.equals("NN") || wordSubString.equals("NNS") ||
                            wordSubString.equals("NNP") || wordSubString.equals("NNPS")) {
                        String s1 = stanfordStemmer.stem(s.substring(0, i));
                        stemmedTermList.add(s1);
                    }
                }
            }
        }
        int a = 0 ;
    }

    public List<String> getStemmedTermList_English(){

        return stemmedTermList;
    }


    public double getDensity(Map<String,Map<String,Integer>> map){
        double density = 0;
        int numberOfEdges = 0;
        int totalNumberOfEdges = (map.size())*(map.size()-1);
        //Map<String,Integer> neighbourMap = new HashMap<>();
        for (String s : map.keySet()){
            numberOfEdges += map.get(s).size();
        }
        density = (double)(numberOfEdges/2) / totalNumberOfEdges;
        if (Double.isNaN(density)){
            int aa = 1;
        }
        return density;
    }

    public HashMap sortByValues(Map<? extends Object, ? extends Object> map) {
        List list = new LinkedList(map.entrySet());
        // Defined Custom Comparator here
        Collections.sort(list, new Comparator() {
            public int compare(Object o1, Object o2) {
                return ((Comparable) ((Map.Entry) (o1)).getValue())
                        .compareTo(((Map.Entry) (o2)).getValue());
            }
        });
        Collections.reverse(list);

        HashMap sortedHashMap = new LinkedHashMap();
        for (Iterator it = list.iterator(); it.hasNext(); ) {
            Map.Entry entry = (Map.Entry) it.next();
            sortedHashMap.put(entry.getKey(), entry.getValue());
        }
        return sortedHashMap;
    }

    //weighted degree
    public Map<String,Integer> computeVertexDegree(){
        Map<String,Integer> vertexTotalWeightHash = new HashMap<>();
        for (String vertex: weightedAdjGraph.keySet()){
            int totalWeight = 0;
            for(String neighbour: weightedAdjGraph.get(vertex).keySet()){
                totalWeight += weightedAdjGraph.get(vertex).get(neighbour);
            }
            vertexTotalWeightHash.put(vertex,totalWeight);
        }
        return vertexTotalWeightHash;
    }


    public void nodesRemover(Map<String,Map<String,Integer>> higherKCoreMap){
            for (String s: higherKCoreMap.keySet()){
                if (weightedAdjGraph.containsKey(s)) {
                    Map<String, Integer> neighboursMap = new HashMap<>(weightedAdjGraph.get(s));
                    weightedAdjGraph.remove(s);
                    //removing vertex s from interviews.graphs.Graph g1
                    for (String neighbour : neighboursMap.keySet()) {
                        interviews.graphs.Edge removedEdge = new interviews.graphs.Edge(stemmedTermList.indexOf(s),
                                stemmedTermList.indexOf(neighbour), neighboursMap.get(neighbour));
                        G1().removeEdge(removedEdge);
                    }

                    for (String s1 : neighboursMap.keySet()) {
                        weightedAdjGraph.get(s1).remove(s);

                    if (weightedAdjGraph.get(s1).isEmpty()) {
                        weightedAdjGraph.remove(s1);
                        int index = stemmedTermList.indexOf(s1);
                        stemmedTermList.set(index, "NA");
                    }
                    }
                }


            }
            for (String s: higherKCoreMap.keySet()){
                while(stemmedTermList.contains(s)) {
                    int index = stemmedTermList.indexOf(s);
                    stemmedTermList.set(index, "NA");
                }
            }

    }

    public int computeElbow(LinkedHashMap<Integer,Double> level_density) {
        boolean isTrue = false;
        boolean isTrue2 = false; //for checking if distancesList contains more than one element with maximum value
        double lineSlope = 0;
        double YIntercept = 0;
        int firstPoint_X = 0;
        double firstPoint_Y = 0;
        int lastPoint_X = 0;
        double lastPoint_Y = 0;
        int point_X = 0;
        double point_Y = 0;
        double maxDistance = 0;
        double point_lineDistance = 0;
        int XElbow = 0 ;
        List<Double> distancesList = new ArrayList<>();
        for (Integer i : level_density.keySet()) {
            if (!isTrue) {
                firstPoint_X = i;
                firstPoint_Y = level_density.get(i);
                isTrue = true;
            }

            lastPoint_X = i;
            lastPoint_Y = level_density.get(i);
        }
        if (level_density.size() > 2) {
            lineSlope = (lastPoint_Y - firstPoint_Y) / (lastPoint_X - firstPoint_X);
            YIntercept = lastPoint_Y - (lineSlope * lastPoint_X);
            for (Integer i : level_density.keySet()) {
                if (i != 1 && i != level_density.size()) {
                    point_X = i;
                    point_Y = level_density.get(i);
                    //computing distance between a point and the line
                    point_lineDistance = Math.abs((lineSlope * point_X) - point_Y + YIntercept) / (Math.sqrt(Math.pow(lineSlope, 2) + 1));
                    distancesList.add(point_lineDistance);
                }
            }
            maxDistance = Collections.max(distancesList);
            List distancesList_Copy = new ArrayList(distancesList);
            distancesList_Copy.set(distancesList_Copy.indexOf(maxDistance), -1);
            if (distancesList_Copy.contains(maxDistance)) {
                isTrue2 = true;
            }
            if (isTrue2) {
                XElbow = firstPoint_X;
            }
            else
                XElbow = distancesList.indexOf(maxDistance) + 2;
        }
        else {
            if (firstPoint_Y > lastPoint_Y)
                XElbow = firstPoint_X;
            else
                XElbow = lastPoint_X;
        }
        return XElbow;
    }

}
